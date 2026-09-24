#!/usr/bin/env python3
"""
ogad.py — OGA Daemon: inference NPU XDNA2 via serveur named pipe.

Architecture:
  llama.cpp → ggml-xdna2 (C++) → named pipe → ogad.py → OGA → RyzenAI → NPU

  Protocol (JSON sur pipe nomme \\\\.\\pipe\\ogad):
  Request:  {"cmd":"load","model":"C:/models/..."}
  Response: {"status":"ok","model_id":"..."}

  Request:  {"cmd":"generate","model_id":"...","prompt":"Hello","max_tokens":50,"temperature":0.7}
  Response: {"status":"ok","text":"...","tokens":[...],"tps":70.9,"ttft_ms":1158}

  Request:  {"cmd":"ping"}
  Response: {"status":"ok"}

Usage:
  python ogad.py                          # demarre le serveur
  python ogad.py --model C:/models/...    # precharge un modele
  python ogad.py --pipe ogad_test         # pipe personnalise
"""
import os, sys, json, time, struct, threading, traceback
import win32pipe, win32file, win32event, pywintypes

# OGA import (avec DLL path)
OGA_AVAILABLE = False
try:
    ryzen_paths = [
        r"C:\Program Files\RyzenAI\1.7.1\deployment",
        r"C:\Program Files\RyzenAI\1.7.1b\deployment",
    ]
    for p in ryzen_paths:
        if os.path.isdir(p):
            os.add_dll_directory(p)
    import onnxruntime_genai as og
    OGA_AVAILABLE = True
except ImportError as e:
    print(f"[ogad] WARNING: OGA non disponible: {e}")

# ---------------------------------------------------------------------------
# Models cache
# ---------------------------------------------------------------------------
class ModelCache:
    """Cache les modeles OGA charges."""
    def __init__(self):
        self.models = {}       # model_id -> (model, tokenizer, config)
        self.next_id = 0
    
    def load(self, model_path: str) -> dict:
        """Charge un modele OGA depuis un dossier."""
        if not OGA_AVAILABLE:
            return {"status": "error", "message": "OGA not installed"}
        
        if not os.path.isdir(model_path):
            return {"status": "error", "message": f"Model not found: {model_path}"}
        
        model_id = f"m{self.next_id}"
        self.next_id += 1
        
        try:
            t0 = time.perf_counter()
            model = og.Model(model_path)
            tokenizer = og.Tokenizer(model)
            dt = (time.perf_counter() - t0) * 1000
            
            self.models[model_id] = {
                "model": model,
                "tokenizer": tokenizer,
                "path": model_path,
                "loaded_ms": dt,
            }
            
            return {
                "status": "ok",
                "model_id": model_id,
                "loaded_ms": dt,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def generate(self, model_id: str, prompt: str, max_tokens: int = 50,
                 temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9) -> dict:
        """Genere du texte depuis un modele charge."""
        if model_id not in self.models:
            return {"status": "error", "message": f"Model {model_id} not loaded"}
        
        entry = self.models[model_id]
        model = entry["model"]
        tokenizer = entry["tokenizer"]
        
        try:
            # Encode prompt
            input_tokens = tokenizer.encode(prompt)
            
            # Create generator params
            params = og.GeneratorParams(model)
            params.set_search_options(
                max_length=max_tokens + len(input_tokens),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            
            # Create generator
            generator = og.Generator(model, params)
            generator.append_tokens(input_tokens)
            
            # Generate
            t0 = time.perf_counter()
            n_tokens = 0
            output_tokens = []
            
            while not generator.is_done() and n_tokens < max_tokens:
                generator.generate_next_token()
                n_tokens += 1
            
            dt = (time.perf_counter() - t0) * 1000
            tps = (n_tokens * 1000.0 / dt) if dt > 0 else 0
            
            # Get output tokens
            full_seq = generator.get_sequence(0)
            # Extract only generated tokens (after prompt)
            gen_tokens = full_seq[len(input_tokens):] if len(full_seq) > len(input_tokens) else full_seq
            
            # Decode
            output_text = tokenizer.decode(gen_tokens) if hasattr(tokenizer, 'decode') else tokenizer.decode(output_tokens) if output_tokens else ""
            
            # Try decoding via tokenizer
            try:
                output_text = tokenizer.decode(gen_tokens)
            except:
                output_text = f"[{len(gen_tokens)} tokens generated]"
            
            # Convert numpy types to native Python for JSON serialization
            token_list = [int(t) for t in gen_tokens] if hasattr(gen_tokens, '__iter__') else []
            
            return {
                "status": "ok",
                "text": str(output_text),
                "tokens": token_list,
                "n_tokens": int(n_tokens),
                "tps": float(round(tps, 1)),
                "infer_ms": float(round(dt, 1)),
                "prompt_tokens": int(len(input_tokens)),
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def unload(self, model_id: str) -> dict:
        """Decharge un modele."""
        if model_id in self.models:
            del self.models[model_id]
            return {"status": "ok"}
        return {"status": "error", "message": f"Model {model_id} not found"}
    
    def list_models(self) -> dict:
        """Liste les modeles charges."""
        info = {}
        for mid, entry in self.models.items():
            info[mid] = {
                "path": entry["path"],
                "loaded_ms": entry["loaded_ms"],
            }
        return {"status": "ok", "models": info}


# ---------------------------------------------------------------------------
# Named pipe server
# ---------------------------------------------------------------------------
PIPE_NAME = r"\\.\pipe\ogad"
PIPE_BUFFER = 65536
PIPE_TIMEOUT = 30000  # 30s

class PipeServer:
    """Serveur named pipe multi-thread."""
    
    def __init__(self, pipe_name: str = PIPE_NAME):
        self.pipe_name = pipe_name
        self.cache = ModelCache()
        self.running = False
        self.threads = []
    
    def start(self):
        """Demarre le serveur (bloquant)."""
        self.running = True
        print(f"[ogad] OGA Daemon v1.0")
        print(f"[ogad] Pipe: {self.pipe_name}")
        print(f"[ogad] OGA: {'DISPONIBLE' if OGA_AVAILABLE else 'NON INSTALLE'}")
        print(f"[ogad] PID: {os.getpid()}")
        print(f"[ogad] En attente de connexions...")
        print()
        
        while self.running:
            try:
                # Create pipe instance
                pipe = win32pipe.CreateNamedPipe(
                    self.pipe_name,
                    win32pipe.PIPE_ACCESS_DUPLEX,
                    win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
                    win32pipe.PIPE_UNLIMITED_INSTANCES,
                    PIPE_BUFFER, PIPE_BUFFER,
                    PIPE_TIMEOUT,
                    None
                )
                
                # Wait for client
                win32pipe.ConnectNamedPipe(pipe, None)
                
                # Handle client in thread
                t = threading.Thread(target=self.handle_client, args=(pipe,), daemon=True)
                t.start()
                self.threads.append(t)
                
            except KeyboardInterrupt:
                print("\n[ogad] Arret demande")
                self.running = False
                break
            except Exception as e:
                print(f"[ogad] Erreur serveur: {e}")
                if not self.running:
                    break
    
    def stop(self):
        self.running = False
    
    def handle_client(self, pipe):
        """Gere une connexion client."""
        try:
            data = b""
            while self.running:
                # Read message (4-byte length prefix + JSON)
                try:
                    result, raw = win32file.ReadFile(pipe, 4)
                    if result != 0:
                        break
                    msg_len = struct.unpack('<I', raw)[0]
                    
                    if msg_len > 1024 * 1024:  # max 1MB
                        break
                    
                    result, raw = win32file.ReadFile(pipe, msg_len)
                    if result != 0:
                        break
                    
                    data = raw.decode('utf-8')
                    
                except pywintypes.error as e:
                    if e.winerror == 109:  # ERROR_BROKEN_PIPE
                        break
                    if e.winerror == 234:  # ERROR_MORE_DATA
                        continue
                    break
                
                # Parse request
                try:
                    req = json.loads(data)
                except json.JSONDecodeError:
                    resp = {"status": "error", "message": "Invalid JSON"}
                    self.send_response(pipe, resp)
                    break
                
                # Process command
                cmd = req.get("cmd", "")
                resp = self.process_command(cmd, req)
                
                # Send response
                self.send_response(pipe, resp)
                
                # Exit on 'exit' command
                if cmd == "exit":
                    break
                    
        except Exception as e:
            print(f"[ogad] Erreur client: {e}")
            traceback.print_exc()
        finally:
            try:
                win32pipe.DisconnectNamedPipe(pipe)
            except:
                pass
    
    def process_command(self, cmd: str, req: dict) -> dict:
        """Traite une commande."""
        if cmd == "ping":
            return {"status": "ok", "oga": OGA_AVAILABLE}
        
        elif cmd == "load":
            return self.cache.load(req.get("model", ""))
        
        elif cmd == "generate":
            return self.cache.generate(
                model_id=req.get("model_id", ""),
                prompt=req.get("prompt", ""),
                max_tokens=req.get("max_tokens", 50),
                temperature=req.get("temperature", 0.7),
                top_k=req.get("top_k", 50),
                top_p=req.get("top_p", 0.9),
            )
        
        elif cmd == "unload":
            return self.cache.unload(req.get("model_id", ""))
        
        elif cmd == "list":
            return self.cache.list_models()
        
        elif cmd == "exit":
            self.running = False
            return {"status": "ok", "message": "shutting down"}
        
        elif cmd == "stats":
            return {
                "status": "ok",
                "models_loaded": len(self.cache.models),
                "pid": os.getpid(),
                "oga_available": OGA_AVAILABLE,
            }
        
        else:
            return {"status": "error", "message": f"Unknown command: {cmd}"}
    
    def send_response(self, pipe, resp: dict):
        """Envoie une reponse JSON avec prefixe taille."""
        data = json.dumps(resp).encode('utf-8')
        header = struct.pack('<I', len(data))
        try:
            win32file.WriteFile(pipe, header + data)
        except:
            pass


# ---------------------------------------------------------------------------
# CLI client (pour tests)
# ---------------------------------------------------------------------------
def send_command(pipe_name: str, cmd: dict) -> dict:
    """Envoie une commande et attend la reponse."""
    try:
        handle = win32file.CreateFile(
            pipe_name,
            win32file.GENERIC_READ | win32file.GENERIC_WRITE,
            0, None,
            win32file.OPEN_EXISTING,
            0, None
        )
        
        data = json.dumps(cmd).encode('utf-8')
        header = struct.pack('<I', len(data))
        win32file.WriteFile(handle, header + data)
        
        result, raw = win32file.ReadFile(handle, 4)
        msg_len = struct.unpack('<I', raw)[0]
        result, raw = win32file.ReadFile(handle, msg_len)
        
        win32file.CloseHandle(handle)
        return json.loads(raw.decode('utf-8'))
    
    except Exception as e:
        return {"status": "error", "message": str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OGA Daemon - NPU XDNA2 inference server")
    parser.add_argument("--pipe", default=PIPE_NAME, help="Named pipe name")
    parser.add_argument("--model", help="Pre-load model at startup")
    parser.add_argument("--command", "-c", help="Send command and exit (client mode)")
    parser.add_argument("--prompt", help="Prompt for generate command")
    parser.add_argument("--max-tokens", type=int, default=50)
    
    args = parser.parse_args()
    
    # Client mode
    if args.command:
        cmd = json.loads(args.command) if args.command.startswith("{") else {"cmd": args.command}
        if args.prompt:
            cmd["prompt"] = args.prompt
        if args.max_tokens:
            cmd["max_tokens"] = args.max_tokens
        resp = send_command(args.pipe, cmd)
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        return
    
    # Server mode
    if args.model:
        # Quick test: load model and test
        print(f"[ogad] Pre-loading model: {args.model}")
        cache = ModelCache()
        result = cache.load(args.model)
        print(f"  {json.dumps(result)}")
        
        if result.get("status") == "ok":
            mid = result["model_id"]
            test = cache.generate(mid, "Hello, my name is", max_tokens=30)
            print(f"  Test: {json.dumps(test, ensure_ascii=False)}")
        return
    
    # Daemon mode
    server = PipeServer(args.pipe)
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
