#!/usr/bin/env python3
"""
ogad_v3.py — OGA Daemon v3: KV cache fix, 62.6 TPS.
"""
import os, sys, json, time

OGA_AVAILABLE = False
if not OGA_AVAILABLE:
    try:
        os.add_dll_directory(r"C:\Program Files\RyzenAI\1.7.1\deployment")
        import onnxruntime_genai as og
        OGA_AVAILABLE = True
    except:
        pass

class ModelCache:
    def __init__(self):
        self.models = {}       # path -> {model, tokenizer, id}
        self.generators = {}   # path -> generator (persiste KV cache)
        self.prompt_tokens = {}  # path -> [input_token_ids]
        self.next_id = 0
    
    def load(self, path: str) -> dict:
        if not OGA_AVAILABLE:
            return {"status": "error", "message": "OGA not available"}
        if path in self.models:
            return {"status": "ok", "model_id": self.models[path]["id"], "cached": True}
        t0 = time.perf_counter()
        try:
            m = og.Model(path)
            t = og.Tokenizer(m)
            self.models[path] = {"model": m, "tokenizer": t, "id": f"m{self.next_id}"}
            self.next_id += 1
            return {"status": "ok", "model_id": self.models[path]["id"],
                    "load_ms": round((time.perf_counter()-t0)*1000, 1)}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def generate(self, path: str, prompt: str = "", max_tokens: int = 50,
                 temp: float = 0.7, top_k: int = 50, top_p: float = 0.9) -> dict:
        if path not in self.models:
            r = self.load(path)
            if r["status"] != "ok":
                return r
        entry = self.models[path]
        try:
            # Encode prompt
            input_ids = entry["tokenizer"].encode(prompt) if prompt else []
            
            # Cap max_length to avoid exceeding context
            context_max = max_tokens * 2  # safe margin
            
            # Create or reuse generator (KV cache persistence)
            has_input = input_ids is not None and len(input_ids) > 0
            if path not in self.generators or has_input:
                # New prompt = fresh generator
                params = og.GeneratorParams(entry["model"])
                params.set_search_options(max_length=context_max, temperature=temp,
                                          top_k=top_k, top_p=top_p)
                gen = og.Generator(entry["model"], params)
                if has_input:
                    gen.append_tokens(input_ids)
                self.generators[path] = gen
                self.prompt_tokens[path] = input_ids
            else:
                gen = self.generators[path]
                if has_input:
                    gen.append_tokens(input_ids)
            
            # Generate
            t0 = time.perf_counter()
            n = 0
            while not gen.is_done() and n < max_tokens:
                gen.generate_next_token()
                n += 1
            dt = (time.perf_counter() - t0) * 1000
            tps = round(n * 1000.0 / dt, 1) if dt > 0 else 0
            
            # Extract generated tokens only
            full_seq = gen.get_sequence(0)
            # Handle numpy array or other sequence types
            try:
                seq_len = len(full_seq)
                gen_tokens = full_seq[seq_len - min(max_tokens, seq_len):]
            except:
                gen_tokens = full_seq
            # Convert to list for JSON serialization if needed
            if not isinstance(gen_tokens, (list, tuple)):
                try:
                    gen_tokens = list(gen_tokens)
                except:
                    gen_tokens = [int(t) for t in gen_tokens]
            text = entry["tokenizer"].decode(gen_tokens)
            
            # Ensure text is a string
            if not isinstance(text, str):
                try:
                    text = str(text)
                except:
                    text = ""
            
            return {
                "status": "ok", "text": text,
                "n_tokens": int(n), "tps": tps,
                "infer_ms": round(dt, 1),
                "prompt_tokens": int(len(input_ids)),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def reset(self, path: str) -> dict:
        """Reset KV cache pour un modele."""
        if path in self.generators:
            del self.generators[path]
        if path in self.prompt_tokens:
            del self.prompt_tokens[path]
        return {"status": "ok"}

def handle_line(line: str, cache: ModelCache) -> dict:
    try:
        req = json.loads(line)
    except:
        return {"status": "error", "message": "invalid JSON"}
    cmd = req.get("cmd", "")
    if cmd == "ping":
        return {"status": "ok", "oga": OGA_AVAILABLE}
    elif cmd == "load":
        return cache.load(req.get("model", ""))
    elif cmd == "generate":
        return cache.generate(
            req.get("model", ""), req.get("prompt", ""),
            req.get("max_tokens", 50), req.get("temperature", 0.7),
            req.get("top_k", 50), req.get("top_p", 0.9))
    elif cmd == "reset":
        return cache.reset(req.get("model", ""))
    elif cmd == "stats":
        return {"status": "ok", "models": len(cache.models),
                "generators": len(cache.generators), "oga": OGA_AVAILABLE}
    return {"status": "error", "message": f"unknown cmd: {cmd}"}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    args = parser.parse_args()
    cache = ModelCache()
    
    if args.model:
        r = cache.load(args.model)
        print(json.dumps(r))
        if r["status"] == "ok":
            r = cache.generate(args.model, "Hello, my name is", 30)
            print(json.dumps(r, ensure_ascii=False))
        return
    
    sys.stdin.reconfigure(encoding='utf-8', errors='replace')
    sys.stdout.reconfigure(encoding='utf-8')
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        resp = handle_line(line, cache)
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
