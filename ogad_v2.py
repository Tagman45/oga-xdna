#!/usr/bin/env python3
"""
ogad_v2.py — OGA Daemon v2: inference NPU via stdin/stdout pipe.

Communication JSON ligne par ligne.
Le processus C++ lance ogad en sous-processus, pipe stdin/stdout.

Usage:
  python ogad_v2.py                          # demarre le serveur
  python ogad_v2.py --model C:/models/...    # charge et test
"""
import os, sys, json, time, threading

OGA_AVAILABLE = False
if not OGA_AVAILABLE:
    try:
        deploy = r"C:\Program Files\RyzenAI\1.7.1\deployment"
        os.add_dll_directory(deploy)
        import onnxruntime_genai as og
        OGA_AVAILABLE = True
    except:
        pass

# ---------------------------------------------------------------------------
# Models cache
# ---------------------------------------------------------------------------
class ModelCache:
    def __init__(self):
        self.models = {}
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
            return {"status": "ok", "model_id": self.models[path]["id"], "load_ms": round((time.perf_counter()-t0)*1000, 1)}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def generate(self, path: str, prompt: str, max_tokens: int = 50,
                 temp: float = 0.7, top_k: int = 50, top_p: float = 0.9) -> dict:
        if path not in self.models:
            # Try auto-load
            r = self.load(path)
            if r["status"] != "ok":
                return r
        entry = self.models[path]
        try:
            input_ids = entry["tokenizer"].encode(prompt)
            params = og.GeneratorParams(entry["model"])
            params.set_search_options(max_length=max_tokens + len(input_ids), temperature=temp, top_k=top_k, top_p=top_p)
            gen = og.Generator(entry["model"], params)
            gen.append_tokens(input_ids)
            t0 = time.perf_counter()
            n = 0
            while not gen.is_done() and n < max_tokens:
                gen.generate_next_token()
                n += 1
            dt = (time.perf_counter() - t0) * 1000
            tokens = gen.get_sequence(0)
            gen_tokens = tokens[len(input_ids):][:max_tokens]
            text = entry["tokenizer"].decode(gen_tokens)
            return {
                "status": "ok", "text": str(text),
                "n_tokens": int(n), "tps": round(n * 1000.0 / dt, 1) if dt > 0 else 0,
                "infer_ms": round(dt, 1), "prompt_tokens": int(len(input_ids)),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

# ---------------------------------------------------------------------------
# Line-based protocol (stdin/stdout)
# ---------------------------------------------------------------------------
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
            req.get("top_k", 50), req.get("top_p", 0.9),
        )
    elif cmd == "stats":
        return {"status": "ok", "models": len(cache.models), "oga": OGA_AVAILABLE}
    return {"status": "error", "message": f"unknown cmd: {cmd}"}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    args = parser.parse_args()
    
    cache = ModelCache()
    
    # Pre-load mode
    if args.model:
        r = cache.load(args.model)
        print(json.dumps(r))
        if r["status"] != "ok":
            return
        r = cache.generate(args.model, "Hello, my name is", 30)
        print(json.dumps(r, ensure_ascii=False))
        return
    
    # Line protocol mode (stdin/stdout)
    sys.stdin.reconfigure(encoding='utf-8', errors='replace')
    sys.stdout.reconfigure(encoding='utf-8')
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        resp = handle_line(line, cache)
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
