#!/usr/bin/env python3
"""
gguf_to_oga.py -- Pont GGUF -> OGA (ONNX Runtime GenAI).

Le vrai verrou du projet n'est plus "comment parler au NPU" (c'est fait, 76 TPS),
mais "comment transformer un GGUF arbitraire en graphe NPU efficace".

Usage:
  python gguf_to_oga.py --gguf model.q4_k_m.gguf --output ./model_oga
  python gguf_to_oga.py --hf Qwen/Qwen2-1.5B --output ./model_oga
"""
import os, sys, json, subprocess, argparse

sys.stdout.reconfigure(encoding='utf-8')

def parse_gguf_header(path: str) -> dict:
    """Parse le header d'un fichier GGUF."""
    if not os.path.exists(path):
        print(f"Fichier introuvable: {path}")
        return None
    try:
        import gguf
        reader = gguf.GGUFReader(path)
        config = {}
        metadata = reader.fields
        for key in ['general.architecture', 'llama.context_length', 'llama.embedding_length',
                     'llama.block_count', 'llama.attention.head_count',
                     'llama.attention.head_count_kv', 'llama.rope.freq_base',
                     'llama.feed_forward_length', 'general.name', 'llama.vocab_size']:
            if key in metadata:
                val = metadata[key]
                try:
                    if hasattr(val, 'value'):
                        config[key] = val.value
                    elif hasattr(val, 'parts') and val.parts:
                        config[key] = val.parts[0]
                except:
                    config[key] = str(val)
        print(f"Modele: {config.get('general.name', '?')}")
        print(f"Arch: {config.get('general.architecture', '?')}")
        print(f"Context: {config.get('llama.context_length', '?')}")
        print(f"Layers: {config.get('llama.block_count', '?')}")
        print(f"Hidden: {config.get('llama.embedding_length', '?')}")
        print(f"Heads: {config.get('llama.attention.head_count', '?')}")
        return config
    except ImportError:
        print("Installation de la lib gguf...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gguf"])
        return parse_gguf_header(path)
    except Exception as e:
        print(f"Erreur: {e}")
        return None

def gguf_to_hf_config(gguf_config: dict) -> dict:
    """Convertit GGUF -> HuggingFace config.json."""
    hidden = gguf_config.get('llama.embedding_length', 2048)
    heads = gguf_config.get('llama.attention.head_count', 32)
    if hasattr(hidden, 'item'): hidden = hidden.item()
    if hasattr(heads, 'item'): heads = heads.item()
    
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": hidden,
        "num_hidden_layers": _v(gguf_config.get('llama.block_count', 22)),
        "num_attention_heads": heads,
        "num_key_value_heads": _v(gguf_config.get('llama.attention.head_count_kv', heads)),
        "intermediate_size": _v(gguf_config.get('llama.feed_forward_length', hidden * 4)),
        "max_position_embeddings": _v(gguf_config.get('llama.context_length', 2048)),
        "rope_theta": float(gguf_config.get('llama.rope.freq_base', 10000)) if 'llama.rope.freq_base' in gguf_config else 10000,
        "vocab_size": _v(gguf_config.get('llama.vocab_size', 32000)),
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
        "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
    }
    return hf_config

def _v(x):
    """Convertit en int natif Python."""
    if hasattr(x, 'item'): return x.item()
    if hasattr(x, 'tolist'): return x.tolist()
    return int(x) if not isinstance(x, int) else x

def export_hf_to_oga(hf_model_id: str, output_dir: str):
    """Exporte HF -> OGA via onnxruntime_genai.models.builder."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nExport HF -> OGA: {hf_model_id}")
    print(f"Output: {output_dir}")
    
    cmd = [
        sys.executable, "-m", "onnxruntime_genai.models.builder",
        "-m", hf_model_id,
        "-o", output_dir,
        "-p", "int4",
        "-e", "cpu",
    ]
    print(f"Lancement: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, timeout=7200)
        print(f"OK! Modele exporte dans: {output_dir}")
        return True
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT (2h) - L'export est tres long pour les gros modeles")
        return False
    except subprocess.CalledProcessError as e:
        print(f"ECHEC: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Pont GGUF -> OGA")
    parser.add_argument("--gguf", help="Fichier GGUF")
    parser.add_argument("--hf", help="ID HuggingFace")
    parser.add_argument("--output", "-o", default="./model_oga")
    args = parser.parse_args()
    
    print("=" * 60)
    print("GGUF -> OGA Bridge")
    print("=" * 60)
    
    if args.gguf:
        print(f"\n[1] Parsing GGUF: {args.gguf}")
        cfg = parse_gguf_header(args.gguf)
        if not cfg: return 1
        
        hf = gguf_to_hf_config(cfg)
        print(f"\n[2] Config extraite: {hf['hidden_size']} hidden, {hf['num_hidden_layers']} layers")
        
        print(f"\n[3] Pour exporter, lancer depuis HuggingFace:")
        print(f"    python -m onnxruntime_genai.models.builder \\")
        print(f"      -m HF_ID \\")
        print(f"      -o {args.output} \\")
        print(f"      -p int4 -e cpu")
        
        # Write config
        with open(os.path.join(args.output, "config.json"), 'w') as f:
            json.dump(hf, f, indent=2)
        print(f"\nConfig sauvegardee dans: {os.path.join(args.output, 'config.json')}")
    
    elif args.hf:
        export_hf_to_oga(args.hf, args.output)
    else:
        print("\nUsage:")
        print("  python gguf_to_oga.py --gguf modele.gguf --output ./dir")
        print("  python gguf_to_oga.py --hf Qwen/Qwen2-1.5B --output ./dir")
        return 1
    
    print(f"\nPour tester: python ogad_v3.py --model {args.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
