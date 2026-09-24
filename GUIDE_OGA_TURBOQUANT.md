# Guide : Mettre le LLM sur NPU + RTX 5070

---

## A. OGA Hybride (NPU + GPU) — OFFICIEL AMD

### Prérequis
```powershell
# 1. Activer le mode performance NPU
cd C:\Windows\System32\AMD
xrt-smi configure --pmode performance

# 2. Activer l'environnement conda RyzenAI
conda activate ryzen-ai-1.7.1

# 3. Installer torch (nécessaire pour le script Python)
pip install torch==2.7.1
```

### Télécharger un modèle hybride pré-optimisé
```powershell
# DeepSeek-R1-8B hybride (NPU prefill + GPU decode)
git lfs install
git clone https://huggingface.co/amd/DeepSeek-R1-Distill-Llama-8B_rai_1.7.1_hybrid

# Ou Qwen2.5-7B hybride
git clone https://huggingface.co/amd/Qwen2.5-7B-Instruct_rai_1.7.1_hybrid
```

### Lancer l'inférence
```powershell
# Python (avec timings)
python "%RYZEN_AI_INSTALLATION_PATH%\LLM\example\model_chat.py" `
    -m DeepSeek-R1-Distill-Llama-8B_rai_1.7.1_hybrid `
    -pr amd_genai_prompt.txt --timings

# C++ (benchmark)
model_benchmark.exe -i DeepSeek-R1-Distill-Llama-8B_rai_1.7.1_hybrid `
    -f amd_genai_prompt.txt -l "1024"
```

### Export manuel Qwen3.5-9B pour OGA
```powershell
# 1. Télécharger le modèle HF
huggingface-cli download Qwen/Qwen3.5-9B --local-dir C:\models\qwen35-9b

# 2. Exporter en ONNX INT4
pip install onnxruntime-genai-directml-ryzenai==0.11.2
python -m onnxruntime_genai.models.builder -i C:\models\qwen35-9b -o C:\models\qwen35-9b-oga -p int4 -e cpu

# 3. Compiler pour NPU + GPU
pip install model-generate==1.7.1
python -m model_generate --npu --token_fusion --input C:\models\qwen35-9b-oga --output C:\models\qwen35-9b-npu
```

### Config hybride (genai_config.json)
```json
{
  "model": {
    "decoder": {
      "session_options": {
        "provider_options": [{
          "RyzenAI": {
            "hybrid_opt_free_after_prefill": "1",
            "hybrid_opt_max_seq_length": "4096",
            "hybrid_opt_token_backend": "cuda"
          }
        }]
      }
    }
  }
}
```

---

## B. TurboQuant (GPU seul — déjà prêt)

### Installer les binaires pré-compilés
```powershell
# Option 1: Télécharger la release officielle
# https://github.com/TheTom/llama-cpp-turboquant/releases/tag/tqp-v0.2.0
# → turboquant-plus-tqp-v0.2.0-windows-x64-cuda12.4.zip

# Option 2: Utiliser le pack local déjà présent
# Les fichiers sont dans:
ls C:\Users\videl\Documents\gllm onnx\lama 1080-5070\
ls C:\Users\videl\Documents\gllm onnx\lama-tensorRT 1050-5070\
```

### Obtenir le modèle Qwen3.5-9B en GGUF
```powershell
# Télécharger le GGUF depuis HuggingFace
huggingface-cli download Qwen/Qwen3.5-9B-GGUF --local-dir C:\models\qwen35-9b-gguf

# Ou convertir depuis le modèle HF
python convert_hf_to_gguf.py C:\models\qwen35-9b --outfile C:\models\qwen35-9b.gguf
```

### Lancer avec TurboQuant sur RTX 5070
```powershell
# KV-cache quantifié asymétrique (recommandé)
llama-cli -m C:\models\qwen35-9b.gguf `
    --cache-type-k q8_0 --cache-type-v turbo3 `
    -ngl 99 -c 8192 -n 256 -p "Prompt..."

# Poids TQ4_1S (dp4a 3.5× plus rapide sur CUDA)
llama-quantize C:\models\qwen35-9b.gguf C:\models\qwen35-9b-tq4.gguf TQ4_1S
llama-cli -m C:\models\qwen35-9b-tq4.gguf -ngl 99
```

### Benchmark
```powershell
llama-bench -m C:\models\qwen35-9b.gguf -ngl 99
```

---

## Comparaison des approches

| Critère | OGA Hybride (NPU+GPU) | TurboQuant (GPU seul) |
|---------|----------------------|----------------------|
| **TPS estimé Qwen3.5-9B** | 15-25 TPS | **28.3 TPS** ✅ |
| **Utilise NPU** | ✅ Oui | ❌ Non |
| **Utilise RTX 5070** | ✅ Oui | ✅ Oui |
| **Prêt maintenant** | ❌ Export manuel | ✅ Binaires dispo |
| **Complexité** | Moyenne | Faible |
| **Modèles pré-optimisés** | DeepSeek-R1, Llama, Phi | GGUF (tous modèles) |
