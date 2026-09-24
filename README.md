# OGA XDNA — OGA Daemon (ONNX GenAI sur AMD XDNA2 NPU, Windows)

Daemon d'inférence NPU XDNA2 via ONNX Runtime GenAI + provider RyzenAI.
Travail indépendant Windows/XDNA2 (juin-juillet 2026).

## Architecture

```
llama.cpp → ggml-xdna2 (C++) → named pipe \\.\pipe\ogad → ogad.py → OGA → RyzenAI → NPU
```

## Protocole (JSON sur pipe nommé)

```
Request:  {"cmd":"load","model":"C:/models/..."}
Response: {"status":"ok","model_id":"..."}

Request:  {"cmd":"generate","model_id":"...","prompt":"Hello","max_tokens":50,"temperature":0.7}
Response: {"status":"ok","text":"...","tokens":[...],"tps":70.9,"ttft_ms":1158}

Request:  {"cmd":"ping"}
Response: {"status":"ok"}
```

## Usage

```powershell
# Prérequis : conda ryzen-ai-1.7.1, mode performance NPU
cd C:\Windows\System32\AMD
xrt-smi configure --pmode performance
conda activate ryzen-ai-1.7.1

# Démarre le serveur (option --model pour précharger)
python ogad.py
python ogad.py --model C:/models/...
python ogad.py --pipe ogad_test   # pipe personnalisé

# Test client (C++ compilé)
.\ogad_test.exe
```

## Fichiers

| Fichier | Rôle |
|---|---|
| `ogad.py` / `ogad_v2.py` / `ogad_v3.py` | Daemon serveur (named pipe) |
| `ogad_client.cpp` / `ogad_client.h` / `ogad_client_v3.cpp` | Client C++ |
| `oga_npu_test.cpp` | Test NPU direct |
| `gguf_to_oga.py` | Convertisseur GGUF → ONNX |
| `GUIDE_OGA_TURBOQUANT.md` | Guide OGA hybride NPU+GPU |
| `qwen35_9b_balanced_ogad.json` | Config OGA Qwen3.5 9B |
| `ogad_test.exe` / `ogad_test_v3.exe` / `oga_npu_test.exe` | Binaires compilés |

## Modèles

Les wrappers ONNX (`model.onnx`, tokenizer, config) sont inclus pour
`tinyllama_oga` et `qwen2_oga`. Les poids (`model.onnx.data`, 0.9-14 GB) ne
sont pas versionnés (limite GitHub) — à générer via `gguf_to_oga.py` ou à
télécharger depuis le hub ONNX correspondant.