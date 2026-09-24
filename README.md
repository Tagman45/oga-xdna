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

## Licence et dépendances

Ce dépôt contient le code de l'auteur (interface de démonstration pour OGA /
ONNX Runtime GenAI sur NPU XDNA2). Il **n'inclut pas de code copié d'autres
projets** :
- Le code utilise les API des bibliothèques tierces suivantes (sans les
  incorporer) :
  - onnxruntime-genai (ONNX Runtime GenAI) - licence MIT
  - gguf (gguf-py) - licence MIT
  - win32pipe / pywin32 - licence PSF
  - DLL système Windows (KERNEL32)
- Aucun marqueur copyright / SPDX d'un tiers n'est présent dans le code.
- Les binaires compilés ne lient que KERNEL32.dll et onnxruntime-genai.dll.


## Sources des connaissances (provenance)

Ce projet a été construit à partir de la **documentation publique officielle**
AMD et ONNX Runtime GenAI — pas de code privé copié :

| Connaissance | Source (officielle/public) |
|---|---|
| API onnxruntime_genai (Model, Tokenizer, Generator) | Documentation ONNX Runtime GenAI (Microsoft) |
| Env RyzenAI conda 
yzen-ai-1.7.1 + DLL deployment path | Installation AMD Ryzen AI Software (officielle) |
| xrt-smi configure --pmode performance | Outil officiel AMD XRT |
| Export ONNX INT4 + onnxruntime_genai.models.builder | Doc officielle ONNX Runtime GenAI |
| Compilation NPU model_generate --npu --token_fusion | Outil officiel AMD model-generate (RyzenAI 1.7.1) |
| Modèles hybrides pré-optimisés (md/*_hybrid sur HF) | AMD sur HuggingFace (officiel) |
| Format GGUF (lecture gguf.GGUFReader) | Bibliothèque open source gguf-py (MIT) |
| Named pipes Windows | API Windows standard (win32pipe/pywin32, PSF) |

**Le code source de ce dépôt est la couche d'intégration de l'auteur** :
daemon named pipe, protocole JSON, cache KV, client C++, convertisseur.
Il n'incorpore pas le code des bibliothèques ci-dessus — il les appelle.
