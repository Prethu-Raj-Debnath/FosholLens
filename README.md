# FosholLens (ফসল লেন্স)

**Multimodal Bangla Voice & Weather-Aware Crop Advisory for Rice and Potato**

Built for **CoU AI Hackathon 2026**.

---

## Team Tesseract

- **Prethu Raj Debnath** (Leader)
- **Md. Kawsar Amin**
- **Nazim Saifullah**

---

## Problem & Solution Summary

### Problem
Smallholder rice (*ধান*) and potato (*আলু*) farmers in Bangladesh face severe yield losses from foliar diseases such as rice blast, bacterial leaf blight, and potato early/late blight. Existing diagnostic apps typically return English-only labels, hallucinate unverified pesticide advice, and ignore local weather—leading farmers to spray right before rainfall or high wind, wasting costly inputs and risking crop and health harm.

### Solution
**FosholLens (ফসল লেন্স)** is a mobile-first Bangla web app and PWA that runs an end-to-end grounded crop advisory pipeline:
1. **ONNX Leaf Disease Vision** (CPU-only, selectable local EfficientNet-B0 or Hugging Face ViT-tiny) that detects supported diseases and explicitly reports `"নিশ্চিতভাবে শনাক্ত করা যায়নি"` when confidence is low or a photo is out of scope—never fabricating certainty or chemical doses.
2. **Weather-Aware Spray-Window Guard** using real-time Open-Meteo forecasts across all 64 districts of Bangladesh, computed deterministically in code (`rain >= 60%` or `wind >= 15 km/h` blocks spraying)—never guessed by the LLM.
3. **Curated BARI/BRRI Grounding** so every symptom, organic measure, chemical active ingredient (with Bangla transliteration), dosage, and prevention step comes strictly from verified Bangladeshi agricultural references.
4. **Bangla Advisory Synthesis & Voice Playback** powered by Google Gemini Flash (with deterministic knowledge-base template fallback when offline or unconfigured) and local Bangla ONNX TTS (`indic-tiny-tts-bengali` -> Opus audio).

---

## Features & How the AI Works

The backend runs a 4-stage asynchronous job pipeline (`reading -> grounding -> writing -> speaking`), persisted in SQLite (WAL) and detailed in [`docs/backend-architecture.md`](docs/backend-architecture.md):

1. **Image Preparation & Privacy (`apps/frontend` & `apps/api/app/routes.py`):** Images are downscaled to `1280px` max edge and re-encoded as JPEG on the server, stripping EXIF metadata so phone GPS coordinates are never stored on disk.
2. **Stage 1 — Vision (`apps/api/app/vision/`):** Runs local ONNX inference (`onnxruntime`, CPU-only) via `VISION_MODEL`:
   - `efficientnet` *(default, committed at `apps/api/models/foshollens_leaf.onnx`, 16 MB, ~40 ms)*: EfficientNet-B0, 9 classes covering rice and potato (including rice bacterial leaf blight, rice blast, potato early blight, potato late blight).
   - `vit` *(22 MB, ~140 ms, downloaded from Hugging Face on first use)*: ViT-tiny (`wambugu71/crop_leaf_diseases_vit` ONNX export), 13 classes across corn, potato, rice, and wheat.
   - Threshold (`VISION_CONFIDENCE_THRESHOLD=0.55`) is applied to raw softmax probabilities *before* crop filtering so wrong-crop photos cannot be inflated into false positives.
3. **Stage 2 — Grounding (`apps/api/app/grounding/`):** Looks up curated disease management data from BRRI and BARI knowledge-base entries (`knowledge_base.py`) and fetches hourly Open-Meteo weather forecasts for the farmer's selected district (`districts.py`, `weather.py`) to compute the deterministic spray verdict.
4. **Stage 3 — Bangla Advisory (`apps/api/app/llm/writer.py`):**
   - **Primary (when `GEMINI_API_KEY` or `LLM_API_KEY` is configured):** Google Gemini Flash (`google-genai` SDK, defaulting to `gemini-3.8-flash` with automatic fallback to `gemini-3.7-flash` -> `gemini-3.5-flash` -> `gemini-3.5-flash-lite` on HTTP 404/429/503) rewrites the grounded facts into clear, empathetic Standard Colloquial Bangla (25–90 words, Bangla digits `০-৯`, PPE reminder, and officer referral when uncertain). An optional OpenAI-compatible provider mode (`LLM_PROVIDER=openai`) is also supported.
   - **Strict Safety Validation:** `validate_narrative()` rejects empty text, non-Bangla text, Latin script, markdown, emojis, out-of-range word counts, and any chemical name not present in the curated facts for that diagnosis.
   - **Automatic Fallback:** If no API key is configured, the network fails, or the model violates safety checks after one retry, `TemplateWriter` (`generatedBy: "template"`) composes safe Bangla advice directly from the grounded facts.
5. **Stage 4 — Bangla Voice Playback (`apps/api/app/tts/engine.py`):** Text results return immediately while `sulabhkatiyar/indic-tiny-tts-bengali` (local ONNX, CPU) synthesizes Bangla speech asynchronously and transcodes via `ffmpeg` to compact Opus audio (`~22 KB/min`). `prepare_text()` repairs `/` ratios and guards against lost chemical names before synthesis.

---

## Tech Stack, Models, APIs & Datasets

- **Frontend (`apps/frontend`):** React 19, TypeScript, Vite 7, React Router, Lucide icons, PWA web manifest, Noto Sans Bengali (with system Bangla fallbacks and zero `letter-spacing` so Bangla conjuncts render cleanly)
- **Shared Contracts (`packages/contracts`):** Pydantic models (`apps/api/app/contracts.py`) mirrored in TypeScript/Zod (`@foshol-lens/contracts`)
- **Backend (`apps/api`):** Python 3.12+, FastAPI, Uvicorn, Pydantic v2, `aiosqlite` (SQLite WAL), HTTPX, Pillow, `onnxruntime`, `huggingface-hub`
- **Vision Models:**
  - Committed EfficientNet-B0 ONNX checkpoint (`apps/api/models/foshollens_leaf.onnx`, 9 classes for Rice & Potato)
  - Selectable Hugging Face Vision Transformer [`wambugu71/crop_leaf_diseases_vit`](https://huggingface.co/wambugu71/crop_leaf_diseases_vit) (`wambugu71/crop_leaf_diseases_vit_onnx`, 13 classes)
- **LLM Advisory:** Google Gemini Flash via `google-genai` SDK (`gemini-3.8-flash` primary with `gemini-3.7-flash` / `gemini-3.5-flash` / `gemini-3.5-flash-lite` fallback; optional OpenAI-compatible `openai` SDK support)
- **Text-to-Speech:** [`sulabhkatiyar/indic-tiny-tts-bengali`](https://huggingface.co/sulabhkatiyar/indic-tiny-tts-bengali) (local ONNX, CPU) + `ffmpeg` -> Opus *(Note: `edge-tts` `bn-BD-NabanitaNeural` was prototyped on Day 1 and replaced by local ONNX TTS for offline/rural reliability)*
- **Weather API:** [Open-Meteo Forecast API](https://open-meteo.com/) (key-free, non-commercial use)
- **Datasets & Agricultural Knowledge Sources:**
  - PlantVillage Dataset (Potato/Tomato leaf diseases)
  - Paddy Doctor Dataset (Rice leaf diseases)
  - **BARI** (*Bangladesh Agricultural Research Institute* — *Krishi Projukti Hatboi* / potato, tomato & chili management guides)
  - **BRRI** (*Bangladesh Rice Research Institute* — *আধুনিক ধানের চাষ*, *ধানের রোগ ও প্রতিকার*)

---

## AI Tools Disclosure

In accordance with CoU AI Hackathon Rule 10 transparency requirements, Team Tesseract discloses the use of the following AI tools during development:
- **Antigravity:** Code implementation assistance, repository merging, refactoring, test verification, and scaffolding.
- **Claude (Anthropic):** Architecture ideation, safety guard design review, and documentation drafting.
- **Google Gemini:** System brainstorming, Bangla advisory prompt engineering, and runtime Gemini Flash inference.

---

## Third-Party Code & Assets Credits

- [wambugu71/crop_leaf_diseases_vit](https://huggingface.co/wambugu71/crop_leaf_diseases_vit) & `wambugu71/crop_leaf_diseases_vit_onnx` — Pretrained Vision Transformer for crop leaf disease classification
- [sulabhkatiyar/indic-tiny-tts-bengali](https://huggingface.co/sulabhkatiyar/indic-tiny-tts-bengali) — Local ONNX Bangla neural text-to-speech synthesis
- [edge-tts](https://github.com/rany2/edge-tts) (`bn-BD-NabanitaNeural`) — Used in our initial Day 1 TTS prototype
- [FastAPI](https://fastapi.tiangolo.com/), [Pydantic](https://docs.pydantic.dev/), and [ONNX Runtime](https://onnxruntime.ai/) — Backend framework, schema validation, and CPU inference runtime
- [React](https://react.dev/), [Vite](https://vite.dev/), [React Router](https://reactrouter.com/), [Zod](https://zod.dev/), and [Lucide](https://lucide.dev/) — Frontend UI, routing, and contract validation
- [Noto Sans Bengali](https://fonts.google.com/noto/specimen/Noto+Sans+Bengali) — SIL Open Font License (OFL)
- [Open-Meteo API](https://open-meteo.com/) — Weather forecast API
- Agricultural treatment references credited to **BARI** (*Krishi Projukti Hatboi*) and **BRRI**, and public leaf datasets **PlantVillage** and **Paddy Doctor**.

---

## Repository Structure

```text
foshollens/
├── apps/
│   ├── api/                      # Python 3.12+ / FastAPI service
│   │   ├── app/
│   │   │   ├── main.py           # FastAPI app, CORS, lifespan startup checks
│   │   │   ├── routes.py         # /api/v1 health, districts, crops, analyses, image, audio
│   │   │   ├── contracts.py      # Pydantic API contracts (source of truth)
│   │   │   ├── config.py         # Environment settings (.env & apps/api/.env)
│   │   │   ├── store.py          # SQLite (WAL) persistence
│   │   │   ├── jobs/pipeline.py  # In-process 4-stage async job runner
│   │   │   ├── vision/           # ONNX leaf classifier (EfficientNet-B0 / ViT)
│   │   │   ├── grounding/        # BARI/BRRI knowledge base, 64 districts, Open-Meteo
│   │   │   ├── llm/writer.py     # Gemini Flash + OpenAI-compat + safety validator + Template fallback
│   │   │   └── tts/engine.py     # indic-tiny-tts-bengali ONNX + ffmpeg Opus encoder
│   │   ├── models/               # Committed default EfficientNet-B0 ONNX checkpoint (16 MB)
│   │   ├── tests/                # pytest suite (pipeline, safety, vision, weather, TTS, CORS)
│   │   └── pyproject.toml
│   └── frontend/                 # @foshol-lens/frontend (React 19 + TypeScript + Vite PWA)
│       ├── index.html
│       ├── public/
│       │   ├── favicon.svg
│       │   └── manifest.webmanifest
│       └── src/
│           ├── components/       # AppShell, Upload, Result, AudioPlayer, WeatherCard, Nav
│           ├── pages/            # HomePage, HistoryPage, AnalysisPage
│           ├── lib/              # API client, image prep, Bangla digit/date formatting
│           └── styles/global.css
├── packages/
│   └── contracts/                # @foshol-lens/contracts (Zod schemas & shared routes)
├── docs/
│   ├── backend-architecture.md
│   └── team-review.md
├── scripts/free-port.sh
├── .env.example
├── .gitignore
├── package.json
└── README.md
```

---

## Local Setup

### Prerequisites
- **Node.js** `>= 22` and `npm`
- **Python 3.12** (recommended via [`uv`](https://docs.astral.sh/uv/))
- **`ffmpeg`** on `PATH` (optional, required for Bangla Opus/MP3 TTS audio output)

### 1. Environment Variables
Copy `.env.example` to `.env` in the repo root (or `apps/api/.env`). Never commit `.env`:

```bash
cp .env.example .env
```

Add `GEMINI_API_KEY` to enable Google Gemini Flash Bangla advisory generation. Without an API key, the app still runs end-to-end using the offline `TemplateWriter` fallback (`generatedBy: "template"`).

### 2. Install Dependencies

```bash
# Root JS workspaces (@foshol-lens/contracts + @foshol-lens/frontend)
npm install

# Python API environment (apps/api)
cd apps/api
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e . pytest pytest-asyncio ruff mypy
# On Windows PowerShell, use:
# uv pip install --python .venv/Scripts/python.exe -e . pytest pytest-asyncio ruff mypy
cd ../..
```

### 3. Run Locally

Start the API (`:4000`) and frontend (`:5173`) in separate terminals (recommended across Linux/macOS/Windows):

```bash
# Terminal 1 — FastAPI backend (port 4000)
cd apps/api
# Linux/macOS:
.venv/bin/uvicorn app.main:app --reload --port 4000
# Windows PowerShell:
# .venv\Scripts\uvicorn.exe app.main:app --reload --port 4000

# Terminal 2 — Frontend + Vite dev server (port 5173, proxies /api -> :4000)
npm run dev:frontend
```

*(On bash/Unix environments, `npm run dev` launches API, contracts watcher, and frontend together.)*

Open `http://localhost:5173` in your browser and verify backend health at `http://localhost:4000/api/v1/health`.

### 4. Tests & Builds

```bash
# Backend pytest suite
cd apps/api && .venv/bin/python -m pytest
# Windows PowerShell:
# cd apps/api; .venv\Scripts\python.exe -m pytest

# Contracts & Frontend TypeScript + Vite production build
npm run build
npm run typecheck
npm run test
```

---

## Links

- **Live Deployment Link:** _[To be added after deployment]_
- **Demo Video Link:** _[To be added before submission]_
