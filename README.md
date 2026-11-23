# FosholLens (ফসল লেন্স)

**Multimodal Bangla Voice & Weather-Aware Crop Advisory for Rice and Potato**

Built for **CoU AI Hackathon 2026**.

---

## Team Tesseract

- **Prethu Raj Debnath**
- **Md. Kawsar Amin**
- **Nazim Saifullah** (`@nazimsaifullah`)

> **GitHub Contributor Transparency Note (Rule 3 / Rule 12):** Commits authored under `@niamt1` belong to a secondary GitHub account operated by registered Team Tesseract member **Nazim Saifullah** (`@nazimsaifullah`). All commits in this repository were authored solely by the three registered team members listed above.

---

## Problem & Solution Summary

### Problem
Smallholder rice (*ধান*) and potato (*আলু*) farmers in Bangladesh face severe yield losses from foliar diseases such as rice blast, bacterial leaf blight, brown spot, tungro, and potato early/late blight. Existing diagnostic tools typically return English-only labels, hallucinate unverified pesticide recommendations (even for bacterial or viral diseases where fungicides do not work), and ignore local weather—leading farmers to spray right before rainfall or strong winds.

### Solution
**FosholLens (ফসল লেন্স)** is a mobile-first Bangla web app and PWA that runs an end-to-end grounded crop advisory pipeline:
1. **ONNX Leaf Disease Vision** (CPU-only, local 8-class EfficientNet-B0) paired with a leaf-framing capture guide, explicitly flagging uncertain photos rather than fabricating diagnoses.
2. **Optional Bangla Voice & Text Question Input** allowing farmers to record a 12-second Bangla voice question or type a question alongside their photo, transcribed natively via Google Gemini Flash audio understanding.
3. **BARI/BRRI Grounded RAG Prescription Engine** (`16` curated chunks across all 8 leaf classes with precomputed Gemini embeddings + offline TF-IDF fallback), enforcing cultural-only guards for bacterial/viral diseases, no-spray healthy guidance, pre-harvest intervals (PHI), PPE reminders, and SAAO referrals.
4. **48-Hour Color-Coded Spray Timeline & GPS Auto-District** using Open-Meteo hourly forecasts across all 64 districts of Bangladesh (with one-tap Haversine GPS district matching), deterministically computing safe (`green`), caution (`amber`), and wash-off/drift risk (`red`) hours plus a pure-Bangla safest daylight spray window.
5. **Bangla Advisory Synthesis & Local Voice Playback** via Google Gemini Flash (with deterministic `TemplateWriter` fallback) and local ONNX Bangla TTS (`sulabhkatiyar/indic-tiny-tts-bengali` + `ffmpeg` -> MP3 by default for iOS Safari/Android/Desktop compatibility, with WAV fallback when `ffmpeg` is absent).

---

## Features & How the AI Works

The backend runs a 4-stage asynchronous job pipeline (`reading -> grounding -> writing -> speaking`), persisted in SQLite (WAL):

1. **Image Prep, Leaf Framing & Privacy (`apps/frontend` & `apps/api/app/routes.py`):**
   - `ImageDropzone.tsx` provides a dashed corner-framed leaf capture guide and Bangla lighting/framing tip.
   - Server-side preprocessing downscales photos to `1280px` max edge and re-encodes to JPEG, stripping EXIF metadata so phone GPS coordinates are never stored on disk.
   - Scope is **rice (`rice`) and potato (`potato`)** only—matching the crops the shipped EfficientNet checkpoint can name (`POST /api/v1/analyses` rejects other crops with `400 unsupported_crop`).
2. **Stage 1 — Vision & Voice Understanding (`apps/api/app/vision/`, `apps/api/app/llm/audio_input.py`):**
   - **Vision:** Runs local CPU ONNX inference (`onnxruntime`):
     - EfficientNet-B0 *(committed at `apps/api/models/foshollens_leaf.onnx`, 16 MB, ~40 ms)*: 8 rice & potato classes (`potato_early_blight`, `potato_healthy`, `potato_late_blight`, `rice_bacterial_leaf_blight`, `rice_blast`, `rice_brown_spot`, `rice_healthy`, and `rice_tungro`). Every label is surfaced to the farmer (`is_healthy` routes healthy plants to no-spray care).
     - Threshold (`VISION_CONFIDENCE_THRESHOLD=0.6`) is checked on the winning class's raw softmax probability, before crop filtering. A clear winner just below it is still accepted when it beats the runner-up by `VISION_MARGIN=0.30`, clears `VISION_MARGIN_FLOOR=0.45`, and the photo is at least `VISION_MARGIN_MIN_CROP_MASS=0.70` the selected crop.
   - **Optional Bangla Voice/Text Question:** Accepts optional `question_text` (up to `300` chars) and `audio` (up to `5 MB`; `audio/webm`, `audio/ogg`, `audio/mp4`, `audio/mpeg`, `audio/wav`, `audio/aac`, `audio/x-m4a`, `audio/x-wav`, `video/mp4`, `video/webm` with codec suffix stripping and MIME normalization). `transcribe_bangla_question()` sends audio bytes natively to Gemini Flash (`types.Part.from_bytes`) and degrades cleanly to `None` if unconfigured or silent.
3. **Stage 2 — BARI/BRRI Hybrid RAG & 48-Hour Weather Guard (`apps/api/app/grounding/`):**
   - **BARI/BRRI Grounded RAG (`rag.py`, `knowledge_base.py`):** Retrieves grounded Bangla/English passages from the 16-chunk corpus (`apps/api/data/bari_brri_corpus.json`) ranked by hybrid cosine similarity over precomputed `gemini-embedding-001` vectors (`apps/api/data/bari_brri_index.json`) and zero-network TF-IDF fallback—prioritizing the farmer's voice/text question when present.
     - Marks `rice_bacterial_leaf_blight` and `rice_tungro` as `cultural_only=True` & `refer_to_officer=True` (explicitly warning that fungicides do not work).
     - Marks `potato_healthy` & `rice_healthy` as no-spray healthy confirmations.
     - Marks uncertain diagnoses across any crop as non-chemical (`cultural_only=True`, `refer_to_officer=True`), retrieving safe scouting passages and advising against blind spraying.
   - **48h Weather Spray Timeline (`weather.py`, `districts.py`):** Evaluates Open-Meteo `forecast_days=3` sliced to the upcoming 48 hours (`rainProb`, precipitation `mm`, wind speed `km/h`; supporting exact GPS coordinates or district centroids and returning `None` if hourly data is empty or expired):
     - `red` (unsafe): `rainProb >= 60%`, `rainMm >= 1.0`, `windKmh >= 15.0`, or heavy rain (`>= 60%` or `>= 1.0 mm`) within the next 3 hours (wash-off lookahead).
     - `amber` (caution): `rainProb >= 30%`, `rainMm > 0.0`, or `windKmh >= 10.0`.
     - `green` (safe): clear conditions, computing a pure-Bangla `safestWindowSummaryBn` for the earliest daylight (`06:00–18:00`) safe window.
4. **Stage 3 — Grounded Bangla Advisory (`apps/api/app/llm/writer.py`):**
   - **Primary:** Google Gemini Flash (`gemini-3.5-flash-lite` default with fallback chain `gemini-3.8-flash` -> `gemini-3.7-flash` -> `gemini-3.5-flash` -> `gemini-3.5-flash-lite` -> `gemini-3.1-flash-lite` on HTTP 404/429/500/502/503/504 or timeout; optional `LLM_PROVIDER=openai` mode) synthesizes a 25–90 word colloquial Bangla advisory directly from retrieved BARI/BRRI passages, weather verdict, and farmer question.
   - **Safety Validation & Offline Fallback:** `validate_narrative()` rejects Latin letters, markdown, emojis, wrong word counts, or any chemical outside the retrieved BARI/BRRI chunks (blocking any chemicals when `cultural_only=True`). `TemplateWriter` composes grounded Bangla advice offline when no key is set or validation fails.
5. **Stage 4 — Local Bangla Voice Playback (`apps/api/app/tts/engine.py`):**
   - Synthesizes Bangla audio asynchronously via `sulabhkatiyar/indic-tiny-tts-bengali` (local ONNX, CPU) and transcodes via `ffmpeg` to compact MP3 audio (`TTS_FORMAT=mp3` default for universal iOS Safari/Android/Desktop playback; `opus` optional; WAV fallback if `ffmpeg` is unavailable). `prepare_text()` repairs `/` dose ratios and verifies medicine names survive before audio is served.

---

## Tech Stack, Models, APIs & Datasets

- **Frontend (`apps/frontend`):** React 19, TypeScript, Vite 7, React Router, Lucide icons, PWA manifest, Noto Sans Bengali, Browser `MediaRecorder` & Geolocation APIs
- **Shared Contracts (`packages/contracts`):** Pydantic v2 models (`apps/api/app/contracts.py`) mirrored in TypeScript/Zod (`@foshol-lens/contracts`)
- **Backend (`apps/api`):** Python 3.12+, FastAPI, Uvicorn, Pydantic v2, `aiosqlite` (SQLite WAL), HTTPX, Pillow, NumPy, `onnxruntime`, `huggingface-hub`
- **Vision Model:**
  - Shipped EfficientNet-B0 ONNX checkpoint (`apps/api/models/foshollens_leaf.onnx`, 8 rice & potato classes)
- **Embeddings & LLM / Audio AI:** Google Gemini (`google-genai` SDK) — `gemini-embedding-001` for precomputed RAG embeddings and `gemini-3.5-flash-lite` (with fallback chain) for Bangla voice transcription and advisory generation
- **Text-to-Speech:** [`sulabhkatiyar/indic-tiny-tts-bengali`](https://huggingface.co/sulabhkatiyar/indic-tiny-tts-bengali) (local ONNX, CPU) + `ffmpeg` -> MP3 (default) / Opus / WAV fallback
- **Weather API:** [Open-Meteo Forecast API](https://open-meteo.com/) (key-free, non-commercial use)
- **Datasets & Agricultural Knowledge Sources:**
  - PlantVillage Dataset (Potato leaf diseases) & Paddy Doctor Dataset (Rice leaf diseases)
  - **BARI** (*Bangladesh Agricultural Research Institute* — *BARI কৃষি প্রযুক্তি হাতবই*)
  - **BRRI** (*Bangladesh Rice Research Institute* — *BRRI আধুনিক ধানের চাষ*, *BRRI Knowledge Bank*) & **DAE** crop protection safety guidance

---

## AI Tools Disclosure

In accordance with **CoU AI Hackathon 2026 Rule 10**, Team Tesseract discloses the AI tools used in building FosholLens:
- **Antigravity:** Codebase exploration, feature implementation, refactoring, RAG/voice/timeline wiring, test execution, and build verification.
- **Claude (Anthropic):** System architecture design, safety guard review (Latin-script TTS checks, cultural-only chemical blocking, wash-off lookahead rules), and documentation structure.
- **Google Gemini:** Prompt engineering for Bangla agricultural advisory & voice transcription, precomputing `gemini-embedding-001` corpus vectors, and runtime `gemini-3.5-flash-lite` inference (with Flash fallback chain).

---

## Third-Party Code & Assets Credits

- [`sulabhkatiyar/indic-tiny-tts-bengali`](https://huggingface.co/sulabhkatiyar/indic-tiny-tts-bengali) — Local ONNX Bangla neural text-to-speech synthesis
- [FastAPI](https://fastapi.tiangolo.com/), [Pydantic](https://docs.pydantic.dev/), [NumPy](https://numpy.org/), and [ONNX Runtime](https://onnxruntime.ai/) — Backend framework, schema validation, vector math, and CPU inference
- [React](https://react.dev/), [Vite](https://vite.dev/), [React Router](https://reactrouter.com/), [Zod](https://zod.dev/), and [Lucide](https://lucide.dev/) — Frontend UI, routing, and contract validation
- [Noto Sans Bengali](https://fonts.google.com/noto/specimen/Noto+Sans+Bengali) — SIL Open Font License (OFL)
- [Open-Meteo API](https://open-meteo.com/) — Hourly weather forecast API
- Agricultural treatment guidance credited to **BARI**, **BRRI**, and **DAE**, and leaf imagery datasets credited to **PlantVillage** and **Paddy Doctor**.

---

## Repository Structure

```text
foshollens/
├── apps/
│   ├── api/                           # Python 3.12+ / FastAPI service
│   │   ├── Dockerfile                 # Production container for Azure Container Apps (port 7860, HEALTHCHECK, UID 1000)
│   │   ├── .dockerignore
│   │   ├── pyproject.toml
│   │   ├── app/
│   │   │   ├── main.py                # FastAPI app, CORS, /health alias, lifespan startup checks
│   │   │   ├── routes.py              # /api/v1 health, districts, crops, analyses, image, audio
│   │   │   ├── contracts.py           # Pydantic API contracts (source of truth)
│   │   │   ├── config.py              # Environment settings (.env & apps/api/.env)
│   │   │   ├── store.py               # SQLite (WAL) persistence
│   │   │   ├── jobs/pipeline.py       # In-process 4-stage async job runner
│   │   │   ├── vision/                # ONNX leaf classifier (EfficientNet-B0)
│   │   │   ├── grounding/
│   │   │   │   ├── rag.py             # Hybrid vector + offline TF-IDF BARI/BRRI retriever
│   │   │   │   ├── knowledge_base.py  # Curated crop/disease KB & aliases
│   │   │   │   ├── weather.py         # Open-Meteo 12h verdict + 48h timeline + wash-off lookahead
│   │   │   │   └── districts.py       # 64 Bangladesh districts with coordinates
│   │   │   ├── llm/
│   │   │   │   ├── writer.py          # Gemini Flash + OpenAI-compat + validator + Template fallback
│   │   │   │   └── audio_input.py     # Gemini Flash native Bangla audio question transcription
│   │   │   └── tts/engine.py          # indic-tiny-tts-bengali ONNX + ffmpeg MP3/Opus encoder
│   │   ├── data/
│   │   │   ├── bari_brri_corpus.json  # 16-chunk curated 8-class BARI/BRRI/DAE knowledge corpus
│   │   │   └── bari_brri_index.json   # Precomputed 768-dim Gemini embeddings
│   │   ├── models/
│   │   │   └── foshollens_leaf.onnx   # Committed 8-class EfficientNet-B0 ONNX checkpoint (16 MB)
│   │   ├── scripts/
│   │   │   └── build_rag_index.py     # Rebuilds bari_brri_index.json via gemini-embedding-001
│   │   └── tests/                     # pytest tests (API, RAG, voice, weather, vision, TTS, CORS)
│   └── frontend/                      # @foshol-lens/frontend (React 19 + TypeScript + Vite PWA)
│       ├── vercel.json                # Standalone SPA rewrite config
│       ├── index.html
│       ├── public/
│       │   ├── favicon.svg
│       │   └── manifest.webmanifest
│       └── src/
│           ├── components/            # Upload, Dropzone framing guide, WeatherCard timeline, Result, Audio
│           ├── pages/                 # HomePage, HistoryPage, AnalysisPage
│           ├── lib/                   # API client, GPS district matcher, image prep, Bangla formatting
│           └── styles/global.css
├── packages/
│   └── contracts/                     # @foshol-lens/contracts (Zod schemas & shared routes)
├── docs/
│   └── backend-architecture.md
├── scripts/free-port.sh
├── .env.example
├── .gitignore
├── package.json
├── vercel.json                        # Root Vercel monorepo build & SPA rewrite config
└── README.md
```

---

## Local Setup

### Prerequisites
- **Node.js** `>= 22` and `npm`
- **Python 3.12** (recommended via [`uv`](https://docs.astral.sh/uv/))
- **`ffmpeg`** on `PATH` (optional, required for Bangla MP3/Opus TTS audio output; falls back to WAV when absent)

### 1. Environment Variables
Copy `.env.example` to `.env` in the repo root (or `apps/api/.env`). Never commit `.env`:

```bash
cp .env.example .env
```

Set `GEMINI_API_KEY` to enable Google Gemini Flash Bangla advisory generation and voice-question transcription. Without an API key, the app still runs end-to-end using offline TF-IDF RAG retrieval and `TemplateWriter` fallback (`generatedBy: "template"`).

### 2. Install Dependencies

```bash
# Root JS workspaces (@foshol-lens/contracts + @foshol-lens/frontend)
npm install

# Python API environment (apps/api)
cd apps/api
uv venv --python 3.12 .venv
# Linux/macOS:
uv pip install --python .venv/bin/python -e . pytest pytest-asyncio ruff mypy
# Windows PowerShell:
# uv pip install --python .venv/Scripts/python.exe -e . pytest pytest-asyncio ruff mypy
cd ../..
```

*(Optional)* Rebuild the precomputed RAG embedding index after editing `apps/api/data/bari_brri_corpus.json`:

```bash
cd apps/api && .venv/bin/python scripts/build_rag_index.py
```

### 3. Run Locally

Start the API (`:4000`) and frontend (`:5173`) in separate terminals:

```bash
# Terminal 1 — FastAPI backend (port 4000)
cd apps/api
.venv/bin/uvicorn app.main:app --reload --port 4000
# Windows PowerShell: .venv\Scripts\uvicorn.exe app.main:app --reload --port 4000

# Terminal 2 — Frontend + Vite dev server (port 5173)
npm run dev:frontend
```

Open `http://localhost:5173` and verify backend health at `http://localhost:4000/api/v1/health`.

### 4. Tests & Builds

```bash
# Backend pytest, ruff, and mypy
cd apps/api && .venv/bin/python -m pytest -v
cd apps/api && .venv/bin/ruff check app tests scripts

# Contracts & Frontend production build
npm run build
npm run typecheck
```

---

## Deployment

### Backend on Azure Container Apps (Docker)
1. Build and push [`apps/api/Dockerfile`](apps/api/Dockerfile) using build context `apps/api` (installs `ffmpeg`, copies `models/foshollens_leaf.onnx` & `data/`, runs as UID `1000`, exposes port `7860` with a built-in Docker `HEALTHCHECK` on `/api/v1/health`, and starts Uvicorn on `0.0.0.0:${PORT:-7860}`; `/health` is also exposed as a root alias).
2. Configure the Container App with:
   - **External Ingress Target Port:** `7860`
   - **Scaling:** `minReplicas=1` (or `0`) and **`maxReplicas=1`** — required so all requests share a single SQLite (WAL) database and local `/app/storage` volume safely.
   - **Resources:** `1.0 vCPU` / `2.0 Gi` RAM (or `2.0 vCPU` / `4.0 Gi` RAM) for CPU ONNX vision + local TTS synthesis.
3. Configure Container App Environment Variables / Secrets:
   - `APP_ENV=production`
   - `GEMINI_API_KEY=<your-gemini-api-key>`
   - `CORS_ORIGINS=http://localhost:5173,https://<your-vercel-domain>.vercel.app`
   - `PUBLIC_BASE_URL=https://<your-container-app-fqdn>`

### Frontend on Vercel
1. Import the repository into Vercel with the **Root Directory** left at the repository root (`.`). Root [`vercel.json`](vercel.json) runs `npm ci`, builds `@foshol-lens/contracts` and `@foshol-lens/frontend`, serves `apps/frontend/dist`, and rewrites SPA routes (`/analysis/:id`, `/history`).
2. Set the Vercel environment variable:
   - `VITE_API_BASE_URL=https://<your-container-app-fqdn>`

---

## Links

- **Live Deployment Link:** _[To be added after deployment]_
- **Demo Video Link:** _[To be added before submission]_
