# FosholLens

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
Smallholder rice (*ধান*) and potato (*আলু*) farmers in Bangladesh face severe yield losses from foliar diseases such as rice blast, brown spot, bacterial leaf blight, and potato early/late blight. Existing diagnostic apps typically return English-only labels, hallucinate unverified pesticide advice, and ignore local weather—leading farmers to spray right before rainfall or high wind, wasting costly inputs and risking crop/health harm.

### Solution
**FosholLens** is a mobile-first Bangla Progressive Web App (PWA) that combines:
1. **Leaf Disease Vision** for Rice and Potato leaf photos.
2. **48-Hour Weather Spray-Window Guard** using real-time Open-Meteo forecasts to flag safe (`green`), cautious (`amber`), and unsafe (`red`) spraying windows.
3. **BARI/BRRI Grounded RAG** so treatment guidance comes strictly from verified Bangladeshi agricultural references.
4. **Bangla Multimodal Advisory & Voice Output** powered by Google Gemini Flash and Bangla TTS, including PPE safety reminders and agriculture officer referral when confidence is uncertain.

---

## Features & How the AI Works

1. **Crop & Leaf Capture (PWA Frontend):** Farmers choose *ধান / Rice* or *আলু / Potato*, capture a leaf photo inside an on-screen framing guide, and optionally ask a question in Bangla text or voice.
2. **Vision Classification (`backend/app/vision.py`):** Preprocesses the leaf photo and runs crop-specific ONNX / Hugging Face plant-disease classification models, returning top predictions and a `confident` / `uncertain` status.
3. **48-Hour Spray-Window Guard (`backend/app/weather.py`):** Queries the Open-Meteo API for the farmer's coordinates (defaulting to Cumilla: `23.4607, 91.1809`) to score rain probability, rainfall (mm), and wind speed (km/h) over the next 48 hours.
4. **Grounded Agricultural Retrieval (`backend/app/rag.py`):** Retrieves relevant excerpts from curated BARI (*Krishi Projukti Hatboi*) and BRRI JSON knowledge bases.
5. **Bangla Advisory Synthesis (`backend/app/llm.py`):** Google Gemini Flash combines the farmer's optional Bangla audio/text question, vision output, weather guard, and RAG sources into clear, grounded Bangla guidance with PPE reminders and officer referral flags.
6. **Bangla Voice Playback (`backend/app/tts.py`):** Synthesizes the Bangla advisory into playable audio via Gemini TTS or `edge-tts` so farmers with limited literacy can listen directly on their phone.

---

## Tech Stack, Models, APIs & Datasets

- **Frontend:** React, Vite, Tailwind CSS, `vite-plugin-pwa`, Noto Sans Bengali
- **Backend:** Python 3.11+, FastAPI, Uvicorn, Pydantic, HTTPX
- **Vision Models:** Hugging Face / ONNX plant-disease classification models (Rice & Potato)
- **LLM & Multimodal Understanding:** Google Gemini Flash
- **Text-to-Speech:** Bangla TTS (Gemini TTS / `edge-tts`)
- **Weather API:** Open-Meteo Forecast API
- **Datasets & Agricultural Knowledge Sources:**
  - PlantVillage Dataset (Potato leaf diseases)
  - Paddy Doctor Dataset (Rice leaf diseases)
  - BARI *Krishi Projukti Hatboi* (Bangladesh Agricultural Research Institute)
  - BRRI Publications (Bangladesh Rice Research Institute)

---

## AI Tools Disclosure

In accordance with hackathon transparency rules, Team Tesseract discloses the use of the following AI tools during development:
- **Claude (Anthropic):** Architecture ideation, prompt/design review, and documentation drafting.
- **Google Gemini:** System architecture brainstorming, Bangla advisory drafting, and runtime multimodal AI inference.
- **Antigravity:** Code implementation assistance, scaffolding, and verification.

---

## Third-Party Code & Assets Credits

- [FastAPI](https://fastapi.tiangolo.com/) & [Pydantic](https://docs.pydantic.dev/) — MIT License
- [React](https://react.dev/), [Vite](https://vite.dev/), [Tailwind CSS](https://tailwindcss.com/), and [vite-plugin-pwa](https://vite-pwa-org.netlify.app/) — MIT License
- [Noto Sans Bengali](https://fonts.google.com/noto/specimen/Noto+Sans+Bengali) — SIL Open Font License (OFL)
- [Open-Meteo API](https://open-meteo.com/) — Free weather forecast API for non-commercial/open use
- Agricultural references credited to **BARI** (*Krishi Projukti Hatboi*) and **BRRI**, and public leaf datasets **PlantVillage** and **Paddy Doctor**.

---

## Repository Structure

```text
foshollens/
├── frontend/                  # React + Vite + Tailwind CSS PWA
├── backend/                   # Python 3.11+ FastAPI service
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI app, CORS, GET /api/health, POST /api/analyze
│   │   ├── schemas.py         # Strict Pydantic request/response models
│   │   ├── vision.py          # Stub: Rice/Potato leaf classifier (ONNX/HF)
│   │   ├── weather.py         # Stub: Open-Meteo 48h spray-window guard
│   │   ├── rag.py             # Stub: BARI/BRRI JSON vector retrieval
│   │   ├── llm.py             # Stub: Gemini Flash (Bangla audio + advisory)
│   │   └── tts.py             # Stub: Bangla TTS (Gemini TTS / edge-tts)
│   ├── data/
│   │   └── .gitkeep
│   └── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Local Setup

### 1. Environment Variables
Copy `.env.example` to `.env` (never commit `.env`):

```bash
cp .env.example .env
```

### 2. Backend Setup (FastAPI)

```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Verify health endpoint:
```bash
curl http://localhost:8000/api/health
# {"status":"ok","service":"foshollens-api"}
```

> **Note:** Until the AI/weather/RAG/TTS modules are plugged in, `POST /api/analyze` intentionally returns **HTTP 501 (Not Implemented)** rather than fabricated diagnoses.

### 3. Frontend Setup (React + Vite PWA)

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser. To build for production:

```bash
npm run build
```

---

## Links

- **Live Deployment Link:** _[To be added after deployment]_
- **Demo Video Link:** _[To be added before submission]_
