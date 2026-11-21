# ফসল লেন্স (Fasol Lens)

ফসল লেন্স — ফসলের ছবি দিয়ে রোগ শনাক্ত করার সহজ ওয়েব অ্যাপ, বাংলাদেশের কৃষকদের জন্য।

## Status

Frontend এবং API দুটোই কাজ করছে, এবং পুরো পাইপলাইন end-to-end চলে।

- **Text এখনই আসে।** কোনো API key ছাড়াই অ্যাপ চলে — LLM কনফিগার না থাকলে
  knowledge base থেকে সরাসরি বাংলা পরামর্শ লেখা হয় (`generatedBy: "template"`)।
- **কণ্ঠও এখনই আসে।** TTS লোকাল, ONNX, CPU-তে চলে; ffmpeg দিয়ে Opus-এ নামানো হয়
  (২২ KB/মিনিটের মতো)।
- **আবহাওয়া বাস্তব।** Open-Meteo থেকে আনা হয়, কিন্তু স্প্রে করার সিদ্ধান্ত কোডে
  হিসাব হয় — LLM কখনো পূর্বাভাস লেখে না।
- **Vision আছে, দুইটি মডেল থেকে বেছে নেওয়া যায়।** `VISION_MODEL` দিয়ে ঠিক হয়:
  `efficientnet` (৯ শ্রেণি, শুধু ধান ও আলু, ১৬ MB, ~৪০ ms, স্থানীয় ফাইল) অথবা
  `vit` (১৩ শ্রেণি, ভুট্টা/গমসহ, ২২ MB, ~১৪০ ms, Hugging Face থেকে নামে)।
  ছবি থেকে কিছু শনাক্ত না হলে ফলাফল স্পষ্টভাবে "নিশ্চিতভাবে শনাক্ত করা যায়নি"
  বলে, কোনো মাত্রা লেখে না, শুধু ফসলের সাধারণ পরিচর্যা দেখায়।

LLM চালু করতে `.env`-এ নিজের key দিন (`LLM_API_KEY`)। key না দিলে অ্যাপ ব্যর্থ হয় না,
শুধু template লেখকটি ব্যবহার হয়।

## Stack

| Layer     | Technology                                                      |
| --------- | --------------------------------------------------------------- |
| Frontend  | React 19, TypeScript, Vite, React Router, Lucide icons          |
| Backend   | Python 3.12 (uv) + FastAPI, SQLite (WAL), in-process job runner |
| Speech    | `indic-tiny-tts-bengali` (ONNX, CPU) + ffmpeg → Opus            |
| Weather   | Open-Meteo (key-free) — verdict computed in code                |
| Contracts | pydantic -> OpenAPI; zod mirror keeps the frontend in step      |

## Layout

```
foshol_lens/
├── apps/
│   ├── api/                      # Python / FastAPI
│   │   ├── app/
│   │   │   ├── routes.py         # upload, status, result, media, audio
│   │   │   ├── jobs/             # in-process stage runner
│   │   │   ├── tts/              # prepare_text + ONNX synthesis + ffmpeg
│   │   │   ├── llm/              # writer, validator, template fallback
│   │   │   ├── grounding/        # knowledge base, districts, weather
│   │   │   ├── contracts.py      # pydantic models (source of truth)
│   │   │   ├── store.py          # SQLite
│   │   │   └── config.py         # settings
│   │   ├── tests/                # pytest: pipeline, safety, weather
│   │   └── storage/              # sqlite db + media (gitignored)
│   └── frontend/                 # @foshol-lens/frontend
│       ├── public/favicon.svg
│       └── src/
│           ├── components/       # AppShell, nav, upload, result, audio, weather
│           ├── pages/            # HomePage, HistoryPage, AnalysisPage
│           ├── lib/              # HTTP client, image prep, formatting
│           ├── styles/           # design tokens + responsive CSS
│           ├── App.tsx           # routes
│           └── main.tsx
├── packages/
│   └── contracts/                # @foshol-lens/contracts (zod + types)
├── docs/
├── tsconfig.base.json
└── package.json
```

### Dependency direction

```
apps/api/contracts.py  ──>  packages/contracts  ──>  apps/frontend
```

`apps/api/app/contracts.py`-ই একমাত্র সত্য। OpenAPI থেকে TypeScript generate করা
এখনো wired up নয়, তাই zod mirror-টি হাতে মিলিয়ে রাখা হয়।

## Getting started

```bash
npm install

# পাইথন API (প্রথমবার)
cd apps/api
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
cd ../..

npm run dev            # API + frontend একসাথে
```

`npm run dev` তিনটি প্রসেস চালায়: FastAPI (`:4000`), contract watcher, এবং Vite
(`:5173`)। আলাদা করতে চাইলে `npm run dev:api` ও `npm run dev:frontend`।

**API না চললে ছবি বিশ্লেষণ হবে না।** ফ্রন্টএন্ড নিজে থেকে খোলে, কিন্তু আপলোড
করলে "সার্ভারে পৌঁছানো যাচ্ছে না" দেখাবে। তাই শুধু `dev:frontend` চালাবেন না।

API নিজে থেকে storage ডিরেক্টরি, ডেটাবেস ও মডেল ডাউনলোড সামলায়। প্রথম TTS অনুরোধে
মডেলটি একবার নামে (~৩০০ MB peak RSS), তারপর ক্যাশ থেকে চলে।

### Environment

`.env.example` কপি করে `apps/api/.env` বানান। একমাত্র ঐচ্ছিক মান `LLM_API_KEY`;
বাকি সব ডিফল্টেই চলে।

## Scripts

| Command                | What it does                         |
| ---------------------- | ------------------------------------ |
| `npm run dev:api`      | FastAPI (uvicorn, port 4000)         |
| `npm run dev:frontend` | Vite dev server                      |
| `npm run build`        | contracts → frontend build           |
| `npm run typecheck`    | frontend + contracts type-check      |
| `npm run test`         | contracts test                       |
| `npm run format`       | Prettier across the JS workspaces    |
| `npm run clean`        | build output + node_modules মুছে ফেলা |

Python পরীক্ষা:

```bash
cd apps/api && .venv/bin/python -m pytest
```

## Frontend design decisions

- **ব্র্যান্ড:** ফসল লেন্স (কৃষিবন্ধু নয়)।
- **Navigation:** desktop-এ top nav (≥768px), mobile-এ fixed bottom tabs — দুই জায়গাতেই
  শুধু হোম ও ইতিহাস।
- **Typography:** কোনো web font নেই — system font stack। প্রতিটি ডিভাইস নিজের পছন্দের
  বাংলা ফন্টে দেখায়, তাই কোনো download বা layout shift হয় না। `letter-spacing` ব্যবহার
  করা হয় না, কারণ তাতে বাংলা যুক্তাক্ষর (যেমন লেন্স) ভেঙে যায়।
- **সংখ্যা:** সব দৃশ্যমান সংখ্যা বাংলা অঙ্কে (০-৯)। আইডি ও URL ASCII-ই থাকে।
- **Image handling:** আপলোডের আগে image → 1280px-এ downscale + re-encode। সার্ভার
  আবার resize করে এবং **EXIF ফেলে দেয়**, কারণ ফোন ছবিতে GPS লেখে।
- **Analysis shape:** confidence না থাকলে জোর করে diagnosis দেখানো হয় না — বরং স্পষ্ট
  "শনাক্ত করা যায়নি" বার্তা ও নতুন ছবি দেওয়ার সুযোগ দেওয়া হয়।

## Backend নকশা, সংক্ষেপে

চারটি ধাপ: `reading → grounding → writing → speaking`। বিস্তারিত
[`docs/backend-architecture.md`](docs/backend-architecture.md)-এ।

- **Grounding-ই একমাত্র তথ্য বলতে পারে।** LLM শুধু একটি field ফেরায় (`spoken`),
  এবং তার আউটপুট কঠোরভাবে যাচাই হয়: বাংলা অক্ষর ছাড়া কিছু নয়, Latin নেই,
  knowledge base-এ না থাকা কোনো ওষুধ নেই।
- **Template fallback প্রথম শ্রেণির পথ।** LLM না থাকলে, rate limit-এ পড়লে বা
  অকেজো লেখা ফেরালে একই তথ্য থেকে সরাসরি বাক্য তৈরি হয়।
- **আবহাওয়ার সিদ্ধান্ত কোডে।** বৃষ্টি ≥ ৬০% বা বাতাস ≥ ১৫ কিমি/ঘণ্টা হলে স্প্রে নয়।
  LLM শুধু বাক্যটি সাজায়।
- **অডিও আলাদা subresource।** লেখা আগে আসে, কণ্ঠ পরে। `/audio` তখন পর্যন্ত `202`
  ফেরায়, এবং প্লেয়ার নিজেই অপেক্ষা করে। অডিও কখনো autoplay হয় না।
- **TTS-এর দুইটি বাস্তব ত্রুটি সামলানো হয়।** মডেলটি `/`-কে "বা" পড়ে এবং Latin
  অক্ষর নিঃশব্দে মুছে ফেলে — তাই `prepare_text` আগেই ঠিক করে, আর কোনো ওষুধের নাম
  হারিয়ে গেলে সেটি শব্দ বানানোর বদলে জোরে ব্যর্থ হয়।
