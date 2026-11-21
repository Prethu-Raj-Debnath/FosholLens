import { useEffect, useRef, useState } from 'react';
import { analyzeCrop, fetchHealth } from './api';

const SAFETY_STYLES = {
  green: 'bg-emerald-100 text-emerald-800 border-emerald-200',
  amber: 'bg-amber-100 text-amber-800 border-amber-200',
  red: 'bg-rose-100 text-rose-800 border-rose-200',
};

export default function App() {
  const [crop, setCrop] = useState('rice');
  const [imageFile, setImageFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [questionText, setQuestionText] = useState('');
  const [health, setHealth] = useState({ state: 'checking', service: '' });
  const [submitting, setSubmitting] = useState(false);
  const [statusMessage, setStatusMessage] = useState(null);
  const [result, setResult] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    let active = true;
    fetchHealth()
      .then((data) => {
        if (active) {
          setHealth({ state: 'online', service: data.service || 'foshollens-api' });
        }
      })
      .catch(() => {
        if (active) {
          setHealth({ state: 'offline', service: '' });
        }
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function handleImageChange(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setImageFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setStatusMessage(null);
  }

  function handleClearImage() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setImageFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (!imageFile) {
      setStatusMessage({
        type: 'warning',
        text: 'অনুগ্রহ করে পাতার একটি ছবি তুলুন বা আপলোড করুন। (Please upload a leaf photo first.)',
      });
      return;
    }

    setSubmitting(true);
    setStatusMessage(null);
    try {
      const data = await analyzeCrop({
        imageFile,
        crop,
        questionText,
      });
      setResult(data);
    } catch (err) {
      setResult(null);
      setStatusMessage({
        type: err.status === 501 ? 'info' : 'error',
        text:
          err.status === 501
            ? `Backend stub active (HTTP 501): ${err.message}`
            : err.message || 'Analysis request failed.',
      });
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      {/* Top Header */}
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/95 backdrop-blur">
        <div className="mx-auto flex max-w-xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2.5">
            <img src="/icon.svg" alt="FosholLens" className="h-9 w-9 rounded-xl shadow-xs" />
            <div>
              <h1 className="text-base font-bold leading-tight text-emerald-900">
                FosholLens <span className="font-semibold text-emerald-700">| ফসললেন্স</span>
              </h1>
              <p className="text-xs text-slate-500">
                ধান ও আলু রোগ শনাক্তকরণ ও আবহাওয়া-ভিত্তিক পরামর্শ
              </p>
            </div>
          </div>

          {/* Backend Health Badge */}
          <span
            title={health.service || 'Backend API status'}
            className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${
              health.state === 'online'
                ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
                : health.state === 'offline'
                  ? 'border-rose-200 bg-rose-50 text-rose-700'
                  : 'border-slate-200 bg-slate-100 text-slate-600'
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                health.state === 'online'
                  ? 'bg-emerald-500'
                  : health.state === 'offline'
                    ? 'bg-rose-500'
                    : 'bg-slate-400 animate-pulse'
              }`}
            />
            {health.state === 'online'
              ? 'API Online'
              : health.state === 'offline'
                ? 'API Offline'
                : 'Checking...'}
          </span>
        </div>
      </header>

      <main className="mx-auto max-w-xl space-y-5 px-4 py-5 pb-12">
        {/* Input Form Card */}
        <form
          onSubmit={handleSubmit}
          className="space-y-5 rounded-2xl border border-slate-200 bg-white p-4 shadow-xs"
        >
          {/* Crop Toggle */}
          <div>
            <label className="mb-2 block text-sm font-semibold text-slate-700">
              ফসল নির্বাচন করুন / Select Crop
            </label>
            <div className="grid grid-cols-2 gap-2 rounded-xl bg-slate-100 p-1">
              <button
                type="button"
                onClick={() => setCrop('rice')}
                className={`rounded-lg py-2.5 text-sm font-semibold transition ${
                  crop === 'rice'
                    ? 'bg-emerald-700 text-white shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                🌾 ধান / Rice
              </button>
              <button
                type="button"
                onClick={() => setCrop('potato')}
                className={`rounded-lg py-2.5 text-sm font-semibold transition ${
                  crop === 'potato'
                    ? 'bg-emerald-700 text-white shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                🥔 আলু / Potato
              </button>
            </div>
          </div>

          {/* Leaf Photo Upload + Framing Guide */}
          <div>
            <label className="mb-2 block text-sm font-semibold text-slate-700">
              আক্রান্ত পাতার ছবি / Leaf Photo
            </label>
            <div className="relative overflow-hidden rounded-xl border-2 border-dashed border-emerald-300 bg-emerald-50/40">
              {previewUrl ? (
                <div className="relative">
                  <img
                    src={previewUrl}
                    alt="Selected leaf preview"
                    className="h-60 w-full object-cover"
                  />
                  <button
                    type="button"
                    onClick={handleClearImage}
                    className="absolute top-2.5 right-2.5 rounded-lg bg-slate-900/75 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-900"
                  >
                    পরিবর্তন করুন / Change
                  </button>
                </div>
              ) : (
                <label className="flex h-56 cursor-pointer flex-col items-center justify-center p-4 text-center">
                  {/* Simple Framing Guide Box */}
                  <div className="relative mb-3 flex h-28 w-44 items-center justify-center">
                    <span className="absolute top-0 left-0 h-5 w-5 rounded-tl-md border-t-3 border-l-3 border-emerald-600" />
                    <span className="absolute top-0 right-0 h-5 w-5 rounded-tr-md border-t-3 border-r-3 border-emerald-600" />
                    <span className="absolute bottom-0 left-0 h-5 w-5 rounded-bl-md border-b-3 border-l-3 border-emerald-600" />
                    <span className="absolute right-0 bottom-0 h-5 w-5 rounded-br-md border-r-3 border-b-3 border-emerald-600" />
                    <span className="px-3 text-xs font-medium text-emerald-800">
                      পাতাটি ফ্রেমের মাঝখানে রাখুন
                      <br />
                      Center leaf inside frame
                    </span>
                  </div>
                  <span className="rounded-lg bg-emerald-700 px-4 py-2 text-xs font-semibold text-white">
                    📷 ছবি তুলুন বা আপলোড করুন / Capture or Upload
                  </span>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    capture="environment"
                    onChange={handleImageChange}
                    className="hidden"
                  />
                </label>
              )}
            </div>
          </div>

          {/* Optional Text Question + Bangla Voice Placeholder */}
          <div className="space-y-2.5">
            <label
              htmlFor="question-input"
              className="block text-sm font-semibold text-slate-700"
            >
              আপনার প্রশ্ন (ঐচ্ছিক) / Question (Optional)
            </label>
            <div className="flex gap-2">
              <input
                id="question-input"
                type="text"
                value={questionText}
                onChange={(e) => setQuestionText(e.target.value)}
                placeholder="যেমন: পাতায় বাদামি দাগ হয়েছে, এখন কী করব?"
                className="w-full rounded-xl border border-slate-300 bg-white px-3.5 py-2.5 text-sm placeholder:text-slate-400 focus:border-emerald-600 focus:outline-none"
              />
              <button
                type="button"
                disabled
                title="Bangla voice recording (5-10s) will be wired in the next milestone"
                className="shrink-0 cursor-not-allowed rounded-xl border border-slate-200 bg-slate-100 px-3.5 py-2.5 text-xs font-semibold text-slate-500"
              >
                🎙️ ভয়েস (শীঘ্রই)
              </button>
            </div>
            <p className="text-xs text-slate-500">
              ডিফল্ট অবস্থান: কুমিল্লা (২৩.৪৬০৭° উ, ৯১.১৮০৯° পূ) · Default location: Cumilla
            </p>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={submitting}
            className="w-full rounded-xl bg-emerald-700 py-3 text-sm font-semibold text-white shadow-xs transition hover:bg-emerald-800 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {submitting
              ? 'বিশ্লেষণ করা হচ্ছে... / Analyzing...'
              : 'বিশ্লেষণ করুন / Analyze Leaf'}
          </button>

          {statusMessage && (
            <div
              className={`rounded-xl border px-3.5 py-2.5 text-xs leading-relaxed ${
                statusMessage.type === 'error'
                  ? 'border-rose-200 bg-rose-50 text-rose-800'
                  : statusMessage.type === 'warning'
                    ? 'border-amber-200 bg-amber-50 text-amber-800'
                    : 'border-sky-200 bg-sky-50 text-sky-900'
              }`}
            >
              {statusMessage.text}
            </div>
          )}
        </form>

        {/* Results Card (Empty-State Ready for Wired Pipeline) */}
        <section className="space-y-4 rounded-2xl border border-slate-200 bg-white p-4 shadow-xs">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <div>
              <h2 className="text-sm font-bold text-slate-800">
                বিশ্লেষণের ফলাফল / Advisory Results
              </h2>
              <p className="text-xs text-slate-500">
                Vision · 48h Spray Window · BARI/BRRI RAG · Bangla Audio
              </p>
            </div>
            {result?.vision?.status && (
              <span className="rounded-full bg-emerald-100 px-2.5 py-0.5 text-xs font-semibold text-emerald-800">
                {result.vision.status}
              </span>
            )}
          </div>

          {/* 1. Vision Section */}
          <div className="rounded-xl border border-slate-100 bg-slate-50 p-3">
            <h3 className="mb-1.5 text-xs font-bold uppercase tracking-wide text-slate-500">
              ১. রোগ শনাক্তকরণ / Vision Diagnosis
            </h3>
            {result?.vision?.top_predictions?.length ? (
              <ul className="space-y-1.5">
                {result.vision.top_predictions.map((pred) => (
                  <li
                    key={pred.label}
                    className="flex items-center justify-between rounded-lg bg-white px-3 py-2 text-sm"
                  >
                    <span className="font-medium text-slate-800">{pred.label}</span>
                    <span className="font-semibold text-emerald-700">
                      {Math.round(pred.confidence * 100)}%
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-xs text-slate-500">
                পাতার ছবি বিশ্লেষণ করলে এখানে সম্ভাব্য রোগ ও আত্মবিশ্বাসের হার দেখা যাবে।
              </p>
            )}
          </div>

          {/* 2. 48h Spray Timeline */}
          <div className="rounded-xl border border-slate-100 bg-slate-50 p-3">
            <h3 className="mb-1.5 text-xs font-bold uppercase tracking-wide text-slate-500">
              ২. ৪৮ ঘণ্টার স্প্রে সময়সূচি / 48h Spray-Window Guard
            </h3>
            {result?.weather ? (
              <div className="space-y-2">
                <p className="text-xs font-medium text-slate-700">
                  {result.weather.safest_window_summary}
                </p>
                <div className="flex gap-2 overflow-x-auto pb-1">
                  {result.weather.next_48h_windows.slice(0, 8).map((slot) => (
                    <div
                      key={slot.time_iso}
                      className={`min-w-24 rounded-lg border px-2.5 py-2 text-xs ${
                        SAFETY_STYLES[slot.safety] || SAFETY_STYLES.amber
                      }`}
                    >
                      <div className="font-semibold">{slot.time_iso.slice(11, 16)}</div>
                      <div>বৃষ্টি: {slot.rain_prob}%</div>
                      <div>বাতাস: {slot.wind_kmh} km/h</div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-500">
                Open-Meteo ৪৮ ঘণ্টার বৃষ্টি ও বাতাসের পূর্বাভাস অনুযায়ী নিরাপদ স্প্রে সময় এখানে দেখা যাবে।
              </p>
            )}
          </div>

          {/* 3. Bangla Advisory */}
          <div className="rounded-xl border border-slate-100 bg-slate-50 p-3">
            <h3 className="mb-1.5 text-xs font-bold uppercase tracking-wide text-slate-500">
              ৩. বাংলা পরামর্শ ও নিরাপত্তা / Bangla Advisory
            </h3>
            {result?.advisory ? (
              <div className="space-y-2 text-sm text-slate-800">
                <p className="whitespace-pre-line leading-relaxed">
                  {result.advisory.bangla_text}
                </p>
                <p className="rounded-lg bg-amber-50 p-2 text-xs text-amber-900">
                  🧤 {result.advisory.ppe_reminder}
                </p>
                {result.advisory.officer_referral && (
                  <p className="rounded-lg bg-rose-50 p-2 text-xs font-semibold text-rose-800">
                    ⚠️ নিশ্চিত হতে নিকটস্থ উপ-সহকারী কৃষি কর্মকর্তার সাথে পরামর্শ করুন।
                  </p>
                )}
              </div>
            ) : (
              <p className="text-xs text-slate-500">
                BARI ও BRRI নির্দেশিকা অনুযায়ী বাংলা পরামর্শ, পিপিই সতর্কতা ও উৎস এখানে প্রদর্শিত হবে।
              </p>
            )}
          </div>

          {/* 4. Bangla Audio Player */}
          <div className="rounded-xl border border-slate-100 bg-slate-50 p-3">
            <h3 className="mb-1.5 text-xs font-bold uppercase tracking-wide text-slate-500">
              ৪. বাংলা অডিও পরামর্শ / Audio Player
            </h3>
            {result?.audio_url_or_base64 ? (
              <audio controls className="w-full" src={result.audio_url_or_base64} />
            ) : (
              <p className="text-xs text-slate-500">
                পরামর্শ প্রস্তুত হলে এখানে বাংলা অডিও প্লেয়ার চালু হবে।
              </p>
            )}
          </div>
        </section>

        <footer className="text-center text-xs text-slate-400">
          CoU AI Hackathon 2026 · Team Tesseract
        </footer>
      </main>
    </div>
  );
}
