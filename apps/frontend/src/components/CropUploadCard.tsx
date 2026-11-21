import type { AnalysisStage } from '@foshol-lens/contracts';
import { ScanSearch } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, ApiError } from '../lib/api-client';
import { prepareImage } from '../lib/image';
import { useDistrict } from '../lib/use-district';
import { AnalysisLoadingState, type UploadPhase } from './AnalysisLoadingState';
import { ImageDropzone } from './ImageDropzone';
import { ImagePreview } from './ImagePreview';

const MAX_BYTES = 8 * 1024 * 1024;

const CROPS = [
  { id: 'rice', name: 'ধান' },
  { id: 'potato', name: 'আলু' },
  { id: 'tomato', name: 'টমেটো' },
  { id: 'chili', name: 'মরিচ' },
] as const;

/**
 * Turn a failure into something a farmer can act on.
 *
 * "আবার চেষ্টা করুন" is only honest if retrying could work. A stopped API is
 * not a bad photo, and telling someone to retake their picture when the server
 * is simply down wastes their time and data.
 */
function describeFailure(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.code === 'network_error') {
      return 'সার্ভারে পৌঁছানো যাচ্ছে না। ইন্টারনেট সংযোগ দেখুন, তারপর আবার চেষ্টা করুন';
    }
    if (error.code === 'timeout') {
      return 'বিশ্লেষণ শেষ হয়নি। একটু পরে আবার চেষ্টা করুন';
    }
    if (error.code === 'analysis_failed') {
      return 'ছবিটি বিশ্লেষণ করা যায়নি। আরও পরিষ্কার ছবি দিয়ে আবার চেষ্টা করুন';
    }
    if (error.status === 413) {
      return 'ছবির আকার বেশি বড়। ছোট ছবি দিয়ে আবার চেষ্টা করুন';
    }
    if (error.status >= 500) {
      return 'সার্ভারে সমস্যা হচ্ছে। কিছুক্ষণ পরে আবার চেষ্টা করুন';
    }
  }
  return 'বিশ্লেষণ করা যায়নি। আবার চেষ্টা করুন';
}

function validateFile(file: File): string | null {
  if (!file.type.startsWith('image/')) {
    return 'শুধুমাত্র ছবি আপলোড করুন';
  }
  if (file.size > MAX_BYTES) {
    return 'ছবির আকার ৮ মেগাবাইটের কম হতে হবে';
  }
  return null;
}

export function CropUploadCard() {
  const navigate = useNavigate();
  const { district, districts, choose } = useDistrict();

  const [file, setFile] = useState<File | null>(null);
  const [crop, setCrop] = useState<string>('rice');
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [phase, setPhase] = useState<UploadPhase>('idle');
  const [stage, setStage] = useState<AnalysisStage | undefined>();
  const [errorMessage, setErrorMessage] = useState<string | undefined>();

  // Object URL for the on-screen preview, revoked on replace/unmount.
  const previewRef = useRef<string | null>(null);
  // Guards against double-submits from rapid taps/keyboard repeat.
  const submittingRef = useRef(false);

  const clearPreview = useCallback(() => {
    if (previewRef.current) {
      URL.revokeObjectURL(previewRef.current);
      previewRef.current = null;
    }
    setPreviewUrl(null);
  }, []);

  useEffect(() => clearPreview, [clearPreview]);

  function handleFile(nextFile: File) {
    if (submittingRef.current) return;

    const problem = validateFile(nextFile);
    if (problem) {
      setErrorMessage(problem);
      setPhase('error');
      return;
    }

    clearPreview();
    previewRef.current = URL.createObjectURL(nextFile);
    setPreviewUrl(previewRef.current);

    setFile(nextFile);
    setErrorMessage(undefined);
    setStage(undefined);
    setPhase('idle');
  }

  function handleReplace() {
    if (submittingRef.current) return;
    clearPreview();
    setFile(null);
    setPhase('idle');
    setStage(undefined);
    setErrorMessage(undefined);
  }

  async function handleAnalyze() {
    if (!file || submittingRef.current) return;

    submittingRef.current = true;
    setPhase('uploading');
    setStage(undefined);
    setErrorMessage(undefined);

    let preparedPreview: string | null = null;

    try {
      const prepared = await prepareImage(file);
      preparedPreview = prepared.previewUrl;

      setPhase('analyzing');
      setStage('queued');

      const created = await api.createAnalysis({
        imageBlob: prepared.blob,
        crop,
        district: district || null,
      });

      const result = await api.waitForResult(created.id, (nextStage) => setStage(nextStage));

      setPhase('success');
      navigate(`/analysis/${result.id}`);
    } catch (error) {
      setPhase('error');
      setErrorMessage(describeFailure(error));
    } finally {
      // The preview we created for upload is not the one on screen; release it.
      if (preparedPreview) URL.revokeObjectURL(preparedPreview);
      submittingRef.current = false;
    }
  }

  const busy = phase === 'uploading' || phase === 'analyzing';
  const hasImage = Boolean(file && previewUrl);

  return (
    <section className="upload-card" aria-labelledby="upload-title">
      <h2 id="upload-title" className="upload-card__title">
        ফসলের ছবি আপলোড করুন
      </h2>
      <p className="upload-card__hint">আপনার ফসলের পরিষ্কার ছবি দিন</p>

      {hasImage && previewUrl ? (
        <ImagePreview
          src={previewUrl}
          alt="নির্বাচিত ফসলের ছবি"
          disabled={busy}
          onReplace={handleReplace}
        />
      ) : (
        <ImageDropzone onFile={handleFile} disabled={busy} />
      )}

      <div className="field-row">
        <label className="field">
          <span className="field__label">ফসল</span>
          <select
            className="field__input"
            value={crop}
            disabled={busy}
            onChange={(event) => setCrop(event.target.value)}
          >
            {CROPS.map((option) => (
              <option key={option.id} value={option.id}>
                {option.name}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span className="field__label">জেলা</span>
          <select
            className="field__input"
            value={district}
            disabled={busy}
            onChange={(event) => choose(event.target.value)}
          >
            <option value="">নির্বাচন করুন</option>
            {districts.map((option) => (
              <option key={option.id} value={option.id}>
                {option.name}
              </option>
            ))}
          </select>
        </label>
      </div>

      <button
        type="button"
        className="button-primary"
        disabled={!file || busy}
        onClick={handleAnalyze}
      >
        {busy ? (
          <span className="spinner" aria-hidden="true" />
        ) : (
          <ScanSearch size={19} strokeWidth={2.1} aria-hidden="true" />
        )}
        {busy ? 'অপেক্ষা করুন...' : 'ছবি বিশ্লেষণ করুন'}
      </button>

      <AnalysisLoadingState phase={phase} stage={stage} errorMessage={errorMessage} />
    </section>
  );
}
