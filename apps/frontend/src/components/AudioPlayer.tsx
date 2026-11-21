import type { AudioInfo } from '@foshol-lens/contracts';
import { Pause, Play } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { api, mediaUrl } from '../lib/api-client';
import { toBengaliDigits } from '../lib/format';

const AUDIO_POLL_MS = 1200;
const AUDIO_POLL_TIMEOUT_MS = 90_000;

function formatDuration(ms: number): string {
  const total = Math.round(ms / 1000);
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${toBengaliDigits(minutes)}:${toBengaliDigits(String(seconds).padStart(2, '0'))}`;
}

/**
 * The spoken narrative.
 *
 * `preload="none"` is deliberate: a 22 KB Opus file is cheap, but on a rural
 * mobile connection even that should be opt-in, and audio is advice rather than
 * an alarm, so it never plays on its own.
 *
 * Synthesis finishes after the text is served, so a `pending` clip is polled
 * rather than written off - the reader is already reading while the voice is
 * being made.
 */
export function AudioPlayer({ audio: initial, analysisId }: { audio: AudioInfo; analysisId: string }) {
  const ref = useRef<HTMLAudioElement | null>(null);
  const [audio, setAudio] = useState<AudioInfo>(initial);
  const [playing, setPlaying] = useState(false);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setAudio(initial);
  }, [initial]);

  // Poll only while there is something to wait for.
  useEffect(() => {
    if (audio.status !== 'pending') return;

    let cancelled = false;
    const startedAt = Date.now();

    async function poll() {
      if (cancelled || Date.now() - startedAt > AUDIO_POLL_TIMEOUT_MS) return;

      try {
        const result = await api.getResult(analysisId);
        if (cancelled) return;
        if (result.audio) setAudio(result.audio);
        if (result.audio && result.audio.status !== 'pending') return;
      } catch {
        // Transient: keep waiting until the deadline.
      }

      if (!cancelled) {
        window.setTimeout(poll, AUDIO_POLL_MS);
      }
    }

    const handle = window.setTimeout(poll, AUDIO_POLL_MS);
    return () => {
      cancelled = true;
      window.clearTimeout(handle);
    };
  }, [analysisId, audio.status]);

  const src = mediaUrl(audio.url);

  useEffect(() => {
    return () => {
      ref.current?.pause();
    };
  }, []);

  if (audio.status === 'failed' || failed) return null;

  if (audio.status === 'pending') {
    return (
      <div className="audio audio--pending" aria-live="polite">
        <span className="spinner" aria-hidden="true" />
        <p className="audio__title">কণ্ঠে রূপ দেওয়া হচ্ছে...</p>
      </div>
    );
  }

  if (audio.status !== 'ready' || !src) return null;

  async function toggle() {
    const element = ref.current;
    if (!element) return;
    if (element.paused) {
      try {
        await element.play();
      } catch {
        setFailed(true);
      }
    } else {
      element.pause();
    }
  }

  return (
    <div className="audio">
      <button
        type="button"
        className="audio__button"
        onClick={toggle}
        aria-label={playing ? 'পড়া বন্ধ করুন' : 'পরামর্শটি শুনুন'}
      >
        {playing ? (
          <Pause size={20} strokeWidth={2.2} aria-hidden="true" />
        ) : (
          <Play size={20} strokeWidth={2.2} aria-hidden="true" />
        )}
      </button>

      <div className="audio__body">
        <p className="audio__title">{playing ? 'শোনা হচ্ছে...' : 'পরামর্শটি শুনুন'}</p>
        {audio.durationMs ? <p className="audio__meta">{formatDuration(audio.durationMs)}</p> : null}
      </div>

      <audio
        ref={ref}
        src={src}
        preload="none"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onError={() => setFailed(true)}
      />
    </div>
  );
}
