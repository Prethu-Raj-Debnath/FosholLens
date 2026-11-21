import type { Weather } from '@foshol-lens/contracts';
import { CloudRain, Wind } from 'lucide-react';
import { toBengaliDigits } from '../lib/format';

/**
 * Weather facts plus the spray verdict.
 *
 * The verdict is computed server-side in code. The LLM only phrases it, so this
 * card and the spoken narrative can never disagree about whether spraying is
 * sensible.
 */
export function WeatherCard({ weather }: { weather: Weather }) {
  const blocking = !weather.sprayWindowOk;

  return (
    <section className={`card ${blocking ? 'card--warn' : 'card--plain'}`}>
      <p className="card__eyebrow">{weather.district} জেলার আবহাওয়া</p>

      <h2 className="card__label">
        {weather.sprayWindowOk ? 'এখন স্প্রে করার অনুকূল সময়' : 'এখনই স্প্রে করবেন না'}
      </h2>

      <p className="card__text">{weather.reason}।</p>

      <div className="weather-facts">
        {typeof weather.rainProbabilityNext12h === 'number' && (
          <span className="weather-fact">
            <CloudRain size={17} strokeWidth={2.1} aria-hidden="true" />
            বৃষ্টির সম্ভাবনা {toBengaliDigits(weather.rainProbabilityNext12h)}%
          </span>
        )}
        {typeof weather.windSpeedMax12h === 'number' && (
          <span className="weather-fact">
            <Wind size={17} strokeWidth={2.1} aria-hidden="true" />
            বাতাস {toBengaliDigits(Math.round(weather.windSpeedMax12h))} কিমি/ঘণ্টা
          </span>
        )}
      </div>

      {weather.source && <p className="confidence__note">তথ্যসূত্র: {weather.source}</p>}
    </section>
  );
}
