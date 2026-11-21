export function formatBengaliDate(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';

  const formatter = new Intl.DateTimeFormat('bn-BD', {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  });

  try {
    return formatter.format(date);
  } catch {
    return date.toLocaleDateString('bn-BD');
  }
}

/**
 * ASCII digits to Bengali ones.
 *
 * Every visible string in this app is Bangla, and `97%` inside an otherwise
 * Bengali sentence reads as a rendering bug rather than a bilingual choice.
 * Applied at the edge, so ids and URLs keep their ASCII digits.
 */
const BENGALI_DIGITS: Record<string, string> = {
  '0': '০',
  '1': '১',
  '2': '২',
  '3': '৩',
  '4': '৪',
  '5': '৫',
  '6': '৬',
  '7': '৭',
  '8': '৮',
  '9': '৯',
};

export function toBengaliDigits(value: number | string): string {
  return String(value).replace(/[0-9]/g, (digit) => BENGALI_DIGITS[digit] ?? digit);
}
