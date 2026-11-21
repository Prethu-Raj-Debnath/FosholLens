const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
).replace(/\/$/, '');

export { API_BASE_URL };

export async function fetchHealth() {
  const response = await fetch(`${API_BASE_URL}/api/health`);
  if (!response.ok) {
    throw new Error(`Health check failed (${response.status})`);
  }
  return response.json();
}

export async function analyzeCrop({ imageFile, crop, questionText, audioBlob, lat, lng }) {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('crop', crop);

  if (questionText && questionText.trim()) {
    formData.append('question_text', questionText.trim());
  }
  if (audioBlob) {
    formData.append('audio', audioBlob, 'question.webm');
  }
  if (typeof lat === 'number') {
    formData.append('lat', String(lat));
  }
  if (typeof lng === 'number') {
    formData.append('lng', String(lng));
  }

  const response = await fetch(`${API_BASE_URL}/api/analyze`, {
    method: 'POST',
    body: formData,
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.detail || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }

  return payload;
}
