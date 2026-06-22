const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export async function predictMango({ file, temperature, humidity, cultivar = "Alphonso" }) {
  const form = new FormData();
  form.append("file", file);
  form.append("temperature", String(temperature));
  form.append("humidity", String(humidity));
  form.append("cultivar", String(cultivar || "Alphonso"));

  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const text = await res.text();
    let message = text || `Predict failed (${res.status})`;
    try {
      const data = JSON.parse(text);
      if (typeof data.detail === "string") {
        message = data.detail;
      }
    } catch {
      // keep raw text fallback
    }
    throw new Error(message);
  }
  return res.json();
}

export function exampleImageUrl(ripenessFolder) {
  // Cache-bust so you see fresh thumbnails.
  const t = Date.now();
  return `${API_BASE}/example/${encodeURIComponent(ripenessFolder)}?t=${t}`;
}

