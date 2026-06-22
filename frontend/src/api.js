const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export async function predictMango({ file, temperature, humidity, cultivar = "Alphonso" }) {
  if (!file) {
    throw new Error("No image file provided.");
  }

  const form = new FormData();
  form.append("file", file);
  form.append("temperature", String(temperature));
  form.append("humidity", String(humidity));
  form.append("cultivar", String(cultivar || "Alphonso"));

  let res;
  try {
    res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: form,
    });
  } catch (networkErr) {
    // fetch throws on network failure / CORS / server unreachable, not on HTTP error status
    throw new Error(
      "Could not reach the prediction server. Check your connection or try again."
    );
  }

  if (!res.ok) {
    let message = `Predict failed (${res.status})`;
    try {
      const data = await res.json();
      if (typeof data?.detail === "string") {
        message = data.detail;
      }
    } catch {
      try {
        const text = await res.text();
        if (text) message = text;
      } catch {
        // keep default status-based message
      }
    }
    throw new Error(message);
  }

  try {
    return await res.json();
  } catch {
    throw new Error("Received an invalid response from the server.");
  }
}

export function exampleImageUrl(ripenessFolder) {
  if (!ripenessFolder) {
    throw new Error("ripenessFolder is required.");
  }
  // Cache-bust so you see fresh thumbnails.
  const t = Date.now();
  return `${API_BASE}/example/${encodeURIComponent(ripenessFolder)}?t=${t}`;
}