function sanitizeDegree(text) {
  // Some terminals may display the degree sign incorrectly; backend message may contain U+FFFD.
  return String(text || "").replace(/\uFFFD/g, "°");
}

function formatPercent(x) {
  const n = Number(x);
  if (Number.isFinite(n)) return `${n.toFixed(2)}%`;
  return String(x ?? "");
}

export function buildTxtReport({ result, temperature, humidity, cultivar }) {
  const processedAt = result?.processed_at ? sanitizeDegree(result.processed_at) : new Date().toISOString();
  const message = sanitizeDegree(result?.message);
  const confidencePct = result?.confidence != null ? (result.confidence * 100).toFixed(2) : "N/A";
  const cv = sanitizeDegree(result?.cultivar || cultivar || "Alphonso");

  const lines = [];
  lines.push("RipenX - Mango Analysis Report");
  lines.push(`Generated At: ${processedAt}`);
  lines.push("");
  lines.push(`Filename: ${result?.filename || "N/A"}`);
  lines.push(`Status: ${result?.ripeness_class || "N/A"}`);
  lines.push(`Confidence: ${confidencePct}%`);
  lines.push(`Cultivar: ${cv}`);
  lines.push(`Current Temp: ${temperature}°C`);
  lines.push(`Current Humidity: ${humidity}%`);
  lines.push(`Harvest Estimate: ${result?.harvest_estimate_days ?? "N/A"} days until ready.`);
  if (result?.recommended_export_destination) {
    lines.push(`Recommended Export Destination: ${sanitizeDegree(result.recommended_export_destination)}`);
    lines.push(`Logistics Action Required: ${sanitizeDegree(result.export_logistics_action)}`);
    if (Array.isArray(result?.mandatory_regulatory_compliance) && result.mandatory_regulatory_compliance.length) {
      lines.push("Mandatory Regulatory Compliance:");
      for (const x of result.mandatory_regulatory_compliance) lines.push(`- ${sanitizeDegree(x)}`);
    }
  }
  lines.push("");
  lines.push("Message (model output):");
  lines.push(message);
  lines.push("");
  if (result?.confidence_percentage) {
    lines.push("All probabilities:");
    for (const [k, v] of Object.entries(result.confidence_percentage)) {
      lines.push(`- ${k}: ${v}`);
    }
  }

  return lines.join("\n");
}

export function buildHtmlReport({ result, temperature, humidity, cultivar }) {
  const message = sanitizeDegree(result?.message);
  const processedAt = result?.processed_at ? sanitizeDegree(result.processed_at) : new Date().toISOString();
  const status = sanitizeDegree(result?.ripeness_class || "N/A");
  const confidencePct = result?.confidence != null ? (result.confidence * 100).toFixed(2) : "N/A";
  const cv = sanitizeDegree(result?.cultivar || cultivar || "Alphonso");
  const exportDest = result?.recommended_export_destination
    ? `<div class="kv"><div class="k">Recommended Export Destination</div><div class="v">${escapeHtml(sanitizeDegree(result.recommended_export_destination))}</div></div>`
    : "";
  const exportLog = result?.export_logistics_action
    ? `<div class="kv" style="grid-column: 1 / -1;"><div class="k">Logistics Action Required</div><div class="v" style="font-weight:600;font-size:14px;">${escapeHtml(sanitizeDegree(result.export_logistics_action))}</div></div>`
    : "";
  const exportCompliance = Array.isArray(result?.mandatory_regulatory_compliance) && result.mandatory_regulatory_compliance.length
    ? `<div class="kv" style="grid-column: 1 / -1;"><div class="k">Mandatory Regulatory Compliance</div><div class="v" style="font-weight:600;font-size:14px;">${result.mandatory_regulatory_compliance.map((x) => escapeHtml(sanitizeDegree(x))).join("<br/>")}</div></div>`
    : "";

  const probsRows = result?.confidence_percentage
    ? Object.entries(result.confidence_percentage)
        .map(([k, v]) => `<tr><td>${sanitizeDegree(k)}</td><td>${sanitizeDegree(v)}</td></tr>`)
        .join("")
    : "";

  const html = `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>RipenX - Mango Report</title>
    <style>
      :root { color-scheme: light; }
      body { font-family: Arial, sans-serif; margin: 24px; color: #1b1b1b; background: #fff; }
      .header { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; border-bottom: 1px solid #e6e6e6; padding-bottom: 10px; }
      h1 { margin: 0; font-size: 20px; }
      .meta { color: #444; font-size: 13px; }
      .card { margin-top: 16px; border: 1px solid #e6e6e6; border-radius: 12px; padding: 16px; }
      .tag { display: inline-block; padding: 6px 10px; border-radius: 999px; background: #fff3cd; border: 1px solid #ffeeba; font-weight: 700; }
      .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }
      .kv { padding: 10px; border: 1px solid #efefef; border-radius: 10px; background: #fafafa; }
      .kv .k { font-size: 12px; color: #555; }
      .kv .v { font-size: 16px; font-weight: 800; margin-top: 4px; }
      pre { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono"; font-size: 13px; background: #0b1320; color: #e9fff6; padding: 14px; border-radius: 12px; }
      table { width: 100%; border-collapse: collapse; margin-top: 10px; }
      td { border-bottom: 1px dashed #e6e6e6; padding: 10px 0; font-size: 14px; }
      td:first-child { width: 60%; color: #111; }
      .muted { color: #666; font-size: 12px; margin-top: 12px; }
    </style>
  </head>
  <body>
    <div class="header">
      <h1>RipenX - Mango Analysis Report</h1>
      <div class="meta">
        Generated At: ${processedAt}
        <div>Filename: ${sanitizeDegree(result?.filename || "N/A")}</div>
      </div>
    </div>

    <div class="card">
      <div><span class="tag">${status}</span></div>
      <div class="grid">
        <div class="kv"><div class="k">Confidence</div><div class="v">${sanitizeDegree(confidencePct)}%</div></div>
        <div class="kv"><div class="k">Harvest Estimate</div><div class="v">${sanitizeDegree(result?.harvest_estimate_days ?? "N/A")} days until ready.</div></div>
        <div class="kv"><div class="k">Cultivar</div><div class="v">${escapeHtml(cv)}</div></div>
        <div class="kv"><div class="k">Current Temp</div><div class="v">${sanitizeDegree(temperature)}°C</div></div>
        <div class="kv"><div class="k">Current Humidity</div><div class="v">${sanitizeDegree(humidity)}%</div></div>
        ${exportDest}
        ${exportLog}
        ${exportCompliance}
      </div>

      <div style="margin-top: 14px;">
        <div class="muted">Message (model output):</div>
        <pre>${escapeHtml(message)}</pre>
      </div>

      <div style="margin-top: 14px;">
        <div class="muted">All probabilities:</div>
        <table>
          ${probsRows}
        </table>
      </div>
    </div>
  </body>
</html>`;
  return html;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function downloadTextFile(filename, content) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export function downloadHtmlFile(filename, html) {
  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

