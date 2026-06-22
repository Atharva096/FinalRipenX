import React, { useEffect, useMemo, useRef, useState } from "react";
import { predictMango, exampleImageUrl } from "./api.js";
import { buildHtmlReport, buildTxtReport, downloadHtmlFile, downloadTextFile } from "./report.js";

const CLASSES = [
  {
    label: "Unripe",
    folder: "unripe",
    hueRange: "80 deg - 130 deg",
    satRange: "40% - 70%",
    notes: [
      "Firm, hard flesh with no give",
      "Uniform dark green skin color",
      "HSV hue dominant in green range",
    ],
    timeline: "Harvest in 10-15 days",
  },
  {
    label: "Partially Ripe",
    folder: "partially_ripe",
    hueRange: "40 deg - 80 deg",
    satRange: "50% - 80%",
    notes: [
      "Slight give when pressed gently",
      "Yellow-green transitional color",
      "HSV hue shifting toward yellow range",
    ],
    timeline: "Harvest in 3-5 days",
  },
  {
    label: "Ripe",
    folder: "ripe",
    hueRange: "15 deg - 45 deg",
    satRange: "70% - 95%",
    notes: [
      "Soft, yielding flesh texture",
      "Golden yellow to orange skin",
      "Peak carotenoid concentration",
    ],
    timeline: "Ready to harvest now",
  },
];

function sanitizeDegree(text) {
  return String(text || "").replace(/\uFFFD/g, "°");
}

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [temperature, setTemperature] = useState(35);
  const [humidity, setHumidity] = useState(50);
  const [cultivar, setCultivar] = useState("Alphonso");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const guidelinesRef = useRef(null);
  const stagesRef = useRef(null);
  const analyzeRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState("");

  const classByLabel = useMemo(() => Object.fromEntries(CLASSES.map((c) => [c.label, c])), []);
  const picked = result ? classByLabel[result.ripeness_class] : null;

  const setImageFile = (f) => {
    setFile(f);
    setPreview(f ? URL.createObjectURL(f) : "");
  };

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  };

  const startCamera = async () => {
    setCameraError("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("Camera access is not supported in this browser.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });
      streamRef.current = stream;
      setCameraActive(true);
    } catch (e) {
      setCameraError(e?.message || "Could not access the camera.");
    }
  };

  const capturePhoto = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        setImageFile(new File([blob], "camera-capture.jpg", { type: "image/jpeg" }));
        stopCamera();
      },
      "image/jpeg",
      0.92
    );
  };

  useEffect(() => {
    const video = videoRef.current;
    const stream = streamRef.current;
    if (cameraActive && video && stream) {
      video.srcObject = stream;
      video.play().catch(() => {});
    }
  }, [cameraActive]);

  useEffect(() => () => stopCamera(), []);

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">RipenX</div>
        <nav className="menu">
          <button onClick={() => guidelinesRef.current?.scrollIntoView({ behavior: "smooth" })}>Guidelines</button>
          <button onClick={() => stagesRef.current?.scrollIntoView({ behavior: "smooth" })}>Stages</button>
          <button onClick={() => analyzeRef.current?.scrollIntoView({ behavior: "smooth" })}>Analyze</button>
        </nav>
      </header>

      <section className="hero">
        <div className="hero-overlay">
          <div className="hero-badge">AI-Powered Ripeness Detection</div>
          <h1 className="hero-title">RipenX</h1>
          <p className="hero-desc">
            Intelligent mango ripeness classification using CNN-powered image analysis.
            Capture, analyze, and get actionable harvest recommendations.
          </p>
          <div className="hero-actions">
            <button className="btn-primary" onClick={() => analyzeRef.current?.scrollIntoView({ behavior: "smooth" })}>
              Analyze Mango
            </button>
            <button className="btn-ghost" onClick={() => guidelinesRef.current?.scrollIntoView({ behavior: "smooth" })}>
              View Guidelines
            </button>
          </div>
        </div>
      </section>

      <main className="content">
        <section className="section" ref={guidelinesRef}>
          <p className="kicker">CAPTURE PROTOCOL</p>
          <h2>Image Capture Guidelines</h2>
          <p className="sub">
            Follow these guidelines to ensure accurate ripeness classification. Proper image quality directly impacts model accuracy.
          </p>
          <div className="grid3">
            {[
              ["Controlled Lighting", "Use diffused white LED lighting (5000-6500K). Avoid direct sunlight and ensure even illumination."],
              ["Camera Setup", "Use a minimum 12MP camera. Set white balance to daylight. Shoot in RAW or high-quality JPEG."],
              ["Distance & Angle", "Position camera 20-30cm from the mango. Capture at 0 deg and 45 deg angles where possible."],
              ["Focus & Clarity", "Ensure the mango surface is in sharp focus. Avoid motion blur with stable support."],
              ["Background", "Use a neutral white or light gray background to avoid color contamination."],
              ["Surface Preparation", "Gently clean the mango surface. Remove droplets, dust, or residue before imaging."],
            ].map(([title, body]) => (
              <article className="guide" key={title}>
                <h3>{title}</h3>
                <p>{body}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="section" ref={stagesRef}>
          <p className="kicker">CLASSIFICATION STAGES</p>
          <h2>Three Ripeness Stages</h2>
          <p className="sub">The model classifies mangoes into three stages based on learned visual features.</p>
          <div className="stages">
            {CLASSES.map((c) => (
              <article className="stage-card" key={c.label}>
                <img src={exampleImageUrl(c.folder)} alt={`${c.label} stage`} />
                <div className="stage-body">
                  <span className="stage-pill">{c.label.toUpperCase()}</span>
                  <div className="pairs">
                    <div>
                      <small>Hue Range</small>
                      <strong>{c.hueRange}</strong>
                    </div>
                    <div>
                      <small>Saturation</small>
                      <strong>{c.satRange}</strong>
                    </div>
                  </div>
                  <ul>
                    {c.notes.map((n) => (
                      <li key={n}>{n}</li>
                    ))}
                  </ul>
                  <div className="timeline">{c.timeline}</div>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="section analyze" ref={analyzeRef}>
          <p className="kicker">IMAGE ANALYSIS</p>
          <h2>Analyze Your Mango</h2>
          <p className="sub">Upload a mango image captured following our guidelines.</p>

          <div className="analyze-card">
            <div className="upload-box">
              <label className="field-label">
                Upload a mango photo
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => {
                    stopCamera();
                    setImageFile(e.target.files?.[0] || null);
                  }}
                />
              </label>

              <div className="camera-box">
                <span className="field-label">Or use your camera</span>
                {!cameraActive ? (
                  <button type="button" className="btn-outline block" onClick={startCamera}>
                    Open Camera
                  </button>
                ) : (
                  <>
                    <video ref={videoRef} className="camera-preview" autoPlay playsInline muted />
                    <canvas ref={canvasRef} hidden />
                    <div className="camera-actions">
                      <button type="button" className="btn-primary" onClick={capturePhoto}>
                        Capture Photo
                      </button>
                      <button type="button" className="btn-ghost" onClick={stopCamera}>
                        Cancel
                      </button>
                    </div>
                  </>
                )}
                {cameraError ? <div className="error">{cameraError}</div> : null}
              </div>
              <div className="env">
                <label>
                  Temp (deg C)
                  <input type="number" value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} />
                </label>
                <label>
                  Humidity (%)
                  <input type="number" value={humidity} onChange={(e) => setHumidity(Number(e.target.value))} />
                </label>
                <label>
                  Cultivar
                  <select value={cultivar} onChange={(e) => setCultivar(e.target.value)}>
                    <option value="Alphonso">Alphonso</option>
                    <option value="Kesar">Kesar</option>
                    <option value="Totapuri">Totapuri</option>
                    <option value="Other">Other</option>
                  </select>
                </label>
              </div>
              <button
                className="btn-primary block"
                disabled={loading}
                onClick={async () => {
                  setError("");
                  setResult(null);
                  if (!file) {
                    setError("Please select a mango image first.");
                    return;
                  }
                  setLoading(true);
                  try {
                    const json = await predictMango({ file, temperature, humidity, cultivar });
                    setResult(json);
                  } catch (e) {
                    const message = e?.message || String(e);
                    if (message.includes("Wrong input entered")) {
                      setError("Wrong input entered");
                      setResult(null);
                    } else {
                      setError(message);
                    }
                  } finally {
                    setLoading(false);
                  }
                }}
              >
                {loading ? "Analyzing..." : "Analyze Mango"}
              </button>
              {error ? <div className="error">{error}</div> : null}
            </div>

            <div className="result-panel">
              <div className="preview-wrap">{preview ? <img src={preview} alt="uploaded mango" /> : <div className="preview-empty">No image selected</div>}</div>
              <div className="result-box">
                {!result ? (
                  <p className="muted">Run analysis to see class, confidence, and harvest estimate.</p>
                ) : (
                  <>
                    <div className="row top">
                      <div>
                        <small>Classification Result</small>
                        <h3>{result.ripeness_class}</h3>
                      </div>
                      <div className="confidence">{(result.confidence * 100).toFixed(1)}% confidence</div>
                    </div>
                    <div className="stats">
                      <div>
                        <small>Current Temp</small>
                        <strong>{temperature} deg C</strong>
                      </div>
                      <div>
                        <small>Current Humidity</small>
                        <strong>{humidity}%</strong>
                      </div>
                      <div>
                        <small>Timeline</small>
                        <strong>{result.harvest_estimate_days} days</strong>
                      </div>
                      <div>
                        <small>Cultivar</small>
                        <strong>{result.cultivar || cultivar}</strong>
                      </div>
                    </div>
                    {result.recommended_export_destination ? (
                      <div className="export-routing">
                        <small>Export routing</small>
                        <p className="export-dest">
                          <strong>Recommended Export Destination:</strong> {sanitizeDegree(result.recommended_export_destination)}
                        </p>
                        <p className="export-logistics">
                          <strong>Logistics Action Required:</strong> {sanitizeDegree(result.export_logistics_action)}
                        </p>
                        {Array.isArray(result.mandatory_regulatory_compliance) && result.mandatory_regulatory_compliance.length ? (
                          <div className="export-compliance">
                            <strong>Mandatory Regulatory Compliance:</strong>
                            <ul>
                              {result.mandatory_regulatory_compliance.map((x) => (
                                <li key={x}>{sanitizeDegree(x)}</li>
                              ))}
                            </ul>
                          </div>
                        ) : null}
                      </div>
                    ) : null}
                    <div className="msg">{sanitizeDegree(result.message)}</div>
                    <div className="report-actions">
                      <button
                        className="btn-green"
                        onClick={() =>
                          downloadTextFile(
                            "mango-report.txt",
                            buildTxtReport({ result, temperature, humidity, cultivar })
                          )
                        }
                      >
                        Download TXT Report
                      </button>
                      <button
                        className="btn-outline"
                        onClick={() =>
                          downloadHtmlFile(
                            "mango-report.html",
                            buildHtmlReport({ result, temperature, humidity, cultivar })
                          )
                        }
                      >
                        Download HTML Report
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

