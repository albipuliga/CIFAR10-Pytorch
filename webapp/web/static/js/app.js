const form = document.getElementById("predict-form");
const modelInput = document.getElementById("model-id");
const topKInput = document.getElementById("top-k");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const fileName = document.getElementById("file-name");
const preview = document.getElementById("upload-preview");
const previewWrap = document.getElementById("preview-wrap");
const predictBtn = document.getElementById("predict-btn");

const predictedClass = document.getElementById("predicted-class");
const confidenceLine = document.getElementById("confidence-line");
const latencyBadge = document.getElementById("latency-badge");
const topKList = document.getElementById("top-k-list");
const errorMessage = document.getElementById("error-message");
const resultEmpty = document.getElementById("result-empty");
const resultContent = document.getElementById("result-content");

const metricsContainer = document.getElementById("metrics-container");
const reportFigure = document.getElementById("report-figure");
const figureCaption = document.getElementById("figure-caption");

let selectedFile = null;
let selectionVersion = 0;
let reportFigures = [];

const SUPPORTED_IMAGE_TYPES = new Set(["image/png", "image/jpeg", "image/jpg"]);
const SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg"];

const readAsDataURL = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Failed to load file preview."));
    reader.readAsDataURL(file);
  });

const preloadImage = (source) =>
  new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve();
    img.onerror = () =>
      reject(new Error("Selected file could not be previewed as an image."));
    img.src = source;
  });

function inferMimeType(file) {
  const normalizedType = file.type?.toLowerCase() || "";
  if (SUPPORTED_IMAGE_TYPES.has(normalizedType)) {
    return normalizedType === "image/jpg" ? "image/jpeg" : normalizedType;
  }

  const lowerName = file.name?.toLowerCase() || "";
  if (lowerName.endsWith(".png")) return "image/png";
  if (lowerName.endsWith(".jpg") || lowerName.endsWith(".jpeg")) return "image/jpeg";
  return normalizedType;
}

function hasSupportedExtension(fileName) {
  const lowerName = (fileName || "").toLowerCase();
  return SUPPORTED_EXTENSIONS.some((extension) => lowerName.endsWith(extension));
}

function cloneForUpload(file) {
  const mimeType = inferMimeType(file);
  const normalizedName = (file.name || "").trim();
  const safeName = normalizedName || (mimeType === "image/png" ? "clipboard-image.png" : "clipboard-image.jpg");
  return new File([file], safeName, {
    type: mimeType || file.type || "application/octet-stream",
    lastModified: file.lastModified || Date.now(),
  });
}

async function setSelectedFile(file) {
  if (!file) return;
  if (!(file instanceof File) || file.size <= 0) {
    throw new Error("Could not read the selected image. Please choose another file.");
  }
  const mimeType = inferMimeType(file);
  if (!SUPPORTED_IMAGE_TYPES.has(mimeType) && !hasSupportedExtension(file.name)) {
    throw new Error("Unsupported file type. Upload a PNG or JPEG image.");
  }

  const currentSelection = ++selectionVersion;
  const stableFile = cloneForUpload(file);
  const previewSource = await readAsDataURL(stableFile);
  await preloadImage(previewSource);

  if (currentSelection !== selectionVersion) return;

  selectedFile = stableFile;
  fileName.textContent = `${stableFile.name} (${(stableFile.size / 1024).toFixed(1)} KB)`;
  preview.src = previewSource;
  preview.hidden = false;
  previewWrap.hidden = false;
  errorMessage.hidden = true;
}

function setLoading(loading) {
  predictBtn.disabled = loading;
  predictBtn.textContent = loading ? "Running..." : "Run Inference";
}

function updateReportFigure(modelId) {
  const suffix = `confusion_matrix_${modelId}.png`;
  const figure = reportFigures.find((f) => f.url.includes(suffix));
  if (figure) {
    reportFigure.src = figure.url;
    reportFigure.hidden = false;
    figureCaption.textContent = figure.name;
  } else {
    reportFigure.removeAttribute("src");
    reportFigure.hidden = true;
    figureCaption.textContent = `No confusion matrix for ${modelId === "cnnv2" ? "CNN V2" : modelId}.`;
  }
}

function renderTopK(topK) {
  topKList.innerHTML = "";
  const maxProb = topK[0]?.probability || 1;
  topK.forEach((item) => {
    const barWidth = ((item.probability / maxProb) * 100).toFixed(1);
    const li = document.createElement("li");
    li.className = "topk-row";
    li.innerHTML = `
      <div class="topk-label">
        <span class="topk-name">${item.class_name}</span>
        <span class="topk-pct">${(item.probability * 100).toFixed(2)}%</span>
      </div>
      <div class="topk-bar-track">
        <div class="topk-bar" style="width: 0%"></div>
      </div>
    `;
    topKList.appendChild(li);
    // Animate bar in next frame so CSS transition fires
    requestAnimationFrame(() => {
      li.querySelector(".topk-bar").style.width = `${barWidth}%`;
    });
  });
}

function renderMetrics(metrics) {
  if (!metrics || typeof metrics !== "object") {
    metricsContainer.textContent = "No benchmark metrics available.";
    return;
  }

  if (metrics.status === "missing" || metrics.status === "invalid") {
    metricsContainer.textContent = metrics.message || "No benchmark metrics available.";
    return;
  }

  const models = metrics.models;
  if (!Array.isArray(models) || models.length === 0) {
    metricsContainer.textContent = "No benchmark metrics available.";
    return;
  }

  const metricDefs = [
    { key: "test_accuracy", label: "Accuracy" },
    { key: "test_precision_macro", label: "Precision" },
    { key: "test_recall_macro", label: "Recall" },
    { key: "test_f1_macro", label: "F1" },
  ];

  const modelName = (m) =>
    m.model === "cnnv2"
      ? "CNN V2"
      : m.model === "baseline"
        ? "Baseline CNN"
        : m.model;

  const bestOverall = models.reduce(
    (bi, m, i, arr) =>
      (m.test_accuracy ?? 0) > (arr[bi].test_accuracy ?? 0) ? i : bi,
    0,
  );

  const headerCells = models
    .map(
      (m, i) => `
      <th>
        <span class="model-badge">
          <span class="model-dot ${i === bestOverall ? "model-dot--best" : "model-dot--other"}"></span>
          ${modelName(m)}
        </span>
      </th>`,
    )
    .join("");

  const cell = (value, isBest) => {
    if (value == null || typeof value !== "number") return `<td><span class="metric-val">—</span></td>`;
    const cls = isBest ? "metric-val metric-val--best" : "metric-val";
    return `<td><span class="${cls}">${(value * 100).toFixed(2)}</span><span class="metric-unit">%</span></td>`;
  };

  const bodyRows = metricDefs
    .map(({ key, label }) => {
      let bestVal = -1;
      let bestModelIdx = 0;
      models.forEach((m, i) => {
        if (typeof m[key] === "number" && m[key] > bestVal) {
          bestVal = m[key];
          bestModelIdx = i;
        }
      });
      const cells = models.map((m, i) => cell(m[key], i === bestModelIdx)).join("");
      return `<tr><td>${label}</td>${cells}</tr>`;
    })
    .join("");

  metricsContainer.innerHTML = `
    <table class="metrics-table" aria-label="Benchmark metrics by model">
      <thead>
        <tr>
          <th>Metric</th>
          ${headerCells}
        </tr>
      </thead>
      <tbody>${bodyRows}</tbody>
    </table>`;
}

async function runPrediction() {
  if (!selectedFile) {
    errorMessage.textContent = "Select an image before running inference.";
    errorMessage.hidden = false;
    return;
  }

  const createFormData = () => {
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model_id", modelInput.value);
    formData.append("top_k", topKInput.value);
    return formData;
  };

  setLoading(true);
  errorMessage.hidden = true;
  try {
    let response = null;
    let networkError = null;
    for (let attempt = 0; attempt < 2; attempt += 1) {
      try {
        response = await fetch("/api/v1/predict", {
          method: "POST",
          body: createFormData(),
        });
        networkError = null;
        break;
      } catch (error) {
        networkError = error;
      }
    }
    if (networkError) {
      throw networkError;
    }

    if (!response.ok) {
      let detail = `Prediction failed (${response.status}).`;
      const rawBody = await response.text();
      if (rawBody) {
        try {
          const payload = JSON.parse(rawBody);
          detail = payload.detail || detail;
        } catch {
          // Some framework-level errors can return plain text.
          detail = rawBody;
        }
      }
      throw new Error(detail);
    }

    const payload = await response.json();
    predictedClass.textContent = payload.predicted_class;
    confidenceLine.textContent = `${(payload.confidence * 100).toFixed(2)}% confidence · ${payload.model_id}`;
    latencyBadge.textContent = `${payload.inference_ms.toFixed(2)} ms`;
    renderTopK(payload.top_k);

    resultEmpty.hidden = true;
    resultContent.hidden = false;
  } catch (error) {
    const message = error instanceof Error ? error.message : "Prediction failed.";
    errorMessage.textContent =
      message === "Failed to fetch"
        ? "Upload failed while sending the image. Please select the image again and retry."
        : message;
    errorMessage.hidden = false;
  } finally {
    setLoading(false);
  }
}

async function loadReports() {
  try {
    const response = await fetch("/api/v1/reports");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed loading reports.");
    }

    renderMetrics(payload.metrics);

    reportFigures = payload.figures ?? [];
    updateReportFigure(modelInput.value);

  } catch (error) {
    metricsContainer.textContent =
      error instanceof Error ? error.message : "Failed loading reports.";
    reportFigure.removeAttribute("src");
    reportFigure.hidden = true;
    figureCaption.textContent = "Figure unavailable.";
  }
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", async (event) => {
  const file = event.dataTransfer?.files?.[0];
  try {
    await setSelectedFile(file);
  } catch (error) {
    errorMessage.textContent =
      error instanceof Error ? error.message : "Failed to select image.";
    errorMessage.hidden = false;
  }
});

window.addEventListener("paste", async (event) => {
  const items = event.clipboardData?.items;
  if (!items) return;

  const imageItem = [...items].find((item) => item.type.startsWith("image/"));
  if (!imageItem) return;
  event.preventDefault();

  const file = imageItem.getAsFile();
  if (!file) {
    errorMessage.textContent = "Clipboard image could not be read. Try copying it again.";
    errorMessage.hidden = false;
    return;
  }

  try {
    await setSelectedFile(file);
  } catch (error) {
    errorMessage.textContent =
      error instanceof Error ? error.message : "Failed to read clipboard image.";
    errorMessage.hidden = false;
  }
});

fileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  try {
    await setSelectedFile(file);
    // Allow selecting the same file again without click-time resets.
    event.target.value = "";
  } catch (error) {
    errorMessage.textContent =
      error instanceof Error ? error.message : "Failed to select image.";
    errorMessage.hidden = false;
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await runPrediction();
});

modelInput.addEventListener("change", () => updateReportFigure(modelInput.value));

loadReports();
