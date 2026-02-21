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

const metricsJson = document.getElementById("metrics-json");
const reportFigure = document.getElementById("report-figure");
const figureCaption = document.getElementById("figure-caption");

let selectedFile = null;

const readAsDataURL = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Failed to load file preview."));
    reader.readAsDataURL(file);
  });

async function setSelectedFile(file) {
  if (!file) return;
  selectedFile = file;
  fileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  preview.src = await readAsDataURL(file);
  preview.hidden = false;
  previewWrap.hidden = false;
  errorMessage.hidden = true;
}

function setLoading(loading) {
  predictBtn.disabled = loading;
  predictBtn.textContent = loading ? "Running..." : "Run Inference";
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

async function runPrediction() {
  if (!selectedFile) {
    errorMessage.textContent = "Select an image before running inference.";
    errorMessage.hidden = false;
    return;
  }

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("model_id", modelInput.value);
  formData.append("top_k", topKInput.value);

  setLoading(true);
  errorMessage.hidden = true;
  try {
    const response = await fetch("/api/v1/predict", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Prediction failed.");
    }

    predictedClass.textContent = payload.predicted_class;
    confidenceLine.textContent = `${(payload.confidence * 100).toFixed(2)}% confidence · ${payload.model_id}`;
    latencyBadge.textContent = `${payload.inference_ms.toFixed(2)} ms`;
    renderTopK(payload.top_k);

    resultEmpty.hidden = true;
    resultContent.hidden = false;
  } catch (error) {
    errorMessage.textContent = error.message;
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

    metricsJson.textContent = JSON.stringify(payload.metrics, null, 2);

    if (payload.figures?.length > 0) {
      const firstFigure = payload.figures[0];
      reportFigure.src = firstFigure.url;
      reportFigure.hidden = false;
      figureCaption.textContent = firstFigure.name;
    } else {
      reportFigure.hidden = true;
      figureCaption.textContent = "No figure found in reports directory.";
    }
  } catch (error) {
    metricsJson.textContent = error.message;
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
  await setSelectedFile(file);
});

fileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  await setSelectedFile(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await runPrediction();
});

loadReports();
