const fileInput = document.querySelector("#fileInput");
const sourceImage = document.querySelector("#sourceImage");
const canvas = document.querySelector("#selectionCanvas");
const ctx = canvas.getContext("2d");
const emptyState = document.querySelector("#emptyState");
const analyzeBtn = document.querySelector("#analyzeBtn");
const matchBtn = document.querySelector("#matchBtn");
const textInput = document.querySelector("#textInput");
const topKInput = document.querySelector("#topK");
const statusEl = document.querySelector("#status");
const resultList = document.querySelector("#resultList");
const candidateInfo = document.querySelector("#candidateInfo");

let imageID = "";
let selection = null;
let dragging = false;
let dragStart = null;
let progressSource = null;

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  setStatus("Uploading...");
  const data = new FormData();
  data.append("image", file);
  const res = await fetch("/api/upload", { method: "POST", body: data });
  if (!res.ok) return setError(await res.text());
  const payload = await res.json();
  imageID = payload.id;
  sourceImage.src = payload.url;
  sourceImage.onload = fitCanvasToImage;
  sourceImage.style.display = "block";
  canvas.style.display = "block";
  emptyState.style.display = "none";
  selection = null;
  textInput.value = "";
  resultList.innerHTML = "";
  candidateInfo.textContent = "";
  analyzeBtn.disabled = true;
  matchBtn.disabled = true;
  setStatus("Drag to select a text region");
});

window.addEventListener("resize", fitCanvasToImage);

canvas.addEventListener("pointerdown", (event) => {
  if (!imageID) return;
  dragging = true;
  dragStart = point(event);
  selection = { x: dragStart.x, y: dragStart.y, w: 0, h: 0 };
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", (event) => {
  if (!dragging || !dragStart) return;
  const p = point(event);
  selection = normalizeRect(dragStart.x, dragStart.y, p.x, p.y);
  drawSelection();
});

canvas.addEventListener("pointerup", () => {
  dragging = false;
  if (selection && selection.w > 6 && selection.h > 6) {
    analyzeBtn.disabled = false;
    matchBtn.disabled = textInput.value.trim() === "";
    setStatus("Region selected. Run OCR or enter text manually");
  }
});

textInput.addEventListener("input", () => {
  matchBtn.disabled = !imageID || !selection || textInput.value.trim() === "";
});

analyzeBtn.addEventListener("click", async () => {
  if (!selection) return;
  setStatus("Running OCR...");
  const res = await postJSON("/api/analyze", { image_id: imageID, box: selection });
  if (!res.ok) return setError(await res.text());
  const payload = await res.json();
  textInput.value = payload.text || "";
  matchBtn.disabled = textInput.value.trim() === "";
  setStatus(payload.warning || `OCR done, confidence ${formatScore(payload.confidence)}`);
});

matchBtn.addEventListener("click", async () => {
  if (!selection) return;
  closeProgress();
  setStatus("Starting match...");
  resultList.innerHTML = "";
  candidateInfo.textContent = "";
  matchBtn.disabled = true;
  const res = await postJSON("/api/match/start", {
    image_id: imageID,
    box: selection,
    text: textInput.value.trim(),
    top_k: Number(topKInput.value || 10),
  });
  if (!res.ok) {
    matchBtn.disabled = false;
    return setError(await res.text());
  }
  const payload = await res.json();
  subscribeProgress(payload.task_id);
});

function subscribeProgress(taskID) {
  progressSource = new EventSource(`/api/match/events/${encodeURIComponent(taskID)}`);
  progressSource.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === "heartbeat") return;
    if (payload.type === "progress") {
      updateProgress(payload);
      return;
    }
    if (payload.type === "done") {
      closeProgress();
      renderResults(payload.result);
      const took = payload.result?.elapsed_ms ? `${payload.result.elapsed_ms}ms` : "";
      setStatus(payload.result?.warning || `Done ${took}`);
      matchBtn.disabled = false;
      return;
    }
    if (payload.type === "error") {
      closeProgress();
      setError(payload.error || "Match failed");
      matchBtn.disabled = false;
    }
  };
  progressSource.onerror = () => {
    closeProgress();
    setError("Progress stream disconnected");
    matchBtn.disabled = false;
  };
}

function updateProgress(payload) {
  if (!payload.total) {
    setStatus(payload.message || payload.phase);
    return;
  }
  const percent = Math.round((payload.done / payload.total) * 100);
  setStatus(`${payload.phase}: ${payload.done}/${payload.total} (${percent}%)`);
}

function closeProgress() {
  if (progressSource) {
    progressSource.close();
    progressSource = null;
  }
}

function fitCanvasToImage() {
  if (!sourceImage.src) return;
  const rect = sourceImage.getBoundingClientRect();
  canvas.width = Math.round(rect.width);
  canvas.height = Math.round(rect.height);
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  drawSelection();
}

function point(event) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = sourceImage.naturalWidth / rect.width;
  const scaleY = sourceImage.naturalHeight / rect.height;
  return {
    x: Math.round((event.clientX - rect.left) * scaleX),
    y: Math.round((event.clientY - rect.top) * scaleY),
  };
}

function normalizeRect(x1, y1, x2, y2) {
  return {
    x: Math.min(x1, x2),
    y: Math.min(y1, y2),
    w: Math.abs(x2 - x1),
    h: Math.abs(y2 - y1),
  };
}

function drawSelection() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!selection || !sourceImage.naturalWidth) return;
  const sx = canvas.width / sourceImage.naturalWidth;
  const sy = canvas.height / sourceImage.naturalHeight;
  ctx.fillStyle = "rgba(22, 102, 197, 0.18)";
  ctx.strokeStyle = "#48a1ff";
  ctx.lineWidth = 2;
  ctx.fillRect(selection.x * sx, selection.y * sy, selection.w * sx, selection.h * sy);
  ctx.strokeRect(selection.x * sx, selection.y * sy, selection.w * sx, selection.h * sy);
}

function renderResults(payload) {
  candidateInfo.textContent = `${payload.candidate_size || 0} candidates`;
  if (!payload.results?.length) {
    resultList.innerHTML = "<p>No font match results. Check the text and font library.</p>";
    return;
  }
  resultList.innerHTML = payload.results.map((item, idx) => `
    <article class="result-card">
      <strong>${idx + 1}. ${escapeHTML(item.font_name)}</strong>
      <div class="score">Score ${formatScore(item.score_total)}</div>
      ${item.preview_url ? `<img class="preview" src="${item.preview_url}" alt="">` : ""}
      <div class="metric-grid">
        <span>SSIM ${formatScore(item.score_ssim)}</span>
        <span>IoU ${formatScore(item.score_iou)}</span>
        <span>Edge ${formatScore(item.score_edge)}</span>
        <span>Shape ${formatScore(item.score_shape)}</span>
      </div>
    </article>
  `).join("");
}

function postJSON(url, payload) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function setStatus(text) {
  statusEl.className = "";
  statusEl.textContent = text;
}

function setError(text) {
  statusEl.className = "error";
  statusEl.textContent = text;
}

function formatScore(value) {
  const n = Number(value || 0);
  return n.toFixed(3);
}

function escapeHTML(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[ch]));
}
