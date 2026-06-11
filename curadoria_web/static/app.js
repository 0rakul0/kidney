const state = {
  items: [],
  currentId: null,
  zoom: 1,
  drawing: false,
  drawingTool: "polygon",
  pointerDown: false,
  points: [],
  segmentationModel: "unet",
  originalWidth: 0,
  originalHeight: 0,
  imageUrl: "",
};
const $ = (id) => document.getElementById(id);
const form = $("review-form");
const reviewerInput = $("reviewer");
const reviewerType = $("reviewer-type");

function reviewer() { return reviewerInput.value.trim(); }
function ensureReviewer() {
  if (reviewer()) return true;
  reviewerInput.focus();
  showMessage("Informe o revisor antes de corrigir a mascara.", true);
  return false;
}
function query(params) { return new URLSearchParams(params).toString(); }
function percent(value) { return value === undefined || value === null || value === "" ? "-" : Number(value).toFixed(3); }
async function request(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Falha na requisicao.");
  return payload;
}
function showMessage(text, error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
}
function showDatabaseMessage(text, error = false) {
  $("database-message").textContent = text;
  $("database-message").classList.toggle("error", error);
}
function viewMode() {
  const superres = $("toggle-superres").checked;
  const clahe = $("toggle-clahe-view").checked;
  if (superres && clahe) return "superres_clahe";
  if (superres) return "superres";
  if (clahe) return "clahe";
  return "original";
}
function updateBaseImage() {
  if (!state.imageUrl) return;
  const params = query({ view: viewMode(), v: Date.now() });
  $("base-image").src = `${state.imageUrl}?${params}`;
}
async function loadMeta() {
  const meta = await request(`/api/meta?${query({ reviewer: reviewer() })}`);
  const completed = meta.total ? (meta.revisados / meta.total) * 100 : 0;
  $("total").textContent = meta.total;
  $("reviewed").textContent = meta.revisados;
  $("pending").textContent = meta.pendentes;
  $("progress-percent").textContent = `${completed.toFixed(1)}%`;
  $("progress-ring").style.setProperty("--percent", completed);
  $("progress-bar").style.width = `${completed}%`;
}
async function loadQueue(preserveSelection = true) {
  state.items = await request(`/api/items?${query({
    reviewer: reviewer(),
    state: $("queue-state").value,
    source: $("source").value,
    annotation: $("annotation").value,
    search: $("search").value.trim(),
    limit: "150",
  })}`);
  const list = $("case-list");
  list.textContent = "";
  state.items.forEach((item, index) => {
    const row = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.classList.toggle("active", item.image_id === state.currentId);
    const origin = item.sem_mascara ? "Sem mascara" : (item.pseudo_mascara ? "Pseudo-mascara" : "Manual");
    button.innerHTML = `<img class="thumb" loading="lazy" src="${item.thumb_url}" alt=""><span class="case-copy"><strong>${item.image_id}</strong><span>${origin}</span></span><span class="done">${item.revisado ? "OK" : index + 1}</span>`;
    button.addEventListener("click", () => selectItem(item.image_id));
    row.append(button);
    list.append(row);
  });
  if (!preserveSelection || !state.items.some((item) => item.image_id === state.currentId)) {
    if (state.items.length) await selectItem(state.items[0].image_id);
  }
}
async function loadCorrections() {
  const list = $("correction-list");
  list.textContent = "";
  try {
    const payload = await request(`/api/corrections?${query({ reviewer: reviewer(), limit: "80" })}`);
    if (!payload.corrections.length) {
      const empty = document.createElement("li");
      empty.className = "empty-change";
      empty.textContent = "Nenhuma modificacao manual registrada.";
      list.append(empty);
      return;
    }
    payload.corrections.forEach((change) => {
      const row = document.createElement("li");
      const button = document.createElement("button");
      const status = change.approval_status || "pendente";
      row.className = `correction-item ${status}`;
      button.type = "button";
      button.innerHTML = `<strong>${change.layer}</strong><span>${change.image_id}</span><small>${change.operation} - ${status}</small>`;
      button.title = change.mask_path || "";
      button.addEventListener("click", () => selectItem(change.image_id));
      row.append(button);
      list.append(row);
    });
  } catch (error) {
    const row = document.createElement("li");
    row.className = "empty-change";
    row.textContent = error.message;
    list.append(row);
  }
}
function applyZoom() {
  $("image-stack").style.transform = `scale(${state.zoom})`;
  $("zoom-label").textContent = `${Math.round(state.zoom * 100)}%`;
}
function setLayer(name, url) {
  const image = $(`layer-${name}`);
  const toggle = $(`toggle-${name}`);
  toggle.checked = Boolean(url);
  toggle.disabled = !url;
  image.style.display = url ? "block" : "none";
  if (url) image.src = `${url}${url.includes("?") ? "&" : "?"}v=${Date.now()}`; else image.removeAttribute("src");
}
function chooseButton(field, value) {
  const group = document.querySelector(`[data-field="${field}"]`);
  if (!group) return;
  group.querySelector(`input[name="${field}"]`).value = value;
  group.querySelectorAll("button").forEach((button) => {
    button.classList.toggle("selected", button.dataset.value === value);
  });
}
function fillReview(review, available) {
  const values = {
    status_rim: available.rim ? "pendente" : "indisponivel",
    status_cortex: available.cortex ? "pendente" : "indisponivel",
    status_medulla: available.medulla ? "pendente" : "indisponivel",
    status_central_echo_complex: available.central_echo_complex ? "pendente" : "indisponivel",
    fibrose: "",
    fonte_fibrose: "",
    observacao: "",
    ...(review || {}),
  };
  ["status_rim", "status_cortex", "status_medulla", "status_central_echo_complex", "fibrose"].forEach((field) => chooseButton(field, values[field]));
  form.elements.fonte_fibrose.value = values.fonte_fibrose;
  form.elements.observacao.value = values.observacao;
  Object.entries(available).forEach(([layer, exists]) => {
    const field = layer === "rim" ? "status_rim" : `status_${layer}`;
    const group = document.querySelector(`[data-field="${field}"]`);
    group.classList.toggle("unavailable", !exists);
    group.querySelectorAll("button").forEach((button) => { button.disabled = !exists; });
  });
}
function showModelMetrics(item) {
  const metrics = item.metricas.medulla;
  const kidney = item.metricas.rim;
  const cortex = item.metricas.cortex;
  const central = item.metricas.central_echo_complex;
  const agreement = item.metricas.concordancia_modelos;
  const hasMetrics = Boolean(metrics.modelo || kidney.modelo || cortex.modelo || central.modelo);
  $("model-empty").style.display = hasMetrics ? "none" : "block";
  $("model-data").style.display = metrics.modelo ? "grid" : "none";
  $("kidney-data").style.display = kidney.modelo ? "flex" : "none";
  $("cortex-data").style.display = cortex.modelo ? "flex" : "none";
  $("central-data").style.display = central.modelo ? "flex" : "none";
  $("model-scope").textContent = metrics.escopo || cortex.escopo || central.escopo || kidney.escopo || "";
  if (kidney.modelo) {
    $("kidney-model-name").textContent = kidney.modelo;
    $("kidney-dice").textContent = percent(kidney.dice);
    $("kidney-iou").textContent = percent(kidney.iou);
    $("kidney-f1").textContent = percent(kidney.f1);
  }
  if (cortex.modelo) {
    $("cortex-model-name").textContent = cortex.modelo;
    $("cortex-dice").textContent = percent(cortex.dice);
    $("cortex-iou").textContent = percent(cortex.iou);
    $("cortex-f1").textContent = percent(cortex.f1);
  }
  if (central.modelo) {
    $("central-model-name").textContent = central.modelo;
    $("central-dice").textContent = percent(central.dice);
    $("central-iou").textContent = percent(central.iou);
    $("central-f1").textContent = percent(central.f1);
  }
  if (metrics.modelo) {
    $("model-name").textContent = metrics.modelo;
    $("model-dice").textContent = percent(metrics.dice);
    $("model-iou").textContent = percent(metrics.iou);
    $("model-f1").textContent = percent(metrics.f1);
    $("agreement-dice").textContent = percent(agreement.dice);
    $("agreement-card").style.display = agreement.dice ? "block" : "none";
  }
}
function polygonColor() {
  return { rim: "#f34f56", cortex: "#41cbd0", medulla: "#ffdb3b", central_echo_complex: "#ff9100" }[$("polygon-layer").value];
}
function resetPolygon() {
  state.points = [];
  state.pointerDown = false;
  $("drawing-layer").replaceChildren();
}
function renderPolygon() {
  const svg = $("drawing-layer");
  svg.replaceChildren();
  if (!state.points.length) return;
  const namespace = "http://www.w3.org/2000/svg";
  const shape = document.createElementNS(namespace, state.drawingTool === "polygon" && state.points.length >= 3 ? "polygon" : "polyline");
  shape.setAttribute("points", state.points.map((point) => point.join(",")).join(" "));
  shape.setAttribute("fill", state.drawingTool === "polygon" && state.points.length >= 3 ? `${polygonColor()}35` : "none");
  shape.setAttribute("stroke", polygonColor());
  shape.setAttribute("stroke-width", state.drawingTool === "polygon" ? "3" : String(Number($("brush-size").value || 12) * 2));
  shape.setAttribute("stroke-linecap", "round");
  shape.setAttribute("stroke-linejoin", "round");
  svg.append(shape);
  if (state.drawingTool !== "polygon") return;
  state.points.forEach(([x, y]) => {
    const marker = document.createElementNS(namespace, "circle");
    marker.setAttribute("cx", x); marker.setAttribute("cy", y); marker.setAttribute("r", "4");
    marker.setAttribute("fill", polygonColor());
    svg.append(marker);
  });
}
function setDrawingTool(tool) {
  const wasActive = state.drawing && state.drawingTool === tool;
  state.drawingTool = tool;
  state.drawing = !wasActive;
  $("polygon-panel").classList.toggle("open", state.drawing);
  $("drawing-layer").classList.toggle("active", state.drawing);
  ["polygon", "brush", "eraser"].forEach((name) => {
    $(`tool-${name}`).classList.toggle("active", state.drawing && state.drawingTool === name);
  });
  $("polygon-operation").value = tool === "eraser" ? "apagar" : ($("polygon-operation").value === "apagar" ? "adicionar" : $("polygon-operation").value);
  $("polygon-operation").disabled = tool === "eraser";
  $("brush-size-field").style.display = tool === "polygon" ? "none" : "flex";
  $("drawing-title").textContent = { polygon: "Corrigir por poligono", brush: "Corrigir com pincel", eraser: "Apagar com borracha" }[tool];
  $("drawing-help").textContent = {
    polygon: "Clique nos vertices; aplicar fecha o poligono.",
    brush: "Arraste sobre a imagem para adicionar mascara.",
    eraser: "Arraste sobre a imagem para apagar a mascara.",
  }[tool];
  if (!state.drawing) resetPolygon();
}
function drawingPoint(event) {
  const bounds = $("drawing-layer").getBoundingClientRect();
  const x = (event.clientX - bounds.left) * state.originalWidth / bounds.width;
  const y = (event.clientY - bounds.top) * state.originalHeight / bounds.height;
  return [Math.round(x), Math.round(y)];
}
async function savePolygon() {
  if (!state.currentId) return;
  if (!ensureReviewer()) return;
  if (state.drawingTool === "polygon" && state.points.length < 3) return showMessage("Desenhe pelo menos tres vertices.", true);
  if (state.drawingTool !== "polygon" && state.points.length < 1) return showMessage("Desenhe com o pincel ou a borracha antes de aplicar.", true);
  try {
    await request("/api/corrections", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: state.currentId,
        reviewer: reviewer(),
        layer: $("polygon-layer").value,
        operation: state.drawingTool === "eraser" ? "apagar" : $("polygon-operation").value,
        tool: state.drawingTool === "polygon" ? "polygon" : "brush",
        radius: Number($("brush-size").value || 12),
        points: state.points,
      }),
    });
    const layer = $("polygon-layer").value;
    resetPolygon();
    await selectItem(state.currentId);
    await loadCorrections();
    chooseButton(layer === "rim" ? "status_rim" : `status_${layer}`, "corrigir");
    showMessage("Mascara corrigida salva; finalize a avaliacao em Salvar e avancar.");
  } catch (error) { showMessage(error.message, true); }
}
async function selectItem(imageId) {
  try {
    state.currentId = imageId;
    state.segmentationModel = $("segmentation-model").value;
    const item = await request(`/api/item/${encodeURIComponent(imageId)}?${query({ reviewer: reviewer(), model: state.segmentationModel })}`);
    $("image-id").textContent = `ID: ${item.image_id}`;
    $("image-id").title = item.image_id;
    state.imageUrl = item.image_url;
    state.originalWidth = Number(item.info.largura || 0);
    state.originalHeight = Number(item.info.altura || 0);
    updateBaseImage();
    $("image-stack").style.display = "block";
    $("empty-state").style.display = "none";
    setLayer("rim", item.layers.rim);
    setLayer("cortex", item.layers.cortex);
    setLayer("medulla", item.layers.medulla);
    setLayer("central_echo_complex", item.layers.central_echo_complex);
    $("item-source").textContent = item.info.origem;
    $("item-dimension").textContent = item.info.dimensao;
    $("item-rim").textContent = item.info.mascara_rim;
    $("item-cortex").textContent = item.info.mascara_cortex;
    $("item-medulla").textContent = item.info.mascara_medulla;
    $("item-central").textContent = item.info.mascara_central_echo_complex;
    if (item.modelo_segmentacao_interna && $("segmentation-model").value !== item.modelo_segmentacao_interna) {
      $("segmentation-model").value = item.modelo_segmentacao_interna;
      state.segmentationModel = item.modelo_segmentacao_interna;
    }
    showModelMetrics(item);
    fillReview(item.review, item.camadas_disponiveis);
    state.zoom = 1;
    resetPolygon();
    applyZoom();
    await loadQueue(true);
    showMessage("");
  } catch (error) { showMessage(error.message, true); }
}
async function saveReview(event) {
  event.preventDefault();
  if (!state.currentId) return;
  const payload = {
    ...Object.fromEntries(new FormData(form).entries()),
    image_id: state.currentId,
    reviewer: reviewer(),
    reviewer_type: reviewerType.value,
  };
  try {
    await request("/api/reviews", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    await loadMeta();
    await loadQueue(false);
    await loadCorrections();
    showMessage("Avaliacao salva.");
  } catch (error) { showMessage(error.message, true); }
}
async function exportDatabase() {
  if (!ensureReviewer()) return;
  try {
    const response = await request("/api/database-export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reviewer: reviewer() }),
    });
    showDatabaseMessage(`${response.export.exported} registros exportados para ${response.export.table}.`);
  } catch (error) {
    showDatabaseMessage(error.message, true);
  }
}
document.querySelectorAll("[data-field] button").forEach((button) => button.addEventListener("click", () => {
  chooseButton(button.closest("fieldset").dataset.field, button.dataset.value);
}));
["rim", "cortex", "medulla", "central_echo_complex"].forEach((name) => $(`toggle-${name}`).addEventListener("change", (event) => {
  $(`layer-${name}`).style.display = event.target.checked ? "block" : "none";
}));
$("zoom-in").addEventListener("click", () => { state.zoom = Math.min(3, state.zoom + 0.25); applyZoom(); });
$("zoom-out").addEventListener("click", () => { state.zoom = Math.max(0.5, state.zoom - 0.25); applyZoom(); });
$("zoom-reset").addEventListener("click", () => { state.zoom = 1; applyZoom(); });
$("tool-zoom").addEventListener("click", () => { state.zoom = Math.min(3, state.zoom + 0.25); applyZoom(); });
$("tool-polygon").addEventListener("click", () => setDrawingTool("polygon"));
$("tool-brush").addEventListener("click", () => setDrawingTool("brush"));
$("tool-eraser").addEventListener("click", () => setDrawingTool("eraser"));
$("polygon-clear").addEventListener("click", resetPolygon);
$("polygon-save").addEventListener("click", savePolygon);
$("polygon-layer").addEventListener("change", renderPolygon);
$("drawing-layer").addEventListener("click", (event) => {
  if (!state.drawing || state.drawingTool !== "polygon" || !state.originalWidth || !state.originalHeight) return;
  state.points.push(drawingPoint(event));
  renderPolygon();
});
$("drawing-layer").addEventListener("pointerdown", (event) => {
  if (!state.drawing || state.drawingTool === "polygon" || !state.originalWidth || !state.originalHeight) return;
  event.preventDefault();
  state.pointerDown = true;
  state.points.push(drawingPoint(event));
  renderPolygon();
});
$("drawing-layer").addEventListener("pointermove", (event) => {
  if (!state.pointerDown || !state.drawing || state.drawingTool === "polygon") return;
  event.preventDefault();
  const next = drawingPoint(event);
  const previous = state.points[state.points.length - 1] || [];
  if (Math.abs(next[0] - previous[0]) + Math.abs(next[1] - previous[1]) < 2) return;
  state.points.push(next);
  renderPolygon();
});
["pointerup", "pointerleave"].forEach((eventName) => $("drawing-layer").addEventListener(eventName, () => {
  state.pointerDown = false;
}));
$("drawing-layer").addEventListener("dblclick", (event) => {
  event.preventDefault();
  if (state.drawingTool === "polygon" && state.points.length >= 3) savePolygon();
});
$("base-image").addEventListener("load", () => {
  const width = state.originalWidth || $("base-image").naturalWidth;
  const height = state.originalHeight || $("base-image").naturalHeight;
  $("drawing-layer").setAttribute("viewBox", `0 0 ${width} ${height}`);
});
$("toggle-superres").addEventListener("change", () => {
  localStorage.setItem("curadoria_view_superres", $("toggle-superres").checked ? "1" : "0");
  updateBaseImage();
});
$("toggle-clahe-view").addEventListener("change", () => {
  localStorage.setItem("curadoria_view_clahe", $("toggle-clahe-view").checked ? "1" : "0");
  updateBaseImage();
});
$("brush-size").addEventListener("input", renderPolygon);
$("segmentation-model").addEventListener("change", async () => {
  state.segmentationModel = $("segmentation-model").value;
  localStorage.setItem("curadoria_segmentation_model", state.segmentationModel);
  if (state.currentId) await selectItem(state.currentId);
});
$("previous").addEventListener("click", () => {
  const index = state.items.findIndex((item) => item.image_id === state.currentId);
  if (index > 0) selectItem(state.items[index - 1].image_id);
});
$("next").addEventListener("click", () => {
  const index = state.items.findIndex((item) => item.image_id === state.currentId);
  if (index >= 0 && index < state.items.length - 1) selectItem(state.items[index + 1].image_id);
});
form.addEventListener("submit", saveReview);
$("refresh-corrections").addEventListener("click", loadCorrections);
$("export-database").addEventListener("click", exportDatabase);
[$("queue-state"), $("source"), $("annotation")].forEach((element) => element.addEventListener("change", async () => {
  await loadMeta(); await loadQueue(false);
}));
$("search").addEventListener("input", () => loadQueue(false));
$("clear-filters").addEventListener("click", async () => {
  $("annotation").value = ""; $("queue-state").value = "pendentes"; $("source").value = ""; $("search").value = "";
  await loadQueue(false);
});
reviewerInput.addEventListener("change", async () => {
  localStorage.setItem("curadoria_reviewer", reviewer()); await loadMeta(); await loadQueue(false); await loadCorrections();
});
reviewerType.addEventListener("change", () => localStorage.setItem("curadoria_reviewer_type", reviewerType.value));
async function start() {
  reviewerInput.value = localStorage.getItem("curadoria_reviewer") || "Revisor 01";
  reviewerType.value = localStorage.getItem("curadoria_reviewer_type") || "especialista";
  $("segmentation-model").value = localStorage.getItem("curadoria_segmentation_model") || "unet";
  $("toggle-superres").checked = localStorage.getItem("curadoria_view_superres") === "1";
  $("toggle-clahe-view").checked = localStorage.getItem("curadoria_view_clahe") === "1";
  $("brush-size-field").style.display = "none";
  state.segmentationModel = $("segmentation-model").value;
  await loadMeta(); await loadQueue(false); await loadCorrections();
}
start().catch((error) => showMessage(error.message, true));
