const state = { items: [], currentId: null, zoom: 1, drawing: false, points: [] };
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
    const origin = item.pseudo_mascara ? "Pseudo-mascara" : "Manual";
    button.innerHTML = `<img class="thumb" loading="lazy" src="${item.thumb_url}" alt=""><span class="case-copy"><strong>${item.image_id}</strong><span>${origin}</span></span><span class="done">${item.revisado ? "OK" : index + 1}</span>`;
    button.addEventListener("click", () => selectItem(item.image_id));
    row.append(button);
    list.append(row);
  });
  if (!preserveSelection || !state.items.some((item) => item.image_id === state.currentId)) {
    if (state.items.length) await selectItem(state.items[0].image_id);
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
  $("drawing-layer").replaceChildren();
}
function renderPolygon() {
  const svg = $("drawing-layer");
  svg.replaceChildren();
  if (!state.points.length) return;
  const namespace = "http://www.w3.org/2000/svg";
  const shape = document.createElementNS(namespace, state.points.length >= 3 ? "polygon" : "polyline");
  shape.setAttribute("points", state.points.map((point) => point.join(",")).join(" "));
  shape.setAttribute("fill", state.points.length >= 3 ? `${polygonColor()}35` : "none");
  shape.setAttribute("stroke", polygonColor());
  shape.setAttribute("stroke-width", "3");
  svg.append(shape);
  state.points.forEach(([x, y]) => {
    const marker = document.createElementNS(namespace, "circle");
    marker.setAttribute("cx", x); marker.setAttribute("cy", y); marker.setAttribute("r", "4");
    marker.setAttribute("fill", polygonColor());
    svg.append(marker);
  });
}
function togglePolygon() {
  state.drawing = !state.drawing;
  $("polygon-panel").classList.toggle("open", state.drawing);
  $("drawing-layer").classList.toggle("active", state.drawing);
  $("tool-polygon").classList.toggle("active", state.drawing);
  if (!state.drawing) resetPolygon();
}
async function savePolygon() {
  if (!state.currentId) return;
  if (!ensureReviewer()) return;
  if (state.points.length < 3) return showMessage("Desenhe pelo menos tres vertices.", true);
  try {
    await request("/api/corrections", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: state.currentId,
        reviewer: reviewer(),
        layer: $("polygon-layer").value,
        operation: $("polygon-operation").value,
        points: state.points,
      }),
    });
    const layer = $("polygon-layer").value;
    resetPolygon();
    await selectItem(state.currentId);
    chooseButton(layer === "rim" ? "status_rim" : `status_${layer}`, "corrigir");
    showMessage("Mascara corrigida salva; finalize a avaliacao em Salvar e avancar.");
  } catch (error) { showMessage(error.message, true); }
}
async function selectItem(imageId) {
  try {
    state.currentId = imageId;
    const item = await request(`/api/item/${encodeURIComponent(imageId)}?${query({ reviewer: reviewer() })}`);
    $("image-id").textContent = `ID: ${item.image_id}`;
    $("base-image").src = item.image_url;
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
    showMessage("Avaliacao salva.");
  } catch (error) { showMessage(error.message, true); }
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
$("tool-polygon").addEventListener("click", togglePolygon);
$("polygon-clear").addEventListener("click", resetPolygon);
$("polygon-save").addEventListener("click", savePolygon);
$("polygon-layer").addEventListener("change", renderPolygon);
$("drawing-layer").addEventListener("click", (event) => {
  if (!state.drawing || !$("base-image").naturalWidth) return;
  const bounds = $("drawing-layer").getBoundingClientRect();
  const x = (event.clientX - bounds.left) * $("base-image").naturalWidth / bounds.width;
  const y = (event.clientY - bounds.top) * $("base-image").naturalHeight / bounds.height;
  state.points.push([Math.round(x), Math.round(y)]);
  renderPolygon();
});
$("drawing-layer").addEventListener("dblclick", (event) => {
  event.preventDefault();
  if (state.points.length >= 3) savePolygon();
});
$("base-image").addEventListener("load", () => {
  $("drawing-layer").setAttribute("viewBox", `0 0 ${$("base-image").naturalWidth} ${$("base-image").naturalHeight}`);
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
[$("queue-state"), $("source"), $("annotation")].forEach((element) => element.addEventListener("change", async () => {
  await loadMeta(); await loadQueue(false);
}));
$("search").addEventListener("input", () => loadQueue(false));
$("clear-filters").addEventListener("click", async () => {
  $("annotation").value = ""; $("queue-state").value = "pendentes"; $("source").value = ""; $("search").value = "";
  await loadQueue(false);
});
reviewerInput.addEventListener("change", async () => {
  localStorage.setItem("curadoria_reviewer", reviewer()); await loadMeta(); await loadQueue(false);
});
reviewerType.addEventListener("change", () => localStorage.setItem("curadoria_reviewer_type", reviewerType.value));
async function start() {
  reviewerInput.value = localStorage.getItem("curadoria_reviewer") || "Revisor 01";
  reviewerType.value = localStorage.getItem("curadoria_reviewer_type") || "especialista";
  await loadMeta(); await loadQueue(false);
}
start().catch((error) => showMessage(error.message, true));
