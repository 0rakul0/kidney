const state = {
  items: [],
  currentId: null,
  zoom: 1,
  drawing: false,
  drawingTool: "polygon",
  pointerDown: false,
  points: [],
  saving: false,
  segmentationModel: "unet",
  originalWidth: 0,
  originalHeight: 0,
  imageUrl: "",
  roiPairs: [],
  roiDirty: false,
  roiDragging: null,
  comparing: false,
  comparatorStart: null,
  comparatorEnd: null,
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
function setFocusMode(enabled) {
  $("workspace").classList.toggle("focus-mode", enabled);
  const button = $("toggle-panels");
  button.setAttribute("aria-expanded", String(!enabled));
  button.title = enabled ? "Mostrar paineis laterais" : "Ocultar paineis laterais";
  localStorage.setItem("curadoria_focus_mode", enabled ? "1" : "0");
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
    observacao: "",
    ...(review || {}),
  };
  ["status_rim", "status_cortex", "status_medulla", "status_central_echo_complex"].forEach((field) => chooseButton(field, values[field]));
  form.elements.observacao.value = values.observacao;
  ["rim", "cortex", "medulla", "central_echo_complex"].forEach((layer) => {
    const exists = available[layer];
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
function showBrightnessMeter(meter, options = {}) {
  const available = Boolean(meter && meter.disponivel);
  const track = document.querySelector(".brightness-track");
  const indicator = $("brightness-indicator");
  const metrics = $("brightness-metrics");
  const distribution = $("brightness-distribution");
  const unavailable = $("brightness-unavailable");
  const roiActions = $("brightness-roi-actions");
  if (!available) {
    $("brightness-value").textContent = "Indisponível";
    unavailable.style.display = "block";
    metrics.style.display = "none";
    distribution.style.display = "none";
    $("brightness-quality").textContent = "";
    $("brightness-pairs").textContent = "";
    roiActions.style.display = "none";
    state.roiPairs = [];
    state.roiDirty = false;
    indicator.style.left = "0%";
    track.setAttribute("aria-valuenow", "0");
    renderBrightnessRois([]);
    return;
  }
  const paired = meter.percentual_pares_mediana;
  const displayedPercent = paired ?? meter.percentual;
  const freeComparison = meter.modo_metrica === "pontos_livres";
  const freePair = meter.pares?.find((pair) => pair.modo === "livre");
  const pointAName = freePair?.regiao_a || "ponto A";
  const pointBName = freePair?.regiao_b || "ponto B";
  state.roiPairs = (meter.pares || []).map((pair) => ({ ...pair, cortex_centro: [...pair.cortex_centro], cec_centro: [...pair.cec_centro] }));
  if (!options.keepDirty) state.roiDirty = false;
  $("brightness-meter-title").textContent = freeComparison ? `Brilho de ${pointAName} (A) em relação a ${pointBName} (B)` : "Ecogenicidade cortical";
  $("brightness-reference").textContent = freeComparison ? `Referência: ${pointBName} (B)` : "Referência: CEC";
  $("brightness-right-label").textContent = freeComparison ? `Brilho de ${pointBName}` : "Brilho do CEC";
  $("brightness-difference-label").textContent = freeComparison ? `Δ ${pointBName} − ${pointAName}` : "Δ CEC − córtex";
  $("brightness-value").textContent = `${displayedPercent.toFixed(1)}% do ${freeComparison ? pointBName : "CEC"}`;
  $("brightness-pairs").textContent = meter.pares?.length
    ? `Mediana de ${meter.pares.length} pares: ${meter.pares.map((pair) => `${pair.faixa} ${pair.percentual.toFixed(1)}%${pair.modo === "livre" ? ` (${pair.regiao_a} / ${pair.regiao_b})` : ""}${pair.diferenca_profundidade_px ? ` · Δprof. ${pair.diferenca_profundidade_px} px` : ""}`).join(" · ")} · IQR ${meter.percentual_pares_iqr.toFixed(1)}%.`
    : "Amostragem pareada indisponível; exibindo descritor global.";
  unavailable.style.display = "none";
  roiActions.style.display = meter.pares?.length ? "flex" : "none";
  $("save-brightness-rois").disabled = !state.roiDirty;
  $("restore-brightness-rois").disabled = meter.origem_pares !== "manual";
  metrics.style.display = "grid";
  distribution.style.display = "grid";
  $("brightness-difference").textContent = `${meter.diferenca_cec_cortex >= 0 ? "+" : ""}${meter.diferenca_cec_cortex.toFixed(1)}`;
  $("brightness-contrast").textContent = `${meter.contraste_normalizado.toFixed(1)}%`;
  $("brightness-cortex-medulla").parentElement.style.display = "";
  $("brightness-cortex-medulla-difference").parentElement.style.display = "";
  $("brightness-ratio-label").textContent = freeComparison ? `${pointAName} (A) / ${pointBName} (B)` : "Córtex / medula";
  $("brightness-secondary-difference-label").textContent = freeComparison ? `Diferença ${pointAName} − ${pointBName}` : "Diferença C − M";
  $("brightness-cortex-label").textContent = freeComparison ? `${pointAName} (A)` : "Córtex";
  $("brightness-cec-label").textContent = freeComparison ? `${pointBName} (B)` : "CEC";
  $("brightness-cortex-medulla").textContent = meter.cortex_medulla_percentual === undefined ? "—" : `${meter.cortex_medulla_percentual.toFixed(1)}%`;
  $("brightness-cortex-medulla-difference").textContent = meter.diferenca_cortex_medulla === undefined ? "—" : `${meter.diferenca_cortex_medulla >= 0 ? "+" : ""}${meter.diferenca_cortex_medulla.toFixed(1)}`;
  $("brightness-cortex-distribution").textContent = `Med. ${meter.cortex_mediana.toFixed(1)} · IQR ${meter.cortex_iqr.toFixed(1)}`;
  $("brightness-cec-distribution").textContent = `Med. ${meter.cec_mediana.toFixed(1)} · IQR ${meter.cec_iqr.toFixed(1)}`;
  $("brightness-quality").textContent = freeComparison ? `Amostra: A ${meter.cortex_pixels.toLocaleString("pt-BR")} px; B ${meter.cec_pixels.toLocaleString("pt-BR")} px. Saturação: ${meter.cortex_saturacao.toFixed(1)}% / ${meter.cec_saturacao.toFixed(1)}%. Calculado nas ROIs marcadas, sem CLAHE.` : `Amostra: córtex ${meter.cortex_pixels.toLocaleString("pt-BR")} px; CEC ${meter.cec_pixels.toLocaleString("pt-BR")} px. Saturação: ${meter.cortex_saturacao.toFixed(1)}% / ${meter.cec_saturacao.toFixed(1)}%.`;
  const displayedPosition = Math.max(0, Math.min(100, displayedPercent));
  indicator.style.left = `${displayedPosition}%`;
  track.setAttribute("aria-valuenow", String(displayedPosition));
  renderBrightnessRois(state.roiPairs);
}
function renderBrightnessRois(pairs) {
  const layer = $("brightness-roi-layer");
  layer.replaceChildren();
  const width = state.originalWidth;
  const height = state.originalHeight;
  layer.setAttribute("viewBox", `0 0 ${width || 1} ${height || 1}`);
  if (!pairs.length) return;
  const namespace = "http://www.w3.org/2000/svg";
  const create = (tag, attributes = {}) => {
    const element = document.createElementNS(namespace, tag);
    Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, String(value)));
    return element;
  };
  layer.classList.toggle("adjusting", $("adjust-brightness-rois").checked);
  pairs.forEach((pair, index) => {
    const [cortexX, cortexY] = pair.cortex_centro;
    const [cecX, cecY] = pair.cec_centro;
    const radius = pair.raio;
    const group = create("g", { "data-pair": index });
    group.append(create("line", { x1: cortexX, y1: cortexY, x2: cecX, y2: cecY, class: "roi-link" }));
    const cortexCircle = create("circle", { cx: cortexX, cy: cortexY, r: radius, class: "roi-cortex", "data-region": "cortex" });
    const cecCircle = create("circle", { cx: cecX, cy: cecY, r: radius, class: "roi-cec", "data-region": "cec" });
    cortexCircle.addEventListener("pointerdown", (event) => beginRoiDrag(event, index, "cortex"));
    cecCircle.addEventListener("pointerdown", (event) => beginRoiDrag(event, index, "cec"));
    group.append(cortexCircle, cecCircle);
    const text = create("text", { x: (cortexX + cecX) / 2, y: Math.min(cortexY, cecY) - radius - 8, "text-anchor": "middle" });
    text.textContent = `${index + 1}. ${pair.faixa}: ${pair.percentual.toFixed(1)}%`;
    group.append(text);
    layer.append(group);
  });
  layer.style.display = $("toggle-brightness-rois").checked ? "block" : "none";
}
function roiPoint(event) {
  const layer = $("brightness-roi-layer");
  const matrix = layer.getScreenCTM();
  const point = layer.createSVGPoint();
  point.x = event.clientX; point.y = event.clientY;
  const local = matrix ? point.matrixTransform(matrix.inverse()) : point;
  return [
    Math.round(Math.min(Math.max(local.x, 0), state.originalWidth - 1)),
    Math.round(Math.min(Math.max(local.y, 0), state.originalHeight - 1)),
  ];
}
function updateRoiPairDrawing(index) {
  const pair = state.roiPairs[index];
  const group = $("brightness-roi-layer").querySelector(`[data-pair="${index}"]`);
  if (!group || !pair) return;
  const [cortexX, cortexY] = pair.cortex_centro;
  const [cecX, cecY] = pair.cec_centro;
  const line = group.querySelector("line");
  line.setAttribute("x1", cortexX); line.setAttribute("y1", cortexY); line.setAttribute("x2", cecX); line.setAttribute("y2", cecY);
  const cortexCircle = group.querySelector('[data-region="cortex"]');
  const cecCircle = group.querySelector('[data-region="cec"]');
  cortexCircle.setAttribute("cx", cortexX); cortexCircle.setAttribute("cy", cortexY);
  cecCircle.setAttribute("cx", cecX); cecCircle.setAttribute("cy", cecY);
  const label = group.querySelector("text");
  label.setAttribute("x", (cortexX + cecX) / 2);
  label.setAttribute("y", Math.min(cortexY, cecY) - pair.raio - 8);
}
function beginRoiDrag(event, index, region) {
  if (!$("adjust-brightness-rois").checked) return;
  event.preventDefault();
  event.stopPropagation();
  state.roiDragging = { index, region, before: JSON.stringify(state.roiPairs) };
  $("brightness-roi-layer").setPointerCapture(event.pointerId);
}
async function finishRoiDrag() {
  const drag = state.roiDragging;
  if (!drag) return;
  state.roiDragging = null;
  try {
    const result = await request("/api/brightness-rois", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: state.currentId, reviewer: reviewer(), model: state.segmentationModel, pares: state.roiPairs, preview: true }),
    });
    state.roiDirty = true;
    showBrightnessMeter(result.medidor_brilho, { keepDirty: true });
    showMessage("ROI ajustada. Salve as ROIs para registrar a seleção do revisor.");
  } catch (error) {
    state.roiPairs = JSON.parse(drag.before);
    renderBrightnessRois(state.roiPairs);
    showMessage(error.message, true);
  }
}
function polygonColor() {
  return { rim: "#f34f56", cortex: "#41cbd0", medulla: "#ffdb3b", central_echo_complex: "#ff9100", anomalia: "#b861ff" }[$("polygon-layer").value];
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
  state.comparing = false;
  state.comparatorStart = null;
  clearComparatorPreview();
  $("comparator-panel").classList.remove("open");
  $("brightness-roi-layer").classList.remove("comparing");
  $("tool-comparator").classList.remove("active");
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
function comparatorRadius() { return Number($("comparator-radius").value || 8); }
function clearComparatorPreview() { $("brightness-roi-layer").querySelector("#comparator-preview")?.remove(); }
function renderComparatorPreview(start, end = start) {
  clearComparatorPreview();
  const layer = $("brightness-roi-layer");
  const namespace = "http://www.w3.org/2000/svg";
  const group = document.createElementNS(namespace, "g");
  group.id = "comparator-preview";
  const make = (tag, attributes) => {
    const element = document.createElementNS(namespace, tag);
    Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, String(value)));
    return element;
  };
  const radius = comparatorRadius();
  group.append(
    make("line", { x1: start[0], y1: start[1], x2: end[0], y2: end[1], class: "roi-link" }),
    make("circle", { cx: start[0], cy: start[1], r: radius, class: "roi-cec" }),
    make("circle", { cx: end[0], cy: end[1], r: radius, class: "roi-cortex" }),
  );
  layer.append(group);
}
function setComparator() {
  const enabled = !state.comparing;
  state.comparing = enabled;
  state.comparatorStart = null;
  state.drawing = false;
  resetPolygon();
  $("polygon-panel").classList.remove("open");
  $("drawing-layer").classList.remove("active");
  ["polygon", "brush", "eraser"].forEach((name) => $(`tool-${name}`).classList.remove("active"));
  $("tool-comparator").classList.toggle("active", enabled);
  $("comparator-panel").classList.toggle("open", enabled);
  $("adjust-brightness-rois").checked = false;
  $("brightness-roi-layer").classList.remove("adjusting");
  $("brightness-roi-layer").classList.toggle("comparing", enabled);
  if (!enabled) clearComparatorPreview();
}
async function addComparatorPair() {
  const start = state.comparatorStart;
  if (!start || !state.currentId) return;
  const end = state.comparatorEnd;
  state.comparatorStart = null;
  state.comparatorEnd = null;
  clearComparatorPreview();
  if (!end) return;
  if (state.roiPairs.length >= 3) {
    showMessage("Há três pares de ROIs. Remova ou ajuste um par existente antes de adicionar outro.", true);
    return;
  }
  const before = JSON.stringify(state.roiPairs);
  state.roiPairs.push({
    faixa: `manual ${state.roiPairs.length + 1}`,
    cec_centro: start,
    cortex_centro: end,
    raio: comparatorRadius(),
    modo: "livre",
  });
  try {
    const result = await request("/api/brightness-rois", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: state.currentId, reviewer: reviewer(), model: state.segmentationModel, pares: state.roiPairs, preview: true }),
    });
    state.roiDirty = true;
    showBrightnessMeter(result.medidor_brilho, { keepDirty: true });
    showMessage("Comparação ponto A → ponto B adicionada. Salve as ROIs para registrar a seleção.");
  } catch (error) {
    state.roiPairs = JSON.parse(before);
    renderBrightnessRois(state.roiPairs);
    showMessage(error.message, true);
  }
}
function drawingPoint(event) {
  const layer = $("drawing-layer");
  // Converte o clique para o proprio sistema de coordenadas do SVG. Isso
  // preserva a posicao correta quando a imagem foi redimensionada, recebeu
  // zoom ou tiver barras laterais por diferenca de proporcao.
  const matrix = layer.getScreenCTM();
  if (matrix) {
    const point = layer.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const local = point.matrixTransform(matrix.inverse());
    return [
      Math.round(Math.min(Math.max(local.x, 0), state.originalWidth - 1)),
      Math.round(Math.min(Math.max(local.y, 0), state.originalHeight - 1)),
    ];
  }
  const bounds = layer.getBoundingClientRect();
  return [
    Math.round((event.clientX - bounds.left) * state.originalWidth / bounds.width),
    Math.round((event.clientY - bounds.top) * state.originalHeight / bounds.height),
  ];
}
async function savePolygon() {
  if (!state.currentId) return;
  if (!ensureReviewer()) return;
  if (state.saving) return;
  if (state.drawingTool === "polygon" && state.points.length < 3) return showMessage("Desenhe pelo menos tres vertices.", true);
  if (state.drawingTool !== "polygon" && state.points.length < 1) return showMessage("Desenhe com o pincel ou a borracha antes de aplicar.", true);
  state.saving = true;
  $("polygon-save").disabled = true;
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
    const reviewField = layer === "rim" ? "status_rim" : `status_${layer}`;
    if (document.querySelector(`[data-field="${reviewField}"]`)) chooseButton(reviewField, "corrigir");
    const label = layer === "anomalia" ? "Marcador de anomalia" : "Mascara manual";
    showMessage(`${label} salvo com versao de auditoria; finalize a avaliacao em Salvar e avancar.`);
  } catch (error) {
    showMessage(error.message, true);
  } finally {
    state.saving = false;
    $("polygon-save").disabled = false;
  }
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
    setLayer("anomalia", item.layers.anomalia);
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
    showBrightnessMeter(item.medidor_brilho);
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
["rim", "cortex", "medulla", "central_echo_complex", "anomalia"].forEach((name) => $(`toggle-${name}`).addEventListener("change", (event) => {
  $(`layer-${name}`).style.display = event.target.checked ? "block" : "none";
}));
$("toggle-brightness-rois").addEventListener("change", (event) => {
  $("brightness-roi-layer").style.display = event.target.checked ? "block" : "none";
});
$("adjust-brightness-rois").addEventListener("change", () => {
  $("brightness-roi-layer").classList.toggle("adjusting", $("adjust-brightness-rois").checked);
});
$("brightness-roi-layer").addEventListener("pointermove", (event) => {
  const drag = state.roiDragging;
  if (!drag) return;
  const pair = state.roiPairs[drag.index];
  if (!pair) return;
  event.preventDefault();
  pair[drag.region === "cortex" ? "cortex_centro" : "cec_centro"] = roiPoint(event);
  updateRoiPairDrawing(drag.index);
});
["pointerup", "pointercancel"].forEach((eventName) => $("brightness-roi-layer").addEventListener(eventName, () => finishRoiDrag()));
$("save-brightness-rois").addEventListener("click", async () => {
  if (!state.currentId || !ensureReviewer() || !state.roiPairs.length) return;
  try {
    const result = await request("/api/brightness-rois", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: state.currentId, reviewer: reviewer(), model: state.segmentationModel, pares: state.roiPairs }),
    });
    state.roiDirty = false;
    showBrightnessMeter(result.medidor_brilho);
    showMessage("ROIs ajustadas salvas para este revisor e esta versão do modelo.");
  } catch (error) { showMessage(error.message, true); }
});
$("restore-brightness-rois").addEventListener("click", async () => {
  if (!state.currentId || !ensureReviewer()) return;
  try {
    await request("/api/brightness-rois", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: state.currentId, reviewer: reviewer(), model: state.segmentationModel, restore: true }),
    });
    await selectItem(state.currentId);
    showMessage("ROIs automáticas restauradas.");
  } catch (error) { showMessage(error.message, true); }
});
$("zoom-in").addEventListener("click", () => { state.zoom = Math.min(3, state.zoom + 0.25); applyZoom(); });
$("zoom-out").addEventListener("click", () => { state.zoom = Math.max(0.5, state.zoom - 0.25); applyZoom(); });
$("zoom-reset").addEventListener("click", () => { state.zoom = 1; applyZoom(); });
$("tool-zoom").addEventListener("click", () => { state.zoom = Math.min(3, state.zoom + 0.25); applyZoom(); });
$("tool-polygon").addEventListener("click", () => setDrawingTool("polygon"));
$("tool-brush").addEventListener("click", () => setDrawingTool("brush"));
$("tool-eraser").addEventListener("click", () => setDrawingTool("eraser"));
$("tool-comparator").addEventListener("click", setComparator);
$("calculate-comparator-pairs").addEventListener("click", async () => {
  if (!state.currentId || !ensureReviewer()) return;
  if (!state.roiPairs.length) {
    showMessage("Marque ao menos um par A → B antes de calcular.", true);
    return;
  }
  try {
    const result = await request("/api/brightness-rois", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: state.currentId, reviewer: reviewer(), model: state.segmentationModel, pares: state.roiPairs, preview: true }),
    });
    state.roiDirty = true;
    showBrightnessMeter(result.medidor_brilho, { keepDirty: true });
    showMessage("Cálculo atualizado a partir das marcações A/B. Salve as ROIs para registrar a seleção.");
  } catch (error) { showMessage(error.message, true); }
});
$("clear-comparator-pairs").addEventListener("click", async () => {
  if (!state.currentId || !ensureReviewer()) return;
  try {
    await request("/api/brightness-rois", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: state.currentId, reviewer: reviewer(), model: state.segmentationModel, restore: true }),
    });
    state.roiPairs = [];
    state.roiDirty = false;
    clearComparatorPreview();
    await selectItem(state.currentId);
    showMessage("Marcações manuais A/B removidas; as ROIs automáticas foram restauradas.");
  } catch (error) { showMessage(error.message, true); }
});
$("comparator-radius").addEventListener("input", () => {
  $("comparator-radius-value").textContent = `${comparatorRadius()} px`;
  if (state.comparatorStart) renderComparatorPreview(state.comparatorStart, state.comparatorEnd || state.comparatorStart);
});
$("brightness-roi-layer").addEventListener("pointerdown", (event) => {
  if (!state.comparing || !state.originalWidth || !state.originalHeight) return;
  event.preventDefault();
  state.comparatorStart = roiPoint(event);
  state.comparatorEnd = state.comparatorStart;
  $("brightness-roi-layer").setPointerCapture(event.pointerId);
  renderComparatorPreview(state.comparatorStart);
});
$("brightness-roi-layer").addEventListener("pointermove", (event) => {
  if (!state.comparing || !state.comparatorStart) return;
  event.preventDefault();
  state.comparatorEnd = roiPoint(event);
  renderComparatorPreview(state.comparatorStart, state.comparatorEnd);
});
function finishComparatorPair(event) {
  if (!state.comparing || !state.comparatorStart) return;
  event?.preventDefault();
  if (event) state.comparatorEnd = roiPoint(event);
  addComparatorPair();
}
$("brightness-roi-layer").addEventListener("pointerup", finishComparatorPair);
// Alguns navegadores encerram o arraste fora do SVG mesmo quando a captura do
// ponteiro está ativa. O listener global torna a soltura confiável nesses casos.
window.addEventListener("pointerup", finishComparatorPair);
$("brightness-roi-layer").addEventListener("pointercancel", () => {
  if (!state.comparing) return;
  state.comparatorStart = null;
  state.comparatorEnd = null;
  clearComparatorPreview();
});
$("toggle-panels").addEventListener("click", () => {
  setFocusMode(!$("workspace").classList.contains("focus-mode"));
});
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
  $("brightness-roi-layer").setAttribute("viewBox", `0 0 ${width} ${height}`);
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
$("toggle-left-details").addEventListener("click", () => {
  const leftRail = document.querySelector(".left");
  const collapsed = leftRail.classList.toggle("details-collapsed");
  $("toggle-left-details").textContent = collapsed ? "Mostrar progresso e filtros" : "Recolher progresso e filtros";
  $("toggle-left-details").setAttribute("aria-expanded", String(!collapsed));
  localStorage.setItem("curadoria_left_details_collapsed", collapsed ? "1" : "0");
});
if (localStorage.getItem("curadoria_left_details_collapsed") === "1") $("toggle-left-details").click();
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
  setFocusMode(localStorage.getItem("curadoria_focus_mode") === "1");
  $("brush-size-field").style.display = "none";
  state.segmentationModel = $("segmentation-model").value;
  await loadMeta(); await loadQueue(false); await loadCorrections();
}
start().catch((error) => showMessage(error.message, true));
