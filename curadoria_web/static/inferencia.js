const $ = (id) => document.getElementById(id);
let selectedFile = null;
let currentInferenceId = "";
let currentMaskProposal = null;

function initAssistantFocus() {
  const panel = document.querySelector(".assistant-panel");
  const backdrop = document.querySelector(".assistant-focus-backdrop");
  const openButton = document.querySelector(".assistant-focus-open");
  const closeButton = document.querySelector(".assistant-focus-close");
  if (!panel || !backdrop || !openButton || !closeButton) return;
  const setFocus = (enabled) => {
    panel.classList.toggle("assistant-focus-active", enabled);
    backdrop.hidden = !enabled;
    document.body.classList.toggle("assistant-focus-open", enabled);
    panel.setAttribute("aria-modal", String(enabled));
    (enabled ? closeButton : openButton).focus();
  };
  openButton.addEventListener("click", () => setFocus(true));
  closeButton.addEventListener("click", () => setFocus(false));
  backdrop.addEventListener("click", () => setFocus(false));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && panel.classList.contains("assistant-focus-active")) setFocus(false);
  });
}
initAssistantFocus();

function message(text, error = false) {
  $("inference-message").textContent = text;
  $("inference-message").classList.toggle("error", error);
}
function chooseFile(file) {
  if (!file) return;
  if (!["image/png", "image/jpeg"].includes(file.type)) return message("Envie uma imagem PNG, JPG ou JPEG.", true);
  if (file.size > 20 * 1024 * 1024) return message("A imagem excede o limite de 20 MB.", true);
  selectedFile = file;
  $("selected-file").hidden = false;
  $("selected-file").textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB`;
  $("run-inference").disabled = false;
  message("");
}
function fileData(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Não foi possível ler a imagem."));
    reader.readAsDataURL(file);
  });
}
function renderMeter(meter) {
  if (!meter || !meter.disponivel) return;
  $("inference-brightness-value").textContent = `${meter.percentual.toFixed(1)}% do CEC`;
  $("inference-brightness-indicator").style.left = `${meter.posicao}%`;
  $("inference-difference").textContent = `${meter.diferenca_cec_cortex >= 0 ? "+" : ""}${meter.diferenca_cec_cortex.toFixed(1)}`;
  $("inference-contrast").textContent = `${meter.contraste_normalizado.toFixed(1)}%`;
  $("inference-cortex-medulla").textContent = meter.cortex_medulla_percentual === undefined ? "—" : `${meter.cortex_medulla_percentual.toFixed(1)}%`;
  $("inference-cortex-medulla-difference").textContent = meter.diferenca_cortex_medulla === undefined ? "—" : `${meter.diferenca_cortex_medulla >= 0 ? "+" : ""}${meter.diferenca_cortex_medulla.toFixed(1)}`;
  $("inference-cortex-distribution").textContent = `Med. ${meter.cortex_mediana.toFixed(1)} · IQR ${meter.cortex_iqr.toFixed(1)}`;
  $("inference-cec-distribution").textContent = `Med. ${meter.cec_mediana.toFixed(1)} · IQR ${meter.cec_iqr.toFixed(1)}`;
  $("inference-quality").textContent = `Amostra: córtex ${meter.cortex_pixels.toLocaleString("pt-BR")} px; CEC ${meter.cec_pixels.toLocaleString("pt-BR")} px. Saturação: ${meter.cortex_saturacao.toFixed(1)}% / ${meter.cec_saturacao.toFixed(1)}%.`;
}
async function runInference() {
  if (!selectedFile) return;
  $("run-inference").disabled = true;
  message("Executando os modelos…");
  try {
    const response = await fetch("/api/inferences", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({filename: selectedFile.name, image_data: await fileData(selectedFile)})});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Falha na inferência.");
    const result = payload.inference;
    currentInferenceId = result.image_id;
    currentMaskProposal = null;
    $("result-name").textContent = result.filename;
    $("result-dimensions").textContent = `${result.info.dimensao} · ${result.info.dispositivo}`;
    $("inference-image").src = `${result.image_url}?v=${Date.now()}`;
    Object.entries(result.layers).forEach(([name, url]) => {
      const layer = $(`inference-${name}`);
      if (!layer) return;
      layer.src = `${url}?v=${Date.now()}`;
      layer.style.display = name === "uncertainty" ? "none" : "block";
    });
    renderMeter(result.medidor_brilho);
    renderInferenceQuality(result);
    $("inference-warning").textContent = result.aviso;
    $("inference-result").hidden = false;
    $("inference-assistant-local").disabled = false;
    $("inference-assistant-cloud").disabled = false;
    $("inference-assistant-compare").disabled = false;
    resetMaskCommand();
    message("Inferência concluída.");
  } catch (error) { message(error.message, true); }
  finally { $("run-inference").disabled = false; }
}
function assistantResultMessage(text, error = false) {
  $("inference-assistant-message").textContent = text;
  $("inference-assistant-message").classList.toggle("error", error);
}
function renderAssistantMarkdown(text) {
  const escaped = String(text || "—").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character]);
  return escaped.replace(/\*\*([\s\S]+?)\*\*/g, "<b>$1</b>");
}
function renderInferenceQuality(result) {
  const quality = result.qualidade || {};
  const status = $("inference-quality-status");
  status.textContent = quality.status === "revisar_com_atencao"
    ? `Atenção: ${(quality.alertas || []).join(" · ")}`
    : "Nenhum alerta automático de confiança ou área foi detectado; a revisão humana continua obrigatória.";
  status.classList.toggle("warning", quality.status === "revisar_com_atencao");
  const grid = $("inference-confidence-grid");
  grid.replaceChildren();
  const confidenceItems = [{label: "Rim", value: quality.confianca_rim}];
  (result.priorizacao_regioes || []).forEach((region) => confidenceItems.push({label: region.rotulo || region.regiao, value: region.confianca}));
  confidenceItems.forEach((item) => {
    const box = document.createElement("div");
    const label = document.createElement("span");
    const value = document.createElement("b");
    label.textContent = item.label;
    value.textContent = Number.isFinite(item.value) ? `${(item.value * 100).toFixed(1)}%` : "—";
    box.append(label, value);
    grid.append(box);
  });
  const list = $("inference-review-priority");
  list.replaceChildren();
  (result.priorizacao_regioes || []).forEach((region) => {
    const item = document.createElement("li");
    item.className = `priority-${region.prioridade || "baixa"}`;
    item.textContent = `${region.rotulo || region.regiao}: prioridade ${region.prioridade || "baixa"}. ${(region.motivos || []).join("; ")}.`;
    list.append(item);
  });
}
function renderInferenceStructured(results) {
  const target = $("inference-assistant-structured");
  const structured = results.flatMap((result) => result.structured ? [result.structured] : []);
  if (!structured.length) {
    target.hidden = true;
    return;
  }
  const mentioned = [...new Set(structured.flatMap((item) => item.regioes_mencionadas || []))];
  const priority = [...new Set(structured.flatMap((item) => (item.regioes_priorizadas || [])
    .filter((region) => region.prioridade === "alta").map((region) => region.regiao)))];
  target.textContent = [
    "Síntese estruturada — revisão humana obrigatória",
    `Regiões mencionadas: ${mentioned.join(", ") || "nenhuma identificada automaticamente"}`,
    `Prioridade alta da segmentação: ${priority.join(", ") || "nenhuma"}`,
    "A priorização é operacional e não representa probabilidade diagnóstica.",
  ].join("\n");
  target.hidden = false;
}
function resetMaskCommand() {
  currentMaskProposal = null;
  const panel = $("inference-mask-command");
  const proposalImage = $("inference-proposal");
  if (panel) panel.hidden = true;
  if (proposalImage) {
    proposalImage.removeAttribute("src");
    proposalImage.style.display = "none";
  }
  if ($("inference-mask-apply")) $("inference-mask-apply").hidden = true;
  if ($("inference-mask-propose")) $("inference-mask-propose").disabled = false;
  if ($("inference-mask-command-result")) $("inference-mask-command-result").textContent = "";
}
function renderMaskCommand(results) {
  const command = results.map((result) => result.mask_command).find((item) => item);
  const panel = $("inference-mask-command");
  if (!command || command.status !== "proposta_disponivel" || !panel) {
    resetMaskCommand();
    return;
  }
  currentMaskProposal = {layer: command.layer, operation: command.operation};
  $("inference-mask-command-title").textContent = `Comando de máscara detectado: ${command.rotulo}`;
  $("inference-mask-command-instruction").textContent = `${command.instruction} A proposta será gerada sem apagar a máscara atual.`;
  $("inference-mask-command-result").textContent = "Aguardando confirmação para gerar a proposta.";
  panel.hidden = false;
}
async function requestMaskProposal() {
  if (!currentInferenceId || !currentMaskProposal) return;
  const reviewer = $("inference-reviewer").value.trim();
  if (!reviewer) return assistantResultMessage("Informe o revisor antes de gerar a proposta.", true);
  const button = $("inference-mask-propose");
  button.disabled = true;
  $("inference-mask-command-result").textContent = "Gerando proposta determinística sobre a máscara atual…";
  try {
    const response = await fetch(`/api/inferences/${encodeURIComponent(currentInferenceId)}/mask-proposals`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({...currentMaskProposal, reviewer}),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Falha ao gerar a proposta.");
    const proposal = payload.proposal;
    currentMaskProposal = proposal;
    const proposalImage = $("inference-proposal");
    proposalImage.src = `${proposal.proposal_url}?v=${Date.now()}`;
    proposalImage.style.display = "block";
    $("inference-mask-command-result").textContent = `Proposta criada: +${proposal.added_pixels.toLocaleString("pt-BR")} pixels em verde e -${proposal.removed_pixels.toLocaleString("pt-BR")} em vermelho. Kernel ${proposal.kernel_size} px. Revise a sobreposição antes de aplicar.`;
    $("inference-mask-apply").hidden = false;
    assistantResultMessage("Proposta de máscara criada; a original permanece preservada.");
  } catch (error) {
    button.disabled = false;
    $("inference-mask-command-result").textContent = error.message;
    assistantResultMessage(error.message, true);
  }
}
async function applyMaskProposal() {
  if (!currentInferenceId || !currentMaskProposal || !currentMaskProposal.proposal_id) return;
  const reviewer = $("inference-reviewer").value.trim();
  if (!reviewer) return assistantResultMessage("Informe o revisor antes de aplicar a proposta.", true);
  const button = $("inference-mask-apply");
  button.disabled = true;
  try {
    const response = await fetch(`/api/inferences/${encodeURIComponent(currentInferenceId)}/mask-proposals/${encodeURIComponent(currentMaskProposal.proposal_id)}/apply`, {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({reviewer}),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Falha ao aplicar a proposta.");
    const applied = payload.proposal;
    const layerImage = $(`inference-${applied.layer}`);
    if (layerImage) layerImage.src = `${applied.layers[applied.layer]}?v=${Date.now()}`;
    $("inference-proposal").style.display = "none";
    $("inference-mask-command-result").textContent = "Proposta aplicada à sessão. A máscara original continua preservada no histórico.";
    button.hidden = true;
    button.disabled = false;
    assistantResultMessage("Máscara atualizada após confirmação do revisor.");
  } catch (error) {
    button.disabled = false;
    assistantResultMessage(error.message, true);
  }
}
function renderInferenceAssistantResults(results, context = {}) {
  $("inference-assistant-evidence").textContent = `Evidências: ${(context.evidence_refs || []).join(" · ")} · Imagem: ${(context.image_size || []).join(" × ")} px`;
  const body = $("inference-assistant-table-body");
  body.replaceChildren();
  results.forEach((result) => {
    const row = document.createElement("tr");
    [result.visual_mode === "full" ? "Imagem completa" : "Recorte da cápsula", result.answer, (result.limitations || []).join(" ")].forEach((value, index) => {
      const cell = document.createElement("td");
      if (index === 1) cell.innerHTML = renderAssistantMarkdown(value);
      else cell.textContent = value || "—";
      if (index === 1) cell.className = "assistant-answer";
      row.append(cell);
    });
    body.append(row);
  });
  renderInferenceStructured(results);
  renderMaskCommand(results);
  $("inference-assistant-result").hidden = false;
}
async function analyzeInferenceLocal() {
  if (!currentInferenceId) return;
  const reviewer = $("inference-reviewer").value.trim();
  if (!reviewer) return assistantResultMessage("Informe o revisor.", true);
  const button = $("inference-assistant-local");
  button.disabled = true;
  assistantResultMessage("Executando o Qwen local com os artefatos recém-gerados…");
  try {
    const response = await fetch("/api/assistant/analyze", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({
      image_id: currentInferenceId,
      reviewer,
      model: "unet",
      visual_mode: $("inference-assistant-mode").value,
      question: $("inference-assistant-question").value,
      provider: "local",
      consent_external_processing: false,
    })});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Falha na análise local.");
    const result = payload.assistant;
    renderInferenceAssistantResults([result], result.evidence_context || {});
    assistantResultMessage("Análise local concluída; revise a resposta.");
  } catch (error) { assistantResultMessage(error.message, true); }
  finally { button.disabled = false; }
}
async function analyzeInferenceCloud() {
  if (!currentInferenceId) return;
  const reviewer = $("inference-reviewer").value.trim();
  if (!reviewer) return assistantResultMessage("Informe o revisor.", true);
  if (!$('inference-assistant-consent').checked) return assistantResultMessage("Marque o consentimento para usar a OpenAI.", true);
  const button = $("inference-assistant-cloud");
  button.disabled = true;
  assistantResultMessage("Enviando a imagem e as evidências autorizadas ao GPT-6 Luna…");
  try {
    const response = await fetch("/api/assistant/analyze", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({
      image_id: currentInferenceId,
      reviewer,
      model: "unet",
      visual_mode: $("inference-assistant-mode").value,
      question: $("inference-assistant-question").value,
      provider: "cloud",
      consent_external_processing: true,
    })});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Falha na análise com a OpenAI.");
    const result = payload.assistant;
    renderInferenceAssistantResults([result], result.evidence_context || {});
    assistantResultMessage("Análise com GPT-6 Luna concluída; revise a resposta.");
  } catch (error) { assistantResultMessage(error.message, true); }
  finally { button.disabled = false; }
}
async function compareInference() {
  if (!currentInferenceId) return;
  const reviewer = $("inference-reviewer").value.trim();
  if (!reviewer) return assistantResultMessage("Informe o revisor.", true);
  if (!$("inference-assistant-consent").checked) return assistantResultMessage("Marque o consentimento para usar a OpenAI.", true);
  const button = $("inference-assistant-compare");
  button.disabled = true;
  assistantResultMessage("Executando os dois modelos com os artefatos recém-gerados…");
  try {
    const response = await fetch("/api/assistant/compare", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({
      image_id: currentInferenceId,
      reviewer,
      model: "unet",
      visual_mode: $("inference-assistant-mode").value,
      question: $("inference-assistant-question").value,
      consent_external_processing: true,
    })});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Falha na comparação.");
    const comparison = payload.comparison;
    renderInferenceAssistantResults(comparison.results || [], comparison.evidence_context || {});
    assistantResultMessage("Comparação concluída; revise as respostas.");
  } catch (error) { assistantResultMessage(error.message, true); }
  finally { button.disabled = false; }
}
$("inference-file").addEventListener("change", (event) => chooseFile(event.target.files[0]));
$("drop-zone").addEventListener("dragover", (event) => { event.preventDefault(); $("drop-zone").classList.add("dragging"); });
$("drop-zone").addEventListener("dragleave", () => $("drop-zone").classList.remove("dragging"));
$("drop-zone").addEventListener("drop", (event) => { event.preventDefault(); $("drop-zone").classList.remove("dragging"); chooseFile(event.dataTransfer.files[0]); });
$("run-inference").addEventListener("click", runInference);
$("inference-assistant-local").addEventListener("click", analyzeInferenceLocal);
$("inference-assistant-cloud").addEventListener("click", analyzeInferenceCloud);
$("inference-assistant-compare").addEventListener("click", compareInference);
$("inference-mask-propose").addEventListener("click", requestMaskProposal);
$("inference-mask-apply").addEventListener("click", applyMaskProposal);
document.querySelectorAll("[data-layer]").forEach((toggle) => toggle.addEventListener("change", () => { $(`inference-${toggle.dataset.layer}`).style.display = toggle.checked ? "block" : "none"; }));
