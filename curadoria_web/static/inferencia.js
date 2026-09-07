const $ = (id) => document.getElementById(id);
let selectedFile = null;

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
    $("result-name").textContent = result.filename;
    $("result-dimensions").textContent = `${result.info.dimensao} · ${result.info.dispositivo}`;
    $("inference-image").src = `${result.image_url}?v=${Date.now()}`;
    Object.entries(result.layers).forEach(([name, url]) => { $(`inference-${name}`).src = `${url}?v=${Date.now()}`; });
    renderMeter(result.medidor_brilho);
    $("inference-warning").textContent = result.aviso;
    $("inference-result").hidden = false;
    message("Inferência concluída.");
  } catch (error) { message(error.message, true); }
  finally { $("run-inference").disabled = false; }
}
$("inference-file").addEventListener("change", (event) => chooseFile(event.target.files[0]));
$("drop-zone").addEventListener("dragover", (event) => { event.preventDefault(); $("drop-zone").classList.add("dragging"); });
$("drop-zone").addEventListener("dragleave", () => $("drop-zone").classList.remove("dragging"));
$("drop-zone").addEventListener("drop", (event) => { event.preventDefault(); $("drop-zone").classList.remove("dragging"); chooseFile(event.dataTransfer.files[0]); });
$("run-inference").addEventListener("click", runInference);
document.querySelectorAll("[data-layer]").forEach((toggle) => toggle.addEventListener("change", () => { $(`inference-${toggle.dataset.layer}`).style.display = toggle.checked ? "block" : "none"; }));
