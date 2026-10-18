"""Seleciona casos reais da Curadoria Web e gera figuras com proveniencia.

Requer manifest.csv e, para revisoes, o banco SQLite local. Nao altera as fontes.
Exemplos: python scripts/figuras/gerar_figuras_curadoria.py listar
          python scripts/figuras/gerar_figuras_curadoria.py revisao --image-id ID --reviewer REV --layer cortex
          python scripts/figuras/gerar_figuras_curadoria.py multiclasse --image-id ID
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "dataset_aumentado/curadoria/miniaturas_completas/manifest.csv"
DEFAULT_DB = PROJECT_ROOT / "dataset_aumentado/curadoria/respostas/curadoria.sqlite3"
DEFAULT_OUT = PROJECT_ROOT / "artigo/artigo_base/figures/geradas"
GENERATOR = Path(__file__).with_name("gerar_figuras.py")
FIELDS = {"rim": "mascara_rim_visual", "cortex": "mascara_cortex_visual",
          "medulla": "mascara_medulla_visual",
          "central_echo_complex": "mascara_central_echo_complex_visual"}
INNER_MANIFEST = {
    "unet": PROJECT_ROOT / "results/intrarenal_model3/intrarenal_unet_multiclass_predictions_dataset_geral/manifest.csv",
    "deeplab": PROJECT_ROOT / "results/intrarenal_model3/intrarenal_multiclass_predictions_dataset_geral_unet085/manifest.csv",
}
PRED_FIELDS = {
    "cortex": "predicted_cortex_mask_path",
    "medulla": "predicted_medulla_mask_path",
    "central_echo_complex": "predicted_central_echo_complex_mask_path",
}


def resolve(value, root=PROJECT_ROOT):
    if not value:
        return None
    raw = str(value).strip()
    path = Path(raw)
    # Old local data manifests may contain D:\kidney\dataset_aumentado\...
    win = re.match(r"^[A-Za-z]:[\\/][^\\/]+[\\/](.+)$", raw)
    if win and not path.exists():
        path = root / win.group(1).replace("\\", "/")
    elif not path.is_absolute():
        path = root / raw.replace("\\", "/")
    return path if path.is_file() else None


def read_index(file):
    if not file.is_file():
        return {}
    with file.open(encoding="utf-8-sig", newline="") as handle:
        return {row["image_id"]: row for row in csv.DictReader(handle)}


def original_image(item, manifest, image_id):
    found = resolve(item.get("imagem_visual"))
    if found:
        return found
    sources = {
        "dataset_loader__": PROJECT_ROOT / "dataset_aumentado/fontes/dataset_loader",
        "monai_curated_png__": PROJECT_ROOT / "dataset_aumentado/fontes/external_data/processed/monai_renal_png/images",
    }
    for prefix, parent in sources.items():
        if image_id.startswith(prefix):
            candidate = parent / (image_id[len(prefix):] + ".png")
            if candidate.is_file():
                return candidate
    for suffix in (".jpg", ".jpeg", ".png"):
        p = manifest.parent / "imagem" / (image_id + suffix)
        if p.is_file():
            return p
    return None


def source_mask(item, image_id, layer, model, model_rows):
    if layer != "rim":
        found = resolve(model_rows.get(image_id, {}).get(PRED_FIELDS[layer], ""))
        if item.get("origem_visual") == "dataset_geral_prediction_space":
            return found
    return resolve(item.get(FIELDS[layer], ""))


def audit_record(output, args, inputs, extra):
    record = {"figure": str(output), "image_id": args.image_id,
              "inputs": {k: str(v) for k, v in inputs.items()},
              "selection": extra, "note": "Arquivos reais; nenhuma correcao simulada."}
    audit = output.with_suffix(".proveniencia.json")
    audit.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Rastreabilidade:", audit)


def run_figure(mode, output, flags):
    command = [sys.executable, str(GENERATOR), mode, "--output", str(output)]
    for key, value in flags.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    subprocess.run(command, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("listar", "revisao", "multiclasse"))
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--database", type=Path, default=DEFAULT_DB)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--image-id", help="Identificador exato do manifesto. Opcional no modo listar.")
    p.add_argument("--reviewer", help="Revisor que salvou uma correcao real.")
    p.add_argument("--layer", choices=tuple(FIELDS), default="cortex")
    p.add_argument("--model", choices=("unet", "deeplab"), default="unet")
    args = p.parse_args()
    if not args.manifest.is_file():
        p.error(f"Manifesto nao encontrado: {args.manifest}")
    items = read_index(args.manifest)
    model_rows = read_index(INNER_MANIFEST[args.model])
    if args.action == "listar":
        if not args.database.is_file():
            print("Banco SQLite ausente: listando apenas imagens disponiveis no manifesto.")
            edits = []
        else:
            with sqlite3.connect(f"file:{args.database.resolve().as_posix()}?mode=ro", uri=True) as db:
                tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                edits = db.execute("SELECT image_id, reviewer, layer, operation, approval_status FROM mask_edits").fetchall() if "mask_edits" in tables else []
        print("Casos no manifesto:", len(items))
        print("Revisoes de mascaras registradas:", len(edits))
        print("Casos revisados elegiveis (ID | revisor | camada | operacao | aprovacao):")
        for image_id, reviewer, layer, operation, approval in edits:
            if image_id in items and original_image(items[image_id], args.manifest, image_id):
                print(" | ".join((image_id, reviewer, layer, operation, approval)))
        print("Use --image-id, --reviewer e --layer para exportar um caso documentado.")
        return
    if not args.image_id or args.image_id not in items:
        p.error("Informe --image-id presente no manifesto (consulte a opcao listar).")
    item = items[args.image_id]
    image = original_image(item, args.manifest, args.image_id)
    if not image:
        p.error("Arquivo de imagem original nao localizado.")
    inputs = {"image": image}
    if args.action == "revisao":
        if not args.reviewer:
            p.error("Revisao exige --reviewer, para vincular a alteracao a um registro real.")
        if not args.database.is_file():
            p.error("Banco SQLite nao encontrado; nao e permitido simular revisoes.")
        with sqlite3.connect(f"file:{args.database.resolve().as_posix()}?mode=ro", uri=True) as db:
            row = db.execute(
                "SELECT mask_path, operation, approval_status, updated_at FROM mask_edits "
                "WHERE image_id=? AND reviewer=? AND layer=?",
                (args.image_id, args.reviewer, args.layer)).fetchone()
        if not row:
            p.error("Nenhuma correcao real correspondente encontrada no banco.")
        automatic = source_mask(item, args.image_id, args.layer, args.model, model_rows)
        reviewed = resolve(row[0])
        if not automatic or not reviewed:
            p.error("Mascara automatica ou corrigida nao localizada; figura nao gerada.")
        inputs.update({"automatic": automatic, "reviewed": reviewed})
        output = args.output_dir / f"revisao_{args.image_id}_{args.layer}.png"
        extra = {"mode": "revisao", "reviewer": args.reviewer, "layer": args.layer,
                 "operation": row[1], "approval_status": row[2], "updated_at": row[3],
                 "model": args.model}
        run_figure("revisao", output, inputs)
    else:
        masks = {}
        for layer in FIELDS:
            candidate = source_mask(item, args.image_id, layer, args.model, model_rows)
            if candidate is None:
                p.error(f"Falta mascara da estrutura {layer}: nao sera criado painel parcial.")
            masks[{"central_echo_complex": "cec"}.get(layer, layer)] = candidate
        inputs.update(masks)
        output = args.output_dir / f"multiclasse_{args.image_id}.png"
        extra = {"mode": "multiclasse", "model": args.model,
                 "origem_visual": item.get("origem_visual", "")}
        run_figure("multiclasse", output, inputs)
    audit_record(output, args, inputs, extra)


if __name__ == "__main__":
    main()
