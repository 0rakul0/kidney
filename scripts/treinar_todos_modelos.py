"""Executa os treinadores existentes sem misturar fontes ou conjuntos de teste.

Simulacao:
 python scripts/treinar_todos_modelos.py --dry-run
 Treinamento supervisionado:
 python scripts/treinar_todos_modelos.py --execute
 Apenas intrarrenais:
 python scripts/treinar_todos_modelos.py --execute --tasks inner_unet inner_deeplab
 Tambem cross-validation historico (execucao separada):
 python scripts/treinar_todos_modelos.py --execute --tasks dataset_geral_cv

Os modelos rodam SEQUENCIALMENTE para evitar concorrencia de VRAM.
Nao reconstroi splits, nao inclui mascaras externas sem curadoria no treino
supervisionado e nao sobrescreve checkpoints ja existentes por padrao.
"""
import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "src/segmentation/experiments"
DATA = ROOT / "dataset_aumentado/dataset_intrarrenal/supervisionado"
MODELS = ROOT / "models"
TASKS = {
    "capsule_unet": ("train_unet.py", "capsule_annotator_1", "capsule"),
    "capsule_unetplusplus": ("train_unetplusplus.py", "capsule_annotator_1", "capsule"),
    "capsule_deeplab": ("train_deeplab.py", "capsule_annotator_1", "capsule"),
    "capsule_segformer": ("train_segformer.py", "capsule_annotator_1", "capsule"),
    "inner_unet": ("train_inner_unet.py", "regions_multiclass_annotator_1", "intrarenal"),
    "inner_deeplab": ("train_inner_deeplab.py", "regions_multiclass_annotator_1", "intrarenal"),
    "medulla_unet": ("train_medulla_roi_unet.py", "medulla_annotator_1", "medulla"),
    "dataset_geral_cv": ("run_dataset_geral_cv.py", "", "cv"),
}
DEFAULT_TASKS = tuple(k for k in TASKS if k != "dataset_geral_cv")


def check_split(root):
    sizes = {}
    for split in ("train", "val", "test"):
        img = root / split / "image"
        mask = root / split / "mask"
        if not img.is_dir() or not mask.is_dir():
            raise FileNotFoundError(f"Dados supervisionados nao encontrados: {img}, {mask}")
        images = {p.name for p in img.glob("*.png")}
        masks = {p.name for p in mask.glob("*.png")}
        if not images or images != masks:
            raise ValueError(f"Inconsistencia de imagens/mascaras em {split}: "
                             f"images={len(images)} masks={len(masks)}")
        sizes[split] = len(images)
    return sizes


def main():
    p = argparse.ArgumentParser(description="Orquestra treino reprodutivel de arquiteturas renais.")
    p.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=DEFAULT_TASKS)
    p.add_argument("--execute", action="store_true", help="Efetivamente executar (padrao: dry-run).")
    p.add_argument("--dry-run", action="store_true", help="Somente exibir e registrar comandos.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4, help="Conservador para RTX 3050; reduza se faltar VRAM.")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output", type=Path, default=ROOT / "results/treinamento_completo")
    p.add_argument("--checkpoint-prefix", default="retrain_2026", help="Prefixo de checkpoints novos.")
    p.add_argument("--overwrite", action="store_true", help="Autoriza substituir checkpoints de destino.")
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.img_size < 32:
        p.error("Parametros numericos invalidos.")
    if not args.checkpoint_prefix or "/" in args.checkpoint_prefix or "\\" in args.checkpoint_prefix:
        p.error("Prefixo de checkpoint invalido.")
    execute = args.execute and not args.dry_run
    args.output.mkdir(parents=True, exist_ok=True)
    plan = []
    for task in args.tasks:
        script, name, kind = TASKS[task]
        cmd = [sys.executable, str(EXPERIMENTS / script)]
        if kind == "cv":
            split_root = ROOT / "dataset_aumentado/dataset_geral_cv/folds"
            if not split_root.is_dir():
                raise FileNotFoundError(f"CV splits ausentes: {split_root}")
            cmd += ["--splits-root", str(split_root), "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size), "--img-size", str(args.img_size),
                    "--lr", str(args.lr), "--seed", str(args.seed),
                    "--num-workers", str(args.num_workers)]
            # Este trainer possui naming e reuso proprios; nunca executar por padrao.
            if not execute:
                cmd.append("--dry-run")
            checkpoint = None
            sizes = {}
        else:
            dataset = DATA / name
            sizes = check_split(dataset)
            checkpoint = MODELS / f"{args.checkpoint_prefix}_{task}.pth"
            if kind == "capsule":
                cmd += ["--dataset-path", str(dataset),
                        "--checkpoint-name", checkpoint.name,
                        "--experiment-name", f"{args.checkpoint_prefix}_{task}",
                        "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size),
                        "--img-size", str(args.img_size),
                        "--lr", str(args.lr), "--seed", str(args.seed),
                        "--num-workers", str(args.num_workers)]
            else:
                cmd += ["--dataset-root", str(dataset), "--checkpoint", str(checkpoint),
                        "--experiment-name", f"{args.checkpoint_prefix}_{task}",
                        "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size),
                        "--img-size", str(args.img_size),
                        "--lr", str(args.lr), "--seed", str(args.seed),
                        "--num-workers", str(args.num_workers)]
        plan.append({"task": task, "command": cmd, "checkpoint": str(checkpoint) if checkpoint else None,
                     "sizes": sizes})
    if execute:
        existing = [x["checkpoint"] for x in plan if x["checkpoint"] and Path(x["checkpoint"]).exists()]
        if existing and not args.overwrite:
            raise FileExistsError("Checkpoints ja existem; use outro --checkpoint-prefix ou --overwrite: " + ", ".join(existing))
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "execute": execute,
                "parameters": vars(args) | {"output": str(args.output)}, "plan": plan}
    (args.output / "plano.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for item in plan:
        print(f"\n[{item['task']}] amostras={item['sizes']}")
        print(shlex.join(item["command"]))
    if not execute:
        print("\nModo simulacao: use --execute para iniciar o treino sequencial.")
        return
    outcomes = []
    for item in plan:
        task = item["task"]
        log = args.output / f"{task}.log"
        start = time.time()
        print(f"\nIniciando: {task} (log: {log})", flush=True)
        with log.open("w", encoding="utf-8") as file:
            result = subprocess.run(item["command"], cwd=ROOT, stdout=file, stderr=subprocess.STDOUT)
        outcome = {"task": task, "returncode": result.returncode,
                   "elapsed_seconds": round(time.time() - start, 2), "log": str(log)}
        outcomes.append(outcome)
        (args.output / "execucao.json").write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
        print(f"Finalizado: {task} | codigo={result.returncode}", flush=True)
        if result.returncode and not args.continue_on_error:
            raise RuntimeError(f"Treino falhou: {task}. Consulte {log}")
    if any(x["returncode"] != 0 for x in outcomes):
        raise SystemExit("Uma ou mais tarefas falharam; confira execucao.json.")


if __name__ == "__main__":
    main()
