"""
Knowledge distillation (somente KD, sem poda/quantizacao) dos baselines YOLOv10
nas bases oxford e caviar. O student e sempre menor que o teacher:

    student yolov10n <- teacher yolov10s
    student yolov10m <- teacher yolov10l
    student yolov10l <- teacher yolov10x

O student parte do seu proprio baseline fine-tuned; o teacher e o baseline
fine-tuned da variante maior (yolov10s so existe em models/epochs_100).
Usa o suporte nativo distill_model=/dis= do ultralytics instalado.

Saida: models_to_deploy/<dataset>/<student>/kd/<student>_kd.pt
Re-executavel (pula o que ja existe).

Uso: venv/bin/python run_kd.py [--epochs 50] [--dis 6.0] [--only yolov10n] [--dataset caviar]
"""
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
PAIRS = [("yolov10n", "yolov10s"), ("yolov10m", "yolov10l"), ("yolov10l", "yolov10x")]
DATASETS = ["caviar", "oxford"]


def baseline(ds, variant):
    p = BASE / "models_to_deploy" / ds / variant / "baseline" / f"{variant}_finetuned.pt"
    if p.exists():
        return p
    p = BASE / "models" / "epochs_100" / ds / f"{variant}_finetuned.pt"
    if p.exists():
        return p
    raise FileNotFoundError(f"baseline nao encontrado: {ds}/{variant}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--dis", type=float, default=6.0, help="peso da loss de distilacao")
    ap.add_argument("--only", help="rodar so este student (ex: yolov10n)")
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("GPU nao encontrada.")

    for ds in ([args.dataset] if args.dataset else DATASETS):
        data_yaml = BASE / "runs" / "yaml" / f"{ds}.yaml"
        for student, teacher in PAIRS:
            if args.only and args.only != student:
                continue
            out = BASE / "models_to_deploy" / ds / student / "kd" / f"{student}_kd.pt"
            if out.exists():
                print(f"[skip] ja existe: {out}")
                continue
            s_path, t_path = baseline(ds, student), baseline(ds, teacher)
            print(f"\n[KD] {ds}: student={student} ({s_path}) <- teacher={teacher} ({t_path})")
            model = YOLO(str(s_path))
            model.train(
                data=str(data_yaml), epochs=args.epochs, imgsz=640, batch=args.batch,
                device=0, workers=8,
                project=str(BASE / "runs" / "train" / "kd"), name=f"{student}_{ds}_kd",
                exist_ok=True, amp=True, optimizer="SGD", lr0=1e-3, lrf=0.05,
                momentum=0.937, weight_decay=0.0005, warmup_epochs=3, patience=0,
                close_mosaic=10, mixup=0.2, mosaic=1.0,
                distill_model=str(t_path), dis=args.dis,
            )
            best = Path(model.trainer.save_dir) / "weights" / "best.pt"
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best, out)
            print(f"[ok] {out}")


if __name__ == "__main__":
    main()
