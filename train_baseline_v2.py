#!/usr/bin/env python3
"""
Treina os baselines YOLOv10 com a receita v2, a mesma usada no ajuste fino da
poda estrutural (prune_models.py --yolo_nbs ...), para que baseline e modelos
comprimidos tenham o mesmo treino.

Receita v2 (RECIPE abaixo):
  - 300 epocas, SGD lr0=0.01 com decaimento cosseno, 3 epocas de aquecimento;
  - lote 8 SEM acumulo de gradiente (nbs = lote): na Oxford (42 imagens de
    treino) o acumulo padrao (nbs=64) dava ~1 passo do otimizador por epoca;
  - aumentacao: mosaico, mixup 0.2, HSV, espelhamento, escala e translacao
    (mesmas do treino original), mosaico desligado nas ultimas 10 epocas.

Controle de overfitting:
  - parada antecipada (patience=50): para se a validacao nao melhora em 50 epocas;
  - o checkpoint salvo e' o melhor na validacao (best.pt), nunca o de teste;
  - o teste nao e' usado no treino nem na escolha do checkpoint;
  - as curvas (results.csv) ficam salvas para conferir treino x validacao.

Entrada: pesos COCO em base_models/<arch>.pt.
Saida:   models/recipe_v2/<dataset>/<arch>_finetuned.pt

Uso:
    venv/bin/python train_baseline_v2.py --dataset oxford_v2 --arch yolov10n
    venv/bin/python train_baseline_v2.py --dataset oxford --arch yolov10n   # base original
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RECIPE = dict(
    epochs=300, batch=8, nbs=8, imgsz=640,
    optimizer="SGD", lr0=0.01, lrf=0.01, cos_lr=True, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=3, patience=50,
    mosaic=1.0, mixup=0.2, close_mosaic=10, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    amp=True, workers=8, seed=0, deterministic=True,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="nome do yaml em runs/yaml (ex. oxford, oxford_v2, caviar)")
    ap.add_argument("--arch", action="append", required=True, help="ex. yolov10n (repetivel)")
    ap.add_argument("--epochs", type=int, default=RECIPE["epochs"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    data = ROOT / "runs" / "yaml" / f"{args.dataset}.yaml"
    for arch in args.arch:
        out = ROOT / "models" / "recipe_v2" / args.dataset / f"{arch}_finetuned.pt"
        if out.exists() and not args.overwrite:
            print(f"[skip] {out.relative_to(ROOT)} ja existe")
            continue
        model = YOLO(str(ROOT / "base_models" / f"{arch}.pt"))
        model.train(data=str(data), device=0, project=str(ROOT / "runs" / "recipe_v2"),
                    name=f"{arch}_{args.dataset}", exist_ok=True, **{**RECIPE, "epochs": args.epochs})
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, out)
        print(f"[ok] {arch} {args.dataset} -> {out.relative_to(ROOT)} (curvas: {model.trainer.save_dir}/results.csv)")


if __name__ == "__main__":
    main()
