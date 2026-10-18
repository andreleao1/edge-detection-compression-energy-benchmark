if __name__ == "__main__":
    import shutil
    import torch
    from pathlib import Path
    from ultralytics import YOLO

    BASE_DIR   = Path("C:/workspace/mestrado/novo_teste")
    MODELS_DIR = BASE_DIR / "base_models"
    # Hiperparâmetros
    num_epochs = 100

    RUNS_DIR   = BASE_DIR / "runs" / "train"
    MODELS_OUT = BASE_DIR / "models" / f"epochs_{num_epochs}"
    DATASETS   = {
        "oxford": str(BASE_DIR / "oxford" / "data.yaml"),
        "caviar":  str(BASE_DIR / "caviar"  / "data.yaml"),
    }
    imgsz      = 640
    lr0        = 1e-3
    lrf        = 0.05
    patience   = 30

    if not torch.cuda.is_available():
        raise RuntimeError("GPU não encontrada. Instale o PyTorch com suporte a CUDA antes de continuar.")

    # Ordena do menor para o maior para detectar problemas cedo
    SIZE_ORDER = ["yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x"]
    all_files  = {p.stem: p for p in MODELS_DIR.glob("*.pt")}
    model_files = [all_files[k] for k in SIZE_ORDER if k in all_files] + \
                  [v for k, v in all_files.items() if k not in SIZE_ORDER]

    if not model_files:
        raise FileNotFoundError(f"Nenhum modelo .pt encontrado em {MODELS_DIR}")

    total = len(model_files) * len(DATASETS)
    count = 0

    for model_path in model_files:
        model_name = model_path.stem

        for dataset_name, data_yaml in DATASETS.items():
            count += 1
            run_name = f"{model_name}_{dataset_name}_finetune"
            print(f"\n[{count}/{total}] {model_name} × {dataset_name} → {run_name}")

            dst_dir = MODELS_OUT / dataset_name
            dst     = dst_dir / f"{model_name}_finetuned.pt"
            if dst.exists():
                print(f"⏭️  Já treinado, pulando.")
                continue

            model = YOLO(str(model_path))

            model.train(
                data=data_yaml,
                epochs=num_epochs,
                imgsz=imgsz,
                batch=-1,               # detecta automaticamente o maior batch que cabe na VRAM
                device=0,
                workers=8,
                project=str(RUNS_DIR),  # caminho absoluto — evita resolver relativo ao cwd errado
                name=run_name,
                exist_ok=True,          # reutiliza pasta existente sem adicionar sufixo numérico
                amp=True,               # mixed precision (FP16) — ~2× mais rápido na GPU
                cache="ram",            # carrega dataset na RAM — elimina gargalo de disco
                optimizer="SGD",
                lr0=lr0,
                lrf=lrf,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=5,
                patience=patience,
                close_mosaic=10,        # desliga mosaic nas últimas 10 épocas (estabiliza)
                hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                mixup=0.2, mosaic=1.0,
            )

            # Copia best.pt para o diretório do dataset usando o caminho real do trainer
            src = Path(model.trainer.save_dir) / "weights" / "best.pt"
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"✅ Concluído: {run_name} → salvo em {dst}")

    print(f"\n✅ Fine-tuning finalizado. {total} experimentos treinados.")
