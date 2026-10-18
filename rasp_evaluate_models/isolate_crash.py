"""
Script de isolamento — roda um modelo .onnx imagem por imagem contra um
dataset, logando o nome do arquivo ANTES de cada inferencia. Use isso
quando main.py travar com "Segmentation fault" sem indicar qual imagem
causou o problema (segfault mata o processo antes do log da imagem seguinte
aparecer — mas como aqui logamos antes de cada session.run(), a ULTIMA
linha impressa antes do crash e a imagem culpada).

Uso:
    python isolate_crash.py <model.onnx> <dataset_dir>

Exemplo:
    python isolate_crash.py \\
        runs/retinanet/quantized/caviar/best_quantized.onnx \\
        caviar/datasets/images/test
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def get_input_spec(session):
    inp = session.get_inputs()[0]
    shape = inp.shape
    has_batch = len(shape) == 4
    h_dim, w_dim = shape[-2], shape[-1]
    if isinstance(h_dim, int) and isinstance(w_dim, int):
        imgsz, square = h_dim, True
    else:
        imgsz, square = 800, False
    return inp.name, has_batch, imgsz, square


def preprocess(img_bgr, imgsz, square):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if square:
        img = cv2.resize(img, (imgsz, imgsz))
    else:
        scale = imgsz / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    arr = img.astype(np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray(arr), (h, w)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    model_path, dataset_dir = sys.argv[1], sys.argv[2]

    print(f"Carregando modelo: {model_path}")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name, has_batch, imgsz, square = get_input_spec(session)
    print(f"input='{input_name}' imgsz={imgsz} square={square} batched={has_batch}")

    image_paths = sorted(
        p for p in Path(dataset_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"{len(image_paths)} imagens em {dataset_dir}\n")

    for i, img_path in enumerate(image_paths, 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[{i}/{len(image_paths)}] AVISO — cv2.imread falhou (None): {img_path.name}")
            continue

        arr, (h, w) = preprocess(img_bgr, imgsz, square)
        if has_batch:
            arr = arr[None, ...]

        # Flush imediato — se travar dentro do session.run(), essa linha
        # ja estara no terminal/log antes do crash.
        print(f"[{i}/{len(image_paths)}] {img_path.name}  (original {w}x{h}, tensor {arr.shape})", flush=True)

        session.run(None, {input_name: arr})

    print("\nConcluido sem crash — todas as imagens processadas com sucesso.")


if __name__ == "__main__":
    main()
