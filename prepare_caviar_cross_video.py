#!/usr/bin/env python3
"""
Prepara sequencias extras do CAVIAR (mesmo corredor de Lisboa do
WalkByShop1cor, outras pessoas/momentos) como conjuntos de TESTE no formato
YOLO, para avaliar os modelos sem o vazamento da divisao atual da Caviar
(treino e teste sorteados quadro a quadro do mesmo video).

Para cada pasta caviar/datasets/cross_video/<Sequencia>/ contendo o .mpg e o
XML de ground truth (CVML) baixados de
https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/ gera:

    images/test/NNNN.jpg   todos os quadros do video
    labels/test/NNNN.txt   classe 0 (pessoa), xc yc w h normalizados — so'
                           as caixas individuais (<object>), sem as de grupo
    data.yaml              test: images/test (train/val apontam para o mesmo
                           lugar so' porque o ultralytics exige as chaves)

A conversao e' a mesma das etiquetas atuais da Caviar: a caixa do XML
(xc, yc, w, h em pixels) dividida pelo tamanho do quadro (384x288).

Dados: EC Funded CAVIAR project/IST 2001 37540 (CC BY-SA).

Uso:
    venv/bin/python prepare_caviar_cross_video.py
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
CROSS_DIR = ROOT / "caviar" / "datasets" / "cross_video"


def read_boxes(xml_path: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    """{numero do quadro: [(xc, yc, w, h) em pixels]} — so' <object>, sem <group>."""
    root = ET.parse(xml_path).getroot()
    out = {}
    for frame in root.iter("frame"):
        boxes = []
        for obj in frame.findall("objectlist/object"):
            b = obj.find("box")
            if b is not None:
                boxes.append(tuple(float(b.get(k)) for k in ("xc", "yc", "w", "h")))
        out[int(frame.get("number"))] = boxes
    return out


def to_yolo(box, width: int, height: int) -> str | None:
    xc, yc, w, h = box
    x1, y1 = max(0.0, xc - w / 2), max(0.0, yc - h / 2)
    x2, y2 = min(float(width), xc + w / 2), min(float(height), yc + h / 2)
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None
    return (f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")


def prepare(seq_dir: Path) -> None:
    mpg = next(seq_dir.glob("*.mpg"))
    xml_path = next(seq_dir.glob("*.xml"))
    boxes = read_boxes(xml_path)
    img_dir, lbl_dir = seq_dir / "images" / "test", seq_dir / "labels" / "test"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(mpg))
    n = n_boxes = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        name = f"{n:04d}"
        cv2.imwrite(str(img_dir / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = [ln for ln in (to_yolo(b, w, h) for b in boxes.get(n, [])) if ln]
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_boxes += len(lines)
        n += 1
    if n != len(boxes):
        print(f"  [aviso] {seq_dir.name}: video tem {n} quadros, XML tem {len(boxes)}")

    (seq_dir / "data.yaml").write_text(
        f"# Sequencia CAVIAR {seq_dir.name} — so' teste (gerado por prepare_caviar_cross_video.py)\n"
        f"path: {seq_dir}\ntrain: images/test\nval: images/test\ntest: images/test\n"
        f"nc: 1\nnames: ['pessoa']\n")
    print(f"  {seq_dir.name:20s} {n} quadros, {n_boxes} caixas -> {seq_dir.relative_to(ROOT)}/data.yaml")


def main():
    seqs = sorted(d for d in CROSS_DIR.iterdir() if d.is_dir() and any(d.glob("*.mpg")))
    if not seqs:
        raise SystemExit(f"nenhuma sequencia em {CROSS_DIR}")
    print(f"Preparando {len(seqs)} sequencia(s):")
    for d in seqs:
        prepare(d)


if __name__ == "__main__":
    main()
