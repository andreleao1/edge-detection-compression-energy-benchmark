#!/usr/bin/env python3
"""
Base Oxford Town Centre v2: mais quadros reais anotados no treino e na
validacao, com o MESMO conjunto de teste.

A base original (oxford/datasets) usa 91 dos 4501 quadros anotados do video
(um a cada 50): 42 de treino, 7 de validacao e 42 de teste, intercalados no
tempo. O annotations.txt tem a anotacao de todos os 4501 quadros.

Regra (sem vazamento):
  - teste: exatamente os 42 quadros originais (nenhum quadro novo);
  - cada quadro candidato (a cada --step quadros) recebe o conjunto do quadro
    original mais proximo (treino ou validacao); quadros cujo vizinho
    original mais proximo e' de teste sao descartados;
  - o candidato so' entra se estiver a pelo menos --gap quadros de qualquer
    quadro de OUTRO conjunto (original ou novo), para que treino, validacao
    e teste nao tenham imagens quase identicas.

Rotulos: caixa do corpo (colunas bodyLeft..bodyBottom do annotations.txt),
recortada na borda da imagem e normalizada (classe 0). O script confere que
os rotulos gerados para os 91 quadros originais batem com os existentes.

Saida: oxford/datasets_v2/{images,labels}/{train,val,test} e
       runs/yaml/oxford_v2.yaml

Uso:
    venv/bin/python prepare_oxford_v2.py --step 10 --gap 50
"""
import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "oxford" / "datasets"
DST = ROOT / "oxford" / "datasets_v2"
W, H = 1920, 1080


def load_boxes() -> dict[int, list[tuple]]:
    boxes = defaultdict(list)
    for line in (SRC / "annotations.txt").read_text().splitlines():
        v = line.split(",")
        if len(v) < 12:
            continue
        frame = int(v[1])
        l, t, r, b = (float(x) for x in v[8:12])
        l, r = max(0.0, l), min(float(W), r)
        t, b = max(0.0, t), min(float(H), b)
        if r <= l or b <= t:
            continue
        boxes[frame].append(((l + r) / 2 / W, (t + b) / 2 / H, (r - l) / W, (b - t) / H))
    return boxes


def label_text(bs) -> str:
    return "".join(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for x, y, w, h in bs)


def check_original(boxes) -> None:
    """Os rotulos gerados devem reproduzir os rotulos originais."""
    bad = 0
    for split in ("train", "val", "test"):
        for f in (SRC / "labels" / split).glob("*.txt"):
            orig = sorted(tuple(round(float(x), 4) for x in l.split()[1:]) for l in f.read_text().splitlines() if l.strip())
            gen = sorted(tuple(round(x, 4) for x in b) for b in boxes.get(int(f.stem), []))
            if len(orig) != len(gen) or any(max(abs(a - b) for a, b in zip(o, g)) > 2e-4 for o, g in zip(orig, gen)):
                bad += 1
                if bad <= 3:
                    print(f"  diferente: {split}/{f.name} (original {len(orig)} caixas, gerado {len(gen)})")
    if bad:
        raise SystemExit(f"{bad} rotulo(s) nao batem com os originais — conversao errada, abortando.")
    print("rotulos conferidos: os 91 quadros originais batem com a conversao.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=int, default=10, help="intervalo entre quadros candidatos")
    ap.add_argument("--gap", type=int, default=50, help="distancia minima (quadros) entre conjuntos diferentes")
    ap.add_argument("--video", type=Path, default=SRC / "oxfordtower.mp4")
    args = ap.parse_args()

    boxes = load_boxes()
    check_original(boxes)

    orig = {s: sorted(int(p.stem) for p in (SRC / "images" / s).glob("*.jpg")) for s in ("train", "val", "test")}
    owner = {f: s for s, fs in orig.items() for f in fs}
    last = max(boxes)
    split = dict(owner)
    for f in range(0, last + 1, args.step):
        if f in split:
            continue
        near = min(owner, key=lambda o: abs(o - f))
        if owner[near] == "test":
            continue
        split[f] = owner[near]
    # descarta candidatos novos perto de outro conjunto (iterativo: nunca remove originais)
    changed = True
    while changed:
        changed = False
        by = defaultdict(list)
        for f, s in split.items():
            by[s].append(f)
        for f, s in list(split.items()):
            if f in owner:
                continue
            if any(abs(f - g) < args.gap for s2, fs in by.items() if s2 != s for g in fs):
                del split[f]
                changed = True

    counts = defaultdict(int)
    for s in split.values():
        counts[s] += 1
    print(f"quadros: treino {counts['train']} (era {len(orig['train'])}), validacao {counts['val']} "
          f"(era {len(orig['val'])}), teste {counts['test']} (igual)")

    if DST.exists():
        shutil.rmtree(DST)
    for s in ("train", "val", "test"):
        (DST / "images" / s).mkdir(parents=True)
        (DST / "labels" / s).mkdir(parents=True)

    # imagens originais: copia as mesmas (mesmo JPEG) para nao mudar o teste
    for f, s in owner.items():
        shutil.copy2(SRC / "images" / s / f"{f:06d}.jpg", DST / "images" / s / f"{f:06d}.jpg")
        shutil.copy2(SRC / "labels" / s / f"{f:06d}.txt", DST / "labels" / s / f"{f:06d}.txt")

    # quadros novos: extraidos do video, mesmo indice do annotations.txt
    new = {f: s for f, s in split.items() if f not in owner}
    cap = cv2.VideoCapture(str(args.video))
    idx, done = 0, 0
    # confere o alinhamento quadro a quadro com uma imagem original
    while done < len(new) and idx <= max(new):
        ok, img = cap.read()
        if not ok:
            raise SystemExit(f"video terminou no quadro {idx}")
        if idx == 0:
            ref = cv2.imread(str(SRC / "images" / owner[0] / "000000.jpg"))
            diff = float(abs(ref.astype(int) - img.astype(int)).mean())
            print(f"alinhamento: diferenca media quadro 0 do video x 000000.jpg = {diff:.2f} (0-255)")
            if diff > 8:
                raise SystemExit("quadro 0 do video nao bate com 000000.jpg — indices desalinhados")
        if idx in new:
            s = new[idx]
            cv2.imwrite(str(DST / "images" / s / f"{idx:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (DST / "labels" / s / f"{idx:06d}.txt").write_text(label_text(boxes[idx]))
            done += 1
        idx += 1
    cap.release()

    yaml = ROOT / "runs" / "yaml" / "oxford_v2.yaml"
    yaml.parent.mkdir(parents=True, exist_ok=True)
    yaml.write_text(f"train: {DST}/images/train\nval: {DST}/images/val\ntest: {DST}/images/test\n"
                    f"nc: 1\nnames: ['pessoa']\n")
    print(f"{done} quadros novos extraidos -> {DST.relative_to(ROOT)}  |  yaml: {yaml.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
