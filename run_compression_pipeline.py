"""
Orquestrador unico da pipeline de compressao + avaliacao + preenchimento das
planilhas, para YOLOv10n/m/l/x, FasterRCNN, RetinaNet e MobileNet, nas bases
oxford e caviar.

Reaproveita (como funcoes importadas, sem duplicar logica de dataloader/
metrica) prune_models.py (poda estrutural/nao-estrutural + KD),
quantize_models.py (quantizacao INT8 via ONNX) e evaluate_models.py
(avaliacao: mAP/Precision/Recall/F1/Inferencia/FPS/FLOPS) e accuracy_db.py
(persistencia + resume — mesma tabela Postgres que evaluate_models.py ja usa,
so que com rotulos de modelo novos e proprios desta pipeline).

Ver o plano completo em:
    C:\\Users\\andre\\.claude\\plans\\delegated-seeking-nebula.md

--------------------------------------------------------------------
USO
--------------------------------------------------------------------
  python run_compression_pipeline.py
      Roda a pipeline inteira (todas as etapas, arquiteturas, datasets e
      percentuais). Dado o volume de jobs (ver aviso de escala no plano),
      espera-se rodar isso em multiplas sessoes — o script pula tudo que ja
      foi gerado/avaliado (resume automatico via arquivo em disco + Postgres).

  Execucao parcial (para rodar por partes / smoke test):
    python run_compression_pipeline.py --stage prune_structured \\
        --dataset caviar --arch mobilenet --pct 20 --ft_epochs 1

  --stage {all, baseline, prune_structured, prune_unstructured, quantize, kd}
      baseline = so' avalia os modelos sem compressao e copia para
      models_to_deploy/ (nenhum treino; os baselines ja sao os fine-tunings
      de models/epochs_100/<dataset>/)
  --dataset {oxford, caviar}          (repetivel; default: os dois)
  --arch <nome interno>               (repetivel; default: todas as 7)
  --pct {20,40,60,80}                 (repetivel; default: as 4)
  --ft_epochs N                       (default: 10)
  --n_calib N                         (imagens de calibracao da quantizacao; default: 100)
  --kd_weight_yolo F                  (default: 0.1 — mesmo valor de kd_test/kd.py)
  --kd_weight_torchvision F           (default: 1.0 — nao tunado, ver plano)
  --overwrite                         (reprocessa mesmo se checkpoint/avaliacao ja existir)

--------------------------------------------------------------------
SAIDAS
--------------------------------------------------------------------
  - Experimentos Metrado Andre.xlsx        — planilha oficial (editada in-place,
    com backup automatico), recebe baseline + Pruning Estruturado + Quantizacao
    + Pruning Estruturado+Quantizacao (colunas Desktop/GPU apenas).
  - Experimentos Metrado Andre - comparacoes.xlsx — planilha nova, recriada a
    cada execucao a partir do Postgres: poda NAO-estruturada (comparacao) e a
    combinacao poda estruturada + KD (+ quantizacao).
  - compressed_models/<dataset>/<arch>/<label>/...  — copia plana de cada
    checkpoint final, pronta para colar em rasp_evaluate_models/settings.yaml.
"""

import argparse
import logging
import re
import shutil
import unicodedata
from copy import copy as copy_style
from datetime import datetime
from pathlib import Path

import openpyxl
import pandas as pd
import torch

import accuracy_db
from prune_models import (
    BASE_DIR, NUM_CLASSES, DEFAULTS,
    load_torchvision_model, run_yolo_pipeline, run_torchvision_pipeline,
    setup_file_logging,
)
from quantize_models import run_yolo_quant_pipeline, run_torchvision_quant_pipeline
from evaluate_models import evaluate_yolo, evaluate_torchvision, evaluate_torchvision_onnx, HAS_THOP

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuracoes
# ---------------------------------------------------------------------------

RUNS_DIR      = BASE_DIR / "runs"
MODELS_DIR    = BASE_DIR / "models" / "epochs_100"
COMPRESSED_DIR = BASE_DIR / "compressed_models"
DEPLOY_DIR     = BASE_DIR / "models_to_deploy"   # tudo que sobe pra Raspberry (baseline + comprimidos)

OFFICIAL_XLSX    = BASE_DIR / "Experimentos Metrado Andre.xlsx"
COMPARISON_XLSX  = BASE_DIR / "Experimentos Metrado Andre - comparacoes.xlsx"

DATASETS = {
    "oxford": BASE_DIR / "oxford",
    "caviar": BASE_DIR / "caviar",
}
DATASET_DB_NAME = {"oxford": "OXFORD_TOWER", "caviar": "CAVIAR"}
SHEET_NAMES     = {"oxford": "Base Oxford Town Centre", "caviar": "Base Caviar"}

# Apenas as variantes/arquiteturas que aparecem na planilha (sem yolov10s/b).
YOLO_VARIANTS      = ["yolov10n", "yolov10m", "yolov10l", "yolov10x"]
TORCHVISION_TYPES  = ["faster_rcnn", "retinanet", "mobilenet"]
ALL_ARCHS          = YOLO_VARIANTS + TORCHVISION_TYPES
PCTS               = [20, 40, 60, 80]

SPREADSHEET_LABEL = {
    "yolov10n": "YOLO10n", "yolov10m": "YOLO10m", "yolov10l": "YOLO10l", "yolov10x": "YOLO10x",
    "faster_rcnn": "FasterRCNN", "retinanet": "RetinaNet", "mobilenet": "MobileNet",
}

# RetinaNet quantizado trava o ONNX Runtime com segfault (ver quantize_models.py)
# — derrubaria o processo inteiro da pipeline, entao nunca quantizar/avaliar
# essa combinacao. YOLO quantizado zerava mAP/Precision/Recall com a cabeca
# quantizada; agora a cabeca fica em float32 (quantize_onnx_static,
# exclude_yolo_head) e o YOLO volta a ser quantizado — CONFERIR os resultados
# na primeira execucao (ainda nao validado).
SKIP_QUANTIZED = {"retinanet"}

# Colunas da planilha oficial que este script escreve (Desktop/GPU apenas).
COL = {
    "Params (M)": 3, "mAP50": 5, "mAP50-95": 6, "Precision": 7, "Recall": 8,
    "F1": 9, "Inferencia (ms)": 10, "FPS": 11, "FLOPS (G)": 12,
}


def _arch_kind(arch: str) -> str:
    return "yolo" if arch in YOLO_VARIANTS else "torchvision"


def _baseline_path(dataset: str, arch: str) -> Path:
    if _arch_kind(arch) == "yolo":
        return MODELS_DIR / dataset / f"{arch}_finetuned.pt"
    return MODELS_DIR / dataset / f"{arch}_finetuned.pth"


def _nome_modelo(arch: str, category: str, pct: int | None = None) -> str:
    return f"{arch}_{category}" + (f"_p{pct}" if pct is not None else "")


def _copy_to_compressed(dataset: str, arch: str, label: str, src: Path) -> Path:
    """Copia o checkpoint para compressed_models/ e models_to_deploy/ (mesma
    estrutura <dataset>/<arch>/<label>/<arquivo>). Retorna o destino em
    compressed_models/."""
    first = None
    for root in (COMPRESSED_DIR, DEPLOY_DIR):
        dst_dir = root / dataset / arch / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / Path(src).name
        if not dst.exists():
            shutil.copy2(src, dst)
        first = first or dst
    return first


# ---------------------------------------------------------------------------
# Avaliacao (wrappers finos sobre evaluate_models.py)
# ---------------------------------------------------------------------------

def _evaluate_checkpoint(arch: str, dataset: str, checkpoint: Path, label: str,
                          compressao: str, pruning_rate: float | None,
                          device: torch.device, device_int: int,
                          template_path: Path | None = None) -> dict:
    if _arch_kind(arch) == "yolo":
        data_yaml = str(DATASETS[dataset] / "data.yaml")
        return evaluate_yolo(checkpoint, data_yaml, label, device_int,
                              compressao=compressao, pruning_rate=pruning_rate,
                              template_path=template_path)

    cfg = DEFAULTS[arch]
    if checkpoint.suffix.lower() == ".onnx":
        return evaluate_torchvision_onnx(checkpoint, label, DATASETS[dataset],
                                          cfg["imgsz"], cfg["square"],
                                          compressao=compressao, pruning_rate=pruning_rate)

    model = load_torchvision_model(arch, checkpoint, NUM_CLASSES, device, cfg["imgsz"]).eval()
    try:
        return evaluate_torchvision(model, label, DATASETS[dataset], device,
                                     cfg["imgsz"], cfg["square"], checkpoint,
                                     compressao=compressao, pruning_rate=pruning_rate)
    finally:
        del model
        torch.cuda.empty_cache()


OVERWRITE = False     # setado em main() a partir de --overwrite
DB_AVAILABLE = False  # setado em main() apos accuracy_db.init_db(); ver _try_init_db

# Cache local (memoria do processo), usado como fallback quando o Postgres
# esta indisponivel — garante que rebuild_comparison_workbook() ainda reflita
# os resultados computados NESTA execucao mesmo sem banco (sem banco, apenas
# nao ha' persistencia entre execucoes diferentes do script).
_LOCAL_CACHE: dict[tuple[str, str], dict] = {}


def _get_cached_result(nome_modelo: str, dataset_db_name: str) -> dict | None:
    if DB_AVAILABLE:
        return accuracy_db.get_result(nome_modelo, dataset_db_name)
    return _LOCAL_CACHE.get((nome_modelo, dataset_db_name))


def _save_cached_result(nome_modelo: str, dataset_db_name: str, row: dict):
    _LOCAL_CACHE[(nome_modelo, dataset_db_name)] = row
    if DB_AVAILABLE:
        try:
            accuracy_db.save_accuracy_result(row, dataset_db_name)
        except Exception:
            logger.error(f"    ERRO gravando no banco: {nome_modelo}", exc_info=True)


def _try_init_db():
    """accuracy_db (Postgres) e' usado so' como cache de resume/persistencia
    entre execucoes — nao e' necessario para gerar checkpoints nem para
    escrever nas planilhas. Se o Postgres nao estiver de pe (ex.: so' sobe
    junto com a stack da Raspberry, arduino/docker-compose), a pipeline
    continua normalmente, so' sem resume entre processos (o skip-existing por
    arquivo em disco continua funcionando do mesmo jeito)."""
    global DB_AVAILABLE
    try:
        accuracy_db.init_db()
        DB_AVAILABLE = True
    except Exception as e:
        DB_AVAILABLE = False
        logger.warning(
            f"  Postgres indisponivel ({e.__class__.__name__}: {e}) — "
            "seguindo sem cache/persistencia de avaliacao (resume por "
            "checkpoint em disco continua ativo; a planilha de comparacao "
            "so' refletira' os resultados desta execucao)."
        )


def _inherit_from_source(row: dict, source_nome: str | None, dataset_db_name: str) -> bool:
    """Modelos quantizados (.onnx) nao tem Params/FLOPS medidos (nao ha
    nn.Module) — herda os do modelo de origem (quantizar nao muda a contagem de
    parametros nem de FLOPs). Retorna True se preencheu algum campo."""
    if not source_nome:
        return False
    src = _get_cached_result(source_nome, dataset_db_name)
    if not src:
        return False
    filled = False
    for key in ("Params (M)", "FLOPS (G)"):
        if row.get(key) is None and src.get(key) is not None:
            row[key] = src[key]
            filled = True
    return filled


def _evaluate_with_resume(nome_modelo: str, dataset: str, inherit_from: str | None = None,
                           **eval_kwargs) -> dict | None:
    """Reaproveita accuracy_db (mesma tabela que evaluate_models.py ja usa)
    como cache de resume: se este nome_modelo+dataset ja foi avaliado antes
    (nesta ou em execucoes anteriores), nao reavalia. Se o Postgres estiver
    indisponivel (DB_AVAILABLE=False), avalia sempre e so' nao persiste.
    inherit_from: nome_modelo de origem de quem herdar Params/FLOPS (quantizados)."""
    dataset_db_name = DATASET_DB_NAME[dataset]

    # Com --overwrite o checkpoint foi regerado, entao o resultado em cache
    # (de um checkpoint anterior) esta' defasado — reavalia. O baseline nao e'
    # regerado pela pipeline, entao continua valendo o cache.
    refresh = OVERWRITE and not nome_modelo.endswith("_baseline")
    existing = None if refresh else _get_cached_result(nome_modelo, dataset_db_name)
    if existing is not None:
        logger.info(f"    Ja avaliado — pulando (resume): {nome_modelo}")
        if _inherit_from_source(existing, inherit_from, dataset_db_name):
            _save_cached_result(nome_modelo, dataset_db_name, existing)
        return existing

    try:
        row = _evaluate_checkpoint(label=nome_modelo, dataset=dataset, **eval_kwargs)
    except Exception:
        logger.error(f"    ERRO avaliando {nome_modelo}", exc_info=True)
        return None
    _inherit_from_source(row, inherit_from, dataset_db_name)

    _save_cached_result(nome_modelo, dataset_db_name, row)
    return row


# ---------------------------------------------------------------------------
# Planilha oficial — leitura/escrita in-place preservando formulas/formatacao
# ---------------------------------------------------------------------------

def _norm(s) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).strip().lower()


def _extract_pct(text: str) -> int | None:
    m = re.search(r"(\d+)\s*%", text)
    return int(m.group(1)) if m else None


def _classify_compressao(text) -> tuple[str, int | None]:
    t = _norm(text)
    if "baseline" in t:
        return ("baseline", None)
    if re.search(r"\bkd\b", t):
        return ("kd_quantizacao" if "quantiza" in t else "pruning_kd", _extract_pct(t))
    if "pruning estrutur" in t:
        return ("pruning_estruturado", _extract_pct(t))
    if "energy aware" in t:
        return ("energy_aware", None)
    if "pruning" in t and "quantiza" in t:
        return ("pruning_quantizacao", _extract_pct(t))
    if "quantiza" in t:
        return ("quantizacao", None)
    return ("unknown", None)


def _build_row_index(ws) -> dict:
    """(modelo normalizado, kind, pct) -> numero da linha."""
    idx = {}
    for row in range(3, ws.max_row + 1):
        modelo = ws.cell(row=row, column=1).value
        compressao = ws.cell(row=row, column=2).value
        if not modelo:
            continue
        kind, pct = _classify_compressao(compressao)
        idx[(_norm(modelo), kind, pct)] = row
    return idx


def _copy_row_style(ws, src_row: int, dst_row: int, max_col: int = 18):
    for col in range(1, max_col + 1):
        src = ws.cell(row=src_row, column=col)
        dst = ws.cell(row=dst_row, column=col)
        if src.has_style:
            dst._style = copy_style(src._style)


def _expand_pruning_quantizacao_block(ws, row_index: dict) -> dict:
    """Migracao idempotente: a planilha original tem 1 linha por arquitetura
    em 'Pruning + Quantizacao' (sem percentual); o usuario pediu que essa
    secao repita os mesmos 4 percentuais do pruning. Expande cada linha
    legada em 4 (20/40/60/80%), copiando estilo (as formulas de D/Q sao
    recriadas depois, por _normalize_compression_formulas — insert_rows do
    openpyxl NAO ajusta referencias de formula nas linhas deslocadas). Se ja
    migrado (nenhuma linha 'pruning_quantizacao' com pct=None sobrando), nao
    faz nada."""
    legacy_rows = sorted(
        row for (_, kind, pct), row in row_index.items()
        if kind == "pruning_quantizacao" and pct is None
    )
    if not legacy_rows:
        return row_index  # ja migrado (ou planilha vazia nessa secao)

    logger.info(f"  Expandindo bloco 'Pruning + Quantizacao' ({len(legacy_rows)} linhas -> "
                f"{len(legacy_rows) * len(PCTS)})...")

    # Processa em ordem decrescente de linha para nao invalidar indices ja
    # calculados dos itens ainda nao processados.
    for legacy_row in sorted(legacy_rows, reverse=True):
        modelo = ws.cell(row=legacy_row, column=1).value

        ws.insert_rows(legacy_row + 1, amount=len(PCTS) - 1)
        for offset, pct in enumerate(PCTS):
            row = legacy_row + offset
            if offset > 0:
                _copy_row_style(ws, legacy_row, row)
            ws.cell(row=row, column=1, value=modelo)
            ws.cell(row=row, column=2, value=f"Pruning + Quantização ({pct}%)")

    return _build_row_index(ws)  # recalcula do zero apos as insercoes


def _ensure_kd_blocks(ws, row_index: dict) -> dict:
    """Acrescenta ao fim da aba os blocos 'Pruning + KD (pct%)' e
    'Pruning + KD + Quantizacao (pct%)' (1 linha por arquitetura x percentual),
    copiando o estilo de uma linha existente. Idempotente: so' cria as linhas
    que ainda nao existem. Acrescentar no fim nao desloca nenhuma formula."""
    style_row = next((r for (_, kind, _), r in row_index.items() if kind == "pruning_quantizacao"),
                     3)
    next_row = ws.max_row + 1
    for kind, label_fmt, skip in [
        ("pruning_kd", "Pruning + KD ({pct}%)", set()),
        ("kd_quantizacao", "Pruning + KD + Quantização ({pct}%)", SKIP_QUANTIZED),
    ]:
        for arch in ALL_ARCHS:
            if arch in skip:
                continue
            modelo = SPREADSHEET_LABEL[arch]
            for pct in PCTS:
                if (_norm(modelo), kind, pct) in row_index:
                    continue
                _copy_row_style(ws, style_row, next_row)
                ws.cell(row=next_row, column=1, value=modelo)
                ws.cell(row=next_row, column=2, value=label_fmt.format(pct=pct))
                next_row += 1
    return _build_row_index(ws)


def _normalize_compression_formulas(ws, row_index: dict):
    """Reescreve as formulas de 'Taxa compressao (%)' (D) e 'Reduction (%)'
    (Q) de TODA linha que nao seja baseline, usando sempre referencia
    absoluta ($C$baseline/$P$baseline) para a linha de baseline da propria
    arquitetura. Roda sempre (idempotente, nao so' apos migracao) porque
    insert_rows do openpyxl desloca fisicamente as linhas abaixo do ponto de
    insercao sem corrigir o texto das formulas nelas — sem este passo, o
    bloco 'Energy Aware Pruning' (que fica abaixo do bloco expandido)
    ficaria com D/Q apontando para a linha errada."""
    baseline_row_by_modelo = {
        modelo: row for (modelo, kind, _pct), row in row_index.items() if kind == "baseline"
    }
    for (modelo, kind, _pct), row in row_index.items():
        if kind in ("baseline", "unknown"):
            continue
        baseline_row = baseline_row_by_modelo.get(modelo)
        if baseline_row is None:
            continue
        # IF(...="";"") evita mostrar 100% quando a coluna de referencia esta'
        # vazia (ex.: Params de modelo ainda nao avaliado, Joules da Raspberry).
        ws.cell(row=row, column=4,
                value=f'=IF(OR(C{row}="",$C${baseline_row}=""),"",'
                      f'100*($C${baseline_row}-C{row})/$C${baseline_row})')
        ws.cell(row=row, column=17,
                value=f'=IF(OR(P{row}="",$P${baseline_row}=""),"",'
                      f'100*($P${baseline_row}-P{row})/$P${baseline_row})')


def _backup_official_workbook():
    stamp = datetime.now().strftime("%Y%m%d")
    backup_path = OFFICIAL_XLSX.parent / f"{OFFICIAL_XLSX.stem}.backup-{stamp}{OFFICIAL_XLSX.suffix}"
    if not backup_path.exists():
        shutil.copy2(OFFICIAL_XLSX, backup_path)
        logger.info(f"Backup da planilha oficial criado em: {backup_path}")


class OfficialSheetWriter:
    """Abre 'Experimentos Metrado Andre.xlsx' uma vez, migra o bloco
    'Pruning + Quantizacao' se necessario, e permite escrever linhas
    individuais (salvando apos cada escrita — arquivo pequeno, custo baixo,
    garante que nada se perde se o processo cair no meio da matriz)."""

    def __init__(self):
        _backup_official_workbook()
        self.wb = openpyxl.load_workbook(OFFICIAL_XLSX)
        self.row_index = {}
        for dataset, sheet_name in SHEET_NAMES.items():
            ws = self.wb[sheet_name]
            idx = _build_row_index(ws)
            idx = _expand_pruning_quantizacao_block(ws, idx)
            idx = _ensure_kd_blocks(ws, idx)
            _normalize_compression_formulas(ws, idx)
            self.row_index[dataset] = idx
        self.wb.save(OFFICIAL_XLSX)

    def write(self, dataset: str, arch: str, kind: str, pct: int | None, row: dict):
        ws = self.wb[SHEET_NAMES[dataset]]
        modelo_norm = _norm(SPREADSHEET_LABEL[arch])
        row_num = self.row_index[dataset].get((modelo_norm, kind, pct))
        if row_num is None:
            logger.warning(
                f"    Nao encontrei linha na planilha oficial para "
                f"({SPREADSHEET_LABEL[arch]}, {kind}, {pct}) — pulando escrita."
            )
            return
        for key, col in COL.items():
            value = row.get(key)
            if value is not None:
                ws.cell(row=row_num, column=col, value=value)
        self.wb.save(OFFICIAL_XLSX)
        logger.info(f"    Planilha oficial atualizada: {SHEET_NAMES[dataset]}!linha {row_num}")


# ---------------------------------------------------------------------------
# Planilha de comparacao — recriada do zero a partir do Postgres a cada chamada
# ---------------------------------------------------------------------------

COMPARISON_CATEGORIES = [
    ("pruned_unstructured",            "Pruning Não-Estruturado ({pct}%)",             True),
    ("pruned_unstructured_quantized",  "Pruning Não-Estruturado ({pct}%) + Quantização", True),
    ("pruned_structured_kd",           "Pruning Estruturado ({pct}%) + KD",             True),
    ("pruned_structured_kd_quantized", "Pruning Estruturado ({pct}%) + KD + Quantização", True),
]


def rebuild_comparison_workbook():
    writer = pd.ExcelWriter(COMPARISON_XLSX, engine="openpyxl")
    for dataset in DATASETS:
        rows = []
        for arch in ALL_ARCHS:
            if arch in SKIP_QUANTIZED:
                categories = [c for c in COMPARISON_CATEGORIES if "quantized" not in c[0]]
            else:
                categories = COMPARISON_CATEGORIES
            for category, _label_fmt, has_pct in categories:
                for pct in PCTS:
                    nome_modelo = _nome_modelo(arch, category, pct)
                    existing = _get_cached_result(nome_modelo, DATASET_DB_NAME[dataset])
                    if existing is None:
                        continue
                    row = dict(existing)
                    row["Modelo"] = SPREADSHEET_LABEL[arch]
                    rows.append(row)
        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name=dataset, index=False)
    writer.close()
    logger.info(f"Planilha de comparacao atualizada: {COMPARISON_XLSX}")


# ---------------------------------------------------------------------------
# Etapa: baseline (sempre avaliado primeiro, pois a formula de Taxa de
# compressao da planilha oficial depende do Params da linha de baseline)
# ---------------------------------------------------------------------------

def ensure_baseline(dataset: str, arch: str, device, device_int, sheet_writer: OfficialSheetWriter):
    baseline = _baseline_path(dataset, arch)
    if not baseline.exists():
        logger.warning(f"  Baseline nao encontrado, pulando: {baseline}")
        return
    _copy_to_compressed(dataset, arch, "baseline", baseline)
    nome_modelo = _nome_modelo(arch, "baseline")
    row = _evaluate_with_resume(
        nome_modelo, dataset, arch=arch, checkpoint=baseline,
        compressao="Nenhum (baseline)", pruning_rate=None,
        device=device, device_int=device_int,
    )
    if row:
        sheet_writer.write(dataset, arch, "baseline", None, row)


# ---------------------------------------------------------------------------
# Etapa: poda (estrutural ou nao-estrutural)
# ---------------------------------------------------------------------------

def run_pruning_stage(mode: str, datasets, archs, pcts, ft_epochs, device, device_int,
                       overwrite: bool, sheet_writer: OfficialSheetWriter):
    dirname = "pruned_structured" if mode == "structured" else "pruned"
    category = f"pruned_{mode}"
    label_fmt = "Pruning Estruturado ({pct}%)" if mode == "structured" else "Pruning Não-Estruturado ({pct}%)"
    kind_official = "pruning_estruturado"  # so' usado quando mode == structured

    for dataset in datasets:
        for arch in archs:
            is_yolo = _arch_kind(arch) == "yolo"
            if mode == "structured" and is_yolo:
                # Decisao do usuario: poda estrutural nao e suportada para YOLO
                # (blocos C2f/C2fCIB/PSA quebram o rastreamento do torch-pruning
                # — ver NotImplementedError em prune_models.run_yolo_pipeline).
                # YOLO usa apenas a poda nao-estrutural, inclusive como resultado
                # "oficial" da planilha (ver bloco mode == "unstructured" abaixo).
                logger.info(f"\n[structured] {arch} — pulando (YOLO usa so' poda nao-estrutural)")
                continue

            ensure_baseline(dataset, arch, device, device_int, sheet_writer)
            baseline = _baseline_path(dataset, arch)
            if not baseline.exists():
                continue

            for pct in pcts:
                sparsity = pct / 100.0
                if _arch_kind(arch) == "yolo":
                    out = RUNS_DIR / "train" / dirname / str(pct) / dataset / f"{arch}_pruned.pt"
                else:
                    out = RUNS_DIR / arch / dirname / str(pct) / dataset / "best_pruned.pth"

                logger.info(f"\n[{mode}] {arch} | {dataset} | {pct}%")
                try:
                    if not out.exists() or overwrite:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        if _arch_kind(arch) == "yolo":
                            run_yolo_pipeline(
                                model_path=baseline, data_yaml=DATASETS[dataset] / "data.yaml",
                                output_path=out, sparsity=sparsity, ft_epochs=ft_epochs,
                                device=device, pruning_mode=mode,
                            )
                        else:
                            run_torchvision_pipeline(
                                model_type=arch, model_path=baseline, dataset_dir=DATASETS[dataset],
                                output_path=out, sparsity=sparsity, ft_epochs=ft_epochs,
                                device=device, pruning_mode=mode,
                            )
                    else:
                        logger.info(f"  Checkpoint ja existe: {out}")
                except Exception:
                    logger.error(f"  ERRO podando {arch}/{dataset}/{pct}% ({mode})", exc_info=True)
                    continue

                _copy_to_compressed(dataset, arch, f"{category}_p{pct}", out)

                # Template so' e' necessario para YOLO nao-estrutural (state_dict
                # puro); estrutural salva checkpoint completo via yolo.save().
                template_path = baseline if (mode == "unstructured" and _arch_kind(arch) == "yolo") else None

                nome_modelo = _nome_modelo(arch, category, pct)
                row = _evaluate_with_resume(
                    nome_modelo, dataset, arch=arch, checkpoint=out,
                    compressao=label_fmt.format(pct=pct), pruning_rate=float(pct),
                    device=device, device_int=device_int, template_path=template_path,
                )
                if row is None:
                    continue

                if mode == "structured" or is_yolo:
                    # Para YOLO, a poda nao-estrutural E' o resultado "oficial"
                    # (nao ha' poda estrutural para essa arquitetura — ver acima).
                    sheet_writer.write(dataset, arch, kind_official, pct, row)
                else:
                    rebuild_comparison_workbook()


# ---------------------------------------------------------------------------
# Etapa: quantizacao (baseline + podado estrutural/nao-estrutural)
# ---------------------------------------------------------------------------

def _quantize_one(arch, dataset, src_checkpoint: Path, out: Path, n_calib: int,
                   template_path: Path | None = None) -> bool:
    if not src_checkpoint.exists():
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    if _arch_kind(arch) == "yolo":
        run_yolo_quant_pipeline(model_path=src_checkpoint, dataset_dir=DATASETS[dataset],
                                 output_path=out, n_calib=n_calib, template_path=template_path)
    else:
        run_torchvision_quant_pipeline(model_type=arch, model_path=src_checkpoint,
                                        dataset_dir=DATASETS[dataset], output_path=out,
                                        n_calib=n_calib)
    return True


def run_quantize_stage(datasets, archs, pcts, n_calib, device, device_int,
                        overwrite: bool, sheet_writer: OfficialSheetWriter):
    for dataset in datasets:
        for arch in archs:
            ensure_baseline(dataset, arch, device, device_int, sheet_writer)
            if arch in SKIP_QUANTIZED:
                logger.info(f"\n[quantize] {arch} — pulando (limitação conhecida, ver README)")
                continue

            # --- baseline quantizado -------------------------------------------------
            baseline = _baseline_path(dataset, arch)
            if baseline.exists():
                if _arch_kind(arch) == "yolo":
                    out = RUNS_DIR / "train" / "quantized" / dataset / f"{arch}_quantized.onnx"
                else:
                    out = RUNS_DIR / arch / "quantized" / dataset / "best_quantized.onnx"
                logger.info(f"\n[quantize] {arch} | {dataset} | baseline")
                try:
                    if not out.exists() or overwrite:
                        _quantize_one(arch, dataset, baseline, out, n_calib)
                except Exception:
                    logger.error(f"  ERRO quantizando baseline {arch}/{dataset}", exc_info=True)
                else:
                    _copy_to_compressed(dataset, arch, "quantized", out)
                    nome_modelo = _nome_modelo(arch, "quantized")
                    row = _evaluate_with_resume(
                        nome_modelo, dataset, arch=arch, checkpoint=out,
                        compressao="Quantização (8 bits)", pruning_rate=None,
                        device=device, device_int=device_int,
                        inherit_from=_nome_modelo(arch, "baseline"),
                    )
                    if row:
                        sheet_writer.write(dataset, arch, "quantizacao", None, row)

            # --- podado (estrutural -> oficial; nao-estrutural -> comparacao) --------
            # Excecao: YOLO nao tem poda estrutural (ver run_pruning_stage) — pra
            # essa arquitetura a poda nao-estrutural E' o resultado "oficial" da
            # planilha (o checkpoint "structured" simplesmente nunca existe no
            # disco pra YOLO, entao o "if not pruned.exists(): continue" abaixo ja
            # pula essa combinacao sozinho).
            is_yolo = _arch_kind(arch) == "yolo"
            for mode, dirname, category, kind_official, label_fmt in [
                ("structured",   "pruned_structured", "pruned_structured_quantized",
                 "pruning_quantizacao", "Pruning Estruturado ({pct}%) + Quantização"),
                ("unstructured", "pruned",            "pruned_unstructured_quantized",
                 "pruning_quantizacao" if is_yolo else None,
                 "Pruning Estruturado ({pct}%) + Quantização" if is_yolo
                 else "Pruning Não-Estruturado ({pct}%) + Quantização"),
            ]:
                if mode == "structured" and is_yolo:
                    # YOLO nunca gera checkpoint "pruned_structured" (ver
                    # run_pruning_stage) — guard explicito em vez de confiar so'
                    # no "if not pruned.exists()" abaixo, que pegaria por engano
                    # um arquivo orfao deixado por uma execucao anterior a essa
                    # regra (ja aconteceu — ver limpeza manual de runs/train/
                    # pruned_structured/).
                    continue

                for pct in pcts:
                    if _arch_kind(arch) == "yolo":
                        pruned = RUNS_DIR / "train" / dirname / str(pct) / dataset / f"{arch}_pruned.pt"
                        out    = RUNS_DIR / "train" / "quantized" / dirname / str(pct) / dataset / f"{arch}_quantized.onnx"
                    else:
                        pruned = RUNS_DIR / arch / dirname / str(pct) / dataset / "best_pruned.pth"
                        out    = RUNS_DIR / arch / "quantized" / dirname / str(pct) / dataset / "best_quantized.onnx"

                    if not pruned.exists():
                        continue  # poda ainda nao rodou para essa combinacao

                    # Poda nao-estrutural de YOLO salva state_dict puro (nao um
                    # checkpoint ultralytics completo) — precisa de um template
                    # pra reconstruir o YOLO() antes de exportar pra ONNX (mesmo
                    # padrao ja usado por evaluate_models.py/_load_yolo_model).
                    template_path = baseline if (mode == "unstructured" and is_yolo) else None

                    logger.info(f"\n[quantize] {arch} | {dataset} | {mode} {pct}%")
                    try:
                        if not out.exists() or overwrite:
                            _quantize_one(arch, dataset, pruned, out, n_calib, template_path=template_path)
                    except Exception:
                        logger.error(f"  ERRO quantizando {arch}/{dataset}/{mode}/{pct}%", exc_info=True)
                        continue

                    _copy_to_compressed(dataset, arch, f"{category}_p{pct}", out)
                    nome_modelo = _nome_modelo(arch, category, pct)
                    row = _evaluate_with_resume(
                        nome_modelo, dataset, arch=arch, checkpoint=out,
                        compressao=label_fmt.format(pct=pct), pruning_rate=float(pct),
                        device=device, device_int=device_int,
                        inherit_from=_nome_modelo(arch, category.replace("_quantized", ""), pct),
                    )
                    if row is None:
                        continue
                    if kind_official:
                        sheet_writer.write(dataset, arch, kind_official, pct, row)
                    else:
                        rebuild_comparison_workbook()


# ---------------------------------------------------------------------------
# Etapa: poda estrutural + knowledge distillation (+ quantizacao)
# ---------------------------------------------------------------------------

def run_kd_stage(datasets, archs, pcts, ft_epochs, n_calib, kd_weight_yolo, kd_weight_tv,
                  device, device_int, overwrite: bool, sheet_writer: OfficialSheetWriter):
    for dataset in datasets:
        for arch in archs:
            baseline = _baseline_path(dataset, arch)
            if not baseline.exists():
                continue
            ensure_baseline(dataset, arch, device, device_int, sheet_writer)

            # YOLO nao suporta poda estrutural (ver run_pruning_stage) — o combo
            # "poda + KD" pra essa arquitetura usa poda nao-estrutural por baixo.
            # O nome interno da categoria ("pruned_structured_kd") fica igual pra
            # todas as arquiteturas so' por simplicidade de bookkeeping (nome_modelo/
            # compressed_models/comparacao); o rotulo humano (compressao) abaixo
            # reflete corretamente qual poda foi de fato usada em cada caso.
            is_yolo = _arch_kind(arch) == "yolo"
            kd_pruning_mode = "unstructured" if is_yolo else "structured"
            kd_label = "Pruning Não-Estruturado" if is_yolo else "Pruning Estruturado"

            for pct in pcts:
                sparsity = pct / 100.0
                if _arch_kind(arch) == "yolo":
                    out = RUNS_DIR / "train" / "pruned_structured_kd" / str(pct) / dataset / f"{arch}_pruned.pt"
                else:
                    out = RUNS_DIR / arch / "pruned_structured_kd" / str(pct) / dataset / "best_pruned.pth"

                logger.info(f"\n[kd] {arch} | {dataset} | {pct}%")
                try:
                    if not out.exists() or overwrite:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        if _arch_kind(arch) == "yolo":
                            run_yolo_pipeline(
                                model_path=baseline, data_yaml=DATASETS[dataset] / "data.yaml",
                                output_path=out, sparsity=sparsity, ft_epochs=ft_epochs,
                                device=device, pruning_mode=kd_pruning_mode,
                                use_kd=True, teacher_path=baseline, kd_weight=kd_weight_yolo,
                            )
                        else:
                            run_torchvision_pipeline(
                                model_type=arch, model_path=baseline, dataset_dir=DATASETS[dataset],
                                output_path=out, sparsity=sparsity, ft_epochs=ft_epochs,
                                device=device, pruning_mode=kd_pruning_mode,
                                use_kd=True, teacher_path=baseline, kd_weight=kd_weight_tv,
                            )
                    else:
                        logger.info(f"  Checkpoint ja existe: {out}")
                except Exception:
                    logger.error(f"  ERRO no KD {arch}/{dataset}/{pct}%", exc_info=True)
                    continue

                _copy_to_compressed(dataset, arch, f"pruned_structured_kd_p{pct}", out)
                nome_modelo = _nome_modelo(arch, "pruned_structured_kd", pct)
                # YOLO usa poda nao-estrutural por baixo (kd_pruning_mode) —
                # salva state_dict puro, precisa de template pra recarregar
                # (mesmo motivo de run_pruning_stage). Torchvision usa poda
                # estrutural (nn.Module inteiro pickled) — load_torchvision_model
                # ja detecta isso sozinho, sem template.
                row = _evaluate_with_resume(
                    nome_modelo, dataset, arch=arch, checkpoint=out,
                    compressao=f"{kd_label} ({pct}%) + KD", pruning_rate=float(pct),
                    device=device, device_int=device_int,
                    template_path=baseline if is_yolo else None,
                )
                if row is not None:
                    sheet_writer.write(dataset, arch, "pruning_kd", pct, row)
                    rebuild_comparison_workbook()

                # --- poda estrutural + KD + quantizacao -----------------------------
                if arch in SKIP_QUANTIZED:
                    continue
                if _arch_kind(arch) == "yolo":
                    out_q = RUNS_DIR / "train" / "quantized" / "pruned_structured_kd" / str(pct) / dataset / f"{arch}_quantized.onnx"
                else:
                    out_q = RUNS_DIR / arch / "quantized" / "pruned_structured_kd" / str(pct) / dataset / "best_quantized.onnx"

                if not out.exists():
                    continue
                logger.info(f"[kd+quantize] {arch} | {dataset} | {pct}%")
                try:
                    if not out_q.exists() or overwrite:
                        # YOLO + KD salva state_dict puro (poda nao-estrutural) —
                        # precisa do baseline como template para exportar o ONNX.
                        _quantize_one(arch, dataset, out, out_q, n_calib,
                                      template_path=baseline if is_yolo else None)
                except Exception:
                    logger.error(f"  ERRO quantizando KD {arch}/{dataset}/{pct}%", exc_info=True)
                    continue

                _copy_to_compressed(dataset, arch, f"pruned_structured_kd_quantized_p{pct}", out_q)
                nome_modelo_q = _nome_modelo(arch, "pruned_structured_kd_quantized", pct)
                row_q = _evaluate_with_resume(
                    nome_modelo_q, dataset, arch=arch, checkpoint=out_q,
                    compressao=f"{kd_label} ({pct}%) + KD + Quantização", pruning_rate=float(pct),
                    device=device, device_int=device_int,
                    inherit_from=_nome_modelo(arch, "pruned_structured_kd", pct),
                )
                if row_q is not None:
                    sheet_writer.write(dataset, arch, "kd_quantizacao", pct, row_q)
                    rebuild_comparison_workbook()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline unica: poda (estrutural/nao-estrutural) + quantizacao "
                     "+ KD + avaliacao + preenchimento das planilhas de experimentos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage", choices=["all", "baseline", "prune_structured", "prune_unstructured", "quantize", "kd"],
                   default="all")
    p.add_argument("--dataset", action="append", choices=list(DATASETS.keys()),
                   help="Repetivel. Default: oxford e caviar.")
    p.add_argument("--arch", action="append", choices=ALL_ARCHS,
                   help="Repetivel. Default: as 7 arquiteturas da planilha.")
    p.add_argument("--pct", action="append", type=int, choices=PCTS,
                   help="Repetivel. Default: 20, 40, 60, 80.")
    p.add_argument("--ft_epochs", type=int, default=10)
    p.add_argument("--n_calib", type=int, default=100)
    p.add_argument("--kd_weight_yolo", type=float, default=0.1)
    p.add_argument("--kd_weight_torchvision", type=float, default=1.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    global OVERWRITE
    OVERWRITE = args.overwrite
    datasets =args.dataset or list(DATASETS.keys())
    archs    = args.arch    or ALL_ARCHS
    pcts     = args.pct     or PCTS

    if not torch.cuda.is_available():
        raise RuntimeError("GPU nao encontrada.")
    device, device_int = torch.device("cuda:0"), 0
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    if not HAS_THOP:
        logger.warning("thop nao instalado — FLOPS sera None. Instale com: pip install thop")

    _try_init_db()
    setup_file_logging(BASE_DIR / "runs" / "compression_pipeline_logs" /
                        f"pipeline_{args.stage}.log")

    sheet_writer = OfficialSheetWriter()

    if args.stage in ("all", "baseline"):
        for dataset in datasets:
            for arch in archs:
                ensure_baseline(dataset, arch, device, device_int, sheet_writer)
    if args.stage in ("all", "prune_structured"):
        run_pruning_stage("structured", datasets, archs, pcts, args.ft_epochs,
                           device, device_int, args.overwrite, sheet_writer)
    if args.stage in ("all", "prune_unstructured"):
        run_pruning_stage("unstructured", datasets, archs, pcts, args.ft_epochs,
                           device, device_int, args.overwrite, sheet_writer)
    if args.stage in ("all", "quantize"):
        run_quantize_stage(datasets, archs, pcts, args.n_calib, device, device_int,
                            args.overwrite, sheet_writer)
    if args.stage in ("all", "kd"):
        run_kd_stage(datasets, archs, pcts, args.ft_epochs, args.n_calib,
                     args.kd_weight_yolo, args.kd_weight_torchvision,
                     device, device_int, args.overwrite, sheet_writer)

    logger.info("\nPipeline concluida (ou pausada por --stage). "
                f"Planilha oficial: {OFFICIAL_XLSX}")


if __name__ == "__main__":
    main()
