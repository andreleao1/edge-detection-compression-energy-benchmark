#!/usr/bin/env python3

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
ATTENTION = ("PSA", "C2PSA", "PSABlock", "Attention")


# ---------------------------------------------------------------------------
# Medicao de energia por janela (API no estilo do ZeusMonitor [6])
# ---------------------------------------------------------------------------

@dataclass
class Measurement:
    time: float     # s
    energy: float   # J


class ZeusMeter:
    """Energia da GPU NVIDIA via Zeus (contador de energia da NVML)."""

    def __init__(self, gpu_index: int = 0):
        from zeus.monitor import ZeusMonitor

        self.gpu = gpu_index
        self.mon = ZeusMonitor(gpu_indices=[gpu_index], cpu_indices=[], approx_instant_energy=True)

    def begin_window(self, key: str):
        self.mon.begin_window(key)

    def end_window(self, key: str) -> Measurement:
        m = self.mon.end_window(key)
        return Measurement(m.time, float(m.gpu_energy[self.gpu]))


class PowerTimeMeter:
    """Energia = potencia x tempo (dispositivos sem contador de energia)."""

    def __init__(self, power_w: float, idle_w: float = 0.0, device: str = "cpu"):
        self.p = power_w - idle_w
        self.cuda = device.startswith("cuda")
        self.t0 = {}

    def begin_window(self, key: str):
        if self.cuda:
            torch.cuda.synchronize()
        self.t0[key] = time.perf_counter()

    def end_window(self, key: str) -> Measurement:
        if self.cuda:
            torch.cuda.synchronize()
        t = time.perf_counter() - self.t0.pop(key)
        return Measurement(t, self.p * t)


def make_meter(args):
    if args.meter == "zeus":
        return ZeusMeter(torch.device(args.device).index or 0)
    return PowerTimeMeter(args.device_power, args.idle_power, args.device)


@torch.no_grad()
def energy_per_call(fn, meter, min_window: float, warmup: int = 3) -> float:
    """Energia (J) de uma chamada: repete fn ate a janela durar >= min_window
    (o contador da NVML tem resolucao de alguns ms; janelas curtas leem 0)."""
    for _ in range(warmup):
        fn()
    meter.begin_window("calib")
    fn()
    t1 = max(meter.end_window("calib").time, 1e-5)
    reps = max(3, math.ceil(min_window / t1))
    meter.begin_window("run")
    for _ in range(reps):
        fn()
    return meter.end_window("run").energy / reps


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def load_yolo(path: Path, device: str) -> nn.Module:
    """Carrega o YOLOv10 e converte os C2f para split-free (exigido pelo
    DepGraph; equivalente numericamente — ver prune_models.py)."""
    from ultralytics import YOLO

    import c2f_splitfree  # noqa: F401  (necessario para carregar modelos ja podados)
    from c2f_splitfree import convert_c2f_splitfree

    model = YOLO(str(path)).model.float().to(device)
    convert_c2f_splitfree(model)
    return model


def ignored_modules(model: nn.Module) -> list[nn.Module]:
    """Cabeca de deteccao e blocos de atencao, com TODOS os submodulos
    (o DepGraph so' ignora o modulo exato; sem expandir, ele enumera os
    grupos da cabeca e trava)."""
    top = [model.model[-1]] + [m for m in model.modules() if m.__class__.__name__ in ATTENTION]
    return [s for t in top for s in t.modules()]


def conv_inputs(model: nn.Module, imgsz: int, device: str) -> dict[str, tuple]:
    """Formato de entrada de cada Conv2d num forward real."""
    shapes, hooks = {}, []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            def hook(mod, inp, out, n=name):  # retorna None: nao altera a saida
                shapes.setdefault(n, tuple(inp[0].shape))
            hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        model.eval()(torch.zeros(1, 3, imgsz, imgsz, device=device))
    for h in hooks:
        h.remove()
    return shapes


# ---------------------------------------------------------------------------
# 1. Perfil de energia por camada
# ---------------------------------------------------------------------------

def _conv_like(conv: nn.Conv2d, cin: int, cout: int) -> nn.Module:
    groups = conv.groups
    if groups > 1 and groups == conv.in_channels == conv.out_channels:  # depthwise
        cin = cout = min(cin, cout)
        groups = cin
    elif groups > 1:
        cin, cout = conv.in_channels, conv.out_channels  # conv agrupada: sem reducao
    return nn.Sequential(
        nn.Conv2d(cin, cout, conv.kernel_size, conv.stride, conv.padding, conv.dilation, groups, conv.bias is not None),
        nn.SiLU(),  # ativacao das Conv do YOLO (custo proporcional aos canais de saida)
    )


def profile(model: nn.Module, meter, imgsz: int, device: str, min_window: float, only: set | None = None) -> dict:
    """Tabela {camada: energia inteira, custo marginal por canal de saida e de entrada}."""
    shapes = conv_inputs(model, imgsz, device)
    table = {}
    convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d) and n in shapes]
    for k, (name, conv) in enumerate(convs, 1):
        if only is not None and name not in only:
            continue
        _, cin, h, w = shapes[name]
        cout = conv.out_channels
        dw = conv.groups > 1 and conv.groups == conv.in_channels == conv.out_channels

        def run(c_in, c_out):
            mod = _conv_like(conv, c_in, c_out).to(device).eval()
            x = torch.randn(1, mod[0].in_channels, h, w, device=device)
            return energy_per_call(lambda: mod(x), meter, min_window)

        e_full = run(cin, cout)
        if cout >= 2:
            e_out = run(cin, cout // 2)
            slope_out = (e_full - e_out) / (cout - cout // 2)
        else:
            slope_out = 0.0
        if dw or conv.groups > 1 or cin < 2:
            slope_in = 0.0  # depthwise: entrada e saida caem juntas (ja em slope_out)
        else:
            e_in = run(cin // 2, cout)
            slope_in = (e_full - e_in) / (cin - cin // 2)
        table[name] = dict(layer=name, cin=cin, cout=cout, groups=conv.groups, k=conv.kernel_size[0],
                           stride=conv.stride[0], h=h, w=w, energy_mJ=1e3 * e_full,
                           slope_out_mJ=1e3 * slope_out, slope_in_mJ=1e3 * slope_in)
        if k % 10 == 0:
            print(f"  perfil: {k}/{len(convs)} convs", flush=True)
    return table


def model_energy(model: nn.Module, meter, imgsz: int, device: str, min_window: float) -> float:
    x = torch.randn(1, 3, imgsz, imgsz, device=device)
    model.eval()
    return energy_per_call(lambda: model(x), meter, min_window)


def write_table(table: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(table.values())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_table(path: Path) -> dict:
    out = {}
    for r in csv.DictReader(path.open()):
        out[r["layer"]] = {k: (r[k] if k == "layer" else float(r[k])) for k in r}
    return out


# ---------------------------------------------------------------------------
# 2-5. Importancia, custo por canal e selecao
# ---------------------------------------------------------------------------

def accumulate_gradients(model: nn.Module, data: Path, src: Path, batches: int, device: str) -> None:
    """Gradientes da perda de deteccao em alguns lotes de treino (Taylor [4])."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    tr = DetectionTrainer(overrides=dict(data=str(data), model=str(src), imgsz=640, batch=8, workers=0,
                                         device=ul_device(device), verbose=False, plots=False))
    tr.model = model
    loader = tr.get_dataloader(tr.data["train"], batch_size=8, rank=-1, mode="train")
    model.args = tr.args
    model.train().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    for i, b in enumerate(loader):
        if i >= batches:
            break
        b["img"] = b["img"].to(device).float() / 255
        for k in ("batch_idx", "cls", "bboxes"):
            b[k] = b[k].to(device)
        loss, _ = model.loss(b)
        loss.sum().backward()
    model.eval()


def group_cost(DG, group, table: dict) -> float:
    """Energia (J) economizada ao remover UM canal do grupo."""
    names = {m: n for n, m in DG.model.named_modules()}
    n_root = len(group[0].idxs)
    cost = 0.0
    for item in group:
        mod = item.dep.target.module
        if not isinstance(mod, nn.Conv2d) or names.get(mod) not in table:
            continue
        t = table[names[mod]]
        frac = len(item.idxs) / n_root
        if DG.is_out_channel_pruning_fn(item.dep.handler):
            cost += 1e-3 * t["slope_out_mJ"] * frac
        elif DG.is_in_channel_pruning_fn(item.dep.handler):
            cost += 1e-3 * t["slope_in_mJ"] * frac
    return cost


def prune_step(model, example, table, budget_j, orig_channels, args, log_layers) -> float:
    """Remove canais ate economizar budget_j (J por imagem). Retorna o estimado."""
    import torch_pruning as tp

    DG = tp.DependencyGraph().build_dependency(model, example_inputs=example)
    imp = tp.importance.GroupTaylorImportance() if args.importance == "taylor" \
        else tp.importance.GroupMagnitudeImportance(p=1)
    names = {m: n for n, m in model.named_modules()}
    cands, groups = [], []
    for g in DG.get_all_groups(ignored_layers=ignored_modules(model), root_module_types=[nn.Conv2d]):
        root = g[0].dep.target.module
        rname = names[root]
        s = imp(g)
        if s is None:
            continue
        c = group_cost(DG, g, table) if args.cost == "energy" else 1.0
        c_eff = c if c > 0 else 1e-9          # custo <= 0 (ruido): poda por ultimo
        energy_c = group_cost(DG, g, table)   # economia real usada no orcamento
        n_now = len(g[0].idxs)
        n0 = orig_channels.setdefault(rname, n_now)
        keep_min = max(args.min_channels, math.ceil(n0 * (1 - args.max_layer_ratio)))
        cap = max(0, n_now - keep_min)
        if cap == 0:
            continue
        gi = len(groups)
        groups.append((root, g[0].dep.handler, rname, cap, energy_c))
        s = s.detach().float().cpu()
        for ch in range(n_now):
            cands.append((float(s[ch]) / c_eff, gi, ch))
    cands.sort()
    chosen, saved = {}, 0.0
    for score, gi, ch in cands:
        if saved >= budget_j:
            break
        root, fn, rname, cap, energy_c = groups[gi]
        sel = chosen.setdefault(gi, [])
        if len(sel) >= cap:
            continue
        sel.append(ch)
        saved += max(energy_c, 0.0)
    for gi, idxs in chosen.items():
        root, fn, rname, cap, energy_c = groups[gi]
        if args.round_to > 1:  # deixa o numero restante de canais multiplo de round_to
            remain = DG.get_out_channels(root) - len(idxs)
            extra = remain % args.round_to
            if extra and len(idxs) + extra <= cap:
                rest = [c for _, g2, c in cands if g2 == gi and c not in idxs][:extra]
                idxs.extend(rest)
        DG.get_pruning_group(root, fn, sorted(idxs)).prune()
        log_layers.append(dict(layer=rname, removed=len(idxs)))
    with torch.no_grad():
        model.eval()(example)  # forward valido apos a poda
    return saved


# ---------------------------------------------------------------------------
# 6-7. Ajuste fino
# ---------------------------------------------------------------------------

def finetune(model, src: Path, data: Path, epochs: int, patience: int, out_dir: Path, name: str, device: str) -> Path:
    from ultralytics.models.yolo.detect import DetectionTrainer

    from train_baseline_v2 import RECIPE

    kw = {**RECIPE, "epochs": epochs, "patience": patience,
          "warmup_epochs": min(RECIPE["warmup_epochs"], max(0, epochs // 5)),
          "close_mosaic": min(RECIPE["close_mosaic"], epochs // 2),
          "data": str(data), "model": str(src), "device": ul_device(device),
          "project": str(out_dir.resolve()), "name": name, "exist_ok": True, "plots": False, "verbose": False}
    tr = DetectionTrainer(overrides=kw)
    tr.model = model.float().train()  # modulo podado: o trainer nao reconstroi pelo yaml
    tr.train()
    best = Path(tr.best)
    return best if best.exists() else Path(tr.last)


def ul_device(device: str):
    """Dispositivo no formato do ultralytics (indice da GPU ou 'cpu')."""
    d = torch.device(device)
    return "cpu" if d.type == "cpu" else (d.index or 0)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_profile(args):
    meter = make_meter(args)
    model = load_yolo(args.model, args.device)
    table = profile(model, meter, args.imgsz, args.device, args.min_window)
    write_table(table, args.out)
    e_model = model_energy(model, meter, args.imgsz, args.device, args.min_window)
    e_conv = sum(t["energy_mJ"] for t in table.values())
    print(f"tabela: {args.out}  |  {len(table)} convs  |  soma das convs {e_conv:.1f} mJ, "
          f"modelo inteiro {1e3 * e_model:.1f} mJ por imagem")


def cmd_prune(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    model = load_yolo(args.model, args.device)
    example = torch.randn(1, 3, args.imgsz, args.imgsz, device=args.device)
    meter = make_meter(args) if (args.energy_table is None or args.measure) else None
    table = read_table(args.energy_table) if args.energy_table else \
        profile(model, meter, args.imgsz, args.device, args.min_window)
    write_table(table, out / "energy_table_inicial.csv")

    e0 = sum(1e-3 * t["energy_mJ"] for t in table.values())
    target = (1 - args.target) * e0
    step_budget = args.step * e0
    est, orig_channels, src, history = e0, {}, args.model, []
    p0 = n_params(model)
    print(f"energia das convs: {1e3 * e0:.1f} mJ/imagem  ->  alvo {1e3 * target:.1f} mJ "
          f"(-{100 * args.target:.0f}%), passos de {1e3 * step_budget:.1f} mJ", flush=True)
    step = 0
    while est > target + 1e-12:
        step += 1
        if args.importance == "taylor":
            accumulate_gradients(model, args.data, src, args.taylor_batches, args.device)
        layers = []
        saved = prune_step(model, example, table, min(step_budget, est - target), orig_channels, args, layers)
        if saved <= 0:
            print("nenhum canal pode ser removido (limite por camada atingido); parando.")
            break
        est -= saved
        rec = dict(step=step, estimated_energy_mJ=1e3 * est, energy_ratio=est / e0,
                   params_M=n_params(model) / 1e6, params_ratio=n_params(model) / p0, layers=layers)
        if args.reprofile and meter is not None:  # NetAdapt: remede no dispositivo
            table = profile(model, meter, args.imgsz, args.device, args.min_window)
            est = sum(1e-3 * t["energy_mJ"] for t in table.values())
            rec["remeasured_energy_mJ"] = 1e3 * est
        print(f"passo {step}: energia estimada {1e3 * est:.1f} mJ ({100 * est / e0:.0f}%), "
              f"params {rec['params_M']:.2f} M ({100 * rec['params_ratio']:.0f}%)", flush=True)
        if args.step_epochs > 0 and est > target + 1e-12:
            src = finetune(model, src, args.data, args.step_epochs, 0, out, f"passo{step}", args.device)
            model = load_yolo(src, args.device)
        history.append(rec)
        with (out / "history.json").open("w") as fh:
            json.dump(history, fh, indent=1)

    print(f"ajuste fino final ({args.final_epochs} epocas, patience {args.patience})...", flush=True)
    best = finetune(model, src, args.data, args.final_epochs, args.patience, out, "final", args.device)
    final = out / "model_energy_pruned.pt"
    shutil.copy2(best, final)

    model = load_yolo(final, args.device)
    summary = dict(model=str(final), source=str(args.model), target=args.target, importance=args.importance,
                   cost=args.cost, steps=step, estimated_energy_ratio=est / e0,
                   params_M=n_params(model) / 1e6, params_ratio=n_params(model) / p0,
                   channels={n: dict(original=orig_channels[n], final=m.out_channels)
                             for n, m in model.named_modules() if n in orig_channels and isinstance(m, nn.Conv2d)})
    if meter is not None and args.measure:
        e_before = model_energy(load_yolo(args.model, args.device), meter, args.imgsz, args.device, args.min_window)
        e_after = model_energy(model, meter, args.imgsz, args.device, args.min_window)
        summary.update(measured_energy_before_mJ=1e3 * e_before, measured_energy_after_mJ=1e3 * e_after,
                       measured_energy_ratio=e_after / e_before)
    with (out / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"modelo: {final}\nresumo: {out / 'summary.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", type=Path, required=True)
        p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
        p.add_argument("--imgsz", type=int, default=640)
        p.add_argument("--meter", choices=("zeus", "power"), default="zeus" if torch.cuda.is_available() else "power")
        p.add_argument("--device-power", type=float, default=3.2,
                       help="[power] potencia media (W) durante a inferencia, medida pelo INA219")
        p.add_argument("--idle-power", type=float, default=0.0, help="[power] potencia em repouso (W) a descontar")
        p.add_argument("--min-window", type=float, default=0.3, help="duracao minima de cada janela de medicao (s)")

    p = sub.add_parser("profile", help="tabela de energia por camada (rodar no dispositivo alvo)")
    common(p)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("prune", help="poda iterativa guiada por energia + ajuste fino")
    common(p)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--energy-table", type=Path, help="CSV de 'profile' (ex. medido na Raspberry)")
    p.add_argument("--target", type=float, default=0.5, help="fracao da energia das convs a remover")
    p.add_argument("--step", type=float, default=0.1, help="fracao da energia original removida por passo")
    p.add_argument("--importance", choices=("taylor", "l1"), default="taylor")
    p.add_argument("--cost", choices=("energy", "uniform"), default="energy",
                   help="uniform = mesma selecao sem dividir pelo custo (comparacao)")
    p.add_argument("--taylor-batches", type=int, default=8)
    p.add_argument("--max-layer-ratio", type=float, default=0.8, help="fracao maxima removida por camada")
    p.add_argument("--min-channels", type=int, default=8)
    p.add_argument("--round-to", type=int, default=1)
    p.add_argument("--step-epochs", type=int, default=10, help="ajuste fino entre passos (0 = sem)")
    p.add_argument("--final-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--reprofile", action="store_true", help="remede a tabela a cada passo (local)")
    p.add_argument("--measure", action="store_true", help="mede a energia antes/depois com --meter")
    p.set_defaults(func=cmd_prune)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
