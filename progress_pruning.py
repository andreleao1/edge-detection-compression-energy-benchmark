#!/usr/bin/env python3
"""Andamento real do run_all_pruning.sh: concluidos, em execucao, restantes e
ETA estimado pela duracao media REAL por modelo (lida dos logs ja concluidos)."""
import os, re, glob, time
from collections import defaultdict

DS, PCTS = ["caviar", "oxford"], [20, 40, 60, 80]
ARCHS = [f"yolov10{v}" for v in "nsmblx"] + ["faster_rcnn", "retinanet", "mobilenet"]

def out_path(a, ds, p):
    if a.startswith("yolov10"):
        return f"runs/train/pruned_structured/{p}/{ds}/{a}_pruned.pt"
    return f"runs/{a}/pruned_structured/{p}/{ds}/best_pruned.pth"

def log_path(a, ds, p): return f"runs/pruning_logs/{a}_{ds}_{p}.log"

def duration(a, ds, p):
    lp = log_path(a, ds, p)
    if not os.path.exists(lp): return None
    st = os.stat(lp)
    birth = getattr(st, "st_birthtime", None)
    if birth is None:  # statx via stat -c %W
        try: birth = float(os.popen(f"stat -c %W '{lp}'").read())
        except Exception: birth = 0
    if not birth: return None
    return st.st_mtime - birth

jobs = [(a, ds, p) for ds in DS for p in PCTS for a in ARCHS]
done, todo, running = [], [], None
for j in jobs:
    (done if os.path.exists(out_path(*j)) else todo).append(j)

durs = defaultdict(list)
for j in done:
    d = duration(*j)
    if d and d > 30: durs[j[0]].append(d)
allv = [d for v in durs.values() for d in v]
gavg = sum(allv) / len(allv) if allv else None

# em execucao = log mais recente ainda sem checkpoint, modificado ha < 15 min
for j in todo:
    lp = log_path(*j)
    if os.path.exists(lp) and time.time() - os.path.getmtime(lp) < 900:
        running = j; break

def est(j):
    v = durs.get(j[0]); 
    return sum(v) / len(v) if v else gavg

print(f"Concluidos : {len(done)} / {len(jobs)}  ({100*len(done)/len(jobs):.0f}%)")
if running:
    tail = open(log_path(*running), errors="ignore").read()[-3000:]
    ep = re.findall(r"(\d+)/(\d+)\s+[\d.]+G", tail) or re.findall(r"Epoch\s+(\d+)/(\d+)", tail)
    print(f"Em execucao: {running[0]} | {running[1]} | {running[2]}%"
          + (f" | epoca {ep[-1][0]}/{ep[-1][1]}" if ep else ""))
else:
    print("Em execucao: nenhum log ativo nos ultimos 15 min (script parado?)")

print("\nDuracao media por modelo (execucoes concluidas):")
for a in ARCHS:
    v = durs.get(a)
    print(f"  {a:12s} {sum(v)/len(v)/60:6.1f} min  (n={len(v)})" if v else f"  {a:12s}   --")

rest = [j for j in todo if est(j)]
if gavg:
    total = sum(est(j) for j in todo if est(j) is not None) + sum(gavg for j in todo if est(j) is None)
    print(f"\nRestam {len(todo)} execucoes; ETA ~ {total/3600:.1f} h "
          f"(modelos sem amostra usam a media geral {gavg/60:.0f} min)")
by = defaultdict(int)
for a, ds, p in todo: by[(ds, p)] += 1
print("\nPendentes por base/percentual:", {f"{k[0]}-{k[1]}%": v for k, v in sorted(by.items())})
