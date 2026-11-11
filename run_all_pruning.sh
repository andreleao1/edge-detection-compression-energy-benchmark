#!/usr/bin/env bash
# Poda estrutural de todos os modelos de models/epochs_100 (caviar e oxford)
# nos percentuais 20/40/60/80. Pula combinacoes ja geradas (re-executavel).
# Uso: ./run_all_pruning.sh [ft_epochs]     (padrao: 30)
set -u
cd "$(dirname "$0")"
FT_EPOCHS="${1:-30}"
PCTS=(20 40 60 80)
LOG_DIR=runs/pruning_logs; mkdir -p "$LOG_DIR" runs/yaml

for ds in caviar oxford; do
  cat > runs/yaml/$ds.yaml <<YAML
train: $PWD/$ds/datasets/images/train
val: $PWD/$ds/datasets/images/val
test: $PWD/$ds/datasets/images/test
nc: 1
names: ['pessoa']
YAML
done

for ds in caviar oxford; do
  for pct in "${PCTS[@]}"; do
    # --- YOLO ---
    for v in n s m b l x; do
      model="models/epochs_100/$ds/yolov10${v}_finetuned.pt"
      out="runs/train/pruned_structured/$pct/$ds/yolov10${v}_pruned.pt"
      [ -f "$model" ] || { echo "[skip] sem modelo: $model"; continue; }
      [ -f "$out" ] && { echo "[skip] ja existe: $out"; continue; }
      echo "[run] yolov10$v | $ds | $pct%"
      python prune_models.py --model_type yolo --model_path "$model" \
        --data_yaml runs/yaml/$ds.yaml --pruning_mode structured \
        --pruning_percentage "$pct" --ft_epochs "$FT_EPOCHS" --output_path "$out" \
        > "$LOG_DIR/yolov10${v}_${ds}_${pct}.log" 2>&1 || echo "[ERRO] ver $LOG_DIR/yolov10${v}_${ds}_${pct}.log"
    done
    # --- torchvision ---
    for arch in faster_rcnn retinanet mobilenet; do
      model="models/epochs_100/$ds/${arch}_finetuned.pth"
      out="runs/$arch/pruned_structured/$pct/$ds/best_pruned.pth"
      [ -f "$model" ] || { echo "[skip] sem modelo: $model"; continue; }
      [ -f "$out" ] && { echo "[skip] ja existe: $out"; continue; }
      echo "[run] $arch | $ds | $pct%"
      python prune_models.py --model_type "$arch" --model_path "$model" \
        --dataset_dir "$ds" --pruning_mode structured \
        --pruning_percentage "$pct" --ft_epochs "$FT_EPOCHS" --output_path "$out" \
        > "$LOG_DIR/${arch}_${ds}_${pct}.log" 2>&1 || echo "[ERRO] ver $LOG_DIR/${arch}_${ds}_${pct}.log"
    done
  done
done
echo "Concluido."
