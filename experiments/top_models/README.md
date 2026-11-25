# Modelos de melhor custo/beneficio

Gerado por `select_top_models.py --export` em 2026-11-25.
Criterio: fronteira de Pareto: mAP50-95 x J/imagem, ambos na pior base (Caviar/Oxford).

| # | Diretorio | Modelo | J/imagem cav / oxf | mAP50-95 cav / oxf | mAP50 cav / oxf | Re-quantizar? |
|---|---|---|---|---|---|---|
| 1 | `01_yolov10n_kd` | YOLO10n — Knowledge Distillation | 4.14 / 3.38 | 0.892 / 0.471 | 0.995 / 0.884 | nao |
| 2 | `02_yolov10m_quantized` | YOLO10m — Quantização (8 bits) | 11.09 / 11.79 | 0.932 / 0.533 | 0.995 / 0.926 | sim (origem: baseline) |
| 3 | `03_yolov10l_quantized` | YOLO10l — Quantização (8 bits) | 19.73 / 19.80 | 0.938 / 0.554 | 0.995 / 0.963 | sim (origem: baseline) |
| 4 | `04_yolov10m_kd` | YOLO10m — Knowledge Distillation | 22.91 / 17.61 | 0.938 / 0.602 | 0.995 / 0.977 | nao |
| 5 | `05_yolov10l_kd` | YOLO10l — Knowledge Distillation | 41.40 / 34.54 | 0.943 / 0.618 | 0.995 / 0.983 | nao |

Estes valores sao a **referencia** de cada modelo: toda modificacao deve ser
comparada com a linha do proprio modelo (mesmo protocolo de medicao na
Raspberry: `main_models.py --duration-minutes 10 --cooldown-minutes 3`).

## Estrutura

```
<NN>_<arch>_<variante>/
  caviar/source/      checkpoint PyTorch de partida para modificar (Caviar)
  caviar/reference/   modelo exatamente medido na Raspberry, quando difere do
                      source (ex.: ONNX quantizado) — so' para comparacao
  oxford/...          idem para a Oxford
  variants/           uma pasta por experimento, ex. variants/relu/
```

Os modelos sao treinados por base: cada modificacao precisa ser feita e
medida separadamente para Caviar e Oxford.

Nos modelos marcados com "Re-quantizar", a modificacao parte do checkpoint
PyTorch anterior a quantizacao; depois de modificar e ajustar, gere o ONNX
quantizado de novo (`quantize_models.py`) antes de medir na Raspberry.

`manifest.json` tem os caminhos de origem e todas as metricas de referencia
(energia, latencia, CPU, RAM, potencia, mAP, data da medicao).
Os pesos (`.pt`, `.onnx`) nao vao para o git.
