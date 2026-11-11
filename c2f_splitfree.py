"""
Conversao C2f / C2fCIB -> variante "split-free" (sem chunk()), rastreavel pelo
torch-pruning.

Por que isso e necessario: o forward padrao do C2f faz
    y = list(self.cv1(x).chunk(2, 1))
e o DependencyGraph do torch-pruning nao rastreia chunk()/split de canais
corretamente — e exatamente por isso que a poda estrutural "nao funcionava"
para YOLOv10. A solucao (mesma ideia do exemplo oficial yolov8_pruning.py do
torch-pruning) e substituir o Conv cv1 (saida 2c) por dois Convs separados
cv0/cv1 (saida c cada), copiando as metades dos pesos — matematicamente
IDENTICO (conv e linear: fatiar os filtros de saida == fatiar a saida), o que
o teste de equivalencia em prune_pipeline.py verifica numericamente.

Cobre C2f e qualquer subclasse com o mesmo forward (C2fCIB do YOLOv10 herda o
forward do C2f), reaproveitando cv2 e a lista m do modulo original — por isso
nao precisa conhecer os argumentos de construcao de cada subclasse.

IMPORTANTE: checkpoints salvos apos a conversao (podados ou nao) contem
instancias de C2fSplitFree picladas. Este arquivo precisa estar importavel
(mesmo nome, no sys.path — ex.: rodar os scripts a partir da raiz do repo)
sempre que um desses checkpoints for carregado.
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules import C2f, Conv


class C2fSplitFree(nn.Module):
    """Equivalente ao C2f/C2fCIB, com cv0/cv1 separados no lugar do chunk."""

    def __init__(self, cv0: nn.Module, cv1: nn.Module, cv2: nn.Module,
                 m: nn.ModuleList, c: int):
        super().__init__()
        self.c = c
        self.cv0 = cv0
        self.cv1 = cv1
        self.cv2 = cv2
        self.m = m

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def _split_conv_in_two(conv_block: Conv, c: int):
    """Divide um bloco Conv (conv+bn+act) com 2c filtros de saida em dois
    blocos com c filtros cada, copiando pesos/estatisticas por fatia."""
    old_conv, old_bn = conv_block.conv, conv_block.bn
    device, dtype = old_conv.weight.device, old_conv.weight.dtype
    halves = []
    for sl in (slice(0, c), slice(c, 2 * c)):
        new = Conv(old_conv.in_channels, c,
                   k=old_conv.kernel_size[0], s=old_conv.stride[0])
        new = new.to(device=device, dtype=dtype)
        with torch.no_grad():
            new.conv.weight.copy_(old_conv.weight[sl])
            if old_conv.bias is not None:
                new.conv.bias.copy_(old_conv.bias[sl])
            new.bn.weight.copy_(old_bn.weight[sl])
            new.bn.bias.copy_(old_bn.bias[sl])
            new.bn.running_mean.copy_(old_bn.running_mean[sl])
            new.bn.running_var.copy_(old_bn.running_var[sl])
            new.bn.num_batches_tracked.copy_(old_bn.num_batches_tracked)
            new.bn.eps = old_bn.eps
            new.bn.momentum = old_bn.momentum
        new.act = conv_block.act  # mesmo modulo de ativacao (SiLU)
        halves.append(new)
    return halves


def convert_c2f_splitfree(model: nn.Module) -> int:
    """Substitui, in-place, todo C2f/C2fCIB do modelo por C2fSplitFree.
    Retorna o numero de modulos convertidos."""
    n = 0
    for name, child in model.named_children():
        if isinstance(child, C2f):
            cv0, cv1 = _split_conv_in_two(child.cv1, child.c)
            new = C2fSplitFree(cv0, cv1, child.cv2, child.m, child.c)
            # atributos de grafo do DetectionModel (_predict_once usa m.f/m.i)
            for attr in ("f", "i", "type", "np"):
                if hasattr(child, attr):
                    setattr(new, attr, getattr(child, attr))
            # modulos recem-criados nascem em train(); herdar o modo do
            # original (senao o BN dos novos cv0/cv1 usa estatisticas do
            # batch dentro de um modelo em eval e a equivalencia quebra)
            new.train(child.training)
            setattr(model, name, new)
            n += 1
        else:
            n += convert_c2f_splitfree(child)
    return n
