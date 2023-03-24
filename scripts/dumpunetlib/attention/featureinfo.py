from dataclasses import dataclass

from torch import Tensor

@dataclass
class AttnFeatureInfo:
    k: Tensor
    qk: Tensor
    vqk: Tensor
