from dataclasses import dataclass

from torch import Tensor

@dataclass
class AttnFeatureInfo:
    qk: Tensor
    vqk: Tensor
