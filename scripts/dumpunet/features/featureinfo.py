from dataclasses import dataclass

import torch
from torch import Tensor

from scripts.dumpunet import layerinfo
from scripts.dumpunet.report import message as E
from scripts.dumpunet.utils import *

@dataclass
class FeatureInfo:
    input_dims: list[torch.Size]
    output_dims: torch.Size
    output: Tensor

class Features:
    
    # layer -> FeatureInfo
    features : dict[str, FeatureInfo]
    
    def __init__(self):
        self.features = dict()
    
    def __getitem__(self, layer: int|str):
        if isinstance(layer, int):
            return self.get_by_index(layer)
        elif isinstance(layer, str):
            return self.get_by_name(layer)
        else:
            raise ValueError(E(f"invalid type: {type(layer)}"))
    
    def __iter__(self):
        return sorted_items(self.features)
    
    def layers(self):
        return sorted_keys(self.features)
    
    def get_by_name(self, layer: str) -> FeatureInfo|None:
        if layer in self.features:
            return self.features[layer]
        return None
    
    def get_by_index(self, layer: int) -> FeatureInfo|None:
        name = layerinfo.name(layer)
        if name is None:
            return None
        return self.get_by_name(name)
    
    def add(self, layer: int|str, info: FeatureInfo):
        if isinstance(layer, int):
            name = layerinfo.name(layer)
            if name is None:
                raise ValueError(E(f"invalid layer name: {layer}"))
            layer = name
        self.features[layer] = info
