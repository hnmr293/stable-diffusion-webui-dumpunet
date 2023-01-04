from dataclasses import dataclass
from collections import defaultdict

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
        v = None
        if isinstance(layer, int):
            v = self.get_by_index(layer)
        elif isinstance(layer, str):
            v = self.get_by_name(layer)
        if v is None:
            raise KeyError(E(f"invalid key: {type(layer)} {layer}"))
        return v
    
    def __iter__(self):
        return sorted_items(self.features)
    
    def __contains__(self, key: int|str):
        if isinstance(key, int):
            key = layerinfo.name(key) or ""
        return key in self.features
    
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

class MultiStepFeatures(defaultdict[int,Features]):
    
    def __init__(self):
        super().__init__(lambda: Features())

class MultiImageFeatures(defaultdict[int,MultiStepFeatures]):
    
    def __init__(self):
        super().__init__(lambda: MultiStepFeatures())

