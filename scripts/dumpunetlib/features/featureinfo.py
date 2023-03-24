from typing import Generic, TypeVar
from dataclasses import dataclass
from collections import defaultdict

import torch
from torch import Tensor
import numpy as np

from scripts.dumpunetlib import layerinfo
from scripts.dumpunetlib.report import message as E
from scripts.dumpunetlib.utils import *

@dataclass
class FeatureInfo:
    input_dims: list[torch.Size]
    output_dims: torch.Size
    output: Tensor

#@dataclass
#class AttnFeatureInfo:
#    qk: np.ndarray
#    vqk: np.ndarray

TInfo = TypeVar("TInfo")

class Features(Generic[TInfo]):
    
    # layer -> UNetFeatureInfo
    features : dict[str, TInfo]
    
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
    
    def get_by_name(self, layer: str) -> TInfo|None:
        if layer in self.features:
            return self.features[layer]
        return None
    
    def get_by_index(self, layer: int) -> TInfo|None:
        name = layerinfo.name(layer)
        if name is None:
            return None
        return self.get_by_name(name)
    
    def add(self, layer: int|str, info: TInfo):
        if isinstance(layer, int):
            name = layerinfo.name(layer)
            if name is None:
                raise ValueError(E(f"invalid layer name: {layer}"))
            layer = name
        self.features[layer] = info

class MultiStepFeatures(Generic[TInfo], defaultdict[int,Features[TInfo]]):
    
    def __init__(self):
        super().__init__(lambda: Features())

class MultiImageFeatures(Generic[TInfo], defaultdict[int,MultiStepFeatures[TInfo]]):
    
    def __init__(self):
        super().__init__(lambda: MultiStepFeatures())

