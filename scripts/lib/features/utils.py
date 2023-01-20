import math
from typing import Generator

from torch import Tensor

from scripts.lib import tutils
from scripts.lib import layerinfo
from scripts.lib.features.featureinfo import FeatureInfo, Features, MultiImageFeatures
from scripts.lib.report import message as E

def feature_diff(
    features1: MultiImageFeatures[FeatureInfo],
    features2: MultiImageFeatures[FeatureInfo],
    abs: bool = False
) -> Generator[tuple[int,int,str,Tensor],None,None]:
    # features1 and features2 must be have same keys...
    for img_idx in sorted(features1.keys()):
        assert img_idx in features1
        assert img_idx in features2
        
        fs1 = features1[img_idx]
        fs2 = features2[img_idx]
        
        for step in sorted(fs1.keys()):
            assert step in fs1
            assert step in fs2
            
            f1 = fs1[step]
            f2 = fs2[step]
            
            for layer in f1.layers():
                assert layer in f1
                assert layer in f2
                
                l1, l2 = f1[layer], f2[layer]
                a, b = l1.output, l2.output
                assert a.size() == b.size()
                assert len(a.size()) == 3
                
                if abs:
                    c = (b - a).abs()
                else:
                    c = b - a
                
                yield img_idx, step, layer, c

def feature_to_grid_images(
    feature: FeatureInfo|Tensor,
    layer: str,
    width: int,
    height: int,
    color: bool
):
    tensor = feature
    if isinstance(feature, FeatureInfo):
        tensor = feature.output
    assert isinstance(tensor, Tensor)
    assert len(tensor.size()) == 3
    
    grid_x, grid_y = _get_grid_num(layer, width, height)
    canvases = tutils.tensor_to_image(tensor, grid_x, grid_y, color)
    return canvases

def save_features(
    feature: FeatureInfo,
    save_dir: str,
    basename: str
):
    tutils.save_tensor(feature.output, save_dir, basename)

def _get_grid_num(layer: str, width: int, height: int):
    assert layer is not None and layer != "", E("<Layers> must not be empty.")
    assert layer in layerinfo.Settings, E(f"Invalid <Layers> value: {layer}.")
    _, (ch, mh, mw) = layerinfo.Settings[layer]
    iw = math.ceil(width  / 64)
    ih = math.ceil(height / 64)
    w = mw * iw
    h = mh * ih 
    # w : width of a feature map
    # h : height of a feature map
    # ch: a number of a feature map
    n = [w, h]
    while ch % 2 == 0:
        n[n[0]>n[1]] *= 2
        ch //= 2
    n[n[0]>n[1]] *= ch
    if n[0] > n[1]:
        while n[0] > n[1] * 2 and (n[0] // w) % 2 == 0:
            n[0] //= 2
            n[1] *= 2
    else:
        while n[0] * 2 < n[1] and (n[1] // h) % 2 == 0:
            n[0] *= 2
            n[1] //= 2
    
    return n[0] // w, n[1] // h
