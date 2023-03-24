from typing import Generator

from torch import Tensor

from scripts.dumpunetlib import tutils
from scripts.dumpunetlib.colorizer import Colorizer
from scripts.dumpunetlib.features.featureinfo import FeatureInfo, MultiImageFeatures

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
    add_average: bool,
    color: Colorizer
):
    tensor = feature
    if isinstance(feature, FeatureInfo):
        tensor = feature.output
    assert isinstance(tensor, Tensor)
    
    canvases = tutils.tensor_to_grid_images(tensor, layer, width, height, color, add_average)
    return canvases

def save_features(
    feature: FeatureInfo,
    save_dir: str,
    basename: str
):
    tutils.save_tensor(feature.output, save_dir, basename)
