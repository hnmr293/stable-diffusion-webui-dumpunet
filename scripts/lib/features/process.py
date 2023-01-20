import os
import math
from typing import Generator

from torch import Tensor
import numpy as np
from PIL import Image

from modules import shared

from scripts.lib import layerinfo
from scripts.lib.features.featureinfo import FeatureInfo, Features, MultiImageFeatures
from scripts.lib.report import message as E

def feature_diff(
    features1: MultiImageFeatures,
    features2: MultiImageFeatures,
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
    feature: FeatureInfo,
    layer: str,
    width: int,
    height: int,
    color: bool
):
    return tensor_to_grid_images(feature.output, layer, width, height, color)

def tensor_to_grid_images(
    tensor: Tensor,
    layer: str,
    width: int,
    height: int,
    color: bool
):
    assert len(tensor.size()) == 3
    
    grid_x, grid_y = _get_grid_num(layer, width, height)
    canvases = tensor_to_image(tensor, grid_x, grid_y, color)
    
    return canvases
    

def save_features(
    feature: FeatureInfo,
    save_dir: str,
    basename: str
):
    save_tensor(feature.output, save_dir, basename)

def tensor_to_image(
            tensor: Tensor,
            grid_x: int,
            grid_y: int,
            color: bool,
):
    # Regardless of wheather --opt-channelslast is enabled or not, 
    # feature.size() seems to return (batch, ch, h, w).
    # Is this intended behaviour???
    
    assert len(tensor.size()) == 3
    
    max_ch, ih, iw = tensor.size()
    width = (grid_x * (iw + 1) - 1)
    height = (grid_y * (ih + 1) - 1)
    
    def each_slice(it: range, n: int):
        cur = []
        for x in it:
            cur.append(x)
            if n == len(cur):
                yield cur
                cur = []
        if 0 < len(cur):
            yield cur
    
    canvases: list[Image.Image] = []
    color_format = "RGB" if color else "L"
    
    for chs in each_slice(range(max_ch), grid_x * grid_y):
        chs = list(chs)
        
        canvas = Image.new(color_format, (width, height), color=0)
        for iy in range(grid_y):
            if len(chs) == 0:
                break
            
            for ix in range(grid_x):
                if shared.state.interrupted:
                    break
                
                if len(chs) == 0:
                    break
                
                ch = chs.pop(0)
                array = tensor[ch].cpu().numpy().astype(np.float32)
                
                # create image
                x = (iw+1) * ix
                y = (ih+1) * iy
                image = _tensor_to_image(array, color)
                canvas.paste(Image.fromarray(image, color_format), (x, y))
        
        canvases.append(canvas)
    return canvases

def save_tensor(
    tensor: Tensor,
    save_dir: str,
    basename: str
):
    assert len(tensor.size()) == 3
    for ch, t in enumerate(tensor):
        filename = basename.format(ch=ch)
        binpath = os.path.join(save_dir, filename + ".bin")
        with open(binpath, "wb") as io:
            array = t.cpu().numpy().astype(np.float32)
            io.write(bytearray(array))
    

def _tensor_to_image(array: np.ndarray, color: bool):
    # array := (-∞, ∞)
    
    if color:
        def colorize(v: float):
            # v = -1 .. 1 を
            # v < 0 のとき青 (0, 0, 1)
            # v > 0 のとき赤 (1 ,0, 0)
            # にする
            rgb = (v if v > 0.0 else 0.0, 0.0, -v if v < 0.0 else 0.0)
            return rgb
        colorize2 = np.vectorize(colorize, otypes=[np.float32, np.float32, np.float32])
        rgb = colorize2(np.clip(array, -1.0, 1.0))
        return np.clip((np.dstack(rgb) * 256), 0, 255).astype(np.uint8)
            
    else:
        return np.clip(np.abs(array) * 256, 0, 255).astype(np.uint8)

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
    
    return [n[0] // w, n[1] // h]
