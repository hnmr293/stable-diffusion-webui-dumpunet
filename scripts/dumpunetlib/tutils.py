import os
import math

import torch
from torch import Tensor
import numpy as np
from PIL import Image

from modules import shared

from scripts.dumpunetlib import layerinfo
from scripts.dumpunetlib.report import message as E
from scripts.dumpunetlib.colorizer import Colorizer

def tensor_to_grid_images(
    tensor: Tensor,
    layer: str,
    width: int,
    height: int,
    color: Colorizer,
    average_type: str|None = None,
):
    grid_x, grid_y = get_grid_num(layer, width, height)
    canvases = tensor_to_image(tensor, grid_x, grid_y, color, average_type)
    return canvases

def tensor_to_image(
            tensor: Tensor,
            grid_x: int,
            grid_y: int,
            color: Colorizer,
            average_type: str|None = None,
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
    
    avg_img = tensor_to_averaged_image(tensor, average_type, color)
    if avg_img is not None:
        canvases.append(avg_img)
    
    for chs in each_slice(range(max_ch), grid_x * grid_y):
        chs = list(chs)
        
        canvas = Image.new(color.format, (width, height), color=0)
        for iy in range(grid_y):
            if len(chs) == 0:
                break
            
            for ix in range(grid_x):
                if shared.state.interrupted:
                    break
                
                if len(chs) == 0:
                    break
                
                ch = chs.pop(0)
                image = tensor2d_to_image(tensor[ch], color)
                
                # create image
                x = (iw+1) * ix
                y = (ih+1) * iy
                canvas.paste(image, (x, y))
        
        canvases.append(canvas)
    return canvases

def tensor2d_to_image(
    tensor: Tensor,
    color: Colorizer,
):
    assert len(tensor.shape) == 2, f"tensor.shape = {tensor.shape}"
    array = tensor.cpu().numpy().astype(np.float32)
    return Image.fromarray(color(array), color.format)

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
    

def get_grid_num(layer: str, width: int, height: int):
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

def averaged_tensor(
    tensor: Tensor,
    average_type: str|None,
):
    average_type = (
        '' if average_type is None
        else average_type.lower()
    )
    
    avg = None
    
    if len(average_type) != 0:
        if average_type == 'sum':
            avg = torch.mean(tensor, 0) # tensor.shape: (ch, h, w) -> (h, w)
        elif average_type == '1-norm':
            avg = torch.linalg.vector_norm(tensor, dim=0, ord=1) / tensor.shape[0]
        elif average_type == '2-norm':
            avg = torch.linalg.vector_norm(tensor, dim=0, ord=2) / tensor.shape[0]

    return avg
    
def tensor_to_averaged_image(
    tensor: Tensor,
    average_type: str|None,
    color: Colorizer
):
    avg = averaged_tensor(tensor, average_type)
    
    if avg is not None:
        avg_img = tensor2d_to_image(avg, color)
        return avg_img
    else:
        return None
