import os

from torch import Tensor
import numpy as np
from PIL import Image

from modules import shared

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
