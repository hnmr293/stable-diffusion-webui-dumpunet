from typing import Callable
import colorsys
import numpy as np
import numpy.typing as npt

"""
colorize    transform    fn (W/B: -∞..+∞ -> 0..1, R/B: -∞..+∞ -> -1..1)
---------------------------------------------------------------------------------------------
White/Black Linear       v -> {|clamp(v, min, max)|-X}/(Y-X)
                                where
                                    X = 0                if sign(min) != sign(max)
                                        MIN(|min|,|max|) otherwise
                                    Y = MAX(|min|,|max|)
            Sigmoid      v -> |sigmoid(v+offset, gain)-0.5|*2
Red/Blue    Linear       v -> [{clamp(v, min, max)-min}/(max-min)-0.5]*2
            Sigmoid      v -> {sigmoid(v+offset, gain)-0.5}*2
"""

def create_convert_linear_abs(min_: float, max_: float):
    assert min_ < max_
    X = 0.0 if np.sign(min_) != np.sign(max_) else \
        min(np.abs(min_), np.abs(max_))
    Y = max(np.abs(min_), np.abs(max_))
    
    #@np.vectorize
    def fn(array: npt.NDArray[np.float32]):
        vs = (np.abs(np.clip(array, min_, max_)) - X) / (Y - X)
        assert np.all(((0.0 <= vs) & (vs <= 1.0)) | np.isnan(vs))
        return vs
    return fn

def create_convert_linear(min: float, max: float):
    assert min < max
    #@np.vectorize
    def fn(array: npt.NDArray[np.float32]):
        vs = ((np.clip(array, min, max) - min) / (max - min) - 0.5) * 2.0
        assert np.all(((-1.0 <= vs) & (vs <= 1.0)) | np.isnan(vs))
        return vs
    return fn

def create_convert_sigmoid_abs(gain: float, offset: float):
    #@np.vectorize
    def fn(array: npt.NDArray[np.float32]):
        vs = 1.0 / (1.0 + np.exp(-gain * (array+offset)))
        vs = np.abs(vs - 0.5) * 2.0
        assert np.all(((0.0 <= vs) & (vs <= 1.0)) | np.isnan(vs))
        return vs
    return fn

def create_convert_sigmoid(gain: float, offset: float):
    #@np.vectorize
    def fn(array: npt.NDArray[np.float32]):
        vs = 1.0 / (1.0 + np.exp(-gain * (array+offset)))
        vs = (vs - 0.5) * 2.0
        assert np.all(((-1.0 <= vs) & (vs <= 1.0)) | np.isnan(vs))
        return vs
    return fn

class Colorizer:
    
    # v -> v
    convert: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]|None
    
    # v -> pixel
    colorize: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]|None
    
    format: str
    
    def __init__(
        self,
        value1: str,
        value2: str,
        rgb: tuple[str,str,str],
        hsl: tuple[str,str,str],
        trans: str,
        linear_minmax: tuple[float,float],
        sigmoid_gain_offset: tuple[float,float]
    ):
        assert value1 in ["White/Black", "Red/Blue", "Custom"]
        if value1 == "White/Black":
            self.format = "L"
            colorize = None
        elif value1 == "Red/Blue":
            self.format = "RGB"
            colorize = create_colorizer(colorize_red_blue_v)
        else:
            assert value2 in ["RGB", "HSL"]
            if value2 == "RGB":
                r, g, b = rgb
                fn = eval(f"lambda v: ( ({r}), ({g}), ({b}) )", { "__builtins__": np }, {})
                fv = np.vectorize(fn, otypes=[np.float32, np.float32, np.float32])
                self.format ="RGB"
            else:
                h, s, l = hsl
                fn = eval(f"lambda v: HLS2RGB( ({h}), ({l}), ({s}) )", { "__builtins__": np, "HLS2RGB": colorsys.hls_to_rgb }, {})
                fv = np.vectorize(fn, otypes=[np.float32, np.float32, np.float32])
                self.format = "RGB"
            colorize = create_colorizer(fv)
        
        assert trans in ["Linear", "Sigmoid"]
        if trans == "Linear":
            if value1 == "White/Black":
                convert = create_convert_linear_abs(*linear_minmax)
            else:
                convert = create_convert_linear(*linear_minmax)
        else:
            if value1 == "White/Black":
                convert = create_convert_sigmoid_abs(*sigmoid_gain_offset)
            else:
                convert = create_convert_sigmoid(*sigmoid_gain_offset)
        
        self.convert = convert
        self.colorize = colorize
    
    def __call__(self, array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if self.convert is not None:
            array = self.convert(array)
        if self.colorize is not None:
            array = self.colorize(array)
        
        #assert np.all(((0.0 <= array) & (array <= 1.0)) | np.isnan(array))
        
        return np.clip(array * 256, 0.0, 255.0).astype(np.uint8)

def colorize_red_blue_(v: float) -> tuple[float, float, float]:
    # v = -∞..+∞ を
    # v < 0 のとき青 (0, 0, v)
    # v > 0 のとき赤 (v ,0, 0)
    # にする
    L = (v if v > 0.0 else 0.0, 0.0, np.abs(v) if v < 0.0 else 0.0)
    return L

colorize_red_blue_v = np.vectorize(colorize_red_blue_, otypes=[np.float32, np.float32, np.float32])

def create_colorizer(fn: np.vectorize):
    def colorizer(array: npt.NDArray[np.float32]):
        return np.dstack(fn(array))
    return colorizer
