"""
utilities for layer operation
"""

Names_IN : tuple[str, ...] = (
    *[f"IN{n:02}" for n in range(12)],
)

Names_M : tuple[str, ...] = (
    "M00",
)

Names_OUT : tuple[str, ...] = (
    *[f"OUT{n:02}" for n in range(12)],
)

Names : tuple[str, ...] = (
    #"IN@@",
    *Names_IN,
    *Names_M,
    *Names_OUT,
    #"OUT$$",
)

Settings : dict[str,tuple[tuple[int,int,int],tuple[int,int,int]]] = {
    #            input shape   output shape
    #"IN@@":   ( (   0, 0, 0), (   4, 8, 8) ),
    "IN00":   ( (   4, 8, 8), ( 320, 8, 8) ),
    "IN01":   ( ( 320, 8, 8), ( 320, 8, 8) ),
    "IN02":   ( ( 320, 8, 8), ( 320, 8, 8) ),
    "IN03":   ( ( 320, 8, 8), ( 320, 4, 4) ),
    "IN04":   ( ( 320, 4, 4), ( 640, 4, 4) ),
    "IN05":   ( ( 640, 4, 4), ( 640, 4, 4) ),
    "IN06":   ( ( 640, 4, 4), ( 640, 2, 2) ),
    "IN07":   ( ( 640, 2, 2), (1280, 2, 2) ),
    "IN08":   ( (1280, 2, 2), (1280, 2, 2) ),
    "IN09":   ( (1280, 2, 2), (1280, 1, 1) ),
    "IN10":   ( (1280, 1, 1), (1280, 1, 1) ),
    "IN11":   ( (1280, 1, 1), (1280, 1, 1) ),
    "M00":    ( (1280, 1, 1), (1280, 1, 1) ),
    "OUT00":  ( (2560, 1, 1), (1280, 1, 1) ),
    "OUT01":  ( (2560, 1, 1), (1280, 1, 1) ),
    "OUT02":  ( (2560, 1, 1), (1280, 2, 2) ),
    "OUT03":  ( (2560, 2, 2), (1280, 2, 2) ),
    "OUT04":  ( (2560, 2, 2), (1280, 2, 2) ),
    "OUT05":  ( (1920, 2, 2), (1280, 4, 4) ),
    "OUT06":  ( (1920, 4, 4), ( 640, 4, 4) ),
    "OUT07":  ( (1280, 4, 4), ( 640, 4, 4) ),
    "OUT08":  ( ( 960, 4, 4), ( 640, 8, 8) ),
    "OUT09":  ( ( 960, 8, 8), ( 320, 8, 8) ),
    "OUT10":  ( ( 640, 8, 8), ( 320, 8, 8) ),
    "OUT11":  ( ( 640, 8, 8), ( 320, 8, 8) ),
    #"OUT$$":  ( (   0, 0, 0), (   4, 8, 8) ),
}

assert set(Names) == set(Settings.keys()), f"Names={Names}, Settings={Settings.keys()}"

def index(layername: str) -> int|None:
    return _index(layername, Names)

def input_index(layername: str) -> int|None:
    return _index(layername, Names_IN)

def middle_index(layername: str) -> int|None:
    return _index(layername, Names_M)

def output_index(layername: str) -> int|None:
    return _index(layername, Names_OUT)

def name(index: int) -> str|None:
    if index < len(Names_IN):
        return Names_IN[index]
    
    index -= len(Names_IN)
    if index < len(Names_M):
        return Names_M[index]
    
    index -= len(Names_M)
    if index < len(Names_OUT):
        return Names_OUT[index]
    
    return None

def _index(layername: str, layers: tuple[str,...]) -> int|None:
    if layername in layers:
        return layers.index(layername)
    else:
        return None
