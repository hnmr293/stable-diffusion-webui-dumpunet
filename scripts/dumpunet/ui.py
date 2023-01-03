import re

from scripts.dumpunet import layerinfo
from scripts.dumpunet.report import message as E

#re_layer_pat = r"((?:IN|OUT)\d\d|M00|IN@@|OUT\$\$)"
re_layer_pat = r"((?:IN|OUT)\d\d|M00)"
re_layer = re.compile(rf"^\s*{re_layer_pat}\s*$")
re_layer_range = re.compile(rf"^\s*{re_layer_pat}\s*-\s*{re_layer_pat}\s*(?:\(\s*\+?\s*(\d+)\s*\))?\s*$")

def retrieve_layers(input: str) -> list[str]:
    if input is None or input == "":
        # all layers
        return list(layerinfo.Names)
    
    def index(name: str):
        v = layerinfo.index(name)
        if v is None:
            raise ValueError(E(f"Invalid layer name: {name}"))
        return v
    
    result : list[int]|None = []
    tokens = input.split(",")
    for token in tokens:
        if token == "":
            continue
        m1 = re_layer.fullmatch(token)
        m2 = re_layer_range.fullmatch(token)
        if m1:
            result.append(index(m1.group(1)))
        elif m2:
            lay1 = index(m2.group(1))
            lay2 = index(m2.group(2))
            step = eval(m2.group(3)) if m2.group(3) else 1
            result.extend(range(lay1, lay2+1, step))
        else:
            raise ValueError(E(f"Invalid layer name: {token}"))
        
    result = list(set(result))
    if len(result) == 0:
        return list(layerinfo.Names)
    else:
        return [layerinfo.Names[n] for n in sorted(result)]

re_num = re.compile(r"^\s*\+?\s*\d+\s*$")
re_range = re.compile(r"^\s*(\+?\s*\d+)\s*-\s*(\+?\s*\d+)\s*(?:\(\s*\+?\s*(\d+)\s*\))?\s*$")

def retrieve_steps(input: str):
    if input is None or input == "":
        return None
    
    steps : list[int]|None = []
    tokens = input.split(",")
    for token in tokens:
        if token == "":
            continue
        m1 = re_num.fullmatch(token)
        m2 = re_range.fullmatch(token)
        if m1:
            steps1 = eval("[" + m1.group(0) + "]")
        elif m2:
            n1 = eval(m2.group(1))
            n2 = eval(m2.group(2))
            n3 = eval(m2.group(3)) if m2.group(3) else 1
            steps1 = list(range(n1, n2+1, n3))
        else:
            raise ValueError(E(f"Invalid input for <Image saving steps>: {token}."))
        steps.extend(steps1)
    
    steps = list(set(steps))
    if len(steps) == 0:
        steps = None # all steps
    else:
        steps.sort()
    
    return steps
