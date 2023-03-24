import re
from dataclasses import dataclass, fields
from abc import abstractmethod
from typing import Generator as tGenerator

from scripts.dumpunetlib import layerinfo
from scripts.dumpunetlib.layer_prompt.parser import parse, bad_prompt

@dataclass
class LayerPrompts:
    IN00 : str
    IN01 : str
    IN02 : str
    IN03 : str
    IN04 : str
    IN05 : str
    IN06 : str
    IN07 : str
    IN08 : str
    IN09 : str
    IN10 : str
    IN11 : str
    M00 : str
    OUT00 : str
    OUT01 : str
    OUT02 : str
    OUT03 : str
    OUT04 : str
    OUT05 : str
    OUT06 : str
    OUT07 : str
    OUT08 : str
    OUT09 : str
    OUT10 : str
    OUT11 : str
    
    def __len__(self):
        return len(fields(self))
    
    def __iter__(self):
        for fld in fields(self):
            yield fld.name
    
    def keys(self) -> tGenerator[str,None,None]:
        for fld in fields(self):
            yield fld.name
    
    def values(self) -> tGenerator[str,None,None]:
        for fld in fields(self):
            yield getattr(self, fld.name)
    
    def items(self) -> tGenerator[tuple[str,str],None,None]:
        for fld in fields(self):
            yield fld.name, getattr(self, fld.name)

re_whole = re.compile(r"\(~\:((?s:.)+?)\:~\)")

class Generator:
    
    @abstractmethod
    def generate(self, prompt: str) -> LayerPrompts:
        return NotImplemented

class LayerPromptGenerator(Generator):
    
    def __init__(self):
        pass
    
    def generate(self, prompt: str):
        saved = prompt
        
        p = LayerPromptParser()
        
        pos = 0
        m = re_whole.search(prompt, pos)
        while m:
            before = prompt[:m.start()]
            content = m.group(1)
            pos = m.end()
            
            p.add(before)
            p.parse_and_add(content)

            m = re_whole.search(prompt, pos)
        
        p.add(prompt[pos:])
        
        result = []
        for idx, layer in enumerate(layerinfo.Names):
            result.append(p.apply(idx))
        
        return LayerPrompts(*result)

class LayerPromptEraseGenerator(Generator):
    
    def __init__(self):
        pass
    
    def generate(self, prompt: str):
        p = re.sub(re_whole, "", prompt)
        return LayerPrompts(*([p] * len(layerinfo.Names)))

# Parser

class LayerPromptParser:
    
    def __init__(self):
        self.buffer: list[str|list[str]] = []
    
    def add(self, s: str):
        self.buffer.append(s)
    
    def parse_and_add(self, s: str):
        contents = self._parse(s)
        self.buffer.append(contents)
    
    def apply(self, n: int) -> str:
        buf = []
        for s in self.buffer:
            if isinstance(s, str):
                buf.append(s)
            else:
                buf.append(s[n])
        return "".join(buf)
    
    def _parse(self, s: str) -> list[str]:
        contents: list[str|None] = [None] * len(layerinfo.Names)
        rest: str = ""
        
        conf = parse(s)
        for layers, content in conf:
            for l_conf in layers:
                left, right = l_conf
                if isinstance(left, tuple):
                    # range
                    l_idx = layerinfo.index(left[1])
                    r_idx = layerinfo.index(right[1])
                    if l_idx is None: bad_prompt(s, left[0], f"Unknown layer name: {left[1]}")
                    if r_idx is None: bad_prompt(s, right[0], f"Unknown layer name: {right[1]}")
                    if l_idx > r_idx:
                        i_idx, r_idx = r_idx, l_idx
                    for idx in range(l_idx, r_idx+1):
                        contents[idx] = content[1]
                elif l_conf[1] == "_":
                    rest = content[1]
                else:
                    idx = layerinfo.index(str(right))
                    if idx is None: bad_prompt(s, left, f"Unknown layer name: {right}")
                    contents[idx] = content[1]
        
        return list(map(lambda x: rest if x is None else x, contents))
        
    
"""
a cute school girl
a (~:IN00-IN02,OUT04:cute:~) school girl
a (~:
     IN00-IN02,OUT04: cute;
     IN03-IN05:       beautiful;
     _:               excellent;
   :~) school girl

"(~:" { layer ":" content ";" }+ layer ":" content ";"? ":~)"

layer   : layer1
        | layer1 "," layer

layer1  : layer2
        | layer2 "-" layer2

layer2  : "IN00" | "IN01" | "IN02" | "IN03" | "IN04" | "IN05" | "IN06" | "IN07" | "IN08" | "IN09" | "IN10" | "IN11"
        | "M00"
        | "OUT00" | "OUT01" | "OUT02" | "OUT03" | "OUT04" | "OUT05" | "OUT06" | "OUT07" | "OUT08" | "OUT09" | "OUT10" | "OUT11"

content : ch +

ch  : character, except for ";"
      to use ";", put "\;"
      to use "\", put "\\"
"""

"""
# https://stackoverflow.com/questions/1751949/python-equivalent-of-rubys-stringscanner
import re

def s_ident(scanner, token): return token
def s_operator(scanner, token): return "op%s" % token
def s_float(scanner, token): return float(token)
def s_int(scanner, token): return int(token)

scanner = re.Scanner([
    (r"[a-zA-Z_]\w*", s_ident),
    (r"\d+\.\d*", s_float),
    (r"\d+", s_int),
    (r"=|\+|-|\*|/", s_operator),
    (r"\s+", None),
    ])

print scanner.scan("sum = 3*foo + 312.50 + bar")
"""
