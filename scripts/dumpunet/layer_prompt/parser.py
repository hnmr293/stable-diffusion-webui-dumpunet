import re

try:
    from scripts.dumpunet.report import message as E
except:
    # for debug
    def E(msg: str, title=""): return msg


# Scanner

re_layer2 = re.compile("|".join(
    [ f"IN{n:02}" for n in range(12) ] +
    [ "M00" ] +
    [ f"OUT{n:02}" for n in range(12) ]
))
re_space = re.compile(r"\s+")
re_comma = re.compile(r"\s*,\s*")
re_hyphen = re.compile(r"\s*-\s*")
re_under_colon = re.compile(r"\s*_\s*:\s*")
re_colon = re.compile(r"\s*(?<!\\)(?:\\\\)*:")
re_until_colon_or_end = re.compile(r"(.*?)(?:(?<!\\)(?:\\\\)*:|$)")

def parse(s: str):
    pos = 0
    
    def fail():
        bad_prompt(s, pos)
    
    def M(re: re.Pattern, group=0) -> tuple[int,str]|None:
        nonlocal pos
        match = re.match(s, pos=pos)
        if match:
            pos = match.end()
            return match.start(), match.group(group)
        else:
            return None
    
    def skip(re, group=0) -> bool:
        return False if M(re) is None else True
    
    def need(re, group=0) -> tuple[int,str]:
        m = M(re, group)
        if m is None: fail()
        return m
    
    def parse_layer2() -> tuple[int,str]:
        # IN00 | IN01 | ...
        return need(re_layer2)
    
    def parse_layer1() -> tuple[tuple[int,str],tuple[int,str]]|tuple[int,str]:
        # layer2 (- layer2)?
        left = parse_layer2()
        if left is None: fail()
        if not skip(re_hyphen):
            return left
        right = parse_layer2()
        if right is None: fail()
        return (left, right)
    
    def parse_layer() -> list[tuple[tuple[int,str],tuple[int,str]]|tuple[int,str]]:
        # layer1 (, layer)*
        left = parse_layer1()
        if left is None: fail()
        if not skip(re_comma):
            return [left]
        right = parse_layer()
        if right:
            right.append(left)
            return right
        else:
            fail()
    
    def parse_content() -> tuple[int,str]:
        return need(re_until_colon_or_end, 1)
    
    def parse_pair() -> tuple[list[tuple[tuple[int,str],tuple[int,str]]|tuple[int,str]], tuple[int,str]]:
        skip(re_space)
        any = M(re_under_colon)
        if any is not None:
            layers: list[tuple[tuple[int,str],tuple[int,str]]|tuple[int,str]] = [(any[0],"_")]
        else:
            layers = parse_layer()
            layers.reverse()
            need(re_colon)
        content = parse_content()
        skip(re_space)
        return layers, content
    
    result: list[tuple[list[tuple[tuple[int,str],tuple[int,str]]|tuple[int,str]], tuple[int,str]]]
    result = []
    while pos < len(s):
        layers, content = parse_pair()
        result.append((layers, content))
    
    return result

def bad_prompt(s: str, pos):
    raise ValueError(E(f"Invalid prompt at pos={pos}: {s}"))
