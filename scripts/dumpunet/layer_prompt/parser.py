import re

try:
    from scripts.dumpunet.report import message as E
except:
    # for debug
    def E(msg: str, title=""): return msg


class BadPromptError(ValueError):
    prompt: str
    pos: int      # 0-origin
    line: int     # 0-origin
    linepos: int  # 0-origin
    msg: str|None
    
    def __init__(self, prompt: str, pos: int, msg: str|None) -> None:
        self.prompt = prompt
        self.pos = pos
        self.msg = msg
        
        title = E("Syntax Error at line={0}, pos={1}")
        
        before = prompt[:pos]
        line = before.count("\n")
        if 0 < line:
            linepos = pos - before.rindex("\n") - 1
        else:
            linepos = pos
        
        self.line = line
        self.linepos = linepos
        
        cur_line = prompt.splitlines()[line]
        show_range = (linepos - 20, linepos + 20)
        leading = " ... "
        trailing = " ... "
        if show_range[0] < 0:
            show_range = (0, 40)
            leading = ""
        if len(cur_line) <= show_range[1]:
            show_range = (show_range[0], len(cur_line))
            trailing = ""
        
        title = title.format(line + 1, pos)
        
        self._message = f"""\
{title}
    {leading}{cur_line[show_range[0]:show_range[1]]}{trailing}
    {" " * (len(leading) + linepos)}^~~~
{msg}
"""
        super().__init__(self._message)
    
    def message(self):
        return self._message


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
    
    def fail(msg: str|None):
        bad_prompt(s, pos, msg)
    
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
    
    def need(re, group=0, ex:str|None=None) -> tuple[int,str]:
        m = M(re, group)
        if m is None: fail(f"expecting {ex}, but not." if ex else None)
        return m
    
    def parse_layer2() -> tuple[int,str]:
        # IN00 | IN01 | ...
        return need(re_layer2, ex="a layer name (IN00, IN01, ...)")
    
    def parse_layer1() -> tuple[tuple[int,str],tuple[int,str]]|tuple[int,str]:
        # layer2 (- layer2)?
        left = parse_layer2()
        if left is None: fail("a layer name (IN00, IN01, ...)")
        if not skip(re_hyphen):
            return left
        right = parse_layer2()
        if right is None: fail("a layer name (IN00, IN01, ...)")
        return (left, right)
    
    def parse_layer() -> list[tuple[tuple[int,str],tuple[int,str]]|tuple[int,str]]:
        # layer1 (, layer)*
        left = parse_layer1()
        if left is None: fail("a layer name (IN00, IN01, ...)")
        if not skip(re_comma):
            return [left]
        right = parse_layer()
        if right:
            right.append(left)
            return right
        else:
            fail("a layer name (IN00, IN01, ...)")
    
    def parse_content() -> tuple[int,str]:
        return need(re_until_colon_or_end, 1, ex="colon (:) or EOS ($)")
    
    def parse_pair() -> tuple[list[tuple[tuple[int,str],tuple[int,str]]|tuple[int,str]], tuple[int,str]]:
        skip(re_space)
        any = M(re_under_colon)
        if any is not None:
            layers: list[tuple[tuple[int,str],tuple[int,str]]|tuple[int,str]] = [(any[0],"_")]
        else:
            layers = parse_layer()
            layers.reverse()
            need(re_colon, ex="colon (:)")
        content = parse_content()
        skip(re_space)
        return layers, content
    
    result: list[tuple[list[tuple[tuple[int,str],tuple[int,str]]|tuple[int,str]], tuple[int,str]]]
    result = []
    while pos < len(s):
        layers, content = parse_pair()
        result.append((layers, content))
    
    return result

def bad_prompt(s: str, pos: int, msg: str|None = None):
    raise BadPromptError(s, pos, msg)
