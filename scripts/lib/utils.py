from typing import TypeVar, Generator, Tuple
from collections import defaultdict

K = TypeVar('K')
V = TypeVar('V')

def sorted_items(obj: dict[K,V]|defaultdict[K,V]) -> Generator[Tuple[K,V],None,None]:
    for k in sorted_keys(obj):
        yield k, obj[k]

def sorted_keys(obj: dict[K,V]|defaultdict[K,V]) -> Generator[K,None,None]:
    for k in sorted(obj.keys()): # type: ignore
        yield k

def sorted_values(obj: dict[K,V]|defaultdict[K,V]) -> Generator[V,None,None]:
    for k in sorted_keys(obj):
        yield obj[k]
