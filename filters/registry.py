"""
registry.py - Filter Registry (PRD Section 11.2).

Single source of truth for all 10 visual filters. Order:
1. Noir
2. Vintage Film
3. Arctic Blue
4. Golden Hour
5. Cyberpunk Duotone
6. Sepia Classic
7. Soft Portrait
8. Cartoon Sketch
9. Cross-Process Pop
10. Infrared Dream
"""

from typing import List, Tuple, Callable
import numpy as np

from filters.noir import apply_noir
from filters.vintage_film import apply_vintage_film
from filters.arctic_blue import apply_arctic_blue
from filters.golden_hour import apply_golden_hour
from filters.cyberpunk_duotone import apply_cyberpunk_duotone
from filters.sepia_classic import apply_sepia_classic
from filters.soft_portrait import apply_soft_portrait
from filters.cartoon_sketch import apply_cartoon_sketch
from filters.cross_process_pop import apply_cross_process_pop
from filters.infrared_dream import apply_infrared_dream


FilterFunc = Callable[[np.ndarray], np.ndarray]

FILTER_REGISTRY: List[Tuple[str, FilterFunc]] = [
    ("1. Noir", apply_noir),
    ("2. Vintage Film", apply_vintage_film),
    ("3. Arctic Blue", apply_arctic_blue),
    ("4. Golden Hour", apply_golden_hour),
    ("5. Cyberpunk Duotone", apply_cyberpunk_duotone),
    ("6. Sepia Classic", apply_sepia_classic),
    ("7. Soft Portrait", apply_soft_portrait),
    ("8. Cartoon Sketch", apply_cartoon_sketch),
    ("9. Cross-Process Pop", apply_cross_process_pop),
    ("10. Infrared Dream", apply_infrared_dream),
]
