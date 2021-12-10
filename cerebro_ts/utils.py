from __future__ import annotations

import datetime
from typing import List


def timestamp():
    return datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')


def positive_int(s: str) -> int:
    i = int(s)
    if i <= 0:
        raise ValueError(f'Expected positive integer, got {s}')
    return i


def comma_separated_positive_ints(s: str) -> List[int]:
    l = [int(i.strip()) for i in s.split(',')]
    if not l or any(i <= 0 for i in l):
        raise ValueError(f'Expected positive integers, got {s}')
    return l
