from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, NamedTuple


class Split(NamedTuple):
    name: str
    value: float


@dataclass
class Timer:
    clock: Callable[[], float] = perf_counter
    splits: Dict[str, List[float]] = field(default_factory=dict)
    base_time: float = float('-inf')
    # splits: List[Split] = field(default_factory=list)

    def start(self):
        self.base_time = self.clock()

    def split(self, name: str = ''):
        splits = self.splits.setdefault(name, [])
        splits.append(self.clock())

    def get(self, name: str = '') -> List[float]:
        return self.splits.get(name, ())

    def get_times(self) -> Dict[str, List[float]]:
        results = {name: None for name in self.splits}
        for name in results:
            splits = self.splits[name]
            results[name] = [v - splits[0] for v in splits] if splits else ()
        return results
