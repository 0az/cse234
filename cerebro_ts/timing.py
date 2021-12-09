from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List


@dataclass
class Timer:
    clock: Callable[[], float] = perf_counter
    splits: Dict[str, List[float]] = field(default_factory=dict)
    base_time: float = float('-inf')

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

    def print_splits(self, print=print) -> None:
        for name, splits in self.get_times().items():
            if splits:
                print(f'{name}: {splits[-1]:.3f}s')

