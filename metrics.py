from collections import deque
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LossSpikeTracker:
    window: int = 100
    std_factor: float = 3.0

    def __post_init__(self):
        self.values = deque(maxlen=self.window)
        self.count = 0
        self.max_amplitude = 0.0

    def update(self, loss_value: float) -> tuple[bool, float]:
        if len(self.values) < max(10, self.window // 5):
            self.values.append(loss_value)
            return False, 0.0
        vals = torch.tensor(list(self.values), dtype=torch.float32)
        threshold = vals.mean().item() + self.std_factor * vals.std(unbiased=False).item()
        amplitude = max(0.0, loss_value - threshold)
        is_spike = amplitude > 0.0
        if is_spike:
            self.count += 1
            self.max_amplitude = max(self.max_amplitude, amplitude)
        self.values.append(loss_value)
        return is_spike, amplitude


def reduce_aux(aux: dict[str, Any]) -> dict[str, float]:
    out = {}
    for key, value in aux.items():
        if isinstance(value, list):
            nums = []
            for item in value:
                if item is None:
                    continue
                if torch.is_tensor(item):
                    nums.append(float(item.detach().float().mean().item()))
                else:
                    nums.append(float(item))
            if nums:
                out[key] = sum(nums) / len(nums)
        elif torch.is_tensor(value):
            out[key] = float(value.detach().float().mean().item())
        elif value is not None:
            out[key] = float(value)
    return out


def tokens_per_second(tokens_per_iter: int, dt: float) -> float:
    if dt <= 0:
        return 0.0
    return tokens_per_iter / dt

