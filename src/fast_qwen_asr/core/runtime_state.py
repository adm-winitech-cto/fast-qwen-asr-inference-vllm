from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeState:
    max_concurrent_decode: int
    max_concurrent_infer: int
    models: dict[str, Any] = field(default_factory=dict)
    model_status: str = "starting"
    model_ready_event: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self.decode_sem = asyncio.Semaphore(self.max_concurrent_decode)
        self.infer_sem = asyncio.Semaphore(self.max_concurrent_infer)

    def has_model(self, model_name: str) -> bool:
        return model_name in self.models

    @property
    def decode_limit(self) -> int:
        return self.max_concurrent_decode

    @property
    def infer_limit(self) -> int:
        return self.max_concurrent_infer
