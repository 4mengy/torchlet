import uuid
import asyncio
from dataclasses import dataclass, field
from enum import Enum


class RequestState(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Request:
    prompt: str
    input_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    output: str = ""
    error: Exception | None = None
    max_new_tokens: int = 128
    req_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    computed_tokens: int = 0
    state: RequestState = RequestState.WAITING
    done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
