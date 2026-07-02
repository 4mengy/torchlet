import uuid
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
    req_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    computed_tokens: int = 0
    state: RequestState = RequestState.WAITING
