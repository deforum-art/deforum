from typing import Optional
from .generation_args import GenerationArgs, SchedulerType


class GenerationArgsAnimation(GenerationArgs):
    max_frames: int = 25
