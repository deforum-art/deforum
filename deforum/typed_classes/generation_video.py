from typing import List

from . import GenerationArgs


class GenerationVideo(GenerationArgs):
    prompt_schedule: List[str]
    negative_prompt_schedule: List[str]
    max_frames: int = 15
    fps: int = 30
    math_stuff: int = 0

    def prepare_schedules(self):
        print("prepare_schedules")
