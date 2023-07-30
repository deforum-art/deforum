from typing import Any, Dict, Optional, Set
from .generation_args import GenerationArgs, SchedulerType


class GenerationArgsTwoStage(GenerationArgs):
    prompt_stage2: Optional[str] = None
    negative_prompt_stage2: Optional[str] = None
    strength_stage2: Optional[float] = 0.21
    guidance_stage2: Optional[float] = 7.5
    sampler_stage2: Optional[SchedulerType] = None
    num_inference_steps_stage2: Optional[int] = 50
    eta_stage2: Optional[float] = 0.0
