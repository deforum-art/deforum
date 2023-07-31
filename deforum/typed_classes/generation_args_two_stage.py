from typing import Optional
from .generation_args import GenerationArgs, SchedulerType


class GenerationArgsTwoStage(GenerationArgs):
    prompt_stage2: Optional[str] = None
    negative_prompt_stage2: Optional[str] = None
    strength_stage2: Optional[float] = None
    guidance_stage2: Optional[float] = None
    sampler_stage2: Optional[SchedulerType] = None
    num_inference_steps_stage2: Optional[int] = None
    eta_stage2: Optional[float] = None
