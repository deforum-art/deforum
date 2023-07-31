import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Union

import numpy as np
import PIL
import torch
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)
from pydantic import Field

from . import DefaultBase


class SchedulerType(Enum):
    EULER_ANCESTRAL = "euler_ancestral"
    EULER = "euler"
    PNDM = "pndm"
    DPMPP_SINGLESTEP = "dpmpp_singlestep"
    DPMPP_MULTISTEP = "dpmpp_multistep"
    LMS = "lms"
    DDIM = "ddim"
    UNIPC = "unipc"
    SDE = "sde"

    def to_scheduler(self):
        opts = {
            self.EULER_ANCESTRAL.value: EulerAncestralDiscreteScheduler,
            self.EULER.value: EulerDiscreteScheduler,
            self.DDIM.value: DDIMScheduler,
            self.PNDM.value: PNDMScheduler,
            self.DPMPP_MULTISTEP.value: DPMSolverMultistepScheduler,
            self.DPMPP_SINGLESTEP.value: DPMSolverSinglestepScheduler,
            self.LMS.value: LMSDiscreteScheduler,
            self.UNIPC.value: UniPCMultistepScheduler,
            self.SDE.value: DPMSolverSDEScheduler,
        }
        return opts.get(self.value, EulerAncestralDiscreteScheduler)


class GenerationArgs(DefaultBase):
    prompt: Union[str, List[str]]
    negative_prompt: Optional[Union[str, List[str]]] = None
    image: Union[torch.FloatTensor, PIL.Image.Image] = None
    mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None
    height: int = 512
    width: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    strength: float = 0.8
    num_images_per_prompt: Optional[int] = 1
    add_predicted_noise: Optional[bool] = False
    eta: float = 0.0
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
    latents: Optional[torch.FloatTensor] = None
    prompt_embeds: Optional[torch.FloatTensor] = None
    negative_prompt_embeds: Optional[torch.FloatTensor] = None
    max_embeddings_multiples: Optional[int] = 3
    output_type: Optional[str] = "pt"
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None
    is_cancelled_callback: Optional[Callable[[], bool]] = None
    callback_steps: int = 1
    cross_attention_kwargs: Optional[Dict[str, Any]] = None
    clip_skip: Optional[int] = None
    sampler: Optional[SchedulerType] = SchedulerType.EULER_ANCESTRAL
    seed: Optional[int] = Field(default_factory=lambda: np.random.randint(0, (2**16) - 1))
    start_time: Optional[float] = Field(default_factory=lambda: datetime.datetime.now().timestamp())
    repeat: Optional[int] = 1
    seed_mode: Optional[Literal["random", "iter", "constant", "ladder"]] = "iter"
    seed_list: Optional[List[int]] = None
    save_intermediates: Optional[bool] = True
    template_save_path: Optional[str] = "samples/$prompt/$timestr/$custom_$index"

    def to_kwargs(
        self,
        exclude: Set[str] = {"output_type"},
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> Dict[str, Any]:
        if self.seed is not None and self.generator is None:
            self.generator = torch.Generator(device=device).manual_seed(self.seed)
        return self.dict(
            exclude={
                "start_time",
                "seed",
                "sampler",
                "repeat",
                "seed_mode",
                "template_save_path",
                "save_intermediates",
            }.union(exclude)
        )
