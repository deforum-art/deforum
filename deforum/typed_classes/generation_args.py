import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Union

import PIL
import torch
from pydantic import Field

from . import DefaultBase


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
    output_type: Optional[str] = "pil"
    return_dict: bool = True
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None
    is_cancelled_callback: Optional[Callable[[], bool]] = None
    callback_steps: int = 1
    cross_attention_kwargs: Optional[Dict[str, Any]] = None
    clip_skip: Optional[int] = None
    seed: Optional[int] = None
    start_time: Optional[float] = Field(default_factory=lambda: datetime.datetime.now().timestamp())

    def to_kwargs(
        self,
        exclude: Set[str] = {"output_type"},
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> Dict[str, Any]:
        if self.seed is not None and self.generator is None:
            self.generator = torch.Generator(device=device).manual_seed(self.seed)
        return self.dict(exclude={"start_time", "seed"}.union(exclude))
