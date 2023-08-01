"""
Module for Defining Image Generation Parameters and Related Functions

This module provides the necessary classes and functions for defining and managing
the parameters of an image generation task. 
It contains the `GenerationArgs` class which is a model for the parameters used in image generating,
the `SchedulerType` class which defines enumerated types for scheduler, and their related methods.

The module has dependencies on several other libraries including torch, PIL, numpy, 
and the built-in datetime and enum libraries.

Attributes:
----------
SchedulerType : class
    Enumerated type for schedulers.

GenerationArgs : class
    Definition of the parameters for image generation.

Functions:
----------
to_scheduler() : function
    Maps the `SchedulerType` classes' enumerated types to the corresponding scheduler objects.

to_kwargs(device, exclude) : function
    Converts the GenerationArgs object to a dictionary for image generation.
"""
import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Union

import numpy as np
import torch
import PIL

from pydantic import Field
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
from . import DefaultBase


class SchedulerType(Enum):
    """
    An enumeration representing different scheduler types. 
    
    Enum Members
    ------------
    EULER_ANCESTRAL, EULER, PNDM, DPMPP_SINGLESTEP, DPMPP_MULTISTEP, 
    LMS, DDIM, UNIPC, SDE 
    
    Methods
    -------
    to_scheduler : 
        returns a dictionary mapping enum members to their respective scheduler classes
    """
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
        """
        Maps the enum's values to the corresponding scheduler objects.

        Returns
        -------
        object
            The scheduler object corresponding to the enum value.
        """
        return {
            self.EULER_ANCESTRAL.value: EulerAncestralDiscreteScheduler,
            self.EULER.value: EulerDiscreteScheduler,
            self.DDIM.value: DDIMScheduler,
            self.PNDM.value: PNDMScheduler,
            self.DPMPP_MULTISTEP.value: DPMSolverMultistepScheduler,
            self.DPMPP_SINGLESTEP.value: DPMSolverSinglestepScheduler,
            self.LMS.value: LMSDiscreteScheduler,
            self.UNIPC.value: UniPCMultistepScheduler,
            self.SDE.value: DPMSolverSDEScheduler,
        }.get(self.value, EulerAncestralDiscreteScheduler)


class GenerationArgs(DefaultBase):
    """
    GenerationArgs is a model for the parameters used in generating images.

    Attributes:
    ----------
    prompt : Union[str, List[str]]
        The main prompts used in generation.
    negative_prompt : Optional[Union[str, List[str]]]
        The negative prompts used in generation (Optional).
    image : Union[torch.FloatTensor, PIL.Image.Image]
        The input image used for generation.
    mask_image : Union[torch.FloatTensor, PIL.Image.Image]
        The mask image used for generation.
    height : int
        The height of the resulting image.
    width : int
        The width of the resulting image.
    num_inference_steps : int
        The number of inference steps in the generation process.
    guidance_scale : float
        The guidance scale used in the generation process.
    strength : float
        The strength factor used in the generation process.
    num_images_per_prompt : Optional[int]
        The number of images generated per prompt.
    add_predicted_noise : Optional[bool]
        Flag to add predicted noise in the generation process.
    eta : float
        The eta parameter used in the generation process.
    generator : Optional[Union[torch.Generator, List[torch.Generator]]]
        The generator or list of generators used in the generation process.
    latents : Optional[torch.FloatTensor]
        The latent space used in the generation process.
    prompt_embeds : Optional[torch.FloatTensor]
        The prompt embeddings used in the generation process.
    negative_prompt_embeds : Optional[torch.FloatTensor]
        The negative prompt embeddings used in the generation process.
    max_embeddings_multiples : Optional[int]
        The maximum multiples of embeddings used in the generation process.
    output_type : Optional[str]
        The type of the output from the generation process.
    callback : Optional[Callable[[int, int, torch.FloatTensor], None]]
        The callback function executed at each step of the generation process.
    is_cancelled_callback : Optional[Callable[[], bool]]
        The callback function used to check whether the generation process is cancelled.
    callback_steps : int
        The number of steps between each call to the callback function.
    cross_attention_kwargs : Optional[Dict[str, Any]]
        Dictionary of keyword arguments for the cross attention layer.
    clip_skip : Optional[int]
        The interval at which to skip frames during video generation.
    sampler : Optional[SchedulerType]
        The scheduler type used in the generation process.
    seed : Optional[int]
        The seed for the random number generator.
    start_time : Optional[float]
        The starting time of the generation process.
    repeat : Optional[int]
        The number of times to repeat the generation process for each prompt.
    seed_mode : Optional[Literal["random", "iter", "constant", "ladder"]]
        The mode of seed generation.
    seed_list : Optional[List[int]]
        The list of seeds to be used in the generation process.
    save_intermediates : Optional[bool]
        Flag to specify whether to save intermediate frames.
    template_save_path : Optional[str]
        The template path for saving the generated images.

    Methods
    -------
    to_kwargs(device: torch.device, exclude: Set[str]):
        Converts the GenerationArgs object to a dictionary usable for image generation.
    """
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
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        exclude: Set[str] = {"output_type"},
    ) -> Dict[str, Any]:
        """
        Converts the GenerationArgs object to a dict usable for image generation.

        If both seed and generator are present but the generator is not populated,
        a generator is created with the given seed.

        Parameters
        ----------
        device : torch.device
            The device on which to create the torch Generator (if needed).
        exclude : Set[str]
            The keys to exclude from the returned dictionary.

        Returns
        -------
        Dict[str, Any]
            A dictionary that can be used to guide image generation.
        """
        if self.seed and not self.generator:
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
