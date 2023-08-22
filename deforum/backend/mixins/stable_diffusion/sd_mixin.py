import inspect
from typing import Optional, Union, List, Tuple

from loguru import logger
from packaging import version
import numpy as np
import PIL
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers import DiffusionPipeline
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    LoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, randn_tensor

from .. import DiffusersBaseMixin
from ....modules import (
    get_weighted_sd_text_embeddings,
)
import torch.nn.functional as F


class StableDiffusionMixin(DiffusionPipeline, DiffusersBaseMixin, TextualInversionLoaderMixin, FromSingleFileMixin, LoraLoaderMixin):

    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel] = None,
    ):
        super().__init__()

        # scheduler
        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        # scheduler
        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        # safety checker
        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        # safety checker
        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        # unet
        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        # controlnet
        if controlnet is not None and isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)

        # register modules
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

        # vae
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        self.register_to_config(
            requires_safety_checker=requires_safety_checker,
        )

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        max_embeddings_multiples=3,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
        lora_scale: Optional[float] = None,
    ):
        
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            clip_skip (`int`, *optional*):
                The layer of the clip encoder's hidden states to use for unet text conditioning. If not specified, the
                last layer will be used.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """

        # lora
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        # get batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # expand negative prompt to batch size
        if negative_prompt_embeds is None:
            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            if batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            
        # get prompt embeddings
        if prompt_embeds is None or negative_prompt_embeds is None:
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)
                if do_classifier_free_guidance and negative_prompt_embeds is None:
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, self.tokenizer)

            prompt_embeds1, negative_prompt_embeds1 = get_weighted_sd_text_embeddings(
                pipe=self,
                prompt=prompt,
                uncond_prompt=negative_prompt if do_classifier_free_guidance else None,
                max_embeddings_multiples=max_embeddings_multiples,
                clip_skip=clip_skip,
            )
            if prompt_embeds is None:
                prompt_embeds = prompt_embeds1
            if negative_prompt_embeds is None:
                negative_prompt_embeds = negative_prompt_embeds1

        bs_embed, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            bs_embed, seq_len, _ = negative_prompt_embeds.shape
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def check_inputs(
        self,
        prompt,
        height,
        width,
        strength,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        clip_skip=None,
        control_image=None,
        control_conditioning_scale=None,
        control_guidance_start=None,
        control_guidance_end=None,
    ):
        
        # check height and width
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # check strength
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        # callback steps
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # check prompt and prompt_embeds
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # check negative_prompt and negative_prompt_embeds
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        
        # check prompt_embeds and negative_prompt_embeds
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            
        # check clip_skip
        if clip_skip is not None and (not isinstance(clip_skip, int) or clip_skip < 0):
            raise ValueError(f"`clip_skip` has to be a integer greater than or equal to 0 but is {clip_skip} of type {type(clip_skip)}.")
        
        if control_image is not None:
            # check controlnets
            # `prompt` needs more sophisticated handling when there are multiple
            # conditionings.
            if isinstance(self.controlnet, MultiControlNetModel):
                if isinstance(prompt, list):
                    logger.warning(
                        f"You have {len(self.controlnet.nets)} ControlNets and you have passed {len(prompt)}"
                        " prompts. The conditionings will be fixed across the prompts."
                    )

            if isinstance(control_guidance_start, tuple):
                control_guidance_start = list(control_guidance_start)

            if isinstance(control_guidance_end, tuple):
                control_guidance_end = list(control_guidance_end)

            # Check `image`
            is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
                self.controlnet, torch._dynamo.eval_frame.OptimizedModule
            )
            if (
                isinstance(self.controlnet, ControlNetModel)
                or is_compiled
                and isinstance(self.controlnet._orig_mod, ControlNetModel)
            ):
                self.check_image(control_image, prompt, prompt_embeds)
            elif (
                isinstance(self.controlnet, MultiControlNetModel)
                or is_compiled
                and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
            ):
                if not isinstance(control_image, list):
                    raise TypeError("For multiple controlnets: `image` must be type `list`")

                # When `image` is a nested list:
                # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
                elif any(isinstance(i, list) for i in control_image):
                    raise ValueError("A single batch of multiple conditionings are supported at the moment.")
                elif len(control_image) != len(self.controlnet.nets):
                    raise ValueError(
                        f"For multiple controlnets: `image` must have the same length as the number of controlnets, but got {len(control_image)} images and {len(self.controlnet.nets)} ControlNets."
                    )

                for image_ in control_image:
                    self.check_image(image_, prompt, prompt_embeds)
            else:
                assert False

            # Check `control_conditioning_scale`
            if (
                isinstance(self.controlnet, ControlNetModel)
                or is_compiled
                and isinstance(self.controlnet._orig_mod, ControlNetModel)
            ):
                if not isinstance(control_conditioning_scale, float):
                    raise TypeError("For single controlnet: `control_conditioning_scale` must be type `float`.")
            elif (
                isinstance(self.controlnet, MultiControlNetModel)
                or is_compiled
                and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
            ):
                if isinstance(control_conditioning_scale, list):
                    if any(isinstance(i, list) for i in control_conditioning_scale):
                        raise ValueError("A single batch of multiple conditionings are supported at the moment.")
                elif isinstance(control_conditioning_scale, list) and len(control_conditioning_scale) != len(
                    self.controlnet.nets
                ):
                    raise ValueError(
                        "For multiple controlnets: When `control_conditioning_scale` is specified as `list`, it must have"
                        " the same length as the number of controlnets"
                    )
            else:
                assert False

            if len(control_guidance_start) != len(control_guidance_end):
                raise ValueError(
                    f"`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
                )

            if isinstance(self.controlnet, MultiControlNetModel):
                if len(control_guidance_start) != len(self.controlnet.nets):
                    raise ValueError(
                        f"`control_guidance_start`: {control_guidance_start} has {len(control_guidance_start)} elements but there are {len(self.controlnet.nets)} controlnets available. Make sure to provide {len(self.controlnet.nets)}."
                    )

            for start, end in zip(control_guidance_start, control_guidance_end):
                if start >= end:
                    raise ValueError(
                        f"control guidance start: {start} cannot be larger or equal to control guidance end: {end}."
                    )
                if start < 0.0:
                    raise ValueError(f"control guidance start: {start} can't be smaller than 0.")
                if end > 1.0:
                    raise ValueError(f"control guidance end: {end} can't be larger than 1.0.")


    def get_timesteps(self, num_inference_steps, strength, device, is_text2img):
        if is_text2img:
            return self.scheduler.timesteps.to(device), num_inference_steps
        else:
            # get the original timestep using init_timestep
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

            return timesteps, num_inference_steps - t_start

    def decode_latents(self, latents, to_numpy=True):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.vae.dtype)).sample.float()
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        if to_numpy:
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        else:
            image = image.permute(0, 2, 3, 1).float()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        image,
        timestep,
        num_images_per_prompt,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if image is None:
            batch_size = batch_size * num_images_per_prompt
            shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                latents = latents.to(device)

            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
            return latents, None, None
        else:
            image = image.to(device=self.device, dtype=self.vae.dtype)
            init_latent_dist = self.vae.encode(image).latent_dist
            init_latents = init_latent_dist.sample(generator=generator).to(dtype)
            init_latents = self.vae.config.scaling_factor * init_latents

            # Expand init_latents for batch_size and num_images_per_prompt
            if init_latents.shape[0] != (num_images_per_prompt * batch_size):
                init_latents = torch.cat([init_latents] * num_images_per_prompt, dim=0)
            init_latents_orig = init_latents
            # add noise to latents using the timesteps
            noise = randn_tensor(init_latents.shape, generator=generator, device=self.device, dtype=dtype)
            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
            latents = init_latents
            return latents, init_latents_orig, noise
        
    def prepare_control_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image
    
    def check_image(self, image, prompt, prompt_embeds):
        image_is_pil = isinstance(image, PIL.Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_np = isinstance(image, np.ndarray)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

        if (
            not image_is_pil
            and not image_is_tensor
            and not image_is_np
            and not image_is_pil_list
            and not image_is_tensor_list
            and not image_is_np_list
        ):
            raise TypeError(
                f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
            )

        if image_is_pil:
            image_batch_size = 1
        else:
            image_batch_size = len(image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )