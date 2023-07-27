"""
This module provides the Deforum class which is used for image and video transformation.

The main functionalities of the Deforum class includes the methods
`animate_simple` and `vid2vid_simple` which offer the basic transformation ability
on both images and videos using the pretrained model.

Classes
-------
Deforum
    Main class providing functionalities for image or video transformation.

Example
-------
deforum = Deforum(model_name="stabilityai/stable-diffusion-xl-base-1.0")
prompt = "A beautiful sunrise over the ocean"
deforum.animate_simple(prompt, width=706, height=1280, max_frames=40, strength=0.5)
"""
import os
import cv2
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from deforum.pipelines.img2img import StableDiffusionXLImg2ImgPipeline


class Deforum:
    """
    Main application class which aids in image or video transformation.

    Parameters
    ----------
    model_name : str
        The name of the pretrained model.
    dtype : torch.dtype
        The data type of torch tensor.
    variant : str
        The parameter variant for the model.
    use_safetensors : bool
        Whether to use safe tensors or not.
    device : str
        The device to be used for computations, e.g. 'cuda' or 'cpu'.
    sample_dir : str
        The directory where the sample images will be saved.
    sample_format : str
        The format of the sample image file names.
    """

    def __init__(
        self,
        model_name="stabilityai/stable-diffusion-xl-base-1.0",
        dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        device="cuda",
        sample_dir="samples",
        sample_format="sample_{:05d}.png",
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.variant = variant
        self.use_safetensors = use_safetensors
        self.device = device
        self.sample_dir = sample_dir
        self.sample_format = sample_format

        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            variant=variant,
            use_safetensors=use_safetensors,
        )
        self.pipe.to(device)

        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

    def animate_simple(
        self,
        prompt,
        width=1024,
        height=1024,
        max_frames=40,
        strength=0.5,
        init=None
    ):
        """
        Animate the image transformation step by step according to the given prompt.

        Parameters
        ----------
        prompt : str
            The prompt on the basis of which image will be transformed.
        width : int
            The width of the image in pixels.
        height : int
            The height of the image in pixels.
        max_frames : int
            The maximum number of frames to generate.
        strength : float
            The strength factor for the transformation.
        init : torch.Tensor
            The initial image tensor. It is None by default and starting image tensor is generated.
        """
        local_strength = 1
        for iframe in range(max_frames):
            image, init = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                image=init,
                strength=local_strength,
                num_inference_steps=50,
            )
            init = image
            local_strength = strength
            image.save(
                os.path.join(self.sample_dir, self.sample_format.format(iframe + 1))
            )

    def vid2vid_simple(
        self,
        prompt,
        input_video_path,
        output_dir,
        width=1024,
        height=1024,
        max_frames=40,
        strength=0.5,
    ):
        """
        Transforms a video according to a given prompt, frame by frame and save to the output directory.

        Parameters
        ----------
        prompt : str
            The prompt on the basis of which video will be transformed.
        input_video_path : str
            The path to the input video file.
        output_dir : str
            The directory where the output images will be saved.
        width : int
            The width of the image in pixels.
        height : int
            The height of the image in pixels.
        max_frames : int
            The maximum number of frames to generate.
        strength : float
            The strength factor for transformation.
        """
        vidcap = cv2.VideoCapture(input_video_path)
        success, image = vidcap.read()
        count = 0
        init = None
        while success and count < max_frames:
            image = cv2.cvtColor(
                image, cv2.COLOR_BGR2RGB
            )  # Convert the BGR image to RGB
            image = Image.fromarray(image)  # Convert the image to PIL format
            image = image.resize((width, height))  # Resize image
            image = TF.to_tensor(image).unsqueeze(
                0
            )  # Convert the PIL Image to a PyTorch tensor
            image = image.to(self.device)
            init = image
            image, init = self.pipe(
                prompt=prompt, width=width, height=height, image=init, strength=strength
            )
            init = image
            image.save(
                os.path.join(output_dir, f"frame_{count:05d}.png")
            )  # Save frame as PNG image
            success, image = vidcap.read()
            count += 1
