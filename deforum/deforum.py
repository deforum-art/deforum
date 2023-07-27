import os
import torch
from deforum.pipelines.img2img import StableDiffusionXLImg2ImgPipeline
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


class Deforum:
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

    def animate_simple(self, prompt, width=706, height=1280, max_frames=40, run_strength=0.5, init=None):
        strength = 1
        for iframe in range(max_frames):
            image, init = self.pipe(
                prompt=prompt, width=width, height=height, image=init, strength=strength, num_inference_steps=50
            )
            init = image
            strength = run_strength
            image.save(os.path.join(self.sample_dir, self.sample_format.format(iframe+1)))
    

    def vid2vid_simple(self, prompt, input_video_path, output_dir, width=1024, height=1024, max_frames=40, strength=0.5):
        """
        Transforms a video according to a given prompt, frame by frame.
        
        The output images are saved in the output_dir directory, with names in the format "frame_{:05d}.png".
        
        input_video_path: Path to the input video file.
        output_dir: Directory to save the output images.
        """
        vidcap = cv2.VideoCapture(input_video_path)
        success, image = vidcap.read()
        count = 0
        init = None
        while success and count < max_frames:
            # Convert the BGR image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Convert the image to PIL format
            image = Image.fromarray(image)
            # Resize image
            image = image.resize((width, height))
            # Convert the PIL Image to a PyTorch tensor
            image = TF.to_tensor(image).unsqueeze(0)
            image = image.to(self.device)
            init = image

            image, init = self.pipe(
                prompt=prompt, width=width, height=height, image=init, strength=strength
            )

            init = image
            # Save frame as PNG image
            image.save(os.path.join(output_dir, f'frame_{count:05d}.png'))
            
            success, image = vidcap.read()
            count += 1
