# Deforum

Deforum is a Python package for diffusion animation toolkit.

## Installation

You can install Deforum using one of the following methods:

### PyPI

Install from PyPI using `pip`:

```
pip install deforum
```

### Bash script (for Linux)

There's a provided bash script if you're on a Linux system.

```
./install-linux.sh
```

### Batch file (for Windows)

There's a provided batch file if you're on a Windows system.

```
install-windows.bat
```

### Requirements

Deforum has two sets of requirements which can be installed using:

```
pip install -r requirements.txt
```

For development requirements:

```
pip install -r requirements-dev.txt
```
## Package Overview

Deforum is structured in following modules:

* `backend`: Contains the actual generation models. Options include `base` for Stable Diffusion 1.5 and `sdxl` for Stable Diffusion XL.

* `data`: Contains helper data for certain types of generation like wildcards, templates, prompts, stopwords, lightweight models.

* `modules`: Contains various helper classes and utilities for animation processing, controlnet auxiliary model processors, image transformation etc..

* `pipelines`: Contains pipeline classes which are used to generate images or videos using helper modules.

* `typed_classes`: Contains typed classes which are used for type validation and help.

* `utils`: Contains utilities for handling images, videos, and text outside of actual generation.


## Usage

Here's a basic example:

```python
import torch
from deforum import Deforum, DeforumConfig, GenerationArgs

config = DeforumConfig(
    model_name="Lykon/AbsoluteReality",
    model_type="sd1.5",
    dtype=torch.float16,
)

deforum = Deforum(config)

args = GenerationArgs(
    prompt="An ethereal cityscape under a starlit night sky",
    negative_prompt="blurry, bright, devoid, and boring",
    guidance_scale=7.5,
    sampler="euler_ancestral",
    num_inference_steps=30,
)

deforum.generate(args)

```

## License

Deforum is licensed under the MIT License.

For more information please refer to [license](https://github.com/deforum-art/deforum/blob/main/LICENSE).

## Notes

```
backend:
    > Actual model to be used for generation. Can be one of the following:
    - 'base': Stable Diffusion 1.5
    - 'sdxl': Stable Diffusion XL
data:
    > Helper data for certain types of generation or misc things
    - 'wildcards'
    - 'templates'
    - 'prompts'
    - 'stopwords'
    - 'light-weight-models'
mixins:
    > Superclasses which are used when instantiating classes from backend
    - TODO: Add mixins to backend directory
modules:
    > Animation helper classes, controlnet auxiliary model processors for use DURING a generation...
    - 'animation'
    - 'controlnet'
    - 'stylegan2'
    - 'loras'
    - 'noise augmentations'
    - 'upscalers'
    - 'keyframing'
    - 'misc image transformations'
    - 'warp functions'
    - 'attn processors' TODO: Move to backend
    - 'video compilation'
    - 'video postprocessing'
    - 'video processing'
    - 'video transformations'
    - 'video utils'
pipelines:
    > Pipeline classes which are used to generate images or videos sample function implementation
    > for the different models in backend, using helper modules from /modules'
    - 'base (txt2img, img2img, two-pass)'
    - 'vid2vid'
    - 'txt2vid'
    - 'prompt interpolation'
    - 'animatediff'
    - '...experimental pipelines'
typed_classes:
    > Typed classes which are used for type validation and type help
    - 'deforum initialization config'
    - 'pipeline specific arguments'
    - 'pipeline specific output classes'
utils:
    > Utilities for working with images / videos / text outside of actual generation
    - 'image handler'
    - 'template parser'
    - 'image filename processing'
    - 'reading / writing video files'
    - 'mixed model initialization'

deforum/
    backend/
        mixins/
    data/
    modules/
    pipelines/
    typed_classes/
    utils/
    deforum.py
```
