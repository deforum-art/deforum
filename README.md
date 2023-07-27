# Deforum

Deforum is a diffusion animation toolkit. You can use it to create simple animations and video conversions.

## Installation

```
pip install deforum
```

## Examples

### Text to Vid Simple

Use txt2vid_simple to generate an animation with your prompt (e.g. "Cat Sushi"), a defined number of frames (e.g. 80), and a strength value (e.g. 0.5).

```py
from deforum import Deforum

dd = Deforum()

dd.txt2vid_simple(
    prompt="Cat Sushi",
    max_frames=80,
    strength=0.5,
)
```

### Video to Video Simple

You can also convert one video to another using the vid2vid_simple method. This requires an input video path, output directory, max number of frames, and a strength value.

```py
dd.vid2vid_simple(
    prompt="Cat Sushi",
    input_video_path="path/to/vid",
    output_dir="out/put/dir",
    max_frames=90,
    strength=0.62,
)
```

## Contributing 

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more details.

## License 

Deforum is licensed under the MIT License. See [LICENSE](LICENSE) for more information.

## Contact

If you have any questions or need further assistance, feel free to reach out directly to deforum.art@gmail.com.