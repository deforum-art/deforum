# Deforum

```py
from deforum import Deforum

# Create an instance of the Deforum class
dd = Deforum()

# Use dd instance to call animate_simple function with the values prompt, max_frames, and strength
dd.animate_simple(
    prompt="Cat Sushi",
    max_frames=80,
    run_strength=0.5,
)
```

For simple vid2vid, do:
```py
dd.vid2vid_simple(
    prompt="Cat Sushi",
    input_video_path="path/to/vid",
    output_dir="out/put/dir",
    max_frames=90,
    height=1024,
    width=1024,
    strength=0.62,
)
```
