# Deforum

```py
from deforum import DeforumDiffusion

# Create an instance of the DeforumDiffusion class
dd = DeforumDiffusion()

# Use d instance to call generate function with the values prompt,max_frames and strength
dd.generate(
    prompt="Cat Sushi",
    max_frames=80,
    strength=0.5,
)
```