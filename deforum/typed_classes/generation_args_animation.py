import json
import os
from typing import Optional, Any, Dict, List
from .generation_args import GenerationArgs, SchedulerType
from scipy.interpolate import CubicSpline
import numpy as np

class Key:
    def __init__(self, frame: int, value: Any[int, float, str] = 1.0, interpolation: str = "linear"):
        self.frame = frame
        self.value = value
        self.interpolation = interpolation

    @classmethod
    def from_dict(cls, data: dict):
        return cls(data.get('frame', 0), data.get('value', 1.0), data.get('interpolation', "linear"))
def bezier_interpolation(t, P0, P1, P2):
    """ Compute the value at t using quadratic Bezier interpolation """
    return (1-t)**2 * P0 + 2*(1-t)*t * P1 + t**2 * P2

def sinusoidal_interpolation(t, start, end):
    """ Sinusoidal interpolation between start and end """
    return start + (end - start) * (1 - np.cos(t * np.pi)) / 2
class KeyChain:
    def __init__(self):
        # keys will be a dict where each entry is a list of keys for a particular type (like "strength" or "prompts")
        self.keys: Dict[str, List[Key]] = {}

    def add_key(self, name: str, key: Key):
        if name not in self.keys:
            self.keys[name] = []
        self.keys[name].append(key)

    def from_dict(self, data: dict):
        for name, key_data_list in data.items():
            for key_data in key_data_list['values']:
                self.add_key(name, Key.from_dict(key_data))

    def interpolate(self, max_frames: int) -> Dict[str, List[Any]]:
        interpolated_values = {}
        for name, key_list in self.keys.items():
            # Sort keys based on their frame number for accurate interpolation
            key_list.sort(key=lambda x: x.frame)

            frames = np.linspace(0, max_frames, max_frames)
            values = np.zeros(max_frames)

            for i in range(1, len(key_list)):
                start_frame = key_list[i-1].frame
                end_frame = key_list[i].frame
                start_value = key_list[i-1].value
                end_value = key_list[i].value

                method = key_list[i].interpolation

                if method == "linear":
                    mask = (frames >= start_frame) & (frames <= end_frame)
                    values[mask] = np.interp(frames[mask], [start_frame, end_frame], [start_value, end_value])

                elif method == "cubic":
                    cs = CubicSpline([start_frame, end_frame], [start_value, end_value])
                    mask = (frames >= start_frame) & (frames <= end_frame)
                    values[mask] = cs(frames[mask])
                elif method == "bezier":
                    P0 = start_value
                    P2 = end_value
                    P1 = (P0 + P2) / 2  # This is a simplification. In a real-world scenario, you'd have a better way to determine P1.
                    mask = (frames >= start_frame) & (frames <= end_frame)
                    t_values = (frames[mask] - start_frame) / (end_frame - start_frame)
                    values[mask] = [bezier_interpolation(t, P0, P1, P2) for t in t_values]

                elif method == "sinusoidal":
                    mask = (frames >= start_frame) & (frames <= end_frame)
                    t_values = (frames[mask] - start_frame) / (end_frame - start_frame)
                    values[mask] = [sinusoidal_interpolation(t, start_value, end_value) for t in t_values]


                # TODO: Add other interpolation methods

            interpolated_values[name] = values

        return interpolated_values

    def get_value_for_frame(self, frame: int, key_type: str) -> Any:
        """ Returns the interpolated value for a given frame and key type """
        # First, ensure the keys have been interpolated
        interpolated_values = self.interpolate(max_frames=frame + 1)  # Assuming 0-based frame indexing

        # Return the value for the given frame and key type
        return interpolated_values.get(key_type, [])[frame]
class GenerationArgsAnimation(GenerationArgs):
    max_frames: int = 25


example_keychain_dict = {
    "strength":{
        "values":[
            {0, 0.0, "linear"},
            {25, 0.0, "linear"},
            {50, 1.2, "linear"},
            {75, 2.0, "linear"},
        ]},
    "prompts":{
        "values":[
            {0, "example prompt", "linear"},
            {25, "example prompt", "linear"},
            {50, "example prompt", "linear"},
            {75, "example prompt", "linear"},
        ]},
    "guidance_scale":{
        "values":[
            {0, 0.0, "cubic"},
            {25, 0.0, "cubic"},
            {50, 1.2, "cubic"},
            {75, 2.0, "cubic"},
        ]}
    }