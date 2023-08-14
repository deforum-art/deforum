from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from loguru import logger
from pydantic import BaseConfig, BaseModel


class MixedModel(BaseModel):
    models: Dict[Union[str, UNet2DConditionModel], float]
    _is_resolved: bool = False
    _state_dicts: Optional[Tuple[Dict[str, torch.Tensor], float]] = None
    _sd_keys: Optional[Set[str]]

    class Config(BaseConfig):
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def _resolve_models(self) -> None:
        new_models = {}
        for model, amt in self.models.items():
            if isinstance(model, UNet2DConditionModel):
                new_models[model] = amt
                continue
            if isinstance(model, (Path, str)):
                try:
                    model = UNet2DConditionModel.from_pretrained(model, torch_dtype=torch.float32)
                except Exception as _:
                    model = UNet2DConditionModel.from_pretrained(model, torch_dtype=torch.float32, subfolder="unet")
                new_models[model] = amt
        self.models = new_models
        self._is_resolved = True

    def _to_statedicts(self) -> Tuple[Tuple[Dict[str, torch.Tensor], float], List[str]]:
        if not self._is_resolved:
            self._resolve_models()
        sds = []
        keys = set()
        total_amt = 0
        for model, amt in self.models.items():
            sd = model.state_dict()
            keys_for_sd = set(sd.keys())
            if len(keys) == 0:
                keys = keys_for_sd
            elif len(keys.intersection(keys_for_sd)) != len(keys):
                logger.warning(f"Keys mismatch on keys: {keys.symmetric_difference(keys_for_sd)}")
                keys = keys.intersection(keys_for_sd)
            total_amt += amt
            sds.append((sd, amt))
        assert (total_amt == 1) or abs(
            total_amt - 1
        ) < 0.01, f"Model accumulated weight should be very close to 1, (1 +/- 0.01) got {abs(total_amt-1)}"
        self._state_dicts = sds
        self._sd_keys = keys
        return sds, keys

    def _merge_on_key(self, key: str) -> torch.Tensor:
        if not self._is_resolved:
            self._to_statedicts()
        new_tensor = None
        for sd, amt in self._state_dicts:
            if new_tensor is None:
                new_tensor = sd[key] * amt
            else:
                new_tensor += sd[key] * amt
        return new_tensor

    def merge_into(self, unet: UNet2DConditionModel) -> UNet2DConditionModel:
        sd = OrderedDict()
        if not self._is_resolved:
            self._to_statedicts()
        for key in unet.state_dict().keys():
            if key in self._sd_keys:
                sd[key] = self._merge_on_key(key)
            else:
                logger.warning(f"Skipping key, since it was not in all state dicts!")
        unet.load_state_dict(sd)
        return unet

    def merge(self) -> UNet2DConditionModel:
        if not self._is_resolved:
            self._to_statedicts()
        model = list(self.models.keys())[0]
        return self.merge_into(model)
