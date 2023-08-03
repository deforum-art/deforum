# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import os
from typing import Callable, List, Dict
from loguru import logger
import torch
from collections import OrderedDict
from sortedcontainers import SortedSet, SortedDict
from torch import nn
from .modules.glora import GLoRAModule
from .modules.locon import LoConModule

DEBUG = os.getenv("DEBUG", False)


def inner_map_build(*parts):
    def call(cast_item):
        return [cast(item) for cast, item in cast_item]

    def inner_map(items) -> dict:
        return dict(map(call, [list(zip(parts, i)) for i in items]))

    return inner_map


class LycorisNetwork(nn.Module):
    """
    LoRA + LoCon
    """

    # Ignore proj_in or proj_out, their channels is only a few.
    UNET_TARGET_REPLACE_MODULE = [
        "Transformer2DModel",
        "Attention",
        "ResnetBlock2D",
        "Downsample2D",
        "Upsample2D",
    ]
    UNET_TARGET_REPLACE_NAME = [
        "conv_in",
        "conv_out",
        "time_embedding.linear_1",
        "time_embedding.linear_2",
    ]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    def __init__(
        self,
        name,
        text_encoder,
        unet,
        multiplier=1.0,
        lora_dim=4,
        conv_lora_dim=4,
        alpha=1,
        conv_alpha=1,
        use_cp=False,
        dropout=0,
        rank_dropout=0,
        module_dropout=0,
        network_module=LoConModule,
        applicable_lora_names: List[str] = [],
        **kwargs,
    ) -> None:
        super().__init__()
        self.logger = logger
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.conv_lora_dim = int(conv_lora_dim)
        if self.conv_lora_dim != self.lora_dim:
            if DEBUG:
                self.logger.debug("Apply different lora dim for conv layer")
                self.logger.debug(f"Conv Dim: {conv_lora_dim}, Linear Dim: {lora_dim}")

        self.alpha = alpha
        self.conv_alpha = float(conv_alpha)
        if self.alpha != self.conv_alpha:
            if DEBUG:
                self.logger.debug("Apply different alpha value for conv layer")
                self.logger.debug(f"Conv alpha: {conv_alpha}, Linear alpha: {alpha}")

        if 1 >= dropout >= 0:
            if DEBUG:
                self.logger.debug(f"Use Dropout value: {dropout}")
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.name = name

        # create module instances
        def create_modules(
            prefix,
            root_module: torch.nn.Module,
            target_replace_modules,
            target_replace_names=[],
        ) -> List[network_module]:
            loras = []
            lora_set = SortedSet()
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        if lora_name not in applicable_lora_names:
                            continue
                        if child_module.__class__.__name__ == "Linear" and lora_dim > 0:
                            if lora_name in lora_set:
                                continue
                            lora_set.add(lora_name)
                            lora = self.add_to_injector(
                                lora_name,
                                network_module,
                                child_module,
                                self.multiplier,
                                self.lora_dim,
                                self.alpha,
                                self.dropout,
                                self.rank_dropout,
                                self.module_dropout,
                                use_cp,
                                **kwargs,
                            )
                        elif child_module.__class__.__name__ == "Conv2d":
                            k_size, *_ = child_module.kernel_size
                            if k_size == 1 and lora_dim > 0:
                                if lora_name in lora_set:
                                    continue
                                lora_set.add(lora_name)
                                lora = self.add_to_injector(
                                    lora_name,
                                    network_module,
                                    child_module,
                                    self.multiplier,
                                    self.lora_dim,
                                    self.alpha,
                                    self.dropout,
                                    self.rank_dropout,
                                    self.module_dropout,
                                    use_cp,
                                    **kwargs,
                                )
                            elif conv_lora_dim > 0:
                                if lora_name in lora_set:
                                    continue
                                lora_set.add(lora_name)
                                lora = self.add_to_injector(
                                    lora_name,
                                    network_module,
                                    child_module,
                                    self.multiplier,
                                    self.conv_lora_dim,
                                    self.conv_alpha,
                                    self.dropout,
                                    self.rank_dropout,
                                    self.module_dropout,
                                    use_cp,
                                    **kwargs,
                                )
                            else:
                                continue
                        else:
                            continue
                        loras.append(lora)
                elif name in target_replace_names:
                    lora_name = prefix + "." + name
                    lora_name = lora_name.replace(".", "_")
                    if lora_name not in applicable_lora_names:
                        continue
                    if module.__class__.__name__ == "Linear" and lora_dim > 0:
                        if lora_name in lora_set:
                            continue
                        lora_set.add(lora_name)
                        lora = self.add_to_injector(
                            lora_name,
                            network_module,
                            child_module,
                            self.multiplier,
                            self.lora_dim,
                            self.alpha,
                            self.dropout,
                            self.rank_dropout,
                            self.module_dropout,
                            use_cp,
                            **kwargs,
                        )

                    elif module.__class__.__name__ == "Conv2d":
                        k_size, *_ = module.kernel_size
                        if k_size == 1 and lora_dim > 0:
                            if lora_name in lora_set:
                                continue
                            lora_set.add(lora_name)
                            lora = self.add_to_injector(
                                lora_name,
                                network_module,
                                child_module,
                                self.multiplier,
                                self.lora_dim,
                                self.alpha,
                                self.dropout,
                                self.rank_dropout,
                                self.module_dropout,
                                use_cp,
                                **kwargs,
                            )
                        elif conv_lora_dim > 0:
                            if lora_name in lora_set:
                                continue
                            lora_set.add(lora_name)
                            lora = self.add_to_injector(
                                lora_name,
                                network_module,
                                child_module,
                                self.multiplier,
                                self.conv_lora_dim,
                                self.conv_alpha,
                                self.dropout,
                                self.rank_dropout,
                                self.module_dropout,
                                use_cp,
                                **kwargs,
                            )
                        else:
                            continue
                    else:
                        continue
                    loras.append(lora)
            return loras

        if network_module == GLoRAModule:
            if DEBUG:
                self.logger.debug("GLoRA enabled, only train transformer")
            # only train transformer (for GLoRA)
            LycorisNetwork.UNET_TARGET_REPLACE_MODULE = [
                "Transformer2DModel",
                "Attention",
            ]
            LycorisNetwork.UNET_TARGET_REPLACE_NAME = []

        if isinstance(text_encoder, list):
            text_encoders = text_encoder
            use_index = True
        else:
            text_encoders = [text_encoder]
            use_index = False

        self.text_encoder_loras = []
        for i, te in enumerate(text_encoders):
            self.text_encoder_loras.extend(
                create_modules(
                    LycorisNetwork.LORA_PREFIX_TEXT_ENCODER + (f"{i+1}" if use_index else ""),
                    te,
                    LycorisNetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE,
                )
            )
        if DEBUG:
            self.logger.debug(f"create LyCORIS for Text Encoder: {len(self.text_encoder_loras)} modules.")

        self.unet_loras = create_modules(
            LycorisNetwork.LORA_PREFIX_UNET,
            unet,
            LycorisNetwork.UNET_TARGET_REPLACE_MODULE,
        )

        if DEBUG:
            self.logger.debug(f"create LyCORIS for U-Net: {len(self.unet_loras)} modules.")

        self.weights_sd = None

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def add_to_injector(
        self,
        lora_name,
        base_lora_class: LoConModule,
        org_module: nn.Module,
        *args,
        **kwargs,
    ):
        lora_module: LoConModule = base_lora_class(lora_name, org_module, name=self.name, *args, **kwargs)
        if not hasattr(org_module, "lora_injector"):
            injector: LoRALayerInjector = LoRALayerInjector(
                lora_name=lora_name,
                org_module=org_module,
            )
            org_module.register_module(
                "lora_injector",
                injector,
            )
        else:
            injector: LoRALayerInjector = org_module.lora_injector

        injector.add_lora(
            lora_module,
            self.name,
            multiplier=args[0],
            lora_dim=lora_module.lora_dim,
            alpha=lora_module.alpha,
        )
        return injector

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.loras[self.name].set_weight(multiplier)
        return self

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open

            self.weights_sd = load_file(file)
        else:
            self.weights_sd = torch.load(file, map_location="cpu")

    def apply_to(self, apply_text_encoder=None, apply_unet=None):
        if self.weights_sd:
            weights_has_text_encoder = weights_has_unet = False
            for key in self.weights_sd.keys():
                if key.startswith(LycorisNetwork.LORA_PREFIX_TEXT_ENCODER):
                    weights_has_text_encoder = True
                elif key.startswith(LycorisNetwork.LORA_PREFIX_UNET):
                    weights_has_unet = True

            if apply_text_encoder is None:
                apply_text_encoder = weights_has_text_encoder
            else:
                assert (
                    apply_text_encoder == weights_has_text_encoder
                ), f"text encoder weights: {weights_has_text_encoder} but text encoder flag: {apply_text_encoder} / 重みとText Encoderのフラグが矛盾しています"

            if apply_unet is None:
                apply_unet = weights_has_unet
            else:
                assert (
                    apply_unet == weights_has_unet
                ), f"u-net weights: {weights_has_unet} but u-net flag: {apply_unet} / 重みとU-Netのフラグが矛盾しています"
        else:
            assert apply_text_encoder is not None and apply_unet is not None, f"internal error: flag not set"

        if apply_text_encoder:
            if DEBUG:
                self.logger.debug("enable LyCORIS for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            if DEBUG:
                self.logger.debug("enable LyCORIS for U-Net")
        else:
            self.unet_loras = []
        self.all_loras = []
        all_missing = set()
        all_incompatible = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            if self.weights_sd and self.name in lora.loras:
                weights = {
                    k.split(lora.lora_name)[-1][1:]: v
                    for k, v in self.weights_sd.items()
                    if k.startswith(lora.lora_name)
                }
                if "alpha" not in weights:
                    if lora.is_linear:
                        weights["alpha"] = torch.tensor(self.alpha)
                    elif lora.is_conv:
                        weights["alpha"] = torch.tensor(self.conv_alpha)
                miss = lora.loras[self.name].load_state_dict(weights, strict=False)
                for mis in miss.missing_keys:
                    all_missing.add(mis)
                for mis in miss.unexpected_keys:
                    all_incompatible.add(mis)
            if self.name in lora.loras:
                lora.enable()
                self.all_loras.append(lora)
            else:
                self.logger.error(f"our name was not in {lora.lora_name}")
        if "scalar" in all_missing:
            all_missing.remove("scalar")
        if len(all_missing) > 0:
            self.logger.warning(f"Missing keys: {all_missing}")
        if len(all_incompatible) > 0:
            self.logger.warning(f"All incompatible: {all_incompatible}")

    def apply_max_norm_regularization(self, max_norm_value, device):
        key_scaled = 0
        norms = []
        for model in self.unet_loras:
            if hasattr(model.loras[self.name], "apply_max_norm"):
                scaled, norm = model.loras[self.name].apply_max_norm(max_norm_value, device)
                norms.append(norm)
                key_scaled += scaled

        for model in self.text_encoder_loras:
            if hasattr(model.loras[self.name], "apply_max_norm"):
                scaled, norm = model.loras[self.name].apply_max_norm(max_norm_value, device)
                norms.append(norm)
                key_scaled += scaled

        return key_scaled, sum(norms) / len(norms), max(norms)


class LoRALayerInjector(torch.nn.Module):
    def __init__(
        self,
        lora_name: torch.nn.Module,
        org_module: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.org_module_name = org_module.__class__.__name__
        self.org_module = [org_module]
        self.active_loras_weights: Dict[str, float] = SortedDict()
        self.org_forward = self.org_module[0].forward
        self.lora_name = lora_name
        self.loras: Dict[str, LoConModule] = torch.nn.ModuleDict({})
        self.active_lora_keys = []
        self.calculate_loras_fn: Callable[[torch.Tensor], torch.Tensor] = self.calculate_loras_empty
        self.active_loras = []
        self.is_linear = isinstance(org_module, nn.Linear)

    def calculate_loras_empty(self, x: torch.Tensor) -> torch.Tensor:
        return 0

    def add_lora(self, lora: LoConModule, loras_key, multiplier=1.0, **kwargs):
        self.loras[loras_key] = lora
        # lora.apply_max_norm(1.0,torch.device('cuda'))
        if multiplier > 0:
            self.active_loras_weights[loras_key] = multiplier

    def enable(self, enable=True):
        if enable:
            if not self.org_module[0].forward == self.forward:
                self.org_module[0].forward = self.forward
        else:
            self.org_module[0].forward = self.org_forward

    def prepare_loras(self, lora_merge_string):
        if not ":" in lora_merge_string:
            self.active_loras = []
            self.calculate_loras_fn = self.calculate_loras_empty
            return self
        merge_map = inner_map_build(str, float)([z.split(":") for z in lora_merge_string.split(",")])

        actives = list(merge_map.keys())

        self.active_loras = [self.loras[act].set_weight(merge_map[act]) for act in actives if act in self.loras]

        if len(self.active_loras) > 0:
            self.calculate_loras_fn = self.calculate_loras
        else:
            self.calculate_loras_fn = self.calculate_loras_empty
            return self

        param = self.org_module[0].parameters().__next__()
        if not self.device == param.device or self.dtype != param.dtype:
            self.to(device=param.device, dtype=param.dtype)
        return self

    @property
    def device(self):
        for lora in self.loras.values():
            return next(lora.parameters()).device
        return torch.device("cpu")

    @property
    def dtype(self):
        for lora in self.loras.values():
            for param in lora.parameters():
                if isinstance(param.dtype, torch.FloatTensor):
                    return param.dtype
        return torch.float32

    def calculate_loras(self, x):
        # if self.device != x.device:
        #     self.to(device=x.device, dtype=x.dtype)
        total = self.active_loras[0](x)
        for lora in self.active_loras[1:]:
            total += lora(x)
        return total

    def to(self, *args, **kwargs):
        self.loras.to(*args, **kwargs)

    def forward(self, x):
        return self.org_forward(x) + self.calculate_loras_fn(x)


def create_network_from_weights(
    name,
    multiplier,
    file,
    vae,
    text_encoder,
    unet,
    weights_sd=None,
    default_modules_dim=4,
    default_modules_alpha=1,
    default_convs_dim=1,
    default_convs_alpha=1,
    **kwargs,
):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    modules_dim = default_modules_dim
    modules_alpha = default_modules_alpha
    modules_cp = False
    convs_dim = default_convs_dim
    convs_alpha = default_convs_alpha
    applicable_lora_names = set()
    normal_keys = set(["alpha", "lora_down", "lora_up", "lora_mid"])
    already_warned_about = set()
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        lora_key = key.split(".", 1)[-1].split(".")[0]
        if "conv" in lora_name:
            if "alpha" in key:
                convs_alpha = value
            elif "lora_down" in key:
                convs_dim = value.size()[0]
        else:
            if "alpha" in key:
                modules_alpha = value.item()
            elif "lora_down" in key:
                dim = value.size()[0]
                modules_dim = dim
            if "lora_mid" in key:
                modules_cp = True
        if not any(k in key for k in normal_keys):
            if not lora_key in already_warned_about:
                if DEBUG:
                    logger.critical(f"Found unusual lora key! {lora_key}")
                already_warned_about.add(lora_key)
        applicable_lora_names.add(lora_name)

    network = LycorisNetwork(
        name,
        text_encoder,
        unet,
        multiplier=multiplier,
        lora_dim=modules_dim,
        alpha=modules_alpha,
        conv_lora_dim=convs_dim,
        conv_alpha=convs_alpha,
        use_cp=modules_cp,
        applicable_lora_names=applicable_lora_names,
    )
    network.weights_sd = weights_sd
    return network


class LoraMixin:
    loras: Dict[str, LycorisNetwork]
    layer_injectors: List[LoRALayerInjector]

    def _post_init_enable_loras(self):
        if not hasattr(self, "loras"):
            self.loras = OrderedDict()
        if not hasattr(self, "layer_injectors"):
            self.layer_injectors = SortedSet(key=lambda x: x.lora_name)

    def disable_loras(self):
        self._post_init_enable_loras()
        for x in self.layer_injectors:
            x.prepare_loras("")

    def load_network(self, name: str, path: str, default_strength: float = 0.6) -> None:
        if not hasattr(self, "loras"):
            self.loras = OrderedDict()
        self.loras[name] = create_network_from_weights(
            name, default_strength, path, self.vae, self.text_encoder, self.unet
        )
        self.loras[name].apply_to()
        if not hasattr(self, "layer_injectors"):
            self.layer_injectors = SortedSet(key=lambda x: x.lora_name)
        for layer in self.loras[name].all_loras:
            if layer not in self.layer_injectors:
                self.layer_injectors.add(layer)

    def set_lora_strengths(self, lora_prompt):
        try:
            self.loras
        except Exception as e:
            self._post_init_enable_loras()
            return
        if len(list(self.loras.keys())) == 0:
            return

        for lora in self.layer_injectors:
            lora.prepare_loras(lora_prompt)
        return lora_prompt
