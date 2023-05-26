import hashlib
import importlib
import os

import safetensors
import torch
from omegaconf import OmegaConf

from deforum_inference.sd_ckpt_utils import get_state_dict_from_checkpoint, instantiate_from_config
from deforum_inference.sd_ckpt_vae_loader import load_vae
from deforum_inference.sd_ckpt_version_check import return_model_version
from deforum_storage import singleton
gs = singleton




def load_model_from_config(ckpt=None, verbose=False):
    gs.force_inpaint = False

    config, version = return_model_version(ckpt)
    if 'Inpaint' in version:
        gs.force_inpaint = True
        print("Forcing Inpaint")
    if config == None:
        config = os.path.splitext(ckpt)[0] + '.yaml'
    else:
        config = os.path.join('configs/sd_deforum', config)

    if "sd" not in gs.models:
        if verbose:
            print(f"Loading model from {ckpt} with config {config}")
        config = OmegaConf.load(config)

        # print(config.model['params'])

        if 'num_heads' in config.model['params']['unet_config']['params']:
            gs.model_version = '1.5'
        elif 'num_head_channels' in config.model['params']['unet_config']['params']:
            gs.model_version = '2.0'
        if config.model['params']['conditioning_key'] == 'hybrid-adm':
            gs.model_version = '2.0'
        if 'parameterization' in config.model['params']:
            gs.model_resolution = 768
        else:
            gs.model_resolution = 512
        print(f'v {gs.model_version} found with resolution {gs.model_resolution}')

        if verbose:
            print(gs.model_version)

        checkpoint_file = ckpt
        _, extension = os.path.splitext(checkpoint_file)
        map_location = "cpu"
        if extension.lower() == ".safetensors":
            pl_sd = safetensors.torch.load_file(checkpoint_file, device=map_location)
        else:
            pl_sd = torch.load(checkpoint_file, map_location=map_location)

        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = get_state_dict_from_checkpoint(pl_sd)
        # sd = pl_sd["state_dict"]

        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
        model.half()
        gs.models["sd"] = model
        gs.models["sd"].cond_stage_model.device = device
        # gs.models["sd"].embedding_manager = EmbeddingManager(gs.models["sd"].cond_stage_model)
        # embedding_path = '001glitch-core.pt'
        # if embedding_path is not None:
        #    gs.models["sd"].embedding_manager.load(
        #        embedding_path
        #    )

        for m in gs.models["sd"].modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m._orig_padding_mode = m.padding_mode

        autoencoder_version = get_autoencoder_version()

        gs.models["sd"].linear_decode = make_linear_decode(autoencoder_version, device)
        del pl_sd
        del sd
        del m, u
        del model
        torch_gc()

        if gs.model_version == '1.5' and not 'Inpaint' in version:
            run_post_load_model_generation_specifics()

        gs.models["sd"].eval()

        # todo make this 'cuda' a parameter
        gs.models["sd"].to(device)
        # todo why we do this here?
        if gs.diffusion.selected_vae != 'None':
            load_vae(gs.diffusion.selected_vae)




def choose_torch_device() -> str:
    '''Convenience routine for guessing which GPU device to run model on'''
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def run_post_load_model_generation_specifics(self):

    gs.model_hijack = backend.hypernetworks.modules.sd_hijack.StableDiffusionModelHijack()

    gs.model_hijack.hijack(gs.models["sd"])
    gs.model_hijack.embedding_db.load_textual_inversion_embeddings()

    aesthetic = AestheticCLIP()
    aesthetic.process_tokens = gs.models["sd"].cond_stage_model.process_tokens
    gs.models["sd"].cond_stage_model.process_tokens = aesthetic

