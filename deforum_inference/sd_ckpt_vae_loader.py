def load_vae(vae_file=None):
    global first_load, vae_dict, vae_list, loaded_vae_file
    # save_settings = False

    if os.path.isfile(vae_file):
        assert os.path.isfile(vae_file), f"VAE file doesn't exist: {vae_file}"
        print(f"Loading VAE weights from: {vae_file}")
        vae_ckpt = torch.load(vae_file, map_location='cpu')
        vae_dict_1 = {k: v for k, v in vae_ckpt["state_dict"].items() if
                      k[0:4] != "loss" and k not in vae_ignore_keys}
        load_vae_dict(gs.models["sd"], vae_dict_1)

        # If vae used is not in dict, update it
        # It will be removed on refresh though
        # vae_opt = get_filename(vae_file)
        # if vae_opt not in vae_dict:
        #    vae_dict[vae_opt] = vae_file
        #    vae_list.append(vae_opt)
    else:
        print(f"VAE file doesn't exist: {vae_file}")

    loaded_vae_file = vae_file
    """
    # Save current VAE to VAE settings, maybe? will it work?
    if save_settings:
        if vae_file is None:
            vae_opt = "None"
        # shared.opts.sd_vae = vae_opt
    """
    first_load = False



#VAE Loading Utils
def load_vae_dict(model, vae_dict_1=None):
    if vae_dict_1:
        store_base_vae(model)
        model.first_stage_model.load_state_dict(vae_dict_1)
    else:
        restore_base_vae()
    model.first_stage_model.to(choose_torch_device())


def get_base_vae(model):
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info and model:
        return base_vae
    return None


def store_base_vae(model):
    global base_vae  # , checkpoint_info
    # if checkpoint_info != model.sd_checkpoint_info:
    base_vae = model.first_stage_model.state_dict().copy()
    # checkpoint_info = model.sd_checkpoint_info


def delete_base_vae():
    global base_vae, checkpoint_info
    base_vae = None
    checkpoint_info = None


def restore_base_vae(model):
    global base_vae, checkpoint_info
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info:
        load_vae_dict(model, base_vae)
    delete_base_vae()
