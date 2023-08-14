import numpy as np
import torch
from diffusers.utils import PIL_INTERPOLATION
from torchvision.transforms import functional as TF


def preprocess_image(image, batch_size):
    if not isinstance(image, torch.FloatTensor):
        image = image.convert("RGB")
        w, h = image.size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
        image = image.resize((w, h), resample=PIL_INTERPOLATION["nearest"])
        image = np.array(image).astype(np.float32) / 255.0
        image = np.vstack([image[None].transpose(0, 3, 1, 2)] * batch_size)
        image = torch.from_numpy(image)
        return 2.0 * image - 1.0
    else:
        valid_image_channel_sizes = [1, 3]
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        # if image channel is fourth tensor dimension, permute dimensions to pytorch standard (B, C, H, W)
        if image.shape[3] in valid_image_channel_sizes:
            image = image.permute(0, 3, 1, 2)
        elif image.shape[1] not in valid_image_channel_sizes:
            raise ValueError(
                f"tensor image channel dimension of size in {valid_image_channel_sizes} should be second or fourth dimension,"
                f" but received tensor image of shape {tuple(image.shape)}"
            )
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        image = image - image.min()
        image = image / image.max()
        return 2.0 * image - 1.0


def preprocess_mask(mask, batch_size, scale_factor=8):
    if not isinstance(mask, torch.FloatTensor):
        mask = mask.convert("L")
        w, h = mask.size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
        mask = mask.resize((w // scale_factor, h // scale_factor), resample=PIL_INTERPOLATION["nearest"])
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = np.tile(mask, (4, 1, 1))
        mask = np.vstack([mask[None]] * batch_size)
        mask = 1 - mask  # repaint white, keep black
        mask = torch.from_numpy(mask)
        return mask

    else:
        valid_mask_channel_sizes = [1, 3]
        # if mask channel is fourth tensor dimension, permute dimensions to pytorch standard (B, C, H, W)
        if mask.shape[3] in valid_mask_channel_sizes:
            mask = mask.permute(0, 3, 1, 2)
        elif mask.shape[1] not in valid_mask_channel_sizes:
            raise ValueError(
                f"Mask channel dimension of size in {valid_mask_channel_sizes} should be second or fourth dimension,"
                f" but received mask of shape {tuple(mask.shape)}"
            )
        # (potentially) reduce mask channel dimension from 3 to 1 for broadcasting to latent shape
        mask = mask.mean(dim=1, keepdim=True)
        h, w = mask.shape[-2:]
        h, w = (x - x % 8 for x in (h, w))  # resize to integer multiple of 8
        mask = TF.resize(mask, (h // scale_factor, w // scale_factor))
        return mask
