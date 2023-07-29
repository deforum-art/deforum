import torch


def enable_optimizations():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")


def channels_last(pipeline):
    pipeline.unet.to(memory_format=torch.channels_last)
    return pipeline
