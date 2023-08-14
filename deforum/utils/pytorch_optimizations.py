import torch


def enable_optimizations(matmul_prec="high"):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision(matmul_prec)


def channels_last(pipeline):
    pipeline.unet.to(memory_format=torch.channels_last)
    pipeline.vae.to(memory_format=torch.contiguous_format)
    return pipeline
