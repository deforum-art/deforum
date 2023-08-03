import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GLoRAModule(nn.Module):
    """
    modifed from kohya-ss/sd-scripts/networks/lora:LoRAModule
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        use_cp=False,
        name=None,
        **kwargs,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        assert name is not None and isinstance(
            name, str
        ), f"Lora module must have a name, and it must be a string (it is how lora multipliers are individually modified)!"

        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.cp = False

        if isinstance(org_module, nn.Conv2d):
            assert org_module.kernel_size == (1, 1)
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            self.a1 = nn.Conv2d(in_dim, lora_dim, (1, 1), bias=False)
            self.a2 = nn.Conv2d(lora_dim, in_dim, (1, 1), bias=False)
            self.b1 = nn.Conv2d(in_dim, lora_dim, (1, 1), bias=False)
            self.b2 = nn.Conv2d(lora_dim, out_dim, (1, 1), bias=False)
        elif isinstance(org_module, nn.Linear):
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.a1 = nn.Linear(in_dim, lora_dim, bias=False)
            self.a2 = nn.Linear(lora_dim, in_dim, bias=False)
            self.b1 = nn.Linear(in_dim, lora_dim, bias=False)
            self.b2 = nn.Linear(lora_dim, out_dim, bias=False)
        else:
            raise NotImplementedError
        self.shape = org_module.weight.shape

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.a1.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.a2.weight)
        torch.nn.init.kaiming_uniform_(self.b1.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.b2.weight)

        self.multiplier = multiplier
        self.org_module = [org_module]

    def set_weight(self, multiplier):
        self.multiplier = multiplier
        return self

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def make_weight(self, device=None):
        wa1 = self.a1.weight.view(self.a1.weight.size(0), -1)
        wa2 = self.a2.weight.view(self.a2.weight.size(0), -1)
        wb1 = self.b1.weight.view(self.b1.weight.size(0), -1)
        wb2 = self.b2.weight.view(self.b2.weight.size(0), -1)
        orig = self.org_module[0].weight.view(self.org_module[0].weight.size(0), -1)
        return (wb2 @ wb1) + ((orig @ wa2) @ wa1)

    def forward(self, x):
        scale = self.scale * self.multiplier
        ax_mid = self.a1(x) * scale
        bx_mid = self.b1(x) * scale
        return self.org_forward(x + self.a2(ax_mid)) + self.b2(bx_mid)
