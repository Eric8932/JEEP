from dataclasses import dataclass
from typing import List

from util.hparams import HyperParams


@dataclass
class FTHyperParams(HyperParams):
    # Method
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Defaults
    batch_size: int = 32#64
    max_length: int = 512#64
    wd_power_law: tuple = None  # Scale weight decay by number of edits
    min_loss: float=0.1