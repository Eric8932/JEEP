from dataclasses import dataclass
from typing import List, Literal

from util.hparams import HyperParams


@dataclass
class JEEPHyperParams(HyperParams):
    # Method
    #LOW
    layers_low: List[int]
    layer_selection_low: Literal["all", "random"]
    fact_token_low: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last","pred_last"
    ]  
    v_weight_decay_low: float
    clamp_norm_factor_low: float
    kl_factor_low: float
    mom2_adjustment_low: bool
    mom2_update_weight_low: float
    rewrite_module_tmp_low: str

    #HIGH
    layers_high: List[int]
    layer_selection_high: Literal["all", "random"]
    fact_token_high: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last","pred_last"
    ]  
    v_weight_decay_high: float
    clamp_norm_factor_high: float
    kl_factor_high: float
    mom2_adjustment_high: bool
    mom2_update_weight_high: float
    rewrite_module_tmp_high: str

    #opt
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int

    # Module templates
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str
    emb_module_tmp: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    min_loss: float=0.1
