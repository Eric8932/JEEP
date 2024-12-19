import os,gc
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from time import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import get_module_input_output_at_words,compute_z_joint
from .jeep_hparams import JEEPHyperParams


# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def apply_jeep_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: JEEPHyperParams,
    copy=False, 
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    model_fp16 = False,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_jeep(model, tok, requests, hparams, cache_template=cache_template,model_fp16 =model_fp16)

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to("cuda"), val_mat.to("cuda")
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
            if "attn" in w_name:
                upd_matrix = upd_matrix.T

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            if model_fp16:
                w[...] += upd_matrix.half()
            else:
                w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy,deltas


def execute_jeep(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: JEEPHyperParams,
    cache_template: Optional[str] = None,
    model_fp16 = False,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the JEEP update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"JEEP request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    print(hparams)
    weights = {
        f"{hparams.rewrite_module_tmp_low.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp_low.format(layer)}.weight"
        )
        for layer in hparams.layers_low
    }
    weights.update({
        f"{hparams.rewrite_module_tmp_high.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp_high.format(layer)}.weight"
        )
        for layer in hparams.layers_high
    })
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer_low = hparams.layers_low[-1]
    z_list_low = []
    z_layer_high = hparams.layers_high[-1]
    z_list_high = []

    for r_id,request in enumerate(requests):
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                     hparams.min_loss,z_layer_low,z_layer_high, hparams.v_lr, hparams.v_num_grad_steps, hparams.clamp_norm_factor_low,hparams.clamp_norm_factor_high
                       ,hparams.v_weight_decay_low,hparams.v_weight_decay_high, hparams.kl_factor_low,hparams.kl_factor_high, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list_high.append(torch.from_numpy(data["v_star_high"]).to("cuda"))
                z_list_low.append(torch.from_numpy(data["v_star_low"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")
        

        # Compute k/v pair if not loaded from cache
        start = time()
        if not data_loaded:
            cur_z_low,cur_z_high = compute_z_joint(
                    model,
                    tok,
                    request,
                    hparams,
                    z_layer_low,
                    z_layer_high,
                    context_templates,
                )
            z_list_low.append(cur_z_low)
            z_list_high.append(cur_z_high)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star_high": cur_z_high.detach().cpu().numpy(),
                        "v_star_low": cur_z_low.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")

        exec_time = time() - start
        print("Execution took", exec_time)
    zs_low = torch.stack(z_list_low, dim=1)
    zs_high = torch.stack(z_list_high, dim=1)
    print("zs_low_shape",zs_low.shape)
    print("zs_high_shape",zs_high.shape)


    for i, layer_low in enumerate(hparams.layers_low):
        print(f"\n\nLOW LAYER {layer_low}\n")
        layer_ks_low = compute_ks(model, tok, requests, hparams, layer_low, context_templates,"low").T 


        cur_zs_low = get_module_input_output_at_words(
            model,
            tok,
            z_layer_low,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.rewrite_module_tmp_low,
            fact_token_strategy=hparams.fact_token_low,
        )[1].T
        targets_low = zs_low - cur_zs_low
        print("z error low", torch.linalg.norm(targets_low, dim=0).mean())

        repeat_factor_low = (layer_ks_low.size(1) // targets_low.size(1))
        targets_low = targets_low.repeat_interleave(repeat_factor_low, dim=1)

        force_recompute = False
        cov_low = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp_low.format(layer_low),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,#float32
            force_recompute=force_recompute,
        )
        cov_low = cov_low.cpu()

        layer_ks_low, targets_low = (
            layer_ks_low.double().cpu(),
            targets_low.double().cpu(),
        )
        
        adj_k_low = torch.linalg.solve(
            hparams.mom2_update_weight_low * cov_low.double() + layer_ks_low @ layer_ks_low.T,
            layer_ks_low,
        )
        resid_low = targets_low / np.sqrt((len(hparams.layers_low) - i))
        
        upd_matrix_low = (resid_low @ adj_k_low.T).cuda()

        weight_name_low = f"{hparams.rewrite_module_tmp_low.format(layer_low)}.weight"
        upd_matrix_low = upd_matrix_match_shape(upd_matrix_low, weights[weight_name_low].shape)

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            if model_fp16:
                weights[weight_name_low][...] = weights_copy[weight_name_low] + upd_matrix_low.half()
            else:
                weights[weight_name_low][...] = weights_copy[weight_name_low] + upd_matrix_low.float()
            deltas[weight_name_low] = (
                adj_k_low.detach().cpu(),
                resid_low.detach().cpu(),
            )

        cov_low.cpu()
        for x in [layer_ks_low, cur_zs_low, targets_low]:
            x.cpu()
            del x

        gc.collect()
        torch.cuda.empty_cache()

    for i,layer_high in enumerate(hparams.layers_high):
        print(f"\n\nHIGH LAYER {layer_high}\n")

        layer_ks_high = compute_ks(model, tok, requests, hparams, layer_high, context_templates,"high").T 
    
        cur_zs_high = get_module_input_output_at_words(
            model,
            tok,
            z_layer_high,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.rewrite_module_tmp_high,
            fact_token_strategy=hparams.fact_token_high,
        )[1].T

        targets_high = zs_high - cur_zs_high
        print("z error high", torch.linalg.norm(targets_high, dim=0).mean())

        repeat_factor_high = (layer_ks_high.size(1) // targets_high.size(1))
        targets_high = targets_high.repeat_interleave(repeat_factor_high, dim=1)

        force_recompute = False
        cov_high = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp_high.format(layer_high),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
        )
        cov_high = cov_high.cpu()

        layer_ks_high, targets_high = (
            layer_ks_high.double().cpu(),
            targets_high.double().cpu(),
        )
        adj_k_high = torch.linalg.solve(
            hparams.mom2_update_weight_high * cov_high.double() + layer_ks_high @ layer_ks_high.T,
            layer_ks_high,
        )

        resid_high = targets_high / (len(hparams.layers_high) - i)
        resid_high = targets_high / (len(hparams.layers_high) - i)
        upd_matrix_high = (resid_high @ adj_k_high.T).cuda()


        weight_name_high = f"{hparams.rewrite_module_tmp_high.format(layer_high)}.weight"
        upd_matrix_high = upd_matrix_match_shape(upd_matrix_high, weights[weight_name_high].shape)
        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            if model_fp16:
                weights[weight_name_high][...] = weights_copy[weight_name_high] + upd_matrix_high.half()
            else:
                weights[weight_name_high][...] = weights_copy[weight_name_high] + upd_matrix_high.float()
            deltas[weight_name_high] = (
                adj_k_high.detach().cpu(),
                resid_high.detach().cpu(),
            )
        cov_high.cpu()
        for x in [layer_ks_high, cur_zs_high, targets_high]:
            x.cpu()
            del x
        gc.collect()
        torch.cuda.empty_cache()
   
    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    # if last_requests is None:
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
    # COV_CACHE[key] = stat.mom2.moment().float().to("cpu")
    stat.cpu_()
    COV_CACHE[key] = stat

    return (
        torch.inverse(COV_CACHE[key].mom2.moment().float().to("cuda")) if inv else COV_CACHE[key].mom2.moment().float().to("cuda")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by JEEP does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
