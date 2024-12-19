from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util import nethook

from .jeep_hparams import JEEPHyperParams


def compute_z_joint(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: JEEPHyperParams,
    layer_low: int,
    layer_high: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Get model parameters
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    if 'gpt-j' not in tok.name_or_path:
        target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")[
            "input_ids"
        ][0][2:]
        rewriting_prompts = [
            context.format(request["prompt"]) +" "[:len(tok.decode(target_ids[:-1]))]+tok.decode(target_ids[:-1])
            for context_types in context_templates
            for context in context_types
        ]
    else:
        target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")[
            "input_ids"
        ][0]
        rewriting_prompts = [
            context.format(request["prompt"]) + tok.decode(target_ids[:-1])
            for context_types in context_templates
            for context in context_types
        ]

    # Compile list of rewriting and KL x/y pairs
    
    kl_prompts = ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids


    # Compute indices of the tokens where the fact is looked up
    print("Subject_last")
    lookup_idxs_low = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token_low, verbose=(i==0)
        )
        for i, prompt in enumerate(all_prompts)
    ]
    print("pred_last")
    original_prompts = [
            context.format(request["prompt"])
            for context_types in context_templates
            for context in context_types
        ]
    lookup_idxs_high = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token_high, verbose=(i==0)
        )
        for i, prompt in enumerate(original_prompts+kl_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer_low,layer_high)
    print(f"Rewrite layer LOW is {layer_low}")
    print(f"Rewrite layer HIGH is {layer_high}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if 'gpt-j' not in tok.name_or_path:
        delta_low = torch.zeros((model.config.hidden_size,), requires_grad=True, device="cuda")
        delta_high = torch.zeros((model.config.hidden_size,), requires_grad=True, device="cuda")
    else:
        delta_low = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
        delta_high = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")

    target_init_high, target_init_low, kl_distr_init = None, None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    if 'gpt-j' not in tok.name_or_path:
        def edit_output_fn(cur_out, cur_layer):
            nonlocal target_init_low
            nonlocal target_init_high
            if cur_layer == hparams.mlp_module_tmp.format(layer_low):
                if target_init_low is None:
                    print("Recording initial value of v* low")
                    target_init_low = cur_out[0, lookup_idxs_low[0]].detach().clone()
                for i, idx in enumerate(lookup_idxs_low):
                    cur_out[i, idx, :] += delta_low

            if cur_layer == hparams.attn_module_tmp.format(layer_high):
                if target_init_high is None:
                    print("Recording initial value of v* high")
                    if "attn" in cur_layer:
                        target_init_high = cur_out[0][0, lookup_idxs_high[0]].detach().clone()
                    else:
                        target_init_high = cur_out[0, lookup_idxs_high[0]].detach().clone()
                for i, idx in enumerate(lookup_idxs_high):
                    if "attn" in cur_layer:
                        cur_out[0][i, idx, :] += delta_high
                    else:
                        cur_out[i, idx, :] += delta_high
                        
            return cur_out
    else:
        def edit_output_fn(cur_out, cur_layer):
            nonlocal target_init_low
            nonlocal target_init_high
            if cur_layer == hparams.mlp_module_tmp.format(layer_low):
                if target_init_low is None:
                    print("Recording initial value of v* low")
                    target_init_low = cur_out[0, lookup_idxs_low[0]].detach().clone()
                for i, idx in enumerate(lookup_idxs_low):
                    cur_out[i, idx, :] += delta_low
            if cur_layer == hparams.attn_module_tmp.format(layer_high):
                if target_init_high is None:
                    print("Recording initial value of v* high")
                    if "attn" in cur_layer:
                        target_init_high = cur_out[0][0, lookup_idxs_high[0]].detach().clone()
                    else:
                        target_init_high = cur_out[0, lookup_idxs_high[0]].detach().clone()
                for i, idx in enumerate(lookup_idxs_high):
                    if "attn" in cur_layer:
                        cur_out[0][i, idx, :] += delta_high
                    else:
                        cur_out[i, idx, :] += delta_high

            return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta_low,delta_high], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    nll_loss_factor=1
    wd_low = hparams.v_weight_decay_low
    wd_high = hparams.v_weight_decay_high
    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer_low),
                hparams.layer_module_tmp.format(layer_low),
                hparams.attn_module_tmp.format(layer_high),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs_low[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        full_repr = tr[hparams.layer_module_tmp.format(loss_layer)].output[0][
            : len(rewriting_prompts)
        ]
        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w + lm_b, dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()
        max_probs = torch.max(log_probs, dim = 2)[0]
        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor_low * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )

        weight_decay_low = wd_low * (
            torch.norm(delta_low) / torch.norm(target_init_low) ** 2
        )
        
        weight_decay_high = wd_high * (
            torch.norm(delta_high) / torch.norm(target_init_high) ** 2
        )

        loss = nll_loss_factor*nll_loss + kl_loss + weight_decay_low + weight_decay_high
        prob = torch.exp(-nll_loss_each).mean().item()
        print(
            f"nll{nll_loss_factor} loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay_low.item(), 3)}+{np.round(weight_decay_high.item(), 3)} "
            f"avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )

        if torch.exp(-nll_loss_each).mean().item()>=hparams.min_loss:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm_low = hparams.clamp_norm_factor_low * target_init_low.norm()
        if delta_low.norm() > max_norm_low:
            with torch.no_grad():
                delta_low[...] = delta_low * max_norm_low / delta_low.norm()

        max_norm_high = hparams.clamp_norm_factor_high * target_init_high.norm()
        if delta_high.norm() > max_norm_high:
            with torch.no_grad():
                delta_high[...] = delta_high * max_norm_high / delta_high.norm()

    target_low = target_init_low + delta_low
    print(
        f"LOW: Init norm {target_init_low.norm()} | Delta norm {delta_low.norm()} | Target norm {target_low.norm()}"
    )
    target_high = target_init_high + delta_high
    print(
        f"HIGH: Init norm {target_init_high.norm()} | Delta norm {delta_high.norm()} | Target norm {target_high.norm()}"
    )
    return target_low,target_high



def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "pred_last":
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = "pred_last"
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )

    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif fact_token_strategy == "pred_last":
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken="pred_last",
        )[0][0]
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret],skip_special_tokens=True),
        )

    return ret

