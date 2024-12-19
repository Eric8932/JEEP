import unicodedata
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.logit_lens import LogitLens


def generate_interactive(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    top_k: int = 5,
    max_out_len: int = 200,
    compare_against: Optional[AutoModelForCausalLM] = None,
    use_logit_lens: bool = False,
    layer_module_tmp: str = "transformer.h.{}",
    ln_f_module: str = "transformer.ln_f",
    lm_head_module: str = "lm_head",
):
    """
    Puts generation in a loop. Allows users to repeatedly provide inputs
    with which text is generated.
    """

    if use_logit_lens:
        llens_gen = LogitLens(
            model,
            tok,
            layer_module_tmp,
            ln_f_module,
            lm_head_module,
            disabled=not use_logit_lens,
        )
        if compare_against:
            llens_vanilla = LogitLens(
                compare_against,
                tok,
                layer_module_tmp,
                ln_f_module,
                lm_head_module,
                disabled=not use_logit_lens,
            )

    while True:
        prompt = input("Enter a prompt: ").strip(" \r\t\n")

        print(
            f"Argument Model: "
            f"{generate_fast(model, tok, [prompt], n_gen_per_prompt=1, top_k=top_k, max_out_len=max_out_len)}"
        )
        if compare_against:
            print(
                f"Baseline Model: "
                f"{generate_fast(compare_against, tok, [prompt], n_gen_per_prompt=1, top_k=top_k, max_out_len=max_out_len)}"
            )

        if use_logit_lens:
            inp_prompt = tok([prompt], padding=True, return_tensors="pt").to(
                next(model.parameters()).device
            )

            with llens_gen:
                model(**inp_prompt)
            print("\n--- Argument Model Logit Lens ---")
            llens_gen.pprint()

            if compare_against:
                with llens_vanilla:
                    compare_against(**inp_prompt)
                print("--- Baseline Model Logit Lens ---")
                llens_vanilla.pprint()

        print()


# from torch.cuda.amp import autocast  # for automatic mixed precision
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer

def generate_fast(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    n_gen_per_prompt: int = 1,
    top_k: int = 5,
    max_out_len: int = 200,
) -> List[str]:
    """
    Fast, parallelized auto-regressive text generation with top-k sampling using Hugging Face's generate method.
    """
    # Unroll prompts and tokenize
    if 'gpt-j' not in tok.name_or_path:
        tok = LlamaTokenizer.from_pretrained(tok.name_or_path,padding_side="left")
        tok.pad_token = '<unk>'
    else:
        tok = AutoTokenizer.from_pretrained(tok.name_or_path,padding_side="left")
        tok.pad_token = tok.eos_token
    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
    inp_tok = tok(inp, padding=True, return_tensors="pt").to(next(model.parameters()).device)

    # Define generation parameters
    gen_args = {
        'max_length': max_out_len, 
        'do_sample': True, 
        'top_k': top_k,
    }

    out_tok = model.generate(inp_tok['input_ids'], **gen_args)

    # Decode and post-process text
    txt = [tok.decode(x,skip_special_tokens = True) for x in out_tok.detach().cpu().numpy().tolist()]
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        for x in txt
    ]

    return txt


