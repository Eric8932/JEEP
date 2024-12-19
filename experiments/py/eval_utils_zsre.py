"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain
from time import time
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer



def compute_rewrite_quality_zsre(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :return: Dictionary containing rewriting metrics
    """
    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]


    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
    ]
    # Flatten all the evaluated prefixes into one list.
    if 'gpt-j' not in tok.name_or_path:
        target_tok = tok(target_new["str"])["input_ids"][1:]
    else:
        target_tok = tok(" " + target_new["str"])["input_ids"]
    inp_prompts_og = list(chain(*prob_prompts))

    if 'gpt-j' in tok.name_or_path:
        inp_prompts = [
            el + tok.decode(target_tok[:i])
            for el in inp_prompts_og
            for i in range(len(target_tok))
        ]
        inp_targets = [
            tok.decode(target_tok[i])
            for _ in range(len(inp_prompts_og))
            for i in range(len(target_tok))
        ]
        stuff_probs = test_batch_prediction_acc(model, tok, inp_prompts, inp_targets)
    else:
        stuff_probs = test_batch_prediction_acc_llama(model, tok, inp_prompts_og,target_new["str"] )

   
    probs = stuff_probs

    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(
        [l * len(target_tok) for l in map(len, prob_prompts)]
    ).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_correct": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
            ]
        )
    }

    return ret


def test_batch_prediction_acc(model, tok, prompts: typing.List[str], target):
    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
        ).to("cuda")

    with torch.no_grad():
        logits = model(**prompt_tok).logits
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)#
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        ans = torch.argmax(gathered, dim=1)

        correct_id = tok(target, padding=True, return_tensors="pt").to("cuda")[
            "input_ids"
        ]
        # Temporary hack to deal with foreign characters.
        correct_id = correct_id[:, 0].squeeze()

    return (ans == correct_id).detach().cpu().numpy().tolist()

def test_batch_prediction_acc_llama(model, tok, prompts: typing.List[str], target):

    prefix_lens = [len(n) for n in tok(prompts)["input_ids"]]
    prompt_tok = tok(
    [
        f"{prefix} {target}"
        for prefix in prompts
    ],
    padding=True,
    return_tensors="pt",
    ).to("cuda")
    target_tok = tok(target)["input_ids"][1:]
    target_len = len(target_tok)
    with torch.no_grad():
        logits = model(**prompt_tok).logits
    res_list = []

    for i in range(logits.size(0)):
        for j in range(target_len):
            if logits[i, prefix_lens[i] + j - 1, :].argmax().item() != target_tok[j]:
                res_list.append(False)
            else:
                res_list.append(True)

    return res_list


def zsre_loc_batch(model, tokenizer, data_loader):
    acc_list = []
    with torch.no_grad():
        model.eval()
        model.to('cuda')
        for _, batch in enumerate(data_loader):
            src = [b["src"] for b in batch["raw"]]
            trg = [b["trg"] for b in batch["raw"]]
            acc_list += test_batch_prediction_acc_batch(model,tokenizer,src,trg)
            
    return acc_list
   


def test_batch_prediction_acc_batch(model, tok, prompts: typing.List[str], target):
    prefix_lens = [len(n) for n in tok(prompts)["input_ids"]]
    
    prompt_tok = tok(
    [
        f"{prompts[i]} {target[i]}"
        for i in range(len(prompts))
    ],
    padding=True,
    return_tensors="pt",
    ).to("cuda")
    if 'gpt-j' not in tok.name_or_path:
        target_tok = [tok(n)["input_ids"][1:] for n in target ]
    else:
        target_tok = [tok(" "+n)["input_ids"] for n in target ]

    target_len = [len(n) for n in target_tok]
    with torch.no_grad():
        logits = model(**prompt_tok).logits
    res_list = []
    for i in range(logits.size(0)):
        temp_list = []
        for j in range(target_len[i]):
            if logits[i, prefix_lens[i] + j - 1, :].argmax().item() != target_tok[i][j]:
                temp_list.append(False)
            else:
                temp_list.append(True)
        res_list.append(temp_list)
    return res_list


