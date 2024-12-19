"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_counterfact` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain
from time import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer


def compute_rewrite_quality_counterfact(
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
    which_correct = [
        [0 for _ in range(len(rewrite_prompts))],
        [0 for _ in range(len(paraphrase_prompts))],
        # [1 for _ in range(len(neighborhood_prompts))],
    ]
    # Flatten all the evaluated prefixes into one list.
    probs, targets_correct = test_batch_prediction(
        model,
        tok,
        list(chain(*prob_prompts)),
        list(chain(*which_correct)),
        target_new["str"],
        target_true["str"],
    )

    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_corrects = [
        targets_correct[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))
    ]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
            ]
        )
    } 
    ret2 = {
        f"{key}_correct": ret_corrects[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
            ]
        )
    }
    ret.update(ret2)
    return ret

def test_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    which_correct: str,
    target_new: str,
    target_true: str,
):
    """
    which_correct: Which target to consider correct. Either 0 for "new" or 1 for "true".
    """

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]

    prompt_tok = tok(
    [
        f"{prefix} {suffix}"
        for prefix in prefixes
        for suffix in [target_new, target_true]
    ],
    padding=True,
    return_tensors="pt",
    ).to("cuda")

    if 'gpt-j' not in tok.name_or_path:
        a_tok, b_tok = (tok(f"{n}")["input_ids"][1:] for n in [target_new, target_true])
    else:
        a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len

        # Compute suffix probabilities
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len

        # Compute accuracy on new targets
        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            for j in range(cur_len):
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]

                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok:
                    correct = False
                    break
            targets_correct.append(correct)

    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ], targets_correct


def test_batch_prediction_batch(
    model,
    tok,
    prefixes: typing.List[str],
    target_new: str,
    target_true: str,
):
    """
    which_correct: Which target to consider correct. Either 0 for "new" or 1 for "true".
    """

    prefix_lens = [len(n)  for i in range(len(prefixes)) for n in tok(prefixes[i])["input_ids"]]

    prompt_tok = tok(
    [
        f"{prefix} {suffix}"
        for i in range(len(prefixes))
        for prefix in prefixes[i]
        for suffix in [target_new[i], target_true[i]]
    ],
    padding=True,
    return_tensors="pt",
    ).to("cuda")

    if 'gpt-j' not in tok.name_or_path:
        a_tok = [tok(f"{n}")["input_ids"][1:] for n in target_new]
        b_tok = [tok(f"{n}")["input_ids"][1:] for n in target_true]
    else:
        a_tok = [tok(f" {n}")["input_ids"] for n in target_new]
        b_tok = [tok(f" {n}")["input_ids"] for n in target_true]
    choice_a_len = [len(n) for n in a_tok]
    choice_b_len = [len(n) for n in b_tok]

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    probs = np.zeros((logits.size(0),), dtype=np.float32)

    for i in range(logits.size(0)):
        cur_len = choice_a_len[i//20] if i % 2 == 0 else choice_b_len[i//20]

        for j in range(cur_len):
            cur_tok = (a_tok[i//20] if i % 2 == 0 else b_tok[i//20])[j]
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len
    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ]


def mcf_loc_batch(model, tokenizer_right,tokenizer, data_loader):
    acc_list = []
    with torch.no_grad():
        model.eval()
        model.to('cuda')
        for _, batch in enumerate(data_loader):

            target_true = [b["trg"] for b in batch["raw"] ]
            target_new = [b["trg_new"] for b in batch["raw"] ]
            input_prompt = [b["src"] for b in batch["raw"] ]
            acc_list += test_batch_prediction_batch(model,tokenizer_right,input_prompt,target_new,target_true)

        return acc_list