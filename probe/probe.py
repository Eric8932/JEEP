import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
import numpy as np
from copy import deepcopy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer,LlamaForCausalLM


from dsets import (
    MENDQADataset,
    MultiCounterFactDataset,
)

from jeep import JEEPHyperParams
from util import nethook
from util.globals import *



DS_DICT = {
    "zsre": (MENDQADataset),
    "mcf": (MultiCounterFactDataset),
}



def get_words_idxs_in_templates(
    tok: AutoTokenizer, context_templates: str, words: str, subtoken: str
) -> int:
    """
    Given list of template strings, each with *one* format specifier
    (e.g. "{} plays basketball"), and words to be substituted into the
    template, computes the post-tokenization index of their last tokens.
    """

    assert all(
        tmp.count("{}") == 1 for tmp in context_templates
    ), "We currently do not support multiple fill-ins for context"

    # Compute prefixes and suffixes of the tokenized context
    fill_idxs = [tmp.index("{}") for tmp in context_templates]
    prefixes, suffixes = [
        tmp[: fill_idxs[i]] for i, tmp in enumerate(context_templates)
    ], [tmp[fill_idxs[i] + 2 :] for i, tmp in enumerate(context_templates)]
    words = deepcopy(words)

    # Pre-process tokens
    for i, prefix in enumerate(prefixes):
        if len(prefix) > 0:
            assert prefix[-1] == " "
            prefix = prefix[:-1]

            prefixes[i] = prefix
            words[i] = f" {words[i].strip()}"

    # Tokenize to determine lengths
    assert len(prefixes) == len(words) == len(suffixes)
    n = len(prefixes)
    batch_tok = tok([*prefixes, *words, *suffixes])
    if 'gpt-j' not in tok.name_or_path:
        prefixes_len, words_len, suffixes_len = [ 
            [len(el) for el in batch_tok.input_ids[i : i + n]] 
            for i in range(0, n * 3, n) 
            ]
    else:
        prefixes_tok, words_tok, suffixes_tok = [
            batch_tok[i : i + n] for i in range(0, n * 3, n)
        ]
        prefixes_len, words_len, suffixes_len = [
            [len(el) for el in tok_list]
            for tok_list in [prefixes_tok, words_tok, suffixes_tok]
        ]

    # Compute indices of last tokens
    if subtoken == "subject_last" :
        if 'gpt-j' not in tok.name_or_path:
            return [ [prefixes_len[i]+ words_len[i]-3 + (1 if prefixes_len[i] ==1 else 0)
                     +(1 if len(words[i])>1 and words[i][1] in [str(number) for number in range(10)]+['Ō','İ',"Ş","Ç","Đ","Æ"] and prefixes_len[i] !=1  else 0 )
                     ]
                for i in range(n)
            ]
        else:
            return [
                [
                    prefixes_len[i]
                    + words_len[i]
                    - (1 if subtoken == "last" or suffixes_len[i] == 0 else 0)
                ]
                for i in range(n)
            ]
    elif subtoken == "subject_first" :
        return [ [prefixes_len[i]] for i in range(n) ]
    elif subtoken == "subject_subseq":
        if 'gpt-j' not in tok.name_or_path:
            return [ [prefixes_len[i]+ words_len[i]-2 + (1 if prefixes_len[i] ==1 else 0)
                     - (1 if suffixes_len[i] == 1 else 0)] for i in range(n) ]
        else:
            return [ [ prefixes_len[i]+ words_len[i] - (1 if suffixes_len[i] == 0 else 0) ] for i in range(n) ]
        
    

def main(
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    model_path = None,
):
    # Set algorithm-specific variables
    params_class = JEEPHyperParams

    new_name = model_name.split('/')[-1]
    new_name += '_'
    new_name += ds_name
    new_name += '_'
    new_name += str(dataset_size_limit)

    save_dir = RESULTS_DIR  / "probe" /new_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {save_dir}")

    # Get run hyperparameters
    params_path =  HPARAMS_DIR / "JEEP" / hparams_fname
    hparams = params_class.from_json(params_path)



    use_llama = 'gpt-j' not in model_path
    
    if type(model_name) is str:
        print("Instantiating model")

        if use_llama:
    
            model = LlamaForCausalLM.from_pretrained(model_path).cuda()
            tok = LlamaTokenizer.from_pretrained(model_path)
            tok.pad_token = '<unk>'
            print(f"vocab length={len(tok.get_vocab())}")
            
            tok_loc = LlamaTokenizer.from_pretrained(model_path,padding_side="left")
            tok_loc.pad_token = '<unk>'
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
            tok = AutoTokenizer.from_pretrained(model_path)
            tok.pad_token = tok.eos_token

            tok_loc = AutoTokenizer.from_pretrained(model_path,padding_side="left")
            tok_loc.pad_token = tok_loc.eos_token

        model.config._name_or_path = model_name
        print(model.config._name_or_path.replace("/", "_"))
    else:
        model, tok = model_name
        model_name = model.config._name_or_path
    print(model.config._name_or_path)
    print(tok.name_or_path)
    
    ds_class = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)
    

    
    result_l= np.load(f"{DATA_DIR}/related_words.npy",allow_pickle=True)
    print(len(result_l))


    save_dic = {}
    record_num = 0
    problem_set = [7,8,132,376]

    stop_words = np.load(f"{DATA_DIR}/stop_words.npy",allow_pickle=True)
    if "" in stop_words:
        stop_words.remove("")
    stop_words = [str(s) for s in stop_words]
    stop_words_tokens = tok(stop_words).input_ids
    sw_pf = []#stopwords_phrase_first
    for ts in stop_words_tokens:
        sw_pf.append(ts[1])
    sw_pf = list(set(sw_pf))

    process_key_list = ['original_answer','target_answer','factual_related', 'nonfactual_related']

    def agg_key(result_l,key_list):
        set_dic_phrase_first_l = []
        for k in key_list:
            temp_l = result_l[record_num][k]
            if "" in temp_l:
                temp_l.remove("")
            if len(temp_l) == 0:
                temp_l = ["1"]
            all_tokens = tok(temp_l).input_ids
            
            for ts in all_tokens:
                set_dic_phrase_first_l.append(ts[1])

        return list(set(set_dic_phrase_first_l))


    for record_chunks in chunks(ds, 1):
        if record_num in problem_set:
            record_num +=1
            continue
        record = record_chunks[0]
        set_dic_phrase_first = {}
        # set_dic_phrase_first['original_answer'] = agg_key(result_l,['original_answer'])
        set_dic_phrase_first['original_answer'] = [tok(record["requested_rewrite"]['target_true']['str']).input_ids[1]]
        
        set_dic_phrase_first['factual_related'] = agg_key(result_l,['factual_related'])
        set_dic_phrase_first['nonfactual_related'] = agg_key(result_l,['nonfactual_related'])
        set_dic_phrase_first['target_answer'] = [tok(record["requested_rewrite"]['target_new']['str']).input_ids[1]]


        left_l = list(set(result_l[record_num]["left"]))
 
        record_num += 1

        

        record_dic = {}

        context_templates = [record["requested_rewrite"]['prompt']]
        text_list = [record["requested_rewrite"]['prompt'].format(record["requested_rewrite"]['subject'])]
        words = [record["requested_rewrite"]['subject']]*len(text_list)

        subject_last_index = get_words_idxs_in_templates(tok,context_templates,words,"subject_last")
        last_index = [[-1]*len(text_list)]


        input_tok = tok(text_list,return_tensors="pt").to("cuda")

        lm_w, ln_f = (
            nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
            nethook.get_module(model, hparams.ln_f_module),
        )
        try:
            lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
        except LookupError as _:
            lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

        with nethook.TraceDict(
            module=model,
            layers=[ hparams.layer_module_tmp.format(l) for l in range(hparams.v_loss_layer+1)] +
                   [ hparams.mlp_module_tmp.format(l) for l in range(hparams.v_loss_layer+1)] +
                   [ hparams.attn_module_tmp.format(l) for l in range(hparams.v_loss_layer+1)]+
                   [hparams.emb_module_tmp],
            retain_input=False,
            retain_output=True,
        ) as tr:
            _ = model(**input_tok).logits

        layer_w_ln = [[(
            torch.softmax(tr[hparams.layer_module_tmp.format(l)].output[0][text_i][subject_last_index[text_i]]@ lm_w + lm_b,dim=-1).detach().cpu(),
            torch.softmax(tr[hparams.layer_module_tmp.format(l)].output[0][text_i][last_index[text_i]]@ lm_w + lm_b,dim=-1).detach().cpu(),
            ) if l !=0
            else
            (
            torch.softmax(tr[hparams.layer_module_tmp.format(l)].output[0][text_i][subject_last_index[text_i]]@ lm_w + lm_b,dim=-1).detach().cpu(),
            torch.softmax(tr[hparams.layer_module_tmp.format(l)].output[0][text_i][last_index[text_i]]@ lm_w + lm_b,dim=-1).detach().cpu(),
            torch.softmax(tr[hparams.emb_module_tmp].output[text_i][subject_last_index[text_i]]@ lm_w + lm_b,dim=-1).detach().cpu(),
            torch.softmax(tr[hparams.emb_module_tmp].output[text_i][last_index[text_i]]@ lm_w + lm_b,dim=-1).detach().cpu(),
            )
                            
                     for l in range(hparams.v_loss_layer+1)] for text_i in range(len(text_list)) ]
        
       
        def get_logits_prob_rank_top(rep_list):
            l = []

            for k in process_key_list:
                for j in range(len(rep_list)):
                    rank_l = rep_list[j][0].sort(descending=True).indices.sort().indices[set_dic_phrase_first[k]].numpy()
                    l.append((np.mean(rank_l),len(set_dic_phrase_first[k])))
                    prob_l = rep_list[j][0][set_dic_phrase_first[k]].numpy()
                    l.append((sum(prob_l)))


            for j in range(len(rep_list)):
                rank_l = rep_list[j][0].sort(descending=True).indices.sort().indices[sw_pf].numpy()
                l.append((np.mean(rank_l),len(sw_pf)))
                prob_l = rep_list[j][0][sw_pf].numpy()
                l.append((sum(prob_l)))


                rank_l = rep_list[j][0].sort(descending=True).indices.sort().indices[left_l].numpy()
                l.append((np.mean(rank_l),len(left_l)))
                prob_l = rep_list[j][0][left_l].numpy()
                l.append((sum(prob_l)))

                l.append((np.mean(rep_list[j][0].numpy())))

            return [l]

        
        pre_w_ln = []
        for i in range(len(layer_w_ln)):
            l_list = []
            for j in range(len(layer_w_ln[0])):
                rep = layer_w_ln[i][j]
                l_list += get_logits_prob_rank_top(rep)

            pre_w_ln.append(l_list)
        record_dic['pre_w_ln'] = pre_w_ln
        del pre_w_ln


        save_dic[record["case_id"]] = record_dic

        if record_num % 100 == 0:
            np.save(save_dir/("probe_res_"+str(record_num)+".npy"),save_dic)
            del save_dic
            save_dic = {}

        torch.cuda.empty_cache()
    print(f"Results be stored at {save_dir}")
    np.save(save_dir/("probe_res.npy"),save_dic)

        
def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        choices=["EleutherAI/gpt-j-6B","llama"],
        default="Llama-7B",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="Llama-7B.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/swh/UER/TencentPretrain/models/vicuna-7b",
        help="Local model path",
        required=True,
    )

    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        model_path = args.model_path,
    )
