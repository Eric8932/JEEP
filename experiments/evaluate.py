import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
import numpy as np
import os,gc

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer,LlamaForCausalLM

from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    MENDQADataset,
    MultiCounterFactDataset,
    ZSRE_Loc,
    MCF_Loc,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact,mcf_loc_batch
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre,zsre_loc_batch
from jeep import JEEPHyperParams, apply_jeep_to_model
from memit import MEMITHyperParams, apply_memit_to_model
from pmet import PMETHyperParams, apply_pmet_to_model
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import *

ALG_DICT = {
    "JEEP": (JEEPHyperParams, apply_jeep_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "PMET": (PMETHyperParams, apply_pmet_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}


DS_DICT = {
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre,zsre_loc_batch),
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact,mcf_loc_batch),
}

LOC_DICT = {
    "zsre": ZSRE_Loc,
    "mcf": MCF_Loc,
}


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    conserve_memory: bool,
    num_edits: int = 1,
    use_cache: bool = False,
    model_path = None,
    loc_data_size=100,
    loc_batch_size = 4,
    model_fp16 = False,
):

    params_class, apply_algo = ALG_DICT[alg_name]

    new_name = model_name.split('/')[-1]
    new_name += '_'

    new_name += ds_name
    new_name += '_'
    new_name += str(dataset_size_limit)
    new_name += '_'
    new_name += str(num_edits)
   

    new_name += "_"+hparams_fname.split(".")[0]
    

    run_dir = RESULTS_DIR / alg_name / new_name
    run_dir.mkdir(parents=True, exist_ok=True)


    print(f"Results will be stored at {run_dir}")

    params_path = HPARAMS_DIR / alg_name / hparams_fname
    hparams = params_class.from_json(params_path)

    shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model

    if type(model_name) is str:
        print("Instantiating model")
        if model_name in ['llama']:
            if model_fp16:
                model = LlamaForCausalLM.from_pretrained(model_path,revision="float16",torch_dtype=torch.float16).cuda()
            else:
                model = LlamaForCausalLM.from_pretrained(model_path).cuda()
            tok = LlamaTokenizer.from_pretrained(model_path)
            tok.pad_token = '<unk>'
            print(f"vocab length={len(tok.get_vocab())}")
            tok_loc = LlamaTokenizer.from_pretrained(model_path,padding_side="left")
            tok_loc.pad_token = '<unk>'
        else:
            if model_fp16:
                model = AutoModelForCausalLM.from_pretrained(model_path,revision="float16",torch_dtype=torch.float16,).cuda()
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


    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method,ds_eval_loc = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, size=dataset_size_limit,tok=tok)
    

    # Get cache templates--Use cached k/v pairs
    cache_template = None
    if use_cache:
        mfp16 = "_fp16" if model_fp16 else ""
        if alg_name == "JEEP":
            cache_template = (
                    KV_DIR
                    / f"{model_name.replace('/', '_')}_{alg_name}{mfp16}"
                    / f"new2_{ds_name}_minloss{{}}_layermlp{{}}_layermlp{{}}_lr{{}}_step{{}}_cnmlp{{}}_cnmhsa{{}}_wdmlp{{}}_wdmhsa{{}}_klmlp{{}}_klmhsa{{}}_case{{}}.npz"
                )
        elif alg_name=="PMET":
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}{mfp16}"
                / f"{ds_name}_layer_{{}}_{{}}_clamp_{{}}_vlr_{hparams.v_lr}_clampnorm_{hparams.clamp_norm_factor}_wdmlp_{hparams.v_weight_decay}_wdattn_{hparams.v_weight_decay_attn}_klmlp_{hparams.kl_factor}_case_{{}}.npz"
            )
        elif alg_name =="MEMIT":
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{'MEMIT'}{mfp16}"
                / f"layer_{ds_name}_layer_{{}}_wd_{{}}_kl_{hparams.kl_factor}_clampnorm_{hparams.clamp_norm_factor}_mom2_{{}}_case_{{}}.npz"
            )
        else:
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}{mfp16}"
                / f"{ds_name}_layer_{{}}_wd_{{}}_mom2_{{}}_case_{{}}.npz"
            )
       
        print(f"Will load cache from {cache_template}")

    # Iterate through dataset
    chunk_time = 0
    for record_chunks in chunks(ds, num_edits):
        loc_data = LOC_DICT[ds_name](tok_loc,DATA_DIR,dataset_size=loc_data_size,num_edits = num_edits, current=chunk_time)
        loc_loader = DataLoader(loc_data, batch_size=loc_batch_size, collate_fn=loc_data.collate_fn)#chunk
        model.train()

        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template)

        start = time()
        model.train()
        edited_model, weights_copy,deltas = apply_algo(
            model,
            tok,
            [
                {"case_id": record["case_id"], **record["requested_rewrite"]}
                for record in record_chunks
            ],
            hparams,
            copy=False,
            return_orig_weights=True,
            model_fp16 = model_fp16,
            **args_conserve_memory,
            **etc_args,
        )
        exec_time = time() - start
        print("Execution took", exec_time)
        model = edited_model
        model.eval()

        gc.collect()
        torch.cuda.empty_cache()

        model.eval()



        final_res_list = []
        for record in record_chunks:
            metrics = {
                "case_id": record["case_id"],
                "post": ds_eval_method(
                    model,
                    tok,
                    record,
                ),
            }
            final_res_list.append(metrics)
            torch.cuda.empty_cache()
        final_res_list.append({"edit_exec_time":exec_time})
        case_result_template = str(run_dir / "edits-case_{}.json")
        out_file_final = Path(case_result_template.format(chunk_time))
        with open(out_file_final, "w") as f:
            json.dump(final_res_list, f, indent=1)


        loc_res = {}
        if args.ds_name == 'zsre':
            equal_acc_prev = ds_eval_loc(
                model,
                tok,
                loc_loader,
            )
            loc_res['acc'] = equal_acc_prev
            
        else:
            res = ds_eval_loc(
                model,
                tok,
                tok_loc,
                loc_loader,
            )
            torch.cuda.empty_cache()
            loc_res['acc'] = res


        np.save(run_dir/(f"loc_{chunk_time}.npy"),loc_res)

        with torch.no_grad():
            for k, v in weights_copy.items():
                w = nethook.get_parameter(model, k)
                w[...] = v
        print("Chunk",chunk_time,"Finished")
        chunk_time +=1
    
    print(f"Results are saved in {run_dir}")


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT", "PMET","ROME", "FT", "MEND","JEEP"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. ",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=[ "EleutherAI/gpt-j-6B","llama"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="llama_7b.json",
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
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Local model path",
        required=True,
    )
    parser.add_argument(
        "--loc_data_size",
        type=int,
        default=100,
        help="Size for loc data",
    )
    parser.add_argument(
        "--loc_batch_size",
        type=int,
        default=4,
        help="Batch-Size for loc dataloader",
    )
    parser.add_argument(
        "--model_fp16",
        action="store_true",
        help="Using fp16 for models",
    )




    parser.set_defaults(conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.conserve_memory,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        model_path = args.model_path,
        loc_data_size = args.loc_data_size,
        loc_batch_size = args.loc_batch_size,
        model_fp16 = args.model_fp16,

    )
