import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from util.globals import *

REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/zsre_mend_eval.json"


class MENDQADataset:
    """
    Dataset of factual knowledge based on zsRE.
    Specifically selected from the QA validation slice from Mitchell et al.
    Project page: http://nlp.cs.washington.edu/zeroshot/
    """
    def __init__(self, data_dir: str, tok: AutoTokenizer, size=None):
        data_dir = Path(data_dir)
        zsre_loc = data_dir / "zsre_mend_eval.json"


        if not zsre_loc.exists():
            print(f"{zsre_loc} does not exist. Downloading from {REMOTE_URL}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(REMOTE_URL, zsre_loc)


        with open(zsre_loc, "r") as f:
            raw = json.load(f)

        data = []
        for i, record in enumerate(raw):
            if 'gpt-j' not in tok.name_or_path:
                
                ans_toks = tok(record["loc_ans"])["input_ids"][1:]
                np = [
                        {
                            "prompt": record["loc"]+ "?" +  " "[:i]+ tok.decode(ans_toks[:i]),
                            "target": tok.decode(ans_toks[i]),
                        }
                        for i in range(len(ans_toks))
                    ]
            else:
                ans_toks = tok(" " + record["loc_ans"])["input_ids"]
                np = [
                        {
                            "prompt": record["loc"]+ "?" + tok.decode(ans_toks[:i]),
                            "target": tok.decode(ans_toks[i]),
                        }
                        for i in range(len(ans_toks))
                    ]
            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": record["src"].replace(record["subject"], "{}"),
                        "subject": record["subject"],
                        "target_new": {"str": record["answers"][0]},
                        "target_true": {"str": "<|endoftext|>"},
                    },
                    "paraphrase_prompts": [record["rephrase"]],
                    "neighborhood_prompts": np,
                }
            )
        self._data = data
        self._data = data[:size]

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
        
