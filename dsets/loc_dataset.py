import os
import jsonlines
import random
import torch
from torch.utils.data import Dataset
import json

def process_prompt(ori_str):
    return ori_str+ "?"

class ZSRE_Loc(Dataset):
    def __init__(self, tokenizer, data_path, max_length=128, dataset_size=1000,num_edits = None,current=None):
        """
        :param tokenizer:
        :param data_path:
        :param max_length:
        :param validation:
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []
        self.size = dataset_size
        self.max_length = max_length

        data_path = os.path.join(data_path, 'zsre_mend_eval.json')

        with open(data_path,'r') as f:
            all_d = json.load(f)
            for case_id,d in enumerate(all_d):
                d1 = {}
                d1['case_id'] = case_id
                d1['input'] = process_prompt(d['loc'])
                d1['output'] = d['loc_ans']
                d1['rephrases']  = []
                    
                self.data.append(d1)
        self.data = self.data[:self.size]
        if num_edits is not None and current is not None:
            self.data = self.data[current*num_edits:(current+1)*num_edits]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {
            "src": self.data[item]["input"],
            "trg": self.data[item]["output"],
            "rephrases": self.data[item]["rephrases"],
            'case_id':self.data[item]['case_id']
        }

    def collate_fn(self, batch):
        batches = {}
        for name in ("src",):
            tokenizer_input = [b[name] for b in batch]
            tokenizer_output = self.tokenizer(
                tokenizer_input, return_tensors="pt",
                padding=True, max_length=self.max_length,
                truncation=True,
            )
            for k, v in tokenizer_output.items():
                batches["{}_{}".format(name, k)] = v
        batches["raw"] = batch
        return batches


class MCF_Loc(Dataset):
    def __init__(self, tokenizer, data_path, max_length=128, dataset_size=1000,num_edits=None,current=None):
        """
        :param tokenizer:
        :param data_path:
        :param max_length:
        :param validation:
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []
        self.size = dataset_size
        self.max_length = max_length

        data_path = os.path.join(data_path, 'multi_counterfact.json')
        
        with open(data_path,'r') as f:
            all_d = json.load(f)
            for i,d in enumerate(all_d):
                d1 = {}
                d1['input'] = d['neighborhood_prompts']
                d1['output'] = d['requested_rewrite']['target_true']['str']
                d1['output_new'] = d['requested_rewrite']['target_new']['str']
                d1['rephrases']  = []
                d1['relation_id'] = d["requested_rewrite"]["relation_id"]
                d1['target_new_id'] = d['requested_rewrite']['target_new']['id']
                d1['subject'] = d['requested_rewrite']['subject']
                d1['case_id'] = d['case_id']
                
                self.data.append(d1)
        self.data = self.data[:self.size]
        if num_edits is not None and current is not None:
            self.data = self.data[current*num_edits:(current+1)*num_edits]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {
            "src": self.data[item]["input"],
            "trg": self.data[item]["output"],
            "trg_new": self.data[item]["output_new"],
            "rephrases": self.data[item]["rephrases"],
            "relation_id": self.data[item]["relation_id"],
            "target_new_id": self.data[item]["target_new_id"],
            "subject": self.data[item]["subject"],
            'case_id':self.data[item]['case_id']
        }

    def collate_fn(self, batch):
        batches = {}
        for name in ("src",):
            tokenizer_input = [sin_src for b in batch for sin_src in b[name]]
            tokenizer_output = self.tokenizer(
                tokenizer_input, return_tensors="pt",
                padding=True, max_length=self.max_length,
                truncation=True,
            )
            for k, v in tokenizer_output.items():
                batches["{}_{}".format(name, k)] = v
        batches["raw"] = batch
        return batches

