import json
import numpy as np
from pathlib import Path
import scipy.stats as stats
import os

def conf_95(data):
    mean = np.mean(data)
    sem = stats.sem(data)
    confidence_level = 0.95
    degrees_freedom = len(data) - 1
    confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean, sem)
    return (np.round(mean,1),np.round(confidence_interval[1]-mean,1))

def pro_zsre(p,i):
    p1 = os.path.join(p,f"edits-case_{i}.json")
    with open(p1,'r') as f:
        edit_res = json.load(f)
    
    edit_l,para_l = [],[]
    for r in edit_res:
        edit_n,para_n = 0,0
        edit_suc_n,para_suc_n = 0,0
        if 'post' not in r:
            continue
        for edit in r['post']['rewrite_prompts_correct']:
            edit_n+=1
            if edit:
                edit_suc_n+=1
        for para in r['post']['paraphrase_prompts_correct']:
            para_n+=1
            if para:
                para_suc_n+=1
        edit_l.append(edit_suc_n/edit_n*100)
        para_l.append(para_suc_n/para_n*100)
    try:
        p2 = os.path.join(p,f"loc_{i}.npy")
        res = np.load(p2,allow_pickle=True).item()
        res_l = [r.count(True)/len(r)*100 for r in res['acc']]
        return np.mean(edit_l),np.mean(para_l) ,np.mean(res_l)
    except:
        pass

def pro_mcf(p,i):
    p1 = os.path.join(p,f"edits-case_{i}.json")
    with open(p1,'r') as f:
        edit_res = json.load(f)
    edit_n,para_n = 0,0
    edit_suc_n,para_suc_n = 0,0
    for r in edit_res:
        if 'post' not in r:
            continue
        for edit in r['post']['rewrite_prompts_probs']:
            edit_n += 1
            if edit['target_new']<edit['target_true']:
                edit_suc_n+=1
        for para in r['post']['paraphrase_prompts_probs']:
            para_n += 1
            if para['target_new']<para['target_true']:
                para_suc_n+=1
    try:
        p2 = p2 = os.path.join(p,f"loc_{i}.npy")
        res_l = np.load(p2,allow_pickle=True).item()
        acc_l = [r['target_new']>r['target_true'] for r in res_l['acc']]
        return edit_suc_n/edit_n*100,para_suc_n/para_n*100,acc_l.count(True)/len(acc_l)*100
    except:
        pass


PRO_DICT = {
    "zsre": pro_zsre,
    "mcf": pro_mcf,
}

def main(
        dataset,
        results_path,
        edit_num
):
    process_func = PRO_DICT[dataset]
    l = [[],[],[]]
    for i in range(edit_num):
        res = process_func(results_path,i)
        if res is None:
            continue
        for j in range(3):
            l[j].append(res[j])
    res= [conf_95(l1) for l1 in l  ]
    print(res)
    print("Score",round(3/(1/res[0][0]+1/res[1][0]+1/res[2][0])))
    print("Efficacy",res[0])
    print("Generalization",res[1])
    print("Locality",res[2])






if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="",
        required=True,
    )

    parser.add_argument(
        "--results_path",
        default="",
        required=True,
    )
    parser.add_argument(
        "--edit_num",
        type=int,
        required=True,
    )


    args = parser.parse_args()

    main(
        args.dataset,
        results_path=args.results_path,
        edit_num = args.edit_num
    )