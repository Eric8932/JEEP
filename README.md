# Joint Knowledge Editing for Information Enrichment and Probability Promotion

Code and Data for the AAAI 2025 paper *Joint Knowledge Editing for Information Enrichment and Probability Promotion*.

## Setup

### Code
The Code is based on [MEMIT](https://github.com/kmeng01/memit). Requirements and Code Structure are consistent with its.

### Data
For the two editing datasets, Multi-COUNTERFACT and zsRE, we download them from [link](https://memit.baulab.info/data/dsets) .
And we put the data in the data/ folder.


## Probe Expriment

### Run the Probe

```
# Example for running the probe on Llama model and mcf datasets for 10 samples.
python3 -m probe.probe \
    --model_name=llama \
    --model_path= \
    --hparams_fname=llama_7b_mcf.json \
    --dataset_size_limit=10 --ds_name mcf   
```

### Plot the probe results in fig.2 and fig.5

```
python3 prob.plot_fig2
python3 prob.plot_fig5
```



## Running the code


### Running JEEP

```
# Example for running JEEP on zsre datasets.

python3 -m experiments.evaluate \
    --alg_name JEEP \
    --model_name=llama \
    --model_path= \
    --hparams_fname=llama_7b_zsre.json \
    --num_edits=10 \
    --dataset_size_limit=10 --ds_name zsre   \
    --loc_data_size 10 --loc_batch_size 8  --model_fp16 ;
```

### Running the baselines

You could replace the alg_name for different baseline methods, ds_name for different datasets.
Note that model_path for GPT-J model should contain "gpt-j".

```
# Example for running MEMIT on mcf datasets.

python3 -m experiments.evaluate \
    --alg_name MEMIT \
    --model_name=llama \
    --model_path= \
    --hparams_fname=llama_7b.json \
    --num_edits=10 \
    --dataset_size_limit=10 --ds_name mcf   \
    --loc_data_size 10 --loc_batch_size 8  --model_fp16 ;
```





### Parse the results

After the editing, you could call experiments.prase_results to show the evaluation metrics.

```
# Example for parse results on JEEP on llama-7b and zsre datasets. edit_num specific the number of results.

python3 -m experiments.parse_results \
    --dataset zsre \
    --results_path results/JEEP/llama_zsre_10_10_llama_7b_zsre \
    --edit_num 1 
```

## How to Cite

```bibtex
@article{shi2024joint,
  title={Joint Knowledge Editing for Information Enrichment and Probability Promotion},
  author={Shi, Wenhang and Chen, Yiren and Bian, Shuqing and Zhang, Xinyi and Zhao, Zhe and Hu, Pengfei and Lu, Wei and Du, Xiaoyong},
  journal={arXiv preprint arXiv:2412.17872},
  year={2024}
}
```

## Contact information

If you have any question, please contact Wenhang Shi via wenhangshi@ruc.edu.cn.
