# Gradient GA: multi objective

This repo contains code for multi-objective experiment for [original GradGA paper](https://arxiv.org/abs/2502.09860). The code is based on [https://github.com/wenhao-gao/mol_opt](https://github.com/wenhao-gao/mol_opt) Product Metric Optimization (PMO).

## Pre-requisites

```bash
pip install torch 
pip install PyTDC 
pip install rdkit
pip install  dgl
```

Recommended torch version: 2.3.1 and dgl using the following command:

```bash
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

```

## Run the code

The below code is for general run which optimizes the combined objective.

```python
# args:
# seed: random seed for the experiment
# oracles: all the oracle objectives
# max_oracle_calls: maximum sample capacity
python run.py dlp_graph_ga --seed 0 --oracles mestranol_similarity amlodipine_mpo --max_oracle_calls 2500
```
If you want to get each objective statistics separately, use the process_single argument as following

```python
python run.py dlp_graph_ga --oracles mestranol_similarity amlodipine_mpo --process_single Y
```
