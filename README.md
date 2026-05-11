# GEL

This repository is not the official implementation of GEL.

## Requirments

To install requirements:

```bash
uv venv -p 3.12
uv pip install torch==2.4.0 scikit-learn torch_geometric tqdm --torch-backend=cpu
uv pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/repo.html
uv pip install edl-pytorch thop
```

## Running the experiments

To train the model(s) in the paper:

```setup
python main.py
```

Basic usage:

```bash
python main.py --dataset Disney --data_dir ~/datasets/GAD/mat --num_trials 5 --seed 10
```

The random seed is controlled by `--seed`; for multi-trial experiments, trial seeds are generated as:

```text
seed, seed + 1, seed + 2, ...
```

### Original GEL configs in command-line format

```bash
python main.py --dataset cora --lr 0.002 --n_hidden 32 --hops 2 --n_layers 5 --pe_dim 15 --n_epochs 300 --lamda 0.001 --gma 1 --num_trials 5 --seed 10

python main.py --dataset citeseer --lr 0.002 --n_hidden 32 --n_layers 2 --hops 2 --pe_dim 15 --n_epochs 2000 --lamda 0.001 --gma 1 --num_trials 5 --seed 10

python main.py --dataset pubmed --lr 0.002 --n_hidden 16 --n_layers 4 --hops 6 --pe_dim 15 --n_epochs 2000 --lamda 0.001 --gma 5 --num_trials 5 --seed 10

python main.py --dataset BlogCatalog --lr 0.002 --n_hidden 32 --hops 1 --n_layers 4 --pe_dim 15 --n_epochs 2000 --lamda 0.001 --gma 1 --num_trials 5 --seed 10

python main.py --dataset Flickr --lr 0.002 --n_hidden 32 --hops 1 --n_layers 5 --pe_dim 15 --n_epochs 2000 --lamda 0.001 --gma 5 --num_trials 5 --seed 10

python main.py --dataset ACM --lr 0.002 --n_hidden 16 --hops 6 --n_layers 5 --pe_dim 15 --n_epochs 2000 --lamda 0.001 --gma 3 --num_trials 5 --seed 10

python main.py --dataset book --lr 0.02 --n_hidden 8 --hops 12 --n_layers 3 --pe_dim 15 --n_epochs 2000 --lamda 0.001 --gma 5 --num_trials 5 --seed 10

python main.py --dataset Disney --lr 0.0072 --n_hidden 16 --hops 12 --n_layers 3 --pe_dim 15 --n_epochs 2000 --lamda 6 --gma 13 --num_trials 5 --seed 10

python main.py --dataset Enron --lr 0.012 --n_hidden 128 --hops 8 --n_layers 4 --pe_dim 15 --n_epochs 2000 --lamda 0.001 --gma 5 --num_trials 5 --seed 10

python main.py --dataset Reddit --lr 0.02 --n_hidden 128 --hops 3 --n_layers 3 --pe_dim 15 --n_epochs 2000 --lamda 0 --gma 1 --num_trials 5 --seed 10

python main.py --dataset weibo --lr 0.02 --n_hidden 128 --hops 6 --n_layers 3 --pe_dim 15 --n_epochs 1000 --lamda 10 --gma 15 --num_trials 5 --seed 10
```