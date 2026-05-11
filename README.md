# GEL

This repository is the fork implementation of GEL.

## Abstract

Detecting anomalies in graph-structured data is challenging due to the limited availability of labeled anomaly data, pushing the development of unsupervised techniques. Graph autoencoders (GAEs) are widely used, leveraging reconstruction errors of graph topology and node attributes to identify anomalies. However, dependence on reconstruction errors alone makes these methods prone to noise and overfitting issues. To address this, we introduce Graph Evidential Learning (GEL), a novel probabilistic approach that integrates evidential learning into the reconstruction process. GEL models node attributes and graph structure using evidential distributions, capturing two distinct uncertainty types: graph uncertainty and reconstruction uncertainty. These uncertainties are incorporated into the anomaly scoring mechanism, enhancing detection robustness. Comprehensive experiments show that GEL outperforms existing methods and demonstrates strong robustness to noise and structural variations.

## Requirments

To install requirements:

```bash
uv venv -p 3.12
uv pip install torch==2.4.0 scikit-learn torch_geometric --torch-backend=cpu
uv pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/repo.html
uv pip install edl-pytorch thop
#conda env create -f environment.yml
```

## Running the experiments

To train the model(s) in the paper:

```setup
python main.py
```

# Reference
The codebase of this repo is partially based on [CoCo](https://github.com/eighto6/CoCo) repository.