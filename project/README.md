# CSE 847 Project

In this project we tried benchmarking several Graph Neural Networks (GNNs) on the task of node classification. Specifically, we benchmarked 4 different GNNs on three citation datasets. We tuned the hyperparameters for each GNN and dataset combination on the validation set. We then used the best performing hyperparameters to train a final model and record the test performance. 

## Graph Neural Networks

1. [Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (GCN)
2. [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (GAT)
3. [Simple Graph Convolution](https://proceedings.mlr.press/v97/wu19e.html) (SGC)
4. [Approximate personalized propagation of neural predictions](https://arxiv.org/abs/1810.05997) (APPNP).

## Datasets

The three citation datasets are via [Sen et al.](https://ojs.aaai.org/index.php/aimagazine/article/view/2157). We specifically follow the splits provided by [Yang et al.](https://proceedings.mlr.press/v48/yanga16.html).

1. Cora
2. Citeseer
3. Pubmed

## Running

You can train a model by running the `src/run.py` file while passing the model and dataset. Below is an example of running GAT on Citeseer:
```
python src/run.py --model gat --dataset citeseer
```
This will run the model with the default hyperparameters. Alternate hyperparameter values can be passed as command line arguments. The full list can be accessed by passing the `help` flag to the `src/run.py` file.
```
python src/run.py --help
``` 

### Hyperparameter Tuning

The models are tuned via [Optuna](https://optuna.org/) library. The [Tree-structured Parzen Estimator](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler) algorithm is used to find the optimal hyperparameter values. This is run by specifying the model and dataset to tune and passing the `tune` flag. Below is an example of tuning GCN on the cora dataset.
```
python src/run.py --model gcn --dataset cora --tune
```

**NOTE:** The tuned hyperparameters can already be found in the `hyperparams` folder. Running any model/dataset with `--tune` will not modify these files. You will need to manually edit the hyperparameter files with the output.

### Multiple seeds

In order to account for weight initialization, we run each model and dataset with the tuned hyperparameters for multiple different random seed. We run each with 10 random seeds. This is done via the `tuned_multiple_seeds.sh` script. This uses the hyperparameter values found in the `hyperparams` folder. The script takes the model and dataset as the two arguments.
```
bash tuned_multiple_seeds.sh gcn pubmed
```

## Requirements

The project is implemented in python using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/). The required python packages can be found in the `requirements.txt` file.