import torch 
import optuna
import argparse
import numpy as np

import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from utils import *
from models import *


parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Model to run. One of 'GCN', 'GAT', 'APPNP', 'SGC'")
parser.add_argument("--dataset", help="One of 'cora', 'citeseer', 'pubmed'")

parser.add_argument("--tune", help="Tune Model via Optuna", action='store_true', default=False)
parser.add_argument("--run-from-config", help="Whether run with tuned parms", action="store_true", default=False)

parser.add_argument("--epochs", help="Number of epochs to run", default=200, type=int)
parser.add_argument("--lr", help="Learning rate to use while training", default=1e-3, type=float)
parser.add_argument("--decay", help="LR Decay", default=1, type=float)
parser.add_argument("--l2", help="L2 Regularization", default=0, type=float)

## Hyperparameters
parser.add_argument("--num-hid", help="Num hidden dimension", default=64, type=int)
parser.add_argument("--layers", help="Number of layers for model", default=2, type=int)
parser.add_argument("--dropout", help="Dropout to apply", default=0.2, type=float)
parser.add_argument("--num-heads", help="Number Attention heads to use for GAT", default=8, type=int)
parser.add_argument("--iters", help="Smoothing iterations for APPNP", default=10, type=int)
parser.add_argument("--alpha", help="Teleportation probability for APPNP", default=0.1, type=float)

parser.add_argument("--save-as", help="Model to save model as", default=None, type=str)
parser.add_argument("--device", help="Device to run on", type=str, default="cuda")
parser.add_argument("--seed", help="Random Seed", default=None, type=int)

args = parser.parse_args()

MODEL = args.model.lower()
DEVICE = args.device


def get_dataset():
    """
    Get the specific dataset we are working on
    """
    dataset = args.dataset.lower()

    if dataset not in ['cora', 'citeseer', 'pubmed']:
        raise ValueError(f"No dataset named `{dataset}`")

    return Planetoid(PROJECT_DIR, dataset, "public", transform=T.NormalizeFeatures())



def create_model(dataset):
    """
    Create the model for the params passed

    Parameters:
    -----------
        dataset: torch_geometric.dataset.Planetoid
            dataset

    Returns:
    --------
    torch.nn.Module
        Model implemeted
    """
    num_hidden = args.num_hid
    num_feats = dataset.num_features
    num_classes = dataset.num_classes

    if MODEL == "gcn":
        model = GCN(num_feats, num_hidden, num_classes, dropout=args.dropout, num_layers=args.layers)
    elif MODEL == "gat":
        model = GAT(num_feats, num_hidden, num_classes, heads=args.num_heads, dropout=args.dropout, num_layers=args.layers)
    elif MODEL == "appnp":
        model = APPNP(num_feats, num_hidden, num_classes, args.iters, args.alpha, dropout=args.dropout)
    elif MODEL == "sgc":
        model = SGC(num_feats, num_classes, args.layers)
    else:
        raise ValueError(f"No model named `{args.model}`")
    
    model = model.to(DEVICE)

    return model 


def create_model_from_config(dataset, model_params):
    """
    Create the model using the config params

    Parameters:
    -----------
        dataset: torch_geometric.dataset.Planetoid
            dataset
        model_params: dict
            Dict of model parameters from config

    Returns:
    --------
    torch.nn.Module
        Model implemeted
    """
    num_feats = dataset.num_features
    num_classes = dataset.num_classes
    hid_dim = model_params['hidden_dim']

    if MODEL == "gcn":
        model = GCN(num_feats, hid_dim, num_classes, dropout=model_params['dropout'], num_layers=2)
    elif MODEL == "gat":
        model = GAT(num_feats, hid_dim, num_classes, dropout=model_params['dropout'], num_layers=2)
    elif MODEL == "appnp":
        model = APPNP(num_feats, hid_dim, num_classes, model_params['iters'], model_params['alpha'], dropout=model_params['dropout'])
    elif MODEL == "sgc":
        model = SGC(num_feats, num_classes, 2)
    else:
        raise ValueError(f"No model named `{args.model}`")
    
    model = model.to(DEVICE)

    return model 


def eval_model(model, data, mask):
    """
    Evaluate model on val/test set

    Portion taken from here - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py

    Parameters:
    -----------
        model: torch.nn.Mo
            model obj
        data: PyTorch-Geometruc data obj
            data obj
        mask: str
            Either 'val' or 'test'

    Returns:
    --------
    float
        accuracy
    """
    model.eval()
    mask = data.val_mask if mask == "val" else data.test_mask

    out = model(data)
    pred = out[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

    return acc



def train_model(model, data, lr, decay, l2):
    """
    Parameters:
    -----------
        model: torch.nn.Module
            Model we are training
    
    Returns:
    --------
    None
    """
    val_scores = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    if decay != 1:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: decay ** e)
    else:
        lr_scheduler = None
        
    for epoch in range(1, args.epochs+1):
        model.train()
        optimizer.zero_grad()

        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()
        
        val_scores.append(eval_model(model, data, "val"))
        # print(f"Epoch {epoch}: {val_scores[-1]:.2f}")
        
        if lr_scheduler: lr_scheduler.step()
    
    return val_scores[-1]


def run_model():
    """
    Run model either from args or config
    """
    dataset = get_dataset()
    data = dataset[0].to(args.device)

    # If not from config we run via cmd line args
    if args.run_from_config:
        params = load_config_params(MODEL, args.dataset.lower())
        model = create_model_from_config(dataset, params)
        train_model(model, data, params['lr'], params['decay'], params['l2'])
    else:
        model = create_model(dataset)
        train_model(model, data, args.lr, args.decay, args.l2)
    
    print("Test Acc:", eval_model(model, data, "test"))



def run_tuned_model(trial):
    """
    Function run by optuna
    """
    data = get_dataset()

    decay = trial.suggest_categorical("decay", [0.995, 0.999, 1])
    lr = trial.suggest_categorical("lr", [1e-3, 5e-3, 1e-2, 5e-2])
    hid_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64])
    dropout = trial.suggest_categorical("dropout", [0, 0.2, 0.4, 0.6, 0.8])
    l2 = trial.suggest_categorical("l2", [0, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])

    if MODEL == "appnp":
        iters = trial.suggest_categorical("iters", [5, 10, 25, 50, 100])
        alpha = trial.suggest_categorical("alpha", [0, .1, .25, .5, .75, 1])
        model = APPNP(data.num_features, hid_dim, data.num_classes, iters, alpha, dropout=dropout)
    elif MODEL == "gcn":
        model = GCN(data.num_features, hid_dim, data.num_classes, dropout=dropout, num_layers=args.layers)
    elif MODEL == "gat":
        model = GAT(data.num_features, hid_dim, data.num_classes, heads=args.num_heads, dropout=dropout, num_layers=args.layers)
    else:
        model = SGC(data.num_features, data.num_classes, args.layers)
    
    model = model.to(args.device)
    data = data[0].to(args.device)

    return train_model(model, data, lr, decay, l2)


def tune_model():
    """
    Via Optuna
    """
    study = optuna.create_study(
                direction="maximize",
                pruner=optuna.pruners.MedianPruner(),
            )

    study.optimize(run_tuned_model, n_trials=500)

    print(f"{args.dataset.capitalize()} Best Value:", study.best_value, flush=True)
    print(f"{args.dataset.capitalize()} Best Parameters:", study.best_params, flush=True)



def main():
    # Randomness!!!
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.tune:
        tune_model()
    else:
        run_model()


if __name__ == "__main__":
    main()
