import torch 
import argparse
import numpy as np
from tqdm import tqdm

import optuna
from optuna.trial import TrialState

from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

from utils import *
from models import *


parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Model to run. One of 'GCN', 'GAT', 'APPNP', 'SGC")
parser.add_argument("--dataset", help="One of 'cora', 'citeseer', 'pubmed'")

parser.add_argument("--tune", help="Tune Model via Optuna", action='store_true', default=False)

parser.add_argument("--epochs", help="Number of epochs to run", default=250, type=int)
parser.add_argument("--bs", help="Batch size to use for training", default=1, type=int)
parser.add_argument("--lr", help="Learning rate to use while training", default=1e-3, type=float)
parser.add_argument("--decay", help="LR Decay", default=.999, type=float)

## Hyperparameters
parser.add_argument("--layers", help="Number of layers for model", default=2, type=int)
parser.add_argument("--dropout", help="Dropout to apply", default=0.2, type=float)
parser.add_argument("--num-heads", help="Number Attention heads to use for GAT", default=8, type=int)
parser.add_argument("--iters", help="Smoothing iterations for APPNP", default=10, type=int)
parser.add_argument("--alpha", help="Teleportation probability for APPNP", default=0.1, type=float)
parser.add_argument("--mlp-drop", help="MLP Dropout to apply", default=0.1, type=float)

parser.add_argument("--validation", help="Test on validation set every n epochs", type=int, default=5)
parser.add_argument("--early-stop", help="Number of validation scores to wait for an increase before stopping", default=10, type=int)
parser.add_argument("--save-as", help="Model to save model as", default=None, type=str)

parser.add_argument("--test", help="Whether testing model", action='store_true', default=False)
parser.add_argument("--model-run", help="Name of checkpoint file to load", type=str)

parser.add_argument("--device", help="Device to run on", type=str, default="cuda")
args = parser.parse_args()

MODEL = args.model.lower()
DEVICE = args.device

# Randomness!!!
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def get_dataset():
    """
    Get the specific dataset we are working on
    """
    dataset = args.dataset.lower()

    if dataset not in ['cora', 'citeseer', 'pubmed']:
        raise ValueError(f"No dataset named `{dataset}`")

    return Planetoid(PROJECT_DIR, "cora", "public")



def create_model(dataset):
    """
    Create the model for the params passed

    Parameters:
    -----------
        dataset: torch_geometric.dataset.Planetoid
            dataset object

    Returns:
    --------
    torch.nn.Module
        Model implemeted
    """
    # Constant!
    num_feats = dataset.num_features
    num_hidden = 64
    num_classes = dataset.num_classes

    if MODEL == "gcn":
        model = GCN(num_feats, num_hidden, num_classes, dropout=args.dropout, num_layers=args.layers)
    elif MODEL == "gat":
        model = GAT(num_feats, num_hidden, num_classes, heads=args.num_heads, dropout=args.dropout, num_layers=args.layers)
    elif MODEL == "appnp":
        model = APPNP(num_feats, num_hidden, num_classes, args.iters, args.alpha, appnp_drop=args.dropout, mlp_drop=args.mlp_drop)
    elif MODEL == "sgc":
        model = SGC(num_feats, num_classes, args.layers)
    else:
        raise ValueError(f"No model named `{args.model}`")
    
    model = model.to(DEVICE)

    return model 


def eval_model(model, data, mask):
    """
    Evaluate model on val/test set

    Parameters:
    -----------
        model:
        dataloader:

    Returns:
    --------
    float
        accuracy
    """
    mask = data.val_mask if mask == "val" else data.test_mask
    
    model.eval()

    out = model(data)
    acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())

    return acc



# def eval_on_test_set(model, data):
#     """
#     Evaluate the model on the test set and print results

#     Parameters:
#     -----------
#         model: torch.nn.Module
#             Model we are training
    
#     Returns:
#     --------
#     None
#     """
#     dataset
#     test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

#     model = load_model(model, args.model_run)
#     f1score = eval_model(model, test_loader)

#     print("Test F1 Score:", f1score)



def train_model(model, data, lr, decay):
    """
    Parameters:
    -----------
        model: torch.nn.Module
            Model we are training
    
    Returns:
    --------
    None
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: decay ** e)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    val_scores = []

    model.train()
    for epoch in range(1, args.epochs+1):
        optimizer.zero_grad()

        out = model(data)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch} Loss:", round(loss, 4))
        
        # if epoch % args.validation == 0:
        #     val_scores.append(eval_model(model, data, "val"))

        #     print(f"Epoch {epoch} Val Acc Score:", round(val_scores[-1], 4))

        #     # Start checking after accumulate more than val_mrr
        #     if len(val_scores) >= args.early_stop and np.argmax(val_scores[-args.early_stop:]) == 0:
        #         print(f"Validation loss hasn't improved in the last {args.early_stop} validation mean rank scores. Stopping training now!", flush=True)
        #         break
        
        lr_scheduler.step()
    
    # Only use param when passed bec. otherwise None
    if args.save_as:
        save_model(args.save_as, model, optimizer)
    else:
        save_model(args.model.upper(), model, optimizer)




def run_tuned_model(trial):
    """
    Function run by optuna
    """
    data = get_dataset()

    lr = trial.suggest_float("lr", 1e-4, 1e-2)
    dropout = trial.suggest_float("dropout", 0, 0.4)
    decay = trial.suggest_categorical("decay", [0.995, 0.999, 1])
    layers = trial.suggest_categorical("layers", [2, 3, 4])

    if MODEL == "appnp":
        iters = trial.suggest_categorical("iters", [5, 10, 50, 100])
        alpha = trial.suggest_categorical("alpha", [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
        model = APPNP(data.num_features, 64, data.num_classes, iters, alpha, appnp_drop=dropout)
    elif MODEL == "gcn":
        model = GCN(data.num_features, 64, data.num_classes, dropout=dropout, num_layers=layers)
    elif MODEL == "gat":
        model = GAT(data.num_features, 64, data.num_classes, heads=args.num_heads, dropout=dropout, num_layers=layers)
    else:
         model = SGC(data.num_features, data.num_classes, layers)


    train_model(model, data, lr, decay)


def tune_model():
    """
    Via Optuna
    """
    study = optuna.create_study(
                direction="maximize",
                pruner=optuna.pruners.MedianPruner(),
            )

    study.optimize(run_tuned_model, n_trials=250)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))






def main():

    tune_model()
    # model = create_model()
    # dataset = get_dataset()

    # if not args.test:
    #     train_model(model, dataset, args.lr, args.decay)
    # else:
    #     eval_on_test_set(model, dataset)


if __name__ == "__main__":
    main()
