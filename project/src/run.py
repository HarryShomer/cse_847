import torch 
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader

from utils import *
from models import *


parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Model to run. One of 'GCN', 'GAT', 'APPNP'")

parser.add_argument("--epochs", help="Number of epochs to run", default=400, type=int)
parser.add_argument("--bs", help="Batch size to use for training", default=1, type=int)
parser.add_argument("--lr", help="Learning rate to use while training", default=1e-3, type=float)
parser.add_argument("--decay", help="LR Decay", default=.999, type=float)

## Hyperparameters
parser.add_argument("--layers", help="Number of layers for model", default=2, type=int)
parser.add_argument("--dropout", help="Dropout to apply", default=0.2, type=float)
parser.add_argument("--num-heads", help="Number Attention heads to use for GAT", default=8, type=int)
parser.add_argument("--iters", help="Smoothing iterations for APPNP", default=10, type=int)
parser.add_argument("--alpha", help="Teleportation probability for APPNP", default=0.1, type=float)
parser.add_argument("--mlp-drop", help="MLP Dropout to apply", default=0.2, type=float)

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


def create_model():
    """
    Create the model for the params passed

    Returns:
    --------
    torch.nn.Module
        Model implemeted
    """
    # Constant!
    num_feats = 50
    num_hidden = 256
    num_classes = 121

    if MODEL == "gcn":
        model = GCN(num_feats, num_hidden, num_classes, dropout=args.dropout, num_layers=args.layers)
    elif MODEL == "gat":
        model = GAT(num_feats, num_hidden, num_classes, heads=args.num_heads, dropout=args.dropout, num_layers=args.layers)
    elif MODEL == "appnp":
        model = APPNP(num_feats, num_hidden, num_classes, args.iters, args.alpha, appnp_drop=args.dropout, mlp_drop=args.mlp_drop)
    else:
        ValueError(f"No model named `{args.model}`")
    
    model = model.to(DEVICE)

    return model 


def eval_model(model, data_loader):
    """
    Evaluate model on val/test set

    Parameters:
    -----------
        model:
        dataloader:

    Returns:
    --------
    float
        f1 Score
    """
    all_lbls, all_preds = [], []

    model.eval()

    for graph in data_loader:
        graph = graph.to(DEVICE)

        out = model(graph)
        preds = torch.sigmoid(out)
        preds = (preds > 0.5)
        
        all_preds.extend(preds.tolist())
        all_lbls.extend(graph.y.tolist())

    return f1_score(all_lbls, all_preds, average='micro')



def eval_on_test_set(model):
    """
    Evaluate the model on the test set and print results

    Parameters:
    -----------
        model: torch.nn.Module
            Model we are training
    
    Returns:
    --------
    None
    """
    test_data = PPI(os.path.join(PROJECT_DIR, "data"), split='test')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    model = load_model(model, args.model_run)
    f1score = eval_model(model, test_loader)

    print("Test F1 Score:", f1score)



def train_model(model):
    """
    Parameters:
    -----------
        model: torch.nn.Module
            Model we are training
    
    Returns:
    --------
    None
    """
    train_data =  PPI(os.path.join(PROJECT_DIR, "data"), split='train')
    val_data = PPI(os.path.join(PROJECT_DIR, "data"), split='val')
    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: args.decay ** e)

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")

    val_scores = []

    model.train()
    for epoch in range(1, args.epochs+1):
        epoch_loss = 0

        # for graph in tqdm(train_loader, desc=f"Epoch {epoch}"):
        for graph in train_loader:
            optimizer.zero_grad()
            graph = graph.to(DEVICE)

            out = model(graph)
            loss = loss_fn(out, graph.y)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch} Loss:", round(epoch_loss, 4))
        
        if epoch % args.validation == 0:
            val_scores.append(eval_model(model, val_loader))

            print(f"Epoch {epoch} Val F1 Score:", round(val_scores[-1], 4))

            # Start checking after accumulate more than val_mrr
            if len(val_scores) >= args.early_stop and np.argmax(val_scores[-args.early_stop:]) == 0:
                print(f"Validation loss hasn't improved in the last {args.early_stop} validation mean rank scores. Stopping training now!", flush=True)
                break
        
        lr_scheduler.step()
    
    # Only use param when passed bec. otherwise None
    if args.save_as:
        save_model(args.save_as, model, optimizer)
    else:
        save_model(args.model.upper(), model, optimizer)
    

def main():
    model = create_model()

    if not args.test:
        train_model(model)
    else:
        eval_on_test_set(model)


if __name__ == "__main__":
    main()
