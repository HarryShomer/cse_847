import os 
import time
import json
import torch

PROJECT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")



def load_config_params(model_name, dataset_name):
    """
    Load params from `best_params/model_name.json` for specific dataset

    Parameters:
    -----------
        model_name: str
            Name of model
        dataset_name: str
            Name of dataset

    Returns:
    --------
    dict
        Params in file
    """
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "hyperparams")

    with open(os.path.join(folder, f"{model_name}.json"), "r") as f:
        params = json.load(f)[dataset_name]
    
    return params


def save_model(model_name, model, optimizer):
    """
    Save the given model's state
    """
    if not os.path.isdir(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        }, 
        os.path.join(CHECKPOINT_DIR, f"{model_name}_{int(time.time())}.tar")
    )


def load_model(model, model_run):
    """
    Load the saved model
    """
    file_path = os.path.join(CHECKPOINT_DIR, f"{model_run}.tar")

    if not os.path.isfile(file_path):
        raise ValueError(f"The file {file_path} doesn't exist")

    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model
