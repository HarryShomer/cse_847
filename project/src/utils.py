import os 
import time
import torch

PROJECT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")


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
