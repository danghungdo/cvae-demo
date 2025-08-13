import optuna
import torch
import torch.optim as optim
import torch.nn.functional as F
import json
import argparse
from sklearn.model_selection import train_test_split
from src.model import CVAE, loss_function
from src.dataloader import DataLoader
from src.utils import one_hot_encode
from src.dataset import FashionMNISTDataset

def load_hyperparam_ranges(path="src/config/hyperparam_ranges.json"):
    with open(path, "r") as f:
        return json.load(f)

def train(model, device, train_loader, optimizer):
    model.train()
    train_loss = 0
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        labels_one_hot = one_hot_encode(labels, num_classes=10)
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, labels_one_hot)
        loss, bce, kld = loss_function(recon_batch, data, mu, logvar)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    return train_loss / len(train_loader.dataset)

def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            labels_one_hot = one_hot_encode(labels, num_classes=10)
            
            recon_batch, mu, logvar = model(data, labels_one_hot)
            loss, _, _ = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.item()
    
    return test_loss / len(test_loader.dataset)

def objective(trial, device, train_loader, val_loader, ranges, epochs):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial: The Optuna trial object.
        device: The device to run the model on.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        ranges: Dictionary containing hyperparameter ranges.
        epochs: Number of training epochs.

    Returns:
        float: The average validation loss.
    """
    # Suggest hyperparameters from loaded ranges
    hidden_dim = trial.suggest_int('hidden_dim', ranges['hidden_dim']['low'], ranges['hidden_dim']['high'])
    latent_dim = trial.suggest_int('latent_dim', ranges['latent_dim']['low'], ranges['latent_dim']['high'])
    lr = trial.suggest_float('lr', ranges['lr']['low'], ranges['lr']['high'], log=True)

    model = CVAE(hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train(model, device, train_loader, optimizer)

    # Evaluate on validation set
    avg_val_loss = evaluate(model, device, val_loader)
    return avg_val_loss

def optimize(full_dataset, device, n_trials=20, batch_size=64, epochs=30):
    ranges = load_hyperparam_ranges()
    # Split dataset into train and validation
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.3, random_state=42)
    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_subset = torch.utils.data.Subset(full_dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    def wrapped_objective(trial):
        return objective(trial, device, train_loader, val_loader, ranges, epochs)

    study = optuna.create_study(direction="minimize", storage="sqlite:///optuna.db", study_name="cvae_optuna", load_if_exists=True)
    study.optimize(wrapped_objective, n_trials=n_trials, show_progress_bar=True)
    
    # Save best hyperparameters
    best_params = study.best_params
    with open("src/config/best_hyperparams.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print("Best hyperparameters saved to best_hyperparams.json:", best_params)
    return study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hyperparam-ranges", type=str, default="src/config/hyperparam_ranges.json", help="Path to hyperparameter ranges JSON file")
    args = parser.parse_args()

    # Load dataset
    dataset = FashionMNISTDataset(root="./data", train=True)

    # Run optimization
    study = optimize(dataset, args.device, args.n_trials, args.batch_size, args.epochs)

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
    