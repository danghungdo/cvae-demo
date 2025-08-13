import argparse
import json
import torch
from src.model import CVAE, loss_function
from src.utils import one_hot_encode, save_model
from src.dataset import FashionMNISTDataset
from src.dataloader import DataLoader
from src.evaluate import evaluate
from src.wandb_utils import (
    init_wandb, log_losses, log_generated_samples, finish_wandb
)

def train(model, device, train_loader, optimizer, epoch):
    """
    Train the model for one epoch.
    Args:
        model: model to train
        device: device to use
        train_loader: train loader
        optimizer: optimizer
        epoch: current epoch
    Returns:
        avg_loss: average loss for the epoch
        avg_bce: average binary cross-entropy loss
        avg_kld: average Kullback-Leibler divergence loss
    """
    model.train()
    train_loss = 0
    bce_loss_sum = 0
    kld_loss_sum = 0
    num_batches = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        # one hot encoding the labels
        labels_one_hot = one_hot_encode(labels, num_classes=10)

        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data, labels_one_hot)
        loss, bce, kl = loss_function(recon_batch, data, mu, logvar)

        loss.backward()
        train_loss += loss.item()
        bce_loss_sum += bce.item()
        kld_loss_sum += kl.item()
        num_batches += 1
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)]\t"
                f"Loss: {loss.item() / len(data):.6f} "
                f"(BCE: {bce.item() / len(data):.6f}, KL: {kl.item() / len(data):.6f})"
            )

    avg_loss = train_loss / len(train_loader.dataset)
    avg_bce = bce_loss_sum / len(train_loader.dataset)
    avg_kld = kld_loss_sum / len(train_loader.dataset)
    # logger.log(epoch, avg_loss, avg_bce, avg_kld)

    print(
        f"====> Epoch: {epoch} Average loss: {avg_loss:.4f} "
        f"(BCE: {avg_bce:.4f}, KL: {avg_kld:.4f})"
    )

    return avg_loss, avg_bce, avg_kld


def load_hyperparam_config(path="src/config/best_hyperparams.json"):
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="src/config/best_hyperparams.json", help="Path to hyperparameter configuration")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--use-wandb", action="store_true", default=True, help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default="cvae-demo", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-tags", nargs="+", default=None, help="W&B tags for the run")
    args = parser.parse_args()

    hyperparam_config = load_hyperparam_config(args.config_path)
    assert "latent_dim" in hyperparam_config, f"latent_dim must be specified in {args.config_path}"
    assert "hidden_dim" in hyperparam_config, f"hidden_dim must be specified in {args.config_path}"
    assert "lr" in hyperparam_config, f"lr must be specified in {args.config_path}"

    # Add command line arguments to config
    config = hyperparam_config.copy()
    config.update({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
    })
    

    # Initialize W&B if requested
    wandb_run = None
    if args.use_wandb:
        wandb_run = init_wandb(
            config=config,
            project_name=args.wandb_project,
            run_name=args.wandb_run_name,
            tags=args.wandb_tags
        )
        if wandb_run is None:
            print("Warning: W&B initialization failed. Continuing without W&B logging.")
            args.use_wandb = False

    model = CVAE(
        hidden_dim=config.get("hidden_dim"),
        latent_dim=config.get("latent_dim"),
        num_classes=10,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr"))

    # Initialize loss history for combined plotting
    loss_history = {} if args.use_wandb else None

    train_dataset = FashionMNISTDataset(root="./data", train=True)
    test_dataset = FashionMNISTDataset(root="./data", train=False)
    train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size"), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.get("batch_size"), shuffle=False)

    print(f"Starting training for {args.epochs} epochs on {args.device}")
    print(f"Model: CVAE with {config.get('hidden_dim')} hidden dim, {config.get('latent_dim')} latent dim")

    for epoch in range(1, args.epochs + 1):
        # Training
        train_loss, train_bce, train_kld = train(model, args.device, train_loader, optimizer, epoch)

        # Evaluation
        test_loss, test_bce, test_kld = None, None, None
        test_loss, test_bce, test_kld = evaluate(model, args.device, test_loader)
        
        # Log losses and generated samples together
        if args.use_wandb:
            # Get generated samples figure
            generated_samples_fig = log_generated_samples(model, args.device, epoch, return_fig=True)
            # Log everything in one call
            log_losses(epoch, train_loss, train_bce, train_kld, test_loss, test_bce, test_kld, loss_history, generated_samples_fig)


    # Finish W&B logging
    if args.use_wandb:
        finish_wandb()

    print("Training completed!")
    
    # save the model
    save_model(model, "weights", "last_model.pth")
    

if __name__ == "__main__":
    main()
