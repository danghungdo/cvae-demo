import wandb

import matplotlib.pyplot as plt
import os
from src.evaluate import generate_images

from dotenv import load_dotenv
load_dotenv()


def setup_wandb_auth():
    """
    Setup W&B authentication using API key from environment variables
    """
    api_key = os.getenv('WANDB_API_KEY')

    if api_key:
        os.environ['WANDB_API_KEY'] = api_key

        try:
            wandb.login(key=api_key)
            print("W&B authentication successful")
            return True
        except Exception as e:
            print(f"Warning: W&B login failed: {e}")
            print("Please check your WANDB_API_KEY in the .env file")
            return False
    else:
        print("Warning: WANDB_API_KEY not found in environment variables")
        print("Please create a .env file with your W&B API key")
        return False


def init_wandb(config, project_name, run_name, tags):
    """
    Initialize Weights & Biases for experiment tracking

    Args:
        config: Dictionary containing hyperparameters and configuration
        project_name: Name of the W&B project (defaults to env var or config)
        run_name: Optional name for this specific run
        tags: Optional list of tags for organizing runs

    Returns:
        wandb run object or None if authentication fails
    """
    if not setup_wandb_auth():
        print("W&B authentication failed. Continuing without W&B logging.")
        return None

    assert project_name is not None, "project_name must be specified"

    try:
        run = wandb.init(
            project=project_name,
            entity=None,
            name=run_name,
            config=config,
            tags=tags or [],
            mode="online",
            reinit=True
        )

        # Log model architecture info
        wandb.config.update({
            "model_type": "CVAE",
            "dataset": "FashionMNIST",
            "hidden_dim": config.get("hidden_dim"),
            "latent_dim": config.get("latent_dim"),
            "beta_weight": config.get("beta_weight"),
            "num_classes": 10
        })

        print("W&B initialized successfully")
        print(f"Project: {project_name}")
        print(f"Run: {run.name}")
        print(f"URL: {run.url}")

        return run

    except Exception as e:
        print(f"Failed to initialize W&B: {e}")
        return None


def log_losses(epoch, train_loss, train_bce, train_kld,
               test_loss=None, test_bce=None, test_kld=None, loss_history=None, generated_samples_fig=None):
    """
    Log training and validation losses to W&B

    Args:
        epoch: Current epoch number
        train_loss: Training total loss
        train_bce: Training BCE loss
        train_kld: Training KLD loss
        test_loss: Optional test total loss
        test_bce: Optional test BCE loss
        test_kld: Optional test KLD loss
        loss_history: Optional dictionary to store loss history for plotting
        generated_samples_fig: Optional figure with generated samples
    """
    log_dict = {
        "epoch": epoch,
        "train/total_loss": train_loss,
        "train/bce_loss": train_bce,
        "train/kld_loss": train_kld,
    }

    if test_loss is not None:
        log_dict.update({
            "test/total_loss": test_loss,
            "test/bce_loss": test_bce,
            "test/kld_loss": test_kld,
        })

    # Create combined loss plots if history is provided
    if loss_history is not None:
        # Update history
        if 'epochs' not in loss_history:
            loss_history['epochs'] = []
            loss_history['train_total'] = []
            loss_history['train_bce'] = []
            loss_history['train_kld'] = []
            loss_history['test_total'] = []
            loss_history['test_bce'] = []
            loss_history['test_kld'] = []

        loss_history['epochs'].append(epoch)
        loss_history['train_total'].append(train_loss)
        loss_history['train_bce'].append(train_bce)
        loss_history['train_kld'].append(train_kld)

        if test_loss is not None:
            loss_history['test_total'].append(test_loss)
            loss_history['test_bce'].append(test_bce)
            loss_history['test_kld'].append(test_kld)
        else:
            loss_history['test_total'].append(None)
            loss_history['test_bce'].append(None)
            loss_history['test_kld'].append(None)

        # Create combined plots and add to log_dict
        loss_curves_fig = create_loss_plots(loss_history, epoch)
        if loss_curves_fig is not None:
            log_dict["loss_curves"] = wandb.Image(loss_curves_fig)
            plt.close(loss_curves_fig)  # Close the figure after logging
    
    # Add generated samples if provided
    if generated_samples_fig is not None:
        log_dict["generated_images"] = wandb.Image(generated_samples_fig)
        plt.close(generated_samples_fig)  # Close the figure after logging
    
    # Single log call per epoch
    wandb.log(log_dict)


def create_loss_plots(loss_history, epoch):
    """
    Create combined loss plots for train and test

    Args:
        loss_history: Dictionary containing loss history
        epoch: Current epoch number
        
    Returns:
        fig: The matplotlib figure (caller responsible for closing)
    """
    epochs = loss_history['epochs']

    # Filter out None values for test losses
    test_epochs = []
    test_total = []
    test_bce = []
    test_kld = []

    for i, e in enumerate(epochs):
        if loss_history['test_total'][i] is not None:
            test_epochs.append(e)
            test_total.append(loss_history['test_total'][i])
            test_bce.append(loss_history['test_bce'][i])
            test_kld.append(loss_history['test_kld'][i])

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Loss Curves - Epoch {epoch}', fontsize=16)

    # Total Loss
    axes[0].plot(epochs, loss_history['train_total'], 'b-', label='Train', linewidth=2)
    if test_total:
        axes[0].plot(test_epochs, test_total, 'r-', label='Test', linewidth=2)
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # BCE Loss
    axes[1].plot(epochs, loss_history['train_bce'], 'b-', label='Train', linewidth=2)
    if test_bce:
        axes[1].plot(test_epochs, test_bce, 'r-', label='Test', linewidth=2)
    axes[1].set_title('BCE Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # KLD Loss
    axes[2].plot(epochs, loss_history['train_kld'], 'b-', label='Train', linewidth=2)
    if test_kld:
        axes[2].plot(test_epochs, test_kld, 'r-', label='Test', linewidth=2)
    axes[2].set_title('KLD Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
    

def log_generated_samples(model, device, epoch, num_samples_per_class=5, return_fig=False):
    """
    Generate and log sample images for each class

    Args:
        model: The CVAE model
        device: Device to run inference on
        epoch: Current epoch number
        num_samples_per_class: Number of samples to generate per class
        return_fig: If True, return the figure instead of logging directly
        
    Returns:
        fig if return_fig=True, otherwise None
    """
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    fig, axes = plt.subplots(10, num_samples_per_class, figsize=(15, 15))
    for class_idx in range(10):  # 10 classes in Fashion-MNIST
        samples = generate_images(model, device, class_idx)

        for j in range(num_samples_per_class):
            axes[class_idx, j].imshow(samples[j].cpu().numpy(), cmap="gray")
            axes[class_idx, j].set_title(classes[class_idx])
            axes[class_idx, j].axis("off")
    
    if return_fig:
        return fig
    else:
        # direct logging (will create separate step)
        wandb.log({
            "generated_images": wandb.Image(fig),
            "epoch": epoch
        })
        plt.close(fig)
        
        
def finish_wandb():
    """
    Finish the W&B run
    """
    wandb.finish()