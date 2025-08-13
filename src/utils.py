import matplotlib.pyplot as plt
import torch
import os


def save_model(model, folder, filename):
    """
    Save the model to the specified path
    Args:
        model: model to save
        folder: folder to save the model
        filename: filename to save the model
    """
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist.")
    
    torch.save(model.state_dict(), f"{folder}/{filename}")
    print(f"Model saved to {folder}/{filename}")


def load_model(model, path, target_device=None):
    """
    Load the model from the specified path
    Args:
        model: model to load
        path: path to the model
        target_device: device to load the model to (default is None, which detect device automatically)
    Returns:
        model: loaded model
    """
    if target_device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=target_device))
    model.to(target_device)
    print(f"Model loaded from {path}")
    return model


def one_hot_encode(labels, num_classes):
    """
    One-hot encode the labels
    Args:
        labels: labels to encode
        num_classes: number of classes
    Returns:
        one_hot: one-hot encoded labels
    """
    device = labels.device
    labels = labels.long()
    one_hot = torch.zeros(labels.size(0), num_classes, device=device)
    one_hot.scatter_(dim=1, index=labels.unsqueeze(1), value=1)
    return one_hot


class Logger:
    def __init__(self):
        """
        Logger class
        """
        self.train_total_losses = []
        self.train_bce_losses = []
        self.train_kl_losses = []
        self.test_total_losses = []
        self.test_bce_losses = []
        self.test_kl_losses = []
        self.epochs = []

    def log(self, epoch, total_loss, bce_loss, kl_loss):
        if epoch not in self.epochs:
            self.epochs.append(epoch)
        self.train_total_losses.append(total_loss)
        self.train_bce_losses.append(bce_loss)
        self.train_kl_losses.append(kl_loss)

    def log_test(self, epoch, total_loss, bce_loss, kl_loss):
        self.test_total_losses.append(total_loss)
        self.test_bce_losses.append(bce_loss)
        self.test_kl_losses.append(kl_loss)

    def plot_losses(self):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

        # Plot total loss
        axes[0].plot(
            self.epochs, self.train_total_losses, "b-", marker="o", label="Train"
        )
        if len(self.test_total_losses) > 0:
            axes[0].plot(
                self.epochs, self.test_total_losses, "r-", marker="s", label="Test"
            )
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Total Loss")
        axes[0].legend()
        axes[0].grid(True)

        # Plot BCE loss
        axes[1].plot(
            self.epochs, self.train_bce_losses, "b-", marker="o", label="Train"
        )
        if len(self.test_bce_losses) > 0:
            axes[1].plot(
                self.epochs, self.test_bce_losses, "r-", marker="s", label="Test"
            )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("BCE Loss")
        axes[1].legend()
        axes[1].grid(True)

        # Plot KL loss
        axes[2].plot(self.epochs, self.train_kl_losses, "b-", marker="o", label="Train")
        if len(self.test_kl_losses) > 0:
            axes[2].plot(
                self.epochs, self.test_kl_losses, "r-", marker="s", label="Test"
            )
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Loss")
        axes[2].set_title("KL Loss")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()
        

