import torch
from src.utils import one_hot_encode
from src.model import loss_function


def evaluate(model, device, test_loader):
    """
    Evaluate the model
    Args:
        model: model to evaluate
        device: device to use
        test_loader: test loader
    Returns:
        tuple: (avg_test_loss, avg_test_bce, avg_test_kld)
    """
    model.eval()
    test_loss = 0
    test_bce = 0
    test_kld = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)

            # One-hot encode the labels
            labels_one_hot = one_hot_encode(labels, num_classes=10)

            recon_batch, mu, logvar = model(data, labels_one_hot)

            loss, bce, kld = loss_function(recon_batch, data, mu, logvar)

            test_loss += loss.item()
            test_bce += bce.item()
            test_kld += kld.item()

    test_loss /= len(test_loader.dataset)
    test_bce /= len(test_loader.dataset)
    test_kld /= len(test_loader.dataset)

    print(
        f"====> Test set loss: {test_loss:.4f} "
        f"(BCE: {test_bce:.4f}, KL: {test_kld:.4f})"
    )

    return test_loss, test_bce, test_kld


def generate_images(model, device, label_idx, num_images=10):
    """
    Generate images from the model
    Args:
        model: model to generate images from
        device: device to use
        label_idx: label index to generate images from
        num_images: number of images to generate
    Returns:
        generated_images: generated images
    """
    model.eval()
    with torch.no_grad():
        # Create one-hot encoded label
        label = one_hot_encode(
            torch.tensor([label_idx] * num_images, device=device), 10
        )

        # Sample from latent space
        z = torch.randn(num_images, model.latent_dim).to(device)

        # Decode
        sample = model.decode(z, label)

        return sample.view(num_images, 28, 28)