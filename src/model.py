import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, num_classes=10):
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Encoder
        self.fc1 = nn.Linear(input_dim + num_classes, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance

        # Decoder
        self.fc2 = nn.Linear(latent_dim + num_classes, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, condition):
        x_c = torch.cat((x, condition), dim=1)  # Concatenate input and condition
        h = F.relu(self.fc1(x_c))
        h = F.relu(self.fc1(x_c))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z, condition):
        z_c = torch.cat(
            (z, condition), dim=1
        )  # Concatenate latent vector and condition
        h = F.relu(self.fc2(z_c))
        x_recon = torch.sigmoid(self.fc3(h))
        return x_recon

    def forward(self, x, condition):
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, condition)
        return x_recon, mu, logvar


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    # mse = F.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD