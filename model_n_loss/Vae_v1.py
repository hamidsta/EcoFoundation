import torch
import torch.nn as nn
from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        mean = self.fc2_mean(h)
        logvar = self.fc2_logvar(h)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

        # ZINB Parameters
        self.log_theta = nn.Parameter(torch.zeros(1, output_dim))  # Dispersion
        self.dropout = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())
        self.scale = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Softmax(dim=-1))

    def forward(self, z):
        h = self.relu(self.fc1(z))
        scale = self.scale(h)
        dropout = self.dropout(h)
        theta = torch.exp(self.log_theta)
        return scale, dropout, theta

class VAE_v1(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(VAE_v1, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        scale, dropout, theta = self.decoder(z)
        return mean, logvar, scale, dropout, theta
