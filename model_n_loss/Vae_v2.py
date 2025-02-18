import torch 
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    
    """
    A multi-layer encoder with optional batchnorm and dropout.
    hidden_dims: list of hidden layer sizes, e.g. [256, 128]
    """
    def __init__(self, input_dim, latent_dim, hidden_dims=[256, 128], dropout_rate=0.1):
        super(Encoder, self).__init__()

        #  build a stack of Linear -> BatchNorm -> ReLU -> Dropout
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hdim

        self.net = nn.Sequential(*layers)

        # Separate linear layers for mean and logvar
        self.mean_layer   = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        h = self.net(x)
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        return mean, logvar

# -------------------------------------------
# Multi-layer Decoder for ZINB
# -------------------------------------------
class Decoder(nn.Module):
    """
    Multi-layer decoder that outputs:
    - scale (mean expression)
    - dropout (zero-inflation probability)
    - theta (dispersion, gene-specific)
    """
    def __init__(self, latent_dim, output_dim, hidden_dims=[128, 256], dropout_rate=0.1):
        super(Decoder, self).__init__()

        # Build a stack (reverse of encoder, e.g. latent -> 128 -> 256 -> ...)
        layers = []
        prev_dim = latent_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hdim

        self.net = nn.Sequential(*layers)

        # Separate "heads" for scale and dropout
        self.scale_layer   = nn.Linear(hidden_dims[-1], output_dim)
        self.dropout_layer = nn.Linear(hidden_dims[-1], output_dim)

        # Dispersion parameter per gene
        self.log_theta = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, z):
        h = self.net(z)

        # Mean expression (ensure positivity)
        scale = F.relu(self.scale_layer(h)) + 1e-8

        # Zero-inflation probability in [0, 1]
        dropout = torch.sigmoid(self.dropout_layer(h))

        # Convert log_theta -> theta
        theta = torch.exp(self.log_theta)

        return scale, dropout, theta

# -------------------------------------------
# Full VAE
# -------------------------------------------
class VAE_v2(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim=20,
        encoder_hidden_dims=[256, 128],
        decoder_hidden_dims=[128, 256],
        dropout_rate=0.1):
        
        super(VAE_v2, self).__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            dropout_rate=dropout_rate
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=decoder_hidden_dims,
            dropout_rate=dropout_rate
        )

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        scale, dropout, theta = self.decoder(z)
        return mean, logvar, scale, dropout, theta
