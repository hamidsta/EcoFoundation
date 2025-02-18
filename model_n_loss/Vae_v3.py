import torch 
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    A multi-layer encoder with optional batchnorm and dropout.
    hidden_dims: list of hidden layer sizes, e.g. [256, 128]
    """
    def __init__(self, input_dim, latent_dim, hidden_dims=[256, 128], dropout_rate=0.1,
                 n_batches=1, batch_embed_dim=8):
        super(Encoder, self).__init__()

        self.n_batches = n_batches
        if n_batches > 1:
            self.batch_embed = nn.Embedding(n_batches, batch_embed_dim)
            total_input_dim = input_dim + batch_embed_dim
        else:
            self.batch_embed = None
            total_input_dim = input_dim

        #  build a stack of Linear -> BatchNorm -> ReLU -> Dropout
        layers = []
        prev_dim = total_input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.LayerNorm(hdim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hdim

        self.net = nn.Sequential(*layers)

        # Separate linear layers for mean and logvar
        self.mean_layer   = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x, batch_idx=None):
        """
        x: (batch_size, input_dim)
        batch_idx: (batch_size,) integer specifying which batch each cell belongs to
        """
        if self.batch_embed is not None and batch_idx is not None:
            # shape: (batch_size, batch_embed_dim)
            b_vec = self.batch_embed(batch_idx)
            # shape: (batch_size, input_dim + batch_embed_dim)
            x = torch.cat([x, b_vec], dim=1)

        h = self.net(x)
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        return mean, logvar
    
    
    
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
            layers.append(nn.LayerNorm(hdim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hdim

        self.net = nn.Sequential(*layers)
        # try one layer for predicting all 3 outputs 
        # self.output = nn.Linear(hidden_dims[-1], 3 * output_dim) 

        # Separate "heads" for scale and dropout
        self.scale_layer   = nn.Linear(hidden_dims[-1], output_dim)
        self.dropout_layer = nn.Linear(hidden_dims[-1], output_dim)

        # Dispersion parameter per gene
        self.log_theta = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, z, library_size=None):
        """
        z: (batch_size, latent_dim)
        library_size: (batch_size,) if provided, used to scale expression
        """
        h = self.net(z)

        # scale before library multiplication
        #scale_raw = F.relu(self.scale_layer(h)) + 1e-8
        scale_raw = F.softplus(self.scale_layer(h)) 

        # if we have library size, multiply: scale = scale_raw * exp(library_size)
        # library_size could be in log-space or raw. If you have raw, do torch.log first
        if library_size is not None:
            # assume library_size is in log-space
            scale = scale_raw *library_size.unsqueeze(1)
            #torch.exp(library_size).unsqueeze(1)
        else:
            return ValueError("Library size must be provided")

        # Zero-inflation probability
        dropout = torch.sigmoid(self.dropout_layer(h))

        # Convert log_theta -> theta
        #theta = torch.exp(self.log_theta)
        theta = torch.exp(self.log_theta).expand(z.shape[0], -1)

        return scale, dropout, theta
    
    
    
class VAE_v3(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim=20,
        encoder_hidden_dims=[256, 128],
        decoder_hidden_dims=[128, 256],
        dropout_rate=0.1,
        n_batches=1,
        batch_embed_dim=8
    ):
        super(VAE_v3, self).__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            dropout_rate=dropout_rate,
            n_batches=n_batches,
            batch_embed_dim=batch_embed_dim
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

    def forward(self, x, batch_idx=None, library_size=None):
        mean, logvar = self.encoder(x, batch_idx)
        z = self.reparameterize(mean, logvar)
        scale, dropout, theta = self.decoder(z, library_size)
        return mean, logvar, scale, dropout, theta


