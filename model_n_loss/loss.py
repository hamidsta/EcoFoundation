
import torch
from torch.distributions import Normal, NegativeBinomial


def zinb_loss(x, scale, dropout, theta, eps=1e-8):
    """ZINB Negative Log Likelihood."""
    # Ensure valid values for theta and scale
    theta = torch.clamp(theta, min=eps)
    scale = torch.clamp(scale, min=eps)

    # Compute Negative Binomial probability
    nb_prob = theta / (theta + scale + eps)
    nb_prob = torch.clamp(nb_prob, min=0.001, max=0.999)  # Stronger clamping

    # Negative Binomial log likelihood

    log_nb = NegativeBinomial(total_count=theta, probs=nb_prob).log_prob(x)

    # Zero inflation
    zero_inflation = torch.log(dropout + (1 - dropout) * torch.exp(log_nb))
    zinb_log_likelihood = torch.where(
        x < eps, zero_inflation, log_nb + torch.log(1 - dropout + eps)
    )

    return -zinb_log_likelihood.sum(dim=1).mean()


import torch
import torch.nn.functional as F

def zinb_negative_log_likelihood(x, mu, theta, pi, eps=1e-8, gamma_max=1e7):
    """
    ZINB negative log-likelihood using a gamma-based NB formula with careful clamping.
    
    Parameters
    ----------
    x : (batch_size, n_genes)  float or int
        Observed counts.
    mu : (batch_size, n_genes)
        Mean of NB, >= 0.  Typically mu = library_size * scale.
    theta : (batch_size, n_genes) or (1, n_genes) or (n_genes,)
        Dispersion > 0.
    pi : (batch_size, n_genes)
        Zero-inflation probability in [0,1].
    eps : float
        Small constant to avoid log(0).
    gamma_max : float
        Maximum argument to clamp in lgamma to avoid huge overflow.

    Returns
    -------
    scalar : Mean negative log-likelihood across batch.
    """
    # Ensure floating point
    x = x.float()
    mu = mu.float()
    theta = theta.float()
    pi = pi.float()

    # 1) Minimal Clamping
    pi = torch.clamp(pi, min=eps, max=1 - eps)
    mu = torch.clamp(mu, min=eps)
    theta = torch.clamp(theta, min=eps)

    # 2) Compute NB log-prob using explicit gamma formula
    #    NB(x; mu, theta) = Gamma(x+theta)/(Gamma(x+1)*Gamma(theta))
    #                       * (mu/(theta+mu))^x * (theta/(theta+mu))^theta
    #    => log NB = log Gamma(x+theta) - log Gamma(theta) - log Gamma(x+1)
    #                + x * log(mu) - x * log(mu+theta)
    #                + theta * log(theta) - theta * log(mu+theta)
    
    # 2a) if x+theta is extremely large, clamp to gamma_max
    # (rare, but prevents overflow if x is huge)
    xt = torch.clamp(x + theta, max=gamma_max)  
    log_gamma_xt = torch.lgamma(xt)
    
    log_gamma_theta = torch.lgamma(torch.clamp(theta, max=gamma_max))
    log_gamma_xp1 = torch.lgamma(torch.clamp(x + 1, min=eps, max=gamma_max))

    t1 = log_gamma_xt - log_gamma_theta - log_gamma_xp1
    t2 = x * torch.log(mu + eps) - x * torch.log(mu + theta + eps)
    t3 = theta * torch.log(theta + eps) - theta * torch.log(mu + theta + eps)
    nb_log_prob = t1 + t2 + t3  # shape (batch_size, n_genes)

    # 3) Zero-inflation
    #    For x=0 => log( pi + (1-pi)*exp(nb_log_prob) )
    #    For x>0 => nb_log_prob + log(1-pi)
    log_zero_prob    = torch.log(pi + (1. - pi) * torch.exp(nb_log_prob) + eps)
    log_nonzero_prob = nb_log_prob + torch.log(1. - pi + eps)

    is_zero  = (x < eps).float()  # float mask: 1. for zero, 0. for non-zero
    log_prob = is_zero * log_zero_prob + (1. - is_zero) * log_nonzero_prob

    # 4) Negative log-likelihood => sum over genes, mean over batch
    nll = -log_prob.sum(dim=1).mean(dim=0)
    return nll



def zinb_loss_with_sf(x, scale, dropout, theta, sf, eps=1e-8):
    """
    'Approach 1' for library-size correction: multiply scale by sf[i] for each cell.
    x:       (batch_size, n_genes) raw counts
    scale:   (batch_size, n_genes) base mean from decoder
    dropout: (batch_size, n_genes) zero-inflation probability
    theta:   (1, n_genes) or (batch_size, n_genes) dispersion
    sf:      (batch_size,) scale factors for each cell
    """
    # Multiply predicted mean by scale_factors
    scale = scale * sf.unsqueeze(1)  # shape = (batch_size, n_genes)

    # Standard ZINB
    theta = torch.clamp(theta, min=eps)
    scale = torch.clamp(scale, min=eps)

    nb_prob = theta / (theta + scale)
    nb_prob = torch.clamp(nb_prob, min=1e-6, max=1-1e-6)

    log_nb = NegativeBinomial(total_count=theta, probs=nb_prob).log_prob(x)
    zero_inflation = torch.log(dropout + (1 - dropout)*torch.exp(log_nb) + eps)

    zinb_ll = torch.where(
        x < eps,  # effectively x==0 if integer
        zero_inflation,
        log_nb + torch.log(1 - dropout + eps)
    )

    return -zinb_ll.sum(dim=1).mean()
