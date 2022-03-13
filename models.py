import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math

class MultiDAE(nn.Module):
    """
    Container module for Multi-DAE.

    Multi-DAE : Denoising Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiDAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        self.dims = self.q_dims + self.p_dims[1:]
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.dims[:-1], self.dims[1:])])
        self.drop = nn.Dropout(dropout)
        
        self.init_weights()
    
    def forward(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.weights) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar
    
    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)
        
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

# MMD
def compute_kernel(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]
    tiled_x = torch.reshape(x, (x_size, 1, dim)).repeat(1, y_size, 1)
    tiled_y = torch.reshape(y, (1, y_size, dim)).repeat(x_size, 1, 1)
    return torch.exp(-torch.mean(torch.square(tiled_x - tiled_y), 2) / dim)

def compute_mmd(x, y, sigma_sqr=1.0):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

# SWD
def rand_projections(embedding_dim, num_samples=50):
    projections = [w / np.sqrt((w**2).sum())  # L2 normalization
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    return torch.from_numpy(projections).type(torch.FloatTensor)

def sliced_wasserstein_distance(encoded_samples, distribution_samples, num_projections=50, p=2, device='cpu'):
    embedding_dim = distribution_samples.size(1)
    projections = rand_projections(embedding_dim, num_projections).to(device)
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))

    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    return wasserstein_distance.mean()

# CWD
def pairwise_distances(x):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(x, 0, 1)
    y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)

def silverman_rule_of_thumb(N):
    return (4/(3*N))**0.4

def euclidean_norm_squared(X, axis):
    return (X**2).sum(axis)

def cw(X):
    assert len(X.size()) == 2
    N, D = X.size()

    y = silverman_rule_of_thumb(N)

    K = 1.0/(2.0*D-3.0)

    A1 = pairwise_distances(X)
    A = (1/(N**2)) * (1/torch.sqrt(y + K*A1)).sum()

    B1 = euclidean_norm_squared(X, axis=1)
    B = (2/N)*((1/torch.sqrt(y + 0.5 + K*B1))).sum()

    return (1/math.sqrt(1+y)) + A - B

def loss_function(recon_x, x, z, true, mu, logvar, anneal=1.0, device='cpu'):
    """ Function to compute loss """

    # BCE (Binary Cross Entropy)
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))

    lossf = 'KLD' # Options: KLD, MMD, SWD, CWD

    if lossf = 'KLD':
        # KL Divergence
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        lsp = KLD
    elif lossf = 'MMD':
        # Maximimun Mean  Discrepancy
        MMD = 10 * compute_mmd(z, true)
        lsp = MMD
    elif lossf = 'SWD':
        # Sliced Wasserstein Distance
        SWD = 10 * sliced_wasserstein_distance(z, true, device=device)
        lsp = SWD
    elif lossf = 'CWD':
        # Cramer Wold Distance
        CWD = cw(mu)
        lsp = CWD

    return BCE + anneal * lsp
