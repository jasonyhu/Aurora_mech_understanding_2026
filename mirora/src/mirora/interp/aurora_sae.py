"""
A sparse autoencoder for Aurora backbone activations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mirora.config import DEFAULT_EXPANSION_FACTOR, DEFAULT_L1_COEFF

class AuroraSAE(nn.Module):
    def __init__(self, d_model: int = 2048, expansion_factor: int = DEFAULT_EXPANSION_FACTOR,
                 l1_coeff: float = DEFAULT_L1_COEFF):
        super().__init__()
        self.d_hidden = d_model * expansion_factor
        self.l1_coeff = l1_coeff
        
        # Encoder/Decoder weights
        self.W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_model, self.d_hidden)))
        self.b_enc = nn.Parameter(torch.zeros(self.d_hidden))
        self.W_dec = nn.Parameter(self.W_enc.data.clone().t())
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        
        self.normalize_decoder()

    def normalize_decoder(self):
        with torch.no_grad():
            self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True)

    def encode(self, x):
        # x shape: [Tokens, d_model]
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def forward(self, x):
        h = self.encode(x)
        x_reconstruct = h @ self.W_dec + self.b_dec
        return x_reconstruct, h
