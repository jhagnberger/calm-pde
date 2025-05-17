"""CALM-PDE: Continuous and Adaptive Convolutions for Latent Space Modeling of Time-dependent PDEs.

This module implements the processor for latent time-stepping by parametrizing the neural ODE.
"""

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


class CombinedAttention_LN(nn.Module):
    """Combined attention with scaled dot product attention and position attention."""

    def __init__(self, dim, n_heads=8, softmax_temp=1.0, is_periodic=True):
        super().__init__()
        self.n_heads = n_heads
        self.d = dim // n_heads
        self.softmax_temp = softmax_temp
        self.is_periodic = is_periodic
        self.in_proj = nn.Linear(dim, dim * 3)

        self.q_norm = torch.nn.LayerNorm(self.d)
        self.k_norm = torch.nn.LayerNorm(self.d)

        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, x, pos, need_weights=False):
        qkv = self.in_proj(x)
        qkv = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.n_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Norm on query and key
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Scaled dot product
        sdp = torch.einsum("bhid,bhjd->bhij", q, k) / np.sqrt(self.d)

        # Positional information
        dist = torch.abs(pos[:, None, :] - pos[None, ...])
        if self.is_periodic: dist = (dist + 0.5) % 1 - 0.5
        edist = torch.sum(dist**2, dim=-1)
        pos = -edist/self.softmax_temp

        # Self-attention with scaled dot product and positional information
        attn = torch.softmax(pos + sdp, dim=-1)
        x = torch.einsum("bhij,bhjd->bhid", attn, v)
        x = rearrange(x, "b h n d -> b n (h d)")

        x = self.out_proj(x)

        if need_weights:
            return x, torch.mean(attn, dim=1)

        return x


class NeuralODE(nn.Module):
    """Processor network that parametrizes the neural ODE."""

    def __init__(self, dim, hidden_dim, n_heads=8, spatial_dim=2, is_periodic=True):
        super().__init__()
        self.B = nn.Parameter(6.0 * torch.randn(size=(spatial_dim, 64 // 2), requires_grad=False), requires_grad=False)
        self.in_linear = nn.Linear(dim + 64, hidden_dim)

        self.self_attn_1 = CombinedAttention_LN(hidden_dim, n_heads, is_periodic=is_periodic)
        self.mlp_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim))
        
        self.self_attn_2 = CombinedAttention_LN(hidden_dim, n_heads, is_periodic=is_periodic)
        self.mlp_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim))
        
        self.out_linear = nn.Linear(hidden_dim, dim)


    def forward(self, u, latent_pos):
        # Add RFF to the input for positional information
        projection = (2 * np.pi * latent_pos) @ self.B
        projection = torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)
        u = torch.cat((u, projection[None, ...].expand(u.shape[0], -1, -1)), dim=-1)
        
        u = self.in_linear(u)
        
        # First attention block
        u = self.self_attn_1(u, latent_pos) + u
        u = self.mlp_1(u) + u

        # Second attention block
        u = self.self_attn_2(u, latent_pos) + u
        u = self.mlp_2(u) + u

        u = self.out_linear(u)

        return u
