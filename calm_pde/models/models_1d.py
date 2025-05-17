"""CALM-PDE: Continuous and Adaptive Convolutions for Latent Space Modeling of Time-dependent PDEs."""

import torch
import torch.nn as nn
from einops import rearrange
from .layers import CALMEncoderLayer, CALMDecoderLayer, FinalCALMDecoderLayer


class CALM_PDE_Burgers(nn.Module):
    """CALM-PDE model with continuous and adaptive convolutions for reduced-order PDE-solving
    of time-dependent PDEs with neural ODE latent time-stepping.
    """

    def __init__(self, neural_ode, in_channels=1, out_channels=1, dropout=0.0):
        super(CALM_PDE_Burgers, self).__init__()
        
        # Encoder
        self.encoder0 = CALMEncoderLayer(in_channels, 16, 256, 0.05, spatial_dim=1, dropout=dropout, softmax_temp=1)
        self.encoder1 = CALMEncoderLayer(16, 32, 64, 0.1, spatial_dim=1, dropout=dropout, softmax_temp=1)
        self.encoder2 = CALMEncoderLayer(32, 64, 8, 0.5, spatial_dim=1, dropout=dropout, softmax_temp=1)
            
        # NODE for latent time-stepping
        self.neural_ode = neural_ode

        # Decoder
        self.decoder0 = CALMDecoderLayer(64, 32, 64, 1.0, spatial_dim=1, dropout=dropout, softmax_temp=1)
        self.decoder1 = CALMDecoderLayer(32, 16, 256, 0.5, spatial_dim=1, dropout=dropout, softmax_temp=1)
        self.decoder2 = FinalCALMDecoderLayer(16, out_channels, 0.1, spatial_dim=1, softmax_temp=1)

    def encode(self, x, pos):
        x, latent_pos = self.encoder0(x, pos)
        x, latent_pos = self.encoder1(x, latent_pos)
        x, latent_pos = self.encoder2(x, latent_pos)

        return x, latent_pos

    def time_stepping(self, x, steps):
        shape = x.shape
        pred = torch.zeros((shape[0], steps, shape[1], shape[2]), device=x.device)
        for step in range(steps):
            x = self.neural_ode(x, self.encoder2.query_pos) + x
            pred[:, step, ...] = x
        
        return pred
    
    def decode(self, x, latent_pos, query_pos, no_time_dim=False):
        if no_time_dim:
            x = x.unsqueeze(1)

        x, latent_pos = self.decoder0(x, latent_pos)
        x, latent_pos = self.decoder1(x, latent_pos)
        x = self.decoder2(x, latent_pos, query_pos)

        if no_time_dim:
            x = x.squeeze(1)
        
        return x

    def forward(self, x, pos, query_pos, steps):
        # input shape: b, n, c
        # output shape b, t, n, c

        x, pos = self.encode(x, pos)
        x = self.time_stepping(x, steps)
        x = self.decode(x, pos, query_pos)

        return x
