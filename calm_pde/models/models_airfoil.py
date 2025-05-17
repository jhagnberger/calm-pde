"""CALM-PDE: Continuous and Adaptive Convolutions for Latent Space Modeling of Time-dependent PDEs."""

import torch
import torch.nn as nn
from einops import rearrange
from .layers import FirstCALMEncoderLayer, CALMEncoderLayer, CALMDecoderLayer, FinalCALMDecoderLayer


class CALM_PDE_Airfoil(nn.Module):
    """CALM-PDE model with continuous and adaptive convolutions for reduced-order PDE-solving
    of time-dependent PDEs with neural ODE latent time-stepping.
    """

    def __init__(self, neural_ode, in_channels=4, out_channels=4, init_mesh=None, dropout=0.0):
        super(CALM_PDE_Airfoil, self).__init__()
        
        # Encoder
        self.encoder0 = FirstCALMEncoderLayer(in_channels, 32, 1024, 0.01, dropout=dropout, softmax_temp=1, is_periodic=False,
                        same_grid_per_sample=False, init_query_pos=init_mesh[torch.randperm(init_mesh.shape[0])[:1024], :])
        self.encoder1 = CALMEncoderLayer(32, 64, 256, 0.01, dropout=dropout, softmax_temp=1, is_periodic=False)
        self.encoder2 = CALMEncoderLayer(64, 128, 16, 0.5, dropout=dropout, softmax_temp=1/10, is_periodic=False)
            
        # NODE for latent time-stepping
        self.neural_ode = neural_ode

        # Decoder
        self.decoder0 = CALMDecoderLayer(128, 64, 256, 0.5, dropout=dropout, softmax_temp=1/10, is_periodic=False)
        self.decoder1 = CALMDecoderLayer(64, 32, 1024, 0.01, dropout=dropout, softmax_temp=1, is_periodic=False,
                        init_query_pos=init_mesh[torch.randperm(init_mesh.shape[0])[:1024], :])
        self.decoder2 = FinalCALMDecoderLayer(32, out_channels, 0.01, softmax_temp=1, same_grid_per_sample=False, is_periodic=False)

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
