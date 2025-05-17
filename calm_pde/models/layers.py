"""CALM-PDE: Continuous and Adaptive Convolutions for Latent Space Modeling of Time-dependent PDEs.

This module implements the CALM layers used in the CALM-PDE model. The layers compute continuous and adaptive convolutions.
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from opt_einsum import contract


def uniform_distribution(shape, a=-0.5, b=0.5):
    """Returns a tensor with the given shape sampled from a uniform distribution U(a,b)."""
    x = torch.zeros(shape)
    x = x.uniform_(a, b)
    return x


class FirstCALMEncoderLayer(nn.Module):
    """Continuous and adaptive convolution layer to compress point clouds for the use in an encoder.

    Applies continuous and adaptive convolution to the incoming point cloud and compresses it into a smaller 
    (i.e., less points) latent point cloud. The layer supports both periodic and non-periodic boundaries and can
    handle cases where the position is the same for each sample in a batch or different across samples in a batch.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_query_points (int): Number of query points.
        receptive_field (float): A threshold value to limit the receptive field by the p-th percentile.
        softmax_temp (float): Temperature parameter for the distance weighting.
        spatial_dim (int): Spatial dimension of the input data, e.g., 2 for 2D data.
        is_periodic (bool): Whether the boundary is periodic or not.
        same_grid_per_sample (bool): Whether the positions is the same across all samples in a batch or not. Setting
                                     this flag to True will make the layer more efficient for samples with the same positions.
        init_query_pos (torch.Tensor): Predefined positions to initialize the query points. Used for mesh prior initialization.
        dropout (float): Dropout probability for the channels.
        rff_std (float): Standard deviation for the random Fourier features used in the kernel function.
    """

    def __init__(self, in_channels=32, out_channels=64, num_query_points=512, receptive_field=0.1, softmax_temp=1.0,
                spatial_dim=2, is_periodic=True, same_grid_per_sample=True, init_query_pos=None, dropout=0.0, rff_std=6.0):
        super(FirstCALMEncoderLayer, self).__init__()
        
        self.receptive_field = receptive_field
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.softmax_temp = softmax_temp
        self.eps = 1e-8

        self.is_periodic = is_periodic
        self.same_grid_per_sample = same_grid_per_sample

        # Parameters for query positions and modulation
        self.query_pos = nn.Parameter(init_query_pos) if init_query_pos is not None else nn.Parameter(torch.rand(num_query_points, spatial_dim, dtype=torch.float32))
        self.query_modulation_weight = nn.Parameter(torch.rand(num_query_points, 32, dtype=torch.float32))
        self.query_modulation_offset = nn.Parameter(torch.zeros(num_query_points, 32, dtype=torch.float32))

        # Channelwise linear transformation and MLP
        self.linear = nn.Linear(in_channels, in_channels)
        self.mlp = nn.Sequential(nn.Linear(out_channels, out_channels * 4), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(out_channels * 4, out_channels), nn.Dropout(dropout))

        # MLP to parametrize the kernel
        self.B = nn.Parameter(rff_std * torch.randn(size=(spatial_dim, 16), requires_grad=False), requires_grad=False)
        self.linear1 = nn.Linear(32, 32)
        self.linear2 = nn.Linear(32, in_channels * out_channels, bias=False)
        self.filter = nn.Parameter(uniform_distribution(in_channels * out_channels, -np.sqrt(3/in_channels), np.sqrt(3/in_channels)))
        
        # Helpers for the receptive field
        self.slice = torch.vmap(lambda x, ind: torch.index_select(x, 0, ind), in_dims=0, out_dims=0)
        if same_grid_per_sample:
            self.kernel_transform = torch.vmap(lambda k, ind, x: contract("vcd,bvc->bd", k, x[:, ind, :]), in_dims=(0,0,None), out_dims=1)
        else:
            def func(k, ind, x):
                x = torch.stack([x[i, ind[i], :] for i in range(x.shape[0])])
                return contract("bvcd,bvc->bd", k, x)
            self.kernel_transform = torch.vmap(func, in_dims=(1,1,None), out_dims=1)

        # Bias and dropout
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)

    def _forward_same_grid(self, x, pos):
        # x shape: b, n, c
        q = self.query_pos

        # Position matrix
        dist = q[:, None, :] - pos[None, ...]
        if self.is_periodic: dist = (dist + 0.5) % 1 - 0.5
        edist = torch.sum(dist**2, dim=-1)
        
        # Limit receptive field by epsilon environment
        threshold = torch.quantile(edist, self.receptive_field, dim=-1, keepdim=True)
        mask = edist <= threshold
        max_neighbourhood = torch.max(torch.sum(mask, dim=-1))
        _, ind = torch.topk(-edist, k=max_neighbourhood, dim=-1)

        # Compute receptive field
        dist = self.slice(dist, ind)
        edist = self.slice(edist, ind)[..., None]

        # Gaussian kernel
        edist -= edist.min(-2, keepdim=True)[0]
        edist /= edist.max(-2, keepdim=True)[0] + self.eps
        k_distance = torch.softmax(-edist/self.softmax_temp, dim=-2)
        
        # Computed kernel
        projection = (2 * np.pi * dist) @ self.B
        k = torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)
        k = self.linear2(F.gelu(self.linear1(k) * self.query_modulation_weight[:, None, :] + self.query_modulation_offset[:, None, :])) + self.filter[None, None, :]
        
        # Combine both kernels
        k = k * k_distance
        k = rearrange(k, "q v (c d) -> q v c d", c=self.in_channels, d=self.out_channels)
        
        # Kernel transform
        x = self.linear(x)
        x = self.dropout(F.gelu(self.kernel_transform(k, ind, x) + self.bias[None, None, :]))

        # Pointwise MLP
        x = self.mlp(x) + x

        return x, q
    
    def _forward_different_grids(self, x, pos):
        # x shape: b, n, c
        q = self.query_pos

        # Position matrix
        dist = q[None, :, None, :] - pos[:, None, ...]
        if self.is_periodic: dist = (dist + 0.5) % 1 - 0.5
        edist = torch.sum(dist**2, dim=-1)
        
        # Limit receptive field by epsilon environment
        threshold = torch.quantile(edist, self.receptive_field, dim=-1, keepdim=True)
        mask = edist <= threshold
        max_neighbourhood = torch.max(torch.sum(mask, dim=-1))
        _, ind = torch.topk(-edist, k=max_neighbourhood, dim=-1)

        # Compute receptive field
        dist = rearrange(dist, "b q v d -> (b q) v d")
        edist = rearrange(edist, "b q v -> (b q) v")
        ind = rearrange(ind, "b q v -> (b q) v")
        dist = self.slice(dist, ind)
        edist = self.slice(edist, ind)[..., None]
        dist = rearrange(dist, "(b q) v d -> b q v d", q=q.shape[0])
        edist = rearrange(edist, "(b q) v d -> b q v d", q=q.shape[0])
        ind = rearrange(ind, "(b q) v -> b q v", q=q.shape[0])

        # Gaussian kernel
        edist -= edist.min(-2, keepdim=True)[0]
        edist /= edist.max(-2, keepdim=True)[0] + self.eps
        k_distance = torch.softmax(-edist/self.softmax_temp, dim=-2)
        
        # Computed kernel
        projection = (2 * np.pi * dist) @ self.B
        k = torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)
        k = self.linear2(F.gelu(self.linear1(k) * self.query_modulation_weight[:, None, :] + self.query_modulation_offset[:, None, :])) + self.filter[None, None, :]
    
        # Combine both kernels
        k = k * k_distance
        k = rearrange(k, "b q v (c d) -> b q v c d", c=self.in_channels, d=self.out_channels)

        # Kernel transform
        x = self.linear(x)
        x = self.dropout(F.gelu(self.kernel_transform(k, ind, x) + self.bias[None, None, :]))

        # Pointwise MLP
        x = self.mlp(x) + x

        return x, q
    
    def forward(self, x, pos):
        """Computes continuous and adaptive convolution on the input tensor x with the positions pos.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_points, in_channels).
            pos (torch.Tensor): Position tensor of shape (batch_size, num_points, spatial_dim) 
                                or (num_points, spatial_dim) depending on the position type.
        
        Returns:
            tuple: A tuple containing the processed output and positions:
                torch.Tensor: Processed output tensor of shape (batch_size, num_points, out_channels).
                torch.Tensor: Corresponding query positions of shape (num_points, out_channels).
        """

        if self.same_grid_per_sample:
            x, q = self._forward_same_grid(x, pos)
        else:
            x, q = self._forward_different_grids(x, pos)

        return x, q
    

class CALMEncoderLayer(nn.Module):
    """Continuous and adaptive convolution layer to compress point clouds for the use in an encoder.

    Applies continuous and adaptive convolution to the incoming point cloud and compresses it into a smaller 
    (i.e., less points) latent point cloud. The layer supports both periodic and non-periodic boundaries. It
    only supports cases where the positions for each sample in a batch is the same, which can be useful for
    deeper layers of the network. It corresponds to the a FirstCALMEncoderLayer with the flag same_grid_per_sample = True.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_query_points (int): Number of query points.
        receptive_field (float): A threshold value to limit the receptive field by the p-th percentile.
        softmax_temp (float): Temperature parameter for the distance weighting.
        spatial_dim (int): Spatial dimension of the input data, e.g., 2 for 2D data.
        is_periodic (bool): Whether the boundary is periodic or not.
        init_query_pos (torch.Tensor): Predefined positions to initialize the query points. Used for mesh prior initialization.
        dropout (float): Dropout probability for the channels.
        rff_std (float): Standard deviation for the random Fourier features used in the kernel function.
    """

    def __init__(self, in_channels=32, out_channels=64, num_query_points=512, receptive_field=0.1, softmax_temp=1.0,
                spatial_dim=2, is_periodic=True, init_query_pos=None, dropout=0.0, rff_std=6.0):
        super(CALMEncoderLayer, self).__init__()
        
        self.receptive_field = receptive_field
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.softmax_temp = softmax_temp
        self.eps = 1e-8

        self.is_periodic = is_periodic

        # Parameters for query positions and modulation
        self.query_pos = nn.Parameter(init_query_pos) if init_query_pos is not None else nn.Parameter(torch.rand(num_query_points, spatial_dim, dtype=torch.float32))
        self.query_modulation_weight = nn.Parameter(torch.rand(num_query_points, 32, dtype=torch.float32))
        self.query_modulation_offset = nn.Parameter(torch.zeros(num_query_points, 32, dtype=torch.float32))

        # Channelwise linear transformation and MLP
        self.linear = nn.Linear(in_channels, in_channels)
        self.mlp = nn.Sequential(nn.Linear(out_channels, out_channels * 4), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(out_channels * 4, out_channels), nn.Dropout(dropout))

        # MLP to parametrize the kernel function
        self.B = nn.Parameter(rff_std * torch.randn(size=(spatial_dim, 16), requires_grad=False), requires_grad=False)
        self.linear1 = nn.Linear(32, 32)
        self.linear2 = nn.Linear(32, in_channels * out_channels, bias=False)
        self.filter = nn.Parameter(uniform_distribution(in_channels * out_channels, -np.sqrt(3/in_channels), np.sqrt(3/in_channels)))

        # Helpers for the receptive field
        self.slice = torch.vmap(lambda x, ind: torch.index_select(x, 0, ind), in_dims=0, out_dims=0)
        self.kernel_transform = torch.vmap(lambda k, ind, x: contract("vcd,bvc->bd", k, x[:, ind, :]), in_dims=(0,0,None), out_dims=1)
        
        # Bias and dropout
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos):
        """Computes continuous and adaptive convolution on the input tensor x with the positions pos.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_points, in_channels).
            pos (torch.Tensor): Position tensor of shape (num_points, spatial_dim).
        
        Returns:
            tuple: A tuple containing the processed output and positions:
                torch.Tensor: Processed output tensor of shape (batch_size, num_points, out_channels).
                torch.Tensor: Corresponding query positions of shape (num_points, out_channels).
        """
        # x shape: b, n, c
        q = self.query_pos

        # Postion matrix
        dist = q[:, None, :] - pos[None, ...]
        if self.is_periodic: dist = (dist + 0.5) % 1 - 0.5
        edist = torch.sum(dist**2, dim=-1)
        
        # Limit receptive field by epsilon environment
        threshold = torch.quantile(edist, self.receptive_field, dim=-1, keepdim=True)
        mask = edist <= threshold
        max_neighbourhood = torch.max(torch.sum(mask, dim=-1))
        _, ind = torch.topk(-edist, k=max_neighbourhood, dim=-1)

        # Compute receptive field
        dist = self.slice(dist, ind)
        edist = self.slice(edist, ind)[..., None]

        # Gaussian kernel
        edist -= edist.min(-2, keepdim=True)[0]
        edist /= edist.max(-2, keepdim=True)[0] + self.eps
        k_distance = torch.softmax(-edist/self.softmax_temp, dim=-2)
        
        # Computed kernel
        projection = (2 * np.pi * dist) @ self.B
        k = torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)
        k = self.linear2(F.gelu(self.linear1(k) * self.query_modulation_weight[:, None, :] + self.query_modulation_offset[:, None, :])) + self.filter[None, None, :]
        
        # Combine both kernels
        k = k * k_distance
        k = rearrange(k, "q v (c d) -> q v c d", c=self.in_channels, d=self.out_channels)
        
        # Kernel transform
        x = self.linear(x)
        x = self.dropout(F.gelu(self.kernel_transform(k, ind, x) + self.bias[None, None, :]))

        # Pointwise MLP
        x = self.mlp(x) + x

        return x, q


class CALMDecoderLayer(nn.Module):
    """Continuous and adaptive convolution layer to increase a point cloud for the use in a decoder.

    Applies continuous and adaptive convolution to the incoming point cloud and increases it to a larger point cloud.
    The layer supports both periodic and non-periodic boundaries. It only supports cases where the positions for each
    sample in a batch is the same, which can be useful for deeper layers of the network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_query_points (int): Number of query points.
        receptive_field (float): A threshold value to limit the receptive field by the p-th percentile.
        softmax_temp (float): Temperature parameter for the distance weighting.
        spatial_dim (int): Spatial dimension of the input data, e.g., 2 for 2D data.
        is_periodic (bool): Whether the boundary is periodic or not.
        init_query_pos (torch.Tensor): Predefined positions to initialize the query points. Used for mesh prior initialization.
        dropout (float): Dropout probability for the channels.
        rff_std (float): Standard deviation for the random Fourier features used in the kernel function.
    """

    def __init__(self, in_channels=32, out_channels=16, num_query_points=512, receptive_field=1/10, softmax_temp=1.0,
                spatial_dim=2, is_periodic=True, init_query_pos=None, dropout=0.0, rff_std=6.0):
        super(CALMDecoderLayer, self).__init__()

        self.receptive_field = receptive_field
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.softmax_temp = softmax_temp
        self.eps = 1e-8

        self.is_periodic = is_periodic

        # Parameters for query positions and modulation
        self.query_pos = nn.Parameter(init_query_pos) if init_query_pos is not None else nn.Parameter(torch.rand(num_query_points, spatial_dim, dtype=torch.float32))
        self.query_modulation_weight = nn.Parameter(torch.rand(num_query_points, 32, dtype=torch.float32))
        self.query_modulation_offset = nn.Parameter(torch.zeros(num_query_points, 32, dtype=torch.float32))

        # Channelwise linear transformation and MLP
        self.linear = nn.Linear(in_channels, in_channels)
        self.mlp = nn.Sequential(nn.Linear(out_channels, out_channels * 4), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(out_channels * 4, out_channels), nn.Dropout(dropout))
        
        # MLP to parametrize the kernel function
        self.B = nn.Parameter(rff_std * torch.randn(size=(spatial_dim, 16), requires_grad=False), requires_grad=False)
        self.linear1 = nn.Linear(32, 32)
        self.linear2 = nn.Linear(32, in_channels * out_channels, bias=False)
        self.filter = nn.Parameter(uniform_distribution(in_channels * out_channels, -np.sqrt(3/in_channels), np.sqrt(3/in_channels)))

        # Helpers for the receptive field
        self.slice = torch.vmap(lambda x, ind: torch.index_select(x, 0, ind), in_dims=0, out_dims=0)
        self.kernel_transform = torch.vmap(lambda k, ind, x: contract("vcd,btvc->btd", k, x[:, :, ind, :]), in_dims=(0,0,None), out_dims=2)
        
        # Bias and dropout
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos):
        """Computes continuous and adaptive convolution on the input tensor x with the positions pos.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_points, in_channels).
            pos (torch.Tensor): Position tensor of shape (num_points, spatial_dim).
        
        Returns:
            tuple: A tuple containing the processed output and positions:
                torch.Tensor: Processed output tensor of shape (batch_size, num_points, out_channels).
                torch.Tensor: Corresponding query positions of shape (num_points, out_channels).
        """

        # x shape: b, n, c
        b = x.shape[0]
        
        q = self.query_pos
        
        # Position matrix
        dist = q[:, None, :] - pos[None, ...]
        if self.is_periodic: dist = (dist + 0.5) % 1 - 0.5
        edist = torch.sum(dist**2, dim=-1)
        
        # Limit receptive field by epsilon environment
        threshold = torch.quantile(edist, self.receptive_field, dim=-1, keepdim=True)
        mask = edist <= threshold
        max_neighbourhood = torch.max(torch.sum(mask, dim=-1))
        _, ind = torch.topk(-edist, k=max_neighbourhood, dim=-1)

        # Compute receptive field
        dist = self.slice(dist, ind)
        edist = self.slice(edist, ind)[..., None]

        # Gaussian kernel
        edist -= edist.min(-2, keepdim=True)[0]
        edist /= edist.max(-2, keepdim=True)[0] + self.eps
        k_distance = torch.softmax(-edist/self.softmax_temp, dim=-2)

        # Computed kernel
        projection = (2 * np.pi * dist) @ self.B
        k = torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)
        k = self.linear2(F.gelu(self.linear1(k) * self.query_modulation_weight[:, None, :] + self.query_modulation_offset[:, None, :])) + self.filter[None, None, :]
        
        # Combine both kernels
        k = k * k_distance
        k = rearrange(k, "q v (c d) -> q v c d", c=self.in_channels, d=self.out_channels)

        # Kernel transform
        x = self.linear(x)
        x = self.dropout(F.gelu(self.kernel_transform(k, ind, x) + self.bias[None, None, None, :]))
        
        # Pointwise MLP
        x = self.mlp(x) + x

        return x, q


class FinalCALMDecoderLayer(nn.Module):
    """Continuous and adaptive convolution layer that queries a point cloud with the given query positions for the use
    in a decoder.

    Applies continuous and adaptive convolution to the incoming point cloud and queries it with the given query positions.
    The layer supports both periodic and non-periodic boundaries and can handle cases where the query position is the
    same for each sample in a batch or different across samples in a batch.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        receptive_field (float): A threshold value to limit the receptive field by the p-th percentile.
        softmax_temp (float): Temperature parameter for the distance weighting.
        spatial_dim (int): Spatial dimension of the input data, e.g., 2 for 2D data.
        is_periodic (bool): Whether the boundary is periodic or not.
        same_grid_per_sample (bool): Whether the positions is the same across all samples in a batch or not. Setting
                                     this flag to True will make the layer more efficient for samples with the same positions.
        rff_std (float): Standard deviation for the random Fourier features used in the kernel function.
    """

    def __init__(self, in_channels=64, out_channels=32, receptive_field=1/20, softmax_temp=1,
                 spatial_dim=2, is_periodic=True, same_grid_per_sample=True, rff_std=6.0):
        super(FinalCALMDecoderLayer, self).__init__()
        
        self.receptive_field = receptive_field
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.softmax_temp = softmax_temp
        self.eps = 1e-8

        self.is_periodic = is_periodic
        self.same_grid_per_sample = same_grid_per_sample

        # Channelweise linear transformation
        self.linear = nn.Linear(in_channels, in_channels)

        # MLP to parametrize the kernel function
        self.B = nn.Parameter(rff_std * torch.randn(size=(spatial_dim, 16), requires_grad=False), requires_grad=False)
        self.linear1 = nn.Linear(32, 32)
        self.linear2 = nn.Linear(32, in_channels * out_channels, bias=False)
        self.filter = nn.Parameter(uniform_distribution(in_channels * out_channels, -np.sqrt(3/in_channels), np.sqrt(3/in_channels)))

        # Helpers for the receptive field
        self.slice = torch.vmap(lambda x, ind: torch.index_select(x, 0, ind), in_dims=0, out_dims=0)
        if same_grid_per_sample:
            self.kernel_transform = torch.vmap(lambda k, ind, x: contract("vcd,btvc->btd", k, x[:, :, ind, :]), in_dims=(0,0,None), out_dims=2)
        else:
            def func(k, ind, x):
                x = torch.stack([x[i, :, ind[i], :] for i in range(x.shape[0])])
                return contract("bvcd,btvc->btd", k, x)
            self.kernel_transform = torch.vmap(func, in_dims=(1,1,None), out_dims=2)
        
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))

    def _forward_same_grid(self, x, pos, q):
        # x shape: b, (h*w), c
        b = x.shape[0]
        
        # Position matrix
        dist = q[:, None, :] - pos[None, ...]
        if self.is_periodic: dist = (dist + 0.5) % 1 - 0.5
        edist = torch.sum(dist**2, dim=-1)
        
        # Limit receptive field by epsilon environment
        threshold = torch.quantile(edist, self.receptive_field, dim=-1, keepdim=True)
        mask = edist <= threshold
        max_neighbourhood = torch.max(torch.sum(mask, dim=-1))
        _, ind = torch.topk(-edist, k=max_neighbourhood, dim=-1)

        # Compute receptive field
        dist = self.slice(dist, ind)
        edist = self.slice(edist, ind)[..., None]

        # Gaussian kernel
        edist -= edist.min(-2, keepdim=True)[0]
        edist /= edist.max(-2, keepdim=True)[0] + self.eps
        k_distance = torch.softmax(-edist/self.softmax_temp, dim=-2)

        # Computed kernel
        projection = (2 * np.pi * dist) @ self.B
        k = torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)
        k = self.linear2(F.gelu(self.linear1(k))) + self.filter[None, None, :]

        # Combine both kernels
        k = k * k_distance
        k = rearrange(k, "q v (c d) -> q v c d", c=self.in_channels, d=self.out_channels)

        # Kernel transform
        x = self.linear(x)
        x = self.kernel_transform(k, ind, x) + self.bias[None, None, None, :]

        return x
    
    def _forward_different_grids(self, x, pos, q):
        # x shape: b, (h*w), c
        b = x.shape[0]
        
        # Position matrix
        dist = q[:, :, None, :] - pos[None, None, ...]
        if self.is_periodic: dist = (dist + 0.5) % 1 - 0.5
        edist = torch.sum(dist**2, dim=-1)
        
        # Limit receptive field by epsilon environment
        threshold = torch.quantile(edist, self.receptive_field, dim=-1, keepdim=True)
        mask = edist <= threshold
        max_neighbourhood = torch.max(torch.sum(mask, dim=-1))
        _, ind = torch.topk(-edist, k=max_neighbourhood, dim=-1)

        # Compute receptive field
        dist = rearrange(dist, "b q v d -> (b q) v d")
        edist = rearrange(edist, "b q v -> (b q) v")
        ind = rearrange(ind, "b q v -> (b q) v")
        dist = self.slice(dist, ind)
        edist = self.slice(edist, ind)[..., None]
        dist = rearrange(dist, "(b q) v d -> b q v d", q=q.shape[1])
        edist = rearrange(edist, "(b q) v d -> b q v d", q=q.shape[1])
        ind = rearrange(ind, "(b q) v -> b q v", q=q.shape[1])

        # Gaussian kernel
        edist -= edist.min(-2, keepdim=True)[0]
        edist /= edist.max(-2, keepdim=True)[0] + self.eps
        k_distance = torch.softmax(-edist/self.softmax_temp, dim=-2)

        # Computed kernel
        projection = (2 * np.pi * dist) @ self.B
        k = torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)
        k = self.linear2(F.gelu(self.linear1(k))) + self.filter[None, None, :]

        # Combine both kernels
        k = k * k_distance
        k = rearrange(k, "b q v (c d) -> b q v c d", c=self.in_channels, d=self.out_channels)

        # Integral transform
        x = self.linear(x)
        x = self.kernel_transform(k, ind, x) + self.bias[None, None, None, :]

        return x
    
    def forward(self, x, pos, query_pos):
        """Computes continuous and adaptive convolution on the input tensor x with the positions pos.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_points, in_channels).
            pos (torch.Tensor): Position tensor of shape (num_points, spatial_dim).
            query_pos (torch.Tensor): Position tensor of shape (num_query_points, spatial_dim).
        
        Returns:
            torch.Tensor: Processed output tensor of shape (batch_size, num_query_points, out_channels).
        """

        # x shape: b, n, c
        if self.same_grid_per_sample:
            x = self._forward_same_grid(x, pos, query_pos)
        else:
            x = self._forward_different_grids(x, pos, query_pos)

        return x
