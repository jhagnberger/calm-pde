import torch
from models.dynamics_models import NeuralODE
from models.models_2d import CALM_PDE_Vorticity
from models.models_cylinder import CALM_PDE_Cylinder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Example 1: Using the CALM-PDE model for a 2D PDE such as Navier-Stokes. We assume that
# each sample has the same mesh and that the mesh is periodic.
processor = NeuralODE(128, 256, n_heads=8, is_periodic=True)
model = CALM_PDE_Vorticity(processor, in_channels=1, out_channels=1, dropout=0.0).to(device)

# Create random 2D data
u_0 = torch.rand(4, 4096, 1, device=device) # (b, n, c)
input_mesh = torch.rand(4096, 2, device=device) # (n, d)
output_mesh = torch.rand(4096, 2, device=device) # (n, d)
rollout_steps = 20
u = model.forward(u_0, input_mesh, output_mesh, rollout_steps) # (4, 20, 4096, 1)


# Example 2: Using a CALM layer for a 2D PDE with an irregularly sampled spatial domain 
# such as the cylinder flow.
processor = NeuralODE(128, 256, n_heads=8, is_periodic=False)
model = CALM_PDE_Cylinder(processor, in_channels=3, out_channels=16, dropout=0.0).to(device)

# Create random point cloud in 2D with different coordinates for each sample
u_0 = torch.rand(4, 1885, 3, device=device) # (b, n, c)
node_type = torch.zeros(4, 1885, 1, device=device, dtype=torch.int) # (b, n, c)
input_mesh = torch.rand(4, 1885, 2, device=device) # (b, n, d)
output_mesh = torch.rand(4, 1885, 2, device=device) # (b, n, d)
rollout_steps = 20
u = model(u_0, node_type, input_mesh, output_mesh, rollout_steps) # (4, 20, 1885, 16)
