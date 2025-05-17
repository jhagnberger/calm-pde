import torch
from models.layers import FirstCALMEncoderLayer, CALMEncoderLayer, CALMDecoderLayer, FinalCALMDecoderLayer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Example 1: Using CALM layer to encode a point cloud with the same positions for each sample
encoder = CALMEncoderLayer(in_channels=2,
                           out_channels=16,
                           num_query_points=512,
                           receptive_field=0.1,
                           softmax_temp=1.0,
                           spatial_dim=2,
                           is_periodic=False,
                           init_query_pos=None,
                           dropout=0.0).to(device)

# Create random point cloud in 2D with the same coordinates for each sample
features = torch.rand(4, 4096, 2, device=device) # (b, n, c)
positions = torch.rand(4096, 2, device=device) # (n, d)
features_new, positions = encoder(features, positions) # (4, 512, 16), (512, 2)


# Example 2: Using a CALM layer to encode a point cloud with different positions for each sample
encoder = FirstCALMEncoderLayer(in_channels=2,
                                out_channels=16,
                                num_query_points=512,
                                receptive_field=0.1,
                                softmax_temp=1.0,
                                spatial_dim=2,
                                same_grid_per_sample=False,
                                is_periodic=False,
                                init_query_pos=None,
                                dropout=0.0).to(device)

# Create random point cloud in 2D with different coordinates for each sample
features = torch.rand(4, 4096, 2, device=device) # (b, n, c)
positions = torch.rand(4, 4096, 2, device=device) # (b, n, d)
features_new, positions = encoder(features, positions) # (4, 512, 16), (512, 2)


# Example 3: Using a CALM layer to decode a latent point cloud
# You can simply use the CALMEncoderLayer to decode a latent point by using more query points than
# input points and by using less output than input channels.
decoder = CALMEncoderLayer(in_channels=32,
                           out_channels=8,
                           num_query_points=512,
                           receptive_field=0.1,
                           softmax_temp=1.0,
                           spatial_dim=2,
                           is_periodic=False,
                           init_query_pos=None,
                           dropout=0.0).to(device)

# Alternatively, you can use the CALMDecoderLayer to decode a latent point cloud, which supports
# a temporal dimension that is particularly useful for decoding time-dependent PDEs.
decoder_time_dependent = CALMDecoderLayer(in_channels=32,
                                          out_channels=8,
                                          num_query_points=512,
                                          receptive_field=0.1,
                                          softmax_temp=1.0,
                                          spatial_dim=2,
                                          is_periodic=False,
                                          init_query_pos=None,
                                          dropout=0.0).to(device)

# Create random point cloud in 2D with 32 features
features = torch.rand(4, 16, 32, device=device) # (b, n, c)
positions = torch.rand(16, 2, device=device) # (n, d)
features_new, positions = decoder(features, positions) # (4, 512, 16), (512, 2)
features_new, positions = decoder_time_dependent(features.unsqueeze(1), positions) # (4, 1, 512, 16), (512, 2)


# Example 4: Using a CALM layer to decode a time-dependent latent point cloud
decoder_time_dependent = CALMDecoderLayer(in_channels=32,
                                          out_channels=8,
                                          num_query_points=512,
                                          receptive_field=0.1,
                                          softmax_temp=1.0,
                                          spatial_dim=2,
                                          is_periodic=False,
                                          init_query_pos=None,
                                          dropout=0.0).to(device)

# Create random point cloud in 2D with 32 features and 10 time steps
features = torch.rand(4, 10, 16, 32, device=device) # (b, t, n, c)
positions = torch.rand(16, 2, device=device) # (n, d)
features_new, positions = decoder_time_dependent(features, positions) # (4, 10, 512, 16), (512, 2)


# Example 5: Using a CALM layer to decode a time-dependent latent point cloud with external query positions.
# We assume that the query positions are the same for each sample.
decoder_time_dependent = FinalCALMDecoderLayer(in_channels=32,
                                               out_channels=8,
                                               receptive_field=0.1,
                                               softmax_temp=1.0,
                                               spatial_dim=2,
                                               same_grid_per_sample=True,
                                               is_periodic=False).to(device)

# Create random point cloud in 2D with 32 features and 10 time steps
features = torch.rand(4, 10, 16, 32, device=device) # (b, t, n, c)
positions = torch.rand(16, 2, device=device) # (n, d)
query_positions = torch.rand(512, 2, device=device) # (n', d)
features_new = decoder_time_dependent(features, positions, query_positions) # (4, 10, 512, 16)


# Example 6: Using a CALM layer to decode a time-dependent latent point cloud with external query positions.
# We assume that the query positions are different for each sample.
decoder_time_dependent = FinalCALMDecoderLayer(in_channels=32,
                                               out_channels=8,
                                               receptive_field=0.1,
                                               softmax_temp=1.0,
                                               spatial_dim=2,
                                               same_grid_per_sample=False,
                                               is_periodic=False).to(device)

# Create random point cloud in 2D with 32 features and 10 time steps
features = torch.rand(4, 10, 16, 32, device=device) # (b, t, n, c)
positions = torch.rand(16, 2, device=device) # (n, d)
query_positions = torch.rand(4, 512, 2, device=device) # (b, n', d)
features_new = decoder_time_dependent(features, positions, query_positions) # (4, 10, 512, 16)
