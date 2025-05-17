from data.datasets import PDEBenchDataset, FNODataset, AirfoilTimeDataset, CylinderTimeDataset


def get_dataset(config):
    """
    Get the dataset based on the provided configuration.

    Args:
        config: Configuration object containing dataset parameters.

    Returns:
        tuple: A tuple containing:
            - train_data: Training dataset.
            - test_data: Testing dataset.
            - train_mean: Mean of the training dataset.
            - train_std: Standard deviation of the training dataset.
            - spatial_dim: Spatial dimension of the dataset.
            - in_channels: Number of input channels.
            - out_channels: Number of output channels.
    """
    
    data_file = config.data_file
    dataset = config.dataset
    data_path = config.data_path
    reduced_resolution = config.reduced_resolution
    reduced_resolution_t = config.reduced_resolution_t
    truncated_trajectory_length = config.truncated_trajectory_length
    reduced_batch = config.reduced_batch
    initial_step = 1

    print(f"Using dataset {dataset} with file {data_file}")

    if dataset == "PDEBenchBurgers":
        spatial_dim = 1
        in_channels = 1
        out_channels = 1
        train_data = PDEBenchDataset(data_file,
                                reduced_resolution=reduced_resolution,
                                reduced_resolution_t=reduced_resolution_t,
                                reduced_batch=reduced_batch,
                                initial_step=initial_step,
                                saved_folder=data_path)
        test_data =  PDEBenchDataset(data_file,
                            reduced_resolution=reduced_resolution,
                            reduced_resolution_t=reduced_resolution_t,
                            reduced_batch=reduced_batch,
                            initial_step=initial_step,
                            if_test=True,
                            saved_folder=data_path)
    elif dataset == "PDEBenchCNS":
        spatial_dim = 3
        in_channels = 5
        out_channels = 5
        train_data = PDEBenchDataset(data_file,
                                reduced_resolution=reduced_resolution,
                                reduced_resolution_t=reduced_resolution_t,
                                reduced_batch=reduced_batch,
                                initial_step=initial_step,
                                saved_folder=data_path)
        test_data =  PDEBenchDataset(data_file,
                            reduced_resolution=reduced_resolution,
                            reduced_resolution_t=reduced_resolution_t,
                            reduced_batch=reduced_batch,
                            initial_step=initial_step,
                            if_test=True,
                            saved_folder=data_path)
    elif config.dataset == "FNO-1e-4":
        spatial_dim = 2
        in_channels = 1
        out_channels = 1
        train_data = FNODataset(data_file,
                                reduced_resolution_spatial=reduced_resolution,
                                reduced_resolution_temporal=reduced_resolution_t,
                                trajectory_start=10,
                                trajectory_end=truncated_trajectory_length,
                                reduced_batch=reduced_batch,
                                initial_step=initial_step,
                                saved_folder=data_path)
        test_data =  FNODataset(data_file,
                                reduced_resolution_spatial=reduced_resolution,
                                reduced_resolution_temporal=reduced_resolution_t,
                                trajectory_start=10,
                                trajectory_end=truncated_trajectory_length,
                                reduced_batch=reduced_batch,
                                initial_step=initial_step,
                                if_test=True,
                                saved_folder=data_path)
    elif config.dataset == "FNO-1e-5":
        spatial_dim = 2
        in_channels = 1
        out_channels = 1
        train_data = FNODataset(data_file,
                                reduced_resolution_spatial=reduced_resolution,
                                reduced_resolution_temporal=reduced_resolution_t,
                                trajectory_start=0,
                                trajectory_end=-1,
                                reduced_batch=reduced_batch,
                                initial_step=initial_step,
                                saved_folder=data_path)
        test_data =  FNODataset(data_file,
                                reduced_resolution_spatial=reduced_resolution,
                                reduced_resolution_temporal=reduced_resolution_t,
                                trajectory_start=0,
                                trajectory_end=-1,
                                reduced_batch=reduced_batch,
                                initial_step=initial_step,
                                if_test=True,
                                saved_folder=data_path)
    elif config.dataset == "AirfoilTime":
        spatial_dim = 2
        in_channels = 4
        out_channels = 4
        train_data = AirfoilTimeDataset(reduced_resolution=reduced_resolution,
                                reduced_resolution_t=reduced_resolution_t,
                                reduced_batch=reduced_batch,
                                initial_step=initial_step,
                                saved_folder=data_path)
        test_data = AirfoilTimeDataset(reduced_resolution=reduced_resolution,
                                reduced_resolution_t=reduced_resolution_t,
                                reduced_batch=reduced_batch,
                                initial_step=initial_step,
                                saved_folder=data_path,
                                if_test=True)
    elif config.dataset == "CylinderTime":
        spatial_dim = 2
        in_channels = 3
        out_channels = 3
        train_data = CylinderTimeDataset(reduced_resolution=reduced_resolution,
                                reduced_resolution_t=reduced_resolution_t,
                                reduced_batch=reduced_batch,
                                initial_step=initial_step,
                                saved_folder = data_path)
        test_data = CylinderTimeDataset(reduced_resolution=reduced_resolution,
                                reduced_resolution_t=reduced_resolution_t,
                                reduced_batch=reduced_batch,
                                initial_step=initial_step,
                                saved_folder = data_path,
                                if_test=True)
    else:
        raise ValueError(f"Unknown dataset ({config.dataset}) which is not supported!")
    
    return train_data, test_data, train_data.train_mean, train_data.train_std, spatial_dim, in_channels, out_channels
