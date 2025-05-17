"""
PDEBench 1D Burgers: https://darus.uni-stuttgart.de/file.xhtml?fileId=268190&version=8.0
FNO datasets in original mat format: https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-
FNO datasets in numpy format: https://huggingface.co/datasets/jhagnberger/fno-vorticity
PDEBench 3D incompressible Navier-Stokes: https://darus.uni-stuttgart.de/file.xhtml?fileId=173286&version=8.0
MeshGraphNets 2D airfoil: https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets
MeshGraphNets 2D cylinder: https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets
"""

import torch
from torch.utils.data import Dataset
import os
import h5py
import numpy as np
import math as mt
from einops import rearrange
from pathlib import Path


class PDEBenchDataset(Dataset):
    """Code from PDEBench (https://github.com/pdebench/PDEBench/blob/main/pdebench/models/fno/utils.py)
    and modified for use in CALM-PDE.
    
    Args:
        filename (str): filen that contains the data
        initial_step (int): time steps taken as initial condition, defaults to 1
    """

    def __init__(self, filename,
                 initial_step=1,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max=-1,
                 load_stats_from_file=False):
        self.load_stats_from_file = load_stats_from_file

        # Define path to files
        self.filename = filename
        self.file_path = saved_folder
        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        assert filename[-2:] != 'h5', 'HDF5 data is assumed!!'
        
        with h5py.File(root_path, 'r') as f:
            keys = list(f.keys())
            keys.sort()
            if 'tensor' not in keys:
                _data = np.array(f['density'], dtype=np.float32)  # batch, time, x,...
                idx_cfd = _data.shape
                if len(idx_cfd)==3:  # 1D
                    self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                          idx_cfd[2]//reduced_resolution,
                                          mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                          3],
                                        dtype=np.float32)
                    #density
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data[...,0] = _data   # batch, x, t, ch
                    # pressure
                    _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data[...,1] = _data   # batch, x, t, ch
                    # Vx
                    _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data[...,2] = _data   # batch, x, t, ch

                    self.grid = np.array(f["x-coordinate"], dtype=np.float32)
                    self.grid = torch.tensor(self.grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                    print(self.data.shape)
                if len(idx_cfd)==4:  # 2D
                    self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                          idx_cfd[2]//reduced_resolution,
                                          idx_cfd[3]//reduced_resolution,
                                          mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                          4],
                                         dtype=np.float32)
                    # density
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[...,0] = _data   # batch, x, t, ch
                    # pressure
                    _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[...,1] = _data   # batch, x, t, ch
                    # Vx
                    _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[...,2] = _data   # batch, x, t, ch
                    # Vy
                    _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[...,3] = _data   # batch, x, t, ch

                    x = np.array(f["x-coordinate"], dtype=np.float32)
                    y = np.array(f["y-coordinate"], dtype=np.float32)
                    x = torch.tensor(x, dtype=torch.float)
                    y = torch.tensor(y, dtype=torch.float)
                    X, Y = torch.meshgrid(x, y, indexing='ij')
                    self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]
            
                if len(idx_cfd)==5:  # 3D
                    self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                          idx_cfd[2]//reduced_resolution,
                                          idx_cfd[3]//reduced_resolution,
                                          idx_cfd[4]//reduced_resolution,
                                          mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                          5],
                                         dtype=np.float32)
                    # density
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[...,0] = _data   # batch, x, t, ch
                    # pressure
                    _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[...,1] = _data   # batch, x, t, ch
                    # Vx
                    _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[...,2] = _data   # batch, x, t, ch
                    # Vy
                    _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[...,3] = _data   # batch, x, t, ch
                    # Vz
                    _data = np.array(f['Vz'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[...,4] = _data   # batch, x, t, ch

                    x = np.array(f["x-coordinate"], dtype=np.float32)
                    y = np.array(f["y-coordinate"], dtype=np.float32)
                    z = np.array(f["z-coordinate"], dtype=np.float32)
                    x = torch.tensor(x, dtype=torch.float)
                    y = torch.tensor(y, dtype=torch.float)
                    z = torch.tensor(z, dtype=torch.float)
                    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
                    self.grid = torch.stack((X, Y, Z), axis=-1)[::reduced_resolution,\
                                                                ::reduced_resolution,\
                                                                ::reduced_resolution]
                                                                
            else:  # scalar equations
                ## data dim = [t, x1, ..., xd, v]
                _data = np.array(f['tensor'], dtype=np.float32)  # batch, time, x,...
                if len(_data.shape) == 3:  # 1D
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data = _data[:, :, :, None]  # batch, x, t, ch

                    self.grid = np.array(f["x-coordinate"], dtype=np.float32)
                    self.grid = torch.tensor(self.grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                if len(_data.shape) == 4:  # 2D Darcy flow
                    # u: label
                    _data = _data[::reduced_batch,:,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                    #if _data.shape[-1]==1:  # if nt==1
                    #    _data = np.tile(_data, (1, 1, 1, 2))
                    self.data = _data
                    # nu: input
                    _data = np.array(f['nu'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch, None,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                    self.data = np.concatenate([_data, self.data], axis=-1)
                    self.data = self.data[:, :, :, :, None]  # batch, x, y, t, ch

                    x = np.array(f["x-coordinate"], dtype=np.float32)
                    y = np.array(f["y-coordinate"], dtype=np.float32)
                    x = torch.tensor(x, dtype=torch.float)
                    y = torch.tensor(y, dtype=torch.float)
                    X, Y = torch.meshgrid(x, y)
                    self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]

        # train and test split
        num_samples_max = self.data.shape[0]
        test_idx = int(num_samples_max * test_ratio)
        self.data = torch.tensor(self.data)

        # Stats and normalization
        self.compute_stats(self.data[test_idx:num_samples_max])
        self.normalize()

        if if_test:
            self.data = self.data[:test_idx]
        else:
            self.data = self.data[test_idx:num_samples_max]

        # Time steps used as initial conditions
        self.initial_step = initial_step

    def compute_stats(self, data):
        filename = os.path.splitext(self.filename)[0]
        stats_file = Path(os.path.join(self.file_path, filename + "_stats.npy"))
        if stats_file.is_file() and self.load_stats_from_file:
            print("Loading stats")
            data = np.load(stats_file)
            self.train_mean = torch.tensor(data[0])
            self.train_std = torch.tensor(data[1])
        else:
            print("Computing stats")
            if len(self.data.shape) == 4:
                dim = (0,1,2)
            elif len(self.data.shape) == 5:
                dim = (0,1,2,3)
            elif len(self.data.shape) == 6:
                dim = (0,1,2,3,4)
            self.train_std, self.train_mean = torch.std_mean(data, dim=dim, keepdims=False)
            np.save(stats_file, np.array([self.train_mean, self.train_std]))

    def normalize(self):
        # Normalize the data (for each channel)
        print("Normalizing data")
        self.data = (self.data - self.train_mean) / self.train_std
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Retrieves a sample for a given index with initial condition, its full trajectory, and the associated positions.
        
        Args:
            idx (int): Index of the data sample to retrieve.
            
        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The initial condition of the data sample up to `initial_step`.
                - torch.Tensor: The full trajectory including the initial condition.
                - torch.Tensor: The spatial positions associated with the trajectory.
        """
        
        return self.data[idx, ..., :self.initial_step, :], self.data[idx], self.grid


class FNODataset(Dataset):
    """Dataset for FNO vorticity data. Follows test/train split (last 200 samples for testing) proposed in FNO paper"""
    
    def __init__(self, filename,
                 initial_step=1,
                 saved_folder='../data/',
                 if_test=False,
                 reduced_resolution_spatial=1,
                 reduced_resolution_temporal=1,
                 reduced_batch=1,
                 num_samples_max=-1,
                 trajectory_start=-1,
                 trajectory_end=-1,
                 load_stats_from_file=False):
        self.load_stats_from_file = load_stats_from_file
        self.filename = filename
        self.file_path = os.path.abspath(saved_folder)

        # Define path to files
        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        
        data = torch.tensor(np.load(root_path), dtype=torch.float)[..., None] # nt, x, y, nb, nc
        self.data = data.permute(3, 1, 2, 0, 4) # batch, x, y, t, ch

        # We assume that the PDE is periodic in Omega = [0, 1)^2. Therefore, the endpoint must be excluded. This is only
        # important if the CALM layers are used in periodic mode.
        x0, y0 = np.meshgrid(np.linspace(0, 1, 64, endpoint=False), np.linspace(0, 1, 64, endpoint=False))
        xs = np.concatenate((x0[None, ...], y0[None, ...]), axis=0)  # [2, 64, 64]
        grid = torch.tensor(xs, dtype=torch.float)
        self.grid = grid.permute(1, 2, 0)  # [64, 64, 2]

        # limit number of samples
        if num_samples_max > 0:
            num_samples_max = min(num_samples_max, self.data.shape[0])
        else:
            num_samples_max = self.data.shape[0]
        
        # subsample temporal, spatial, batch
        self.grid = self.grid[::reduced_resolution_spatial, ::reduced_resolution_spatial, :]
        self.data = self.data[::reduced_batch, ::reduced_resolution_spatial, ::reduced_resolution_spatial, ::reduced_resolution_temporal, :]

        # Time steps used as initial conditions
        self.initial_step = initial_step

        # truncate trajectory
        trajectory_start = trajectory_start if trajectory_start > 0 else 0
        trajectory_end = trajectory_end if trajectory_end > 0 else self.data.shape[-2]
        self.data = self.data[..., trajectory_start:trajectory_end, :]

        # Stats and normalization
        self.compute_stats(self.data[:-200, ...])
        self.normalize()

        # Train and test split
        if if_test:
            self.data = self.data[-200:, ...]
        else:
            self.data = self.data[:-200, ...]
        
    def compute_stats(self, data):
        filename = os.path.splitext(self.filename)[0]
        stats_file = Path(os.path.join(self.file_path, filename + "_stats.npy"))
        if stats_file.is_file() and self.load_stats_from_file:
            print("Loading stats")
            data = np.load(stats_file)
            self.train_mean = torch.tensor(data[0])
            self.train_std = torch.tensor(data[1])
        else:
            print("Computing stats")
            self.train_std, self.train_mean = torch.std_mean(data, dim=(0,1,2,3), keepdims=False)
            np.save(stats_file, np.array([self.train_mean, self.train_std]))

    def normalize(self):
        # Normalize the data (for each channel)
        print("Normalizing data")
        self.data = (self.data - self.train_mean) / self.train_std

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Retrieves a sample for a given index with initial condition, its full trajectory, and the associated positions.
        
        Args:
            idx (int): Index of the data sample to retrieve.
            
        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The initial condition of the data sample up to `initial_step`.
                - torch.Tensor: The full trajectory including the initial condition.
                - torch.Tensor: The spatial positions associated with the trajectory.
        """
        
        return self.data[idx, ..., :self.initial_step, :], self.data[idx], self.grid


class AirfoilTimeDataset(Dataset):
    """Dataset for MeshGraphNets airfoil data."""

    def __init__(self, initial_step=1,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=2,
                 reduced_batch=1,
                 if_test=False,
                 load_stats_from_file=False):
        self.load_stats_from_file = load_stats_from_file
        self.path = os.path.abspath(saved_folder)
        self.initial_step = initial_step

        if if_test:
            split = "test"
        else:
            split = "train"

        # Load data
        self.data, self.grid, self.node_type = self.get_data(split)
        
        # Stats and normalize
        self.compute_stats()
        self.normalize()

        # Rearrange grid into [0,1]^2 domain to avoid changing the range of the model for each dataset
        self.grid = (self.grid + 20) / 40

        self.data = self.data[::reduced_batch, ::reduced_resolution, ::reduced_resolution_t, :]
        self.grid = self.grid[::reduced_batch, ::reduced_resolution, ::reduced_resolution_t, :]
        self.node_type = self.node_type[::reduced_batch, ::reduced_resolution, ::reduced_resolution_t, :]

    def get_data(self, split):
        data = []
        grid = []
        cells = []
        node_type = []
        data_file = Path(os.path.join(self.path, split + "_data.npy"))
        if data_file.is_file():
            print("Loading preprocessed data")
            data = np.load(data_file)
            grid = np.load(Path(os.path.join(self.path, split + "_grid.npy")))
            cells = np.load(Path(os.path.join(self.path, split + "_cells.npy")))
            node_type = np.load(Path(os.path.join(self.path, split + "_node_type.npy")))
        else:
            print("Loading from tf records")
            # Code from https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/dataset.py
            import tensorflow as tf
            import functools
            import json
            # Disable GPU for Tensorflow
            tf.config.set_visible_devices([], 'GPU')
            def parse(proto, meta):
                """Parses a trajectory from tf.Example."""
                feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta["field_names"]}
                features = tf.io.parse_single_example(proto, feature_lists)
                out = {}
                for key, field in meta["features"].items():
                    data = tf.io.decode_raw(features[key].values, getattr(tf, field["dtype"]))
                    data = tf.reshape(data, field["shape"])
                    if field["type"] == "static":
                        data = tf.tile(data, [meta["trajectory_length"], 1, 1])
                    elif field["type"] == "dynamic_varlen":
                        length = tf.io.decode_raw(features["length_" + key].values, tf.int32)
                        length = tf.reshape(length, [-1])
                        data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
                    elif field["type"] != "dynamic":
                        raise ValueError("invalid data format")
                    out[key] = data
                return out

            with open(os.path.join(self.path, "meta.json"), "r") as fp:
                meta = json.loads(fp.read())
            ds = tf.data.TFRecordDataset(os.path.join(self.path, split + ".tfrecord"))
            ds = ds.map(functools.partial(parse, meta=meta), num_parallel_calls=8)
            ds = ds.prefetch(1)

            # https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/common.py
            # 0: normal, 2: airfoil, 4: inflow
            NODE_TYPE_MAPPING = {0: 0, 2: 1, 4: 2}
            for d in ds:
                sample_pos = d["mesh_pos"].numpy()[::20, ...].transpose(1, 0, 2)
                # map cells to 0: fluid, 1: boundary, 2: geometry
                sample_node_type = d["node_type"].numpy()[::20, ...].transpose(1, 0, 2)
                sample_node_type = np.vectorize(NODE_TYPE_MAPPING.__getitem__)(sample_node_type).astype(np.int32)
                sample_velocity = d["velocity"].numpy()[::20, ...].transpose(1, 0, 2)
                sample_density = d["density"].numpy()[::20, ...].transpose(1, 0, 2)
                sample_pressure = d["pressure"].numpy()[::20, ...].transpose(1, 0, 2)
                sample_cells = d["cells"].numpy()[::20, ...].transpose(1, 0, 2)

                data.append(np.concatenate([sample_velocity, sample_density, sample_pressure], axis=-1))
                grid.append(sample_pos)
                cells.append(sample_cells)
                node_type.append(sample_node_type)
            
            data = np.stack(data, axis=0)
            grid = np.stack(grid, axis=0)
            cells = np.stack(cells, axis=0)
            node_type = np.stack(node_type, axis=0)

            # store preprocessed data
            np.save(Path(os.path.join(self.path, split + "_data.npy")), data)
            np.save(Path(os.path.join(self.path, split + "_grid.npy")), grid)
            np.save(Path(os.path.join(self.path, split + "_cells.npy")), cells)
            np.save(Path(os.path.join(self.path, split + "_node_type.npy")), node_type)
        
        # To pytorch tensor
        data = torch.tensor(data)
        grid = torch.tensor(grid)
        cells = torch.tensor(cells)
        node_type = torch.tensor(node_type)

        return data, grid, node_type

    def compute_stats(self):
        stats_file = Path(os.path.join(self.path, "airfoil_stats.npy"))
        if stats_file.is_file() and self.load_stats_from_file:
            print("Loading stats")
            data = np.load(stats_file)
            self.train_mean = torch.tensor(data[0])
            self.train_std = torch.tensor(data[1])
        else:
            print("Computing stats")
            train_data, _, _ = self.get_data("train")
            self.train_std, self.train_mean = torch.std_mean(train_data, dim=(0,1,2), keepdims=False)
            np.save(stats_file, np.array([self.train_mean, self.train_std]))
            del train_data
    
    def normalize(self):
        # Normalize the data (for each channel)
        print("Normalizing data")
        self.data = (self.data - self.train_mean) / self.train_std
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Retrieves a sample for a given index with initial condition, node_type, its full trajectory,
        and the associated positions.
        
        Args:
            idx (int): Index of the data sample to retrieve.
            
        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The initial condition of the data sample up to `initial_step`.
                - torch.Tensor: The node type of the initial condition.
                - torch.Tensor: The full trajectory including the initial condition.
                - torch.Tensor: The spatial positions associated with the trajectory.
        """
        return self.data[idx, :, :self.initial_step, :], self.node_type[idx, :, :self.initial_step, :], self.data[idx], self.grid[idx]


class CylinderTimeDataset(Dataset):
    """Dataset for MeshGraphNets cylinder data. The data is padded to the maximum number of nodes and cells in the dataset.
    Mask provides a mask to distinguish between the nodes and padding. The padded points have a position of (1000,1000) which
    is outside of the spatial domain and, consequently, will not be ignored by the CALM-PDE model."""

    def __init__(self, initial_step=1,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=2,
                 reduced_batch=1,
                 if_test=False,
                 load_stats_from_file=False):
        self.load_stats_from_file = load_stats_from_file
        self.path = os.path.abspath(saved_folder)
        self.initial_step = initial_step

        if if_test:
            split = "test"
        else:
            split = "train"

        # Load data
        self.data, self.grid, self.node_type, self.mask = self.get_data(split)
        
        # Stats and normalize
        self.compute_stats()
        self.normalize()

        # Rearrange grid into [0,1]^2 domain to avoid changing the range of the model for each dataset
        self.grid = self.grid / torch.tensor([1.6, 0.4])

        self.data = self.data[::reduced_batch, ::reduced_resolution, ::reduced_resolution_t, :]
        self.grid = self.grid[::reduced_batch, ::reduced_resolution, ::reduced_resolution_t, :]
        self.node_type = self.node_type[::reduced_batch, ::reduced_resolution, ::reduced_resolution_t, :]
        self.mask = self.mask[::reduced_batch, ::reduced_resolution, ::reduced_resolution_t, :]

        # mask data (the zeros in the data will be overwritten by the mean due to normalization)
        self.data = self.data * self.mask

    def get_data(self, split):
        data = []
        grid = []
        cells = []
        node_type = []
        num_nodes = []
        max_num_nodes = 0
        num_cells = []
        max_num_cells = 0
        data_file = Path(os.path.join(self.path, split + "_data.npy"))
        if data_file.is_file():
            print("Loading preprocessed data")
            data = np.load(data_file)
            grid = np.load(Path(os.path.join(self.path, split + "_grid.npy")))
            cells = np.load(Path(os.path.join(self.path, split + "_cells.npy")))
            node_type = np.load(Path(os.path.join(self.path, split + "_node_type.npy")))
            num_nodes = np.load(Path(os.path.join(self.path, split + "_num_nodes.npy")))
        else:
            print("Loading from tf records")
            # Code from https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/dataset.py
            import tensorflow as tf
            import functools
            import json
            # Disable GPU for Tensorflow
            tf.config.set_visible_devices([], 'GPU')
            def parse(proto, meta):
                """Parses a trajectory from tf.Example."""
                feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta["field_names"]}
                features = tf.io.parse_single_example(proto, feature_lists)
                out = {}
                for key, field in meta["features"].items():
                    data = tf.io.decode_raw(features[key].values, getattr(tf, field["dtype"]))
                    data = tf.reshape(data, field["shape"])
                    if field["type"] == "static":
                        data = tf.tile(data, [meta["trajectory_length"], 1, 1])
                    elif field["type"] == "dynamic_varlen":
                        length = tf.io.decode_raw(features["length_" + key].values, tf.int32)
                        length = tf.reshape(length, [-1])
                        data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
                    elif field["type"] != "dynamic":
                        raise ValueError("invalid data format")
                    out[key] = data
                return out

            with open(os.path.join(self.path, "meta.json"), "r") as fp:
                meta = json.loads(fp.read())
            ds = tf.data.TFRecordDataset(os.path.join(self.path, split + ".tfrecord"))
            ds = ds.map(functools.partial(parse, meta=meta), num_parallel_calls=8)
            ds = ds.prefetch(1)

            # https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/common.py
            # 0: normal, 4: inflow, 5: outflow, 6: wall
            NODE_TYPE_MAPPING = {0: 0, 4: 1, 5: 2, 6: 3}
            for d in ds:
                sample_pos = d["mesh_pos"].numpy()[::20, ...].transpose(1, 0, 2)
                # map cells to 0: fluid, 1: boundary, 2: geometry
                sample_node_type = d["node_type"].numpy()[::20, ...].transpose(1, 0, 2)
                sample_node_type = np.vectorize(NODE_TYPE_MAPPING.__getitem__)(sample_node_type).astype(np.int32)
                sample_velocity = d["velocity"].numpy()[::20, ...].transpose(1, 0, 2)
                sample_pressure = d["pressure"].numpy()[::20, ...].transpose(1, 0, 2)
                sample_cells = d["cells"].numpy()[::20, ...].transpose(1, 0, 2)

                max_num_nodes = max(max_num_nodes, sample_pos.shape[0])
                num_nodes.append(sample_pos.shape[0])
                max_num_cells = max(max_num_cells, sample_cells.shape[0])
                num_cells.append(sample_cells.shape[0])

                data.append(np.concatenate([sample_velocity, sample_pressure], axis=-1))
                grid.append(sample_pos)
                cells.append(sample_cells)
                node_type.append(sample_node_type)
            
            # pad the data with zeros and set position to (inf, inf)
            for i, data_sample in enumerate(data):
                pad_len = max_num_nodes-data_sample.shape[0]
                data[i] = np.concat([data_sample, np.zeros((pad_len, data_sample.shape[1], data_sample.shape[2]))], axis=0)
                grid[i] = np.concat([grid[i], np.full((pad_len, grid[i].shape[1], grid[i].shape[2]), 1000)], axis=0)
                node_type[i] = np.concat([node_type[i], np.zeros((pad_len, node_type[i].shape[1], node_type[i].shape[2]))], axis=0)

                cells[i] = np.concat([cells[i], np.zeros((max_num_cells-cells[i].shape[0], cells[i].shape[1], cells[i].shape[2]))], axis=0)
                
            data = np.stack(data, axis=0)
            grid = np.stack(grid, axis=0)
            cells = np.stack(cells, axis=0)
            node_type = np.stack(node_type, axis=0)
            num_nodes = np.array(num_nodes)
            num_cells = np.array(num_cells)

            # store preprocessed data
            np.save(Path(os.path.join(self.path, split + "_data.npy")), data)
            np.save(Path(os.path.join(self.path, split + "_grid.npy")), grid)
            np.save(Path(os.path.join(self.path, split + "_cells.npy")), cells)
            np.save(Path(os.path.join(self.path, split + "_node_type.npy")), node_type)
            np.save(Path(os.path.join(self.path, split + "_num_nodes.npy")), num_nodes)
            np.save(Path(os.path.join(self.path, split + "_num_cells.npy")), num_cells)
        
        # To pytorch tensor
        data = torch.tensor(data, dtype=torch.float)
        grid = torch.tensor(grid, dtype=torch.float)
        cells = torch.tensor(cells)
        node_type = torch.tensor(node_type, dtype=torch.int)

        # create mask
        mask = torch.arange(data.shape[1])[None, :] < torch.tensor(num_nodes)[:, None]
        mask = mask[:, :, None, None].expand(-1, -1, data.shape[2], 3)

        return data, grid, node_type, mask

    def compute_stats(self):
        stats_file = Path(os.path.join(self.path, "airfoil_stats.npy"))
        if stats_file.is_file() and self.load_stats_from_file:
            print("Loading stats")
            data = np.load(stats_file)
            self.train_mean = torch.tensor(data[0])
            self.train_std = torch.tensor(data[1])
        else:
            print("Computing stats")
            train_data, grid, _, mask = self.get_data("train")
            train_data = torch.masked.masked_tensor(train_data, mask=mask)
            self.train_std, self.train_mean = torch.std(train_data, dim=(0,1,2), keepdim=False).get_data(), torch.mean(train_data, dim=(0,1,2), keepdim=False).get_data()
            np.save(stats_file, np.array([self.train_mean, self.train_std]))
            del train_data
    
    def normalize(self):
        # Normalize the data (for each channel)
        print("Normalizing data")
        self.data = (self.data - self.train_mean) / self.train_std
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Retrieves a sample for a given index with initial condition, node_type, its full trajectory,
        the associated positions, and the mask.
        
        Args:
            idx (int): Index of the data sample to retrieve.
            
        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The initial condition of the data sample up to `initial_step`.
                - torch.Tensor: The node type of the initial condition.
                - torch.Tensor: The full trajectory including the initial condition.
                - torch.Tensor: The spatial positions associated with the trajectory.
                - torch.Tensor: A mask for the data to distinguish between real and padded points.
        """
        
        return self.data[idx, :, :self.initial_step, :], self.node_type[idx, :, :self.initial_step, :], self.data[idx], self.grid[idx], self.mask[idx]
