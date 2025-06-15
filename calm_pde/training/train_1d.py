import sys
import os
import torch
import numpy as np
import pickle
import torch.nn as nn
import wandb
import math
from einops import rearrange
from timeit import default_timer

# Dataloader and helper functions
from data.get_data import get_dataset
from utils.utils import initialize_gpu, initialize_wandb, get_model_checkpoint_name, count_model_params, get_optimizer_scheduler_loss

# CALM-PDE model
from models.models_1d import CALM_PDE_Burgers
from models.dynamics_models import NeuralODE


def run_training(config=None, wandb_config=None):
    # Set seed and GPU settings
    device = initialize_gpu(config.random_seed)

    # Weights and Biases
    run = initialize_wandb(config, wandb_config)

    # Load data
    train_data, test_data, train_mean, train_std, spatial_dim, in_channels, out_channels = get_dataset(config)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)
    train_mean, train_std = train_mean.to(device), train_std.to(device)
    
    # Create model    
    neural_ode = NeuralODE(64, 128, n_heads=8, spatial_dim=1)
    model = CALM_PDE_Burgers(neural_ode, in_channels=in_channels, out_channels=out_channels, dropout=config.dropout).to(device)
    print(f"Total parameters: {count_model_params(model)}")
    model_checkpoint_name = get_model_checkpoint_name(config)
    print(f"Checkpoint name: {model_checkpoint_name}")

    # Training and evaluation
    optimizer, scheduler, loss_fn, rel_l2_loss_fn = get_optimizer_scheduler_loss(model, config, train_loader, loss_dim=(1,2))
    
    # Training loop
    loss_test_min = np.inf
    for ep in range(config.epochs):
        t1 = default_timer()
        train_losses = {"loss": 0}
        train_parameters = {"full_rollout/trajectory_start": None}
        test_losses = {"loss": 0, "full_rollout/rel_l2": 0}

        model.train()
        for x, y, pos in train_loader:
            # b, w, t, c
            x = x.to(device)
            y = y.to(device)
            pos = pos.to(device)

            # Extract positions
            input_pos = pos[0, ...]
            output_pos = input_pos

            # Autoregressive rollout with increasing length
            epoch_norm = ep / config.epochs
            tstep_norm = 0.5 * (1.0 + math.tanh((epoch_norm - 0.5) / 0.3))
            tstep_ar = int(tstep_norm * (config.rollout_steps+1))
            t = min(config.rollout_steps+1, max(15, 15+tstep_ar))

            # Sample different starting points along trajectory
            subsequence_factor = 1
            if t < config.rollout_steps-subsequence_factor-1:
                bs = y.shape[0]
                start_indices = torch.randint(0, config.rollout_steps-t+2, (bs*subsequence_factor,))
                y_sampled = torch.stack([y[i//subsequence_factor, :, start_idx:start_idx+t] for i, start_idx in enumerate(start_indices)])
            else:
                y_sampled = y

            latent, latent_pos = model.encode(y_sampled[..., 0, :], input_pos)
            propagated_latent = model.time_stepping(latent, t-1)
            
            # Concatenate initial condiiton to propagated latents for decoding
            latent = torch.cat((latent[:, None, ...], propagated_latent), dim=1)
            y_hat = model.decode(latent, latent_pos, output_pos, no_time_dim=False)
            y_hat = rearrange(y_hat, "b t w c -> b w t c")
            
            loss = loss_fn(y_hat, y_sampled[..., :t, :])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Denormalize
            y_hat_physical = (y_hat[..., 1:, :].detach() * train_std) + train_mean
            y_physical = (y[..., 1:, :] * train_std) + train_mean

            # Metrics
            batch_size = y.size(0)
            train_losses["loss"] += loss.item() * batch_size
            train_parameters["full_rollout/trajectory_start"] = t

        # Evaluation
        model.eval()
        with torch.no_grad():
            for x, y, pos in test_loader:
                x = x.to(device)
                y = y.to(device)
                pos = pos.to(device)

                # Extract positions
                input_pos = pos[0, ...]
                output_pos = input_pos
                
                # Model run
                y_hat = model(x[..., 0, :], input_pos, output_pos, config.rollout_steps)
                y_hat = rearrange(y_hat, "b t w c -> b w t c")

                # Denormalize
                y_hat_physical = (y_hat * train_std) + train_mean
                y_physical = (y[..., 1:, :] * train_std) + train_mean

                # metrics
                batch_size = y.size(0)
                test_losses["loss"] += loss_fn(y_hat, y[..., 1:, :]).item() * batch_size
                test_losses["full_rollout/rel_l2"] += rel_l2_loss_fn(y_hat_physical, y_physical).item() * batch_size

            # Divide by total number of samples to get mean
            for loss_name in train_losses.keys():
                train_losses[loss_name] /= len(train_loader.dataset)
            for loss_name in test_losses.keys():
                test_losses[loss_name] /= len(test_loader.dataset)

            # Store best run
            if test_losses["full_rollout/rel_l2"] < loss_test_min:
                loss_test_min = test_losses["full_rollout/rel_l2"]
                torch.save({
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": test_losses["loss"],
                    "rel_l2_loss": test_losses["full_rollout/rel_l2"],
                    }, model_checkpoint_name + ".pt")
                
        t2 = default_timer()
        print(f"epoch: {ep}, t2-t1 (epoch time): {t2-t1:.5f}, train loss: {train_losses['loss']:.5f}, test loss: {test_losses['loss']:.5f}")
        wandb_dict = {"lr": scheduler.get_last_lr()[0]}
        wandb_dict.update({f"train/{key}": value for key, value in train_losses.items() if value != 0.0})
        wandb_dict.update({f"test/{key}": value for key, value in test_losses.items() if value != 0.0})
        wandb_dict.update({f"train/{key}": value for key, value in train_parameters.items() if value is not None})
        wandb.log(wandb_dict)

    # Save model weights
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(model_checkpoint_name + ".pt")
    run.log_artifact(artifact)
    run.finish()
