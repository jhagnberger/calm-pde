import hydra
from omegaconf import DictConfig

# CALM-PDE model for 1D with same positions for all samples in a batch
from training.train_1d import run_training as run_training_calm_1d
# CALM-PDE model for 2D with same positions for all samples in a batch
from training.train_2d import run_training as run_training_calm_2d
# CALM-PDE model for 3D with same positions for all samples in a batch
from training.train_3d import run_training as run_training_calm_3d
# CALM-PDE model with different positions for the samples in a batch and which uses mesh prior for initalization
from training.train_airfoil import run_training as run_training_calm_airfoil
# CALM-PDE model with different positions for the samples in a batch and which allows padding for a different number of points in a batch
from training.train_cylinder import run_training as run_training_calm_cylinder

@hydra.main(version_base="1.2", config_path="config", config_name="2d_fno_1e-4")
def main(cfg: DictConfig):
    experiment = cfg.experiment
    wandb_config = cfg.wandb
    if experiment.name == "CALM-PDE-1D":
        print("CALM-PDE-1D")
        run_training_calm_1d(config=experiment, wandb_config=wandb_config)
    elif experiment.name == "CALM-PDE-2D":
        print("CALM-PDE-2D")
        run_training_calm_2d(config=experiment, wandb_config=wandb_config)
    elif experiment.name == "CALM-PDE-Airfoil":
        print("CALM-PDE-Airfoil")
        run_training_calm_airfoil(config=experiment, wandb_config=wandb_config)
    elif experiment.name == "CALM-PDE-Cylinder":
        print("CALM-PDE-Cylinder")
        run_training_calm_cylinder(config=experiment, wandb_config=wandb_config)
    elif experiment.name == "CALM-PDE-3D":
        print("CALM-PDE-3D")
        run_training_calm_3d(config=experiment, wandb_config=wandb_config)
    else:
        raise ValueError(f"Unknown experiment name: {experiment.name}. Please check the config file.")


if __name__ == "__main__":
    main()
    print("Training done.")
