import wandb
import torch
import matplotlib.pyplot as plt
import numpy as np
from loss.losses import RelL2Loss


def initialize_gpu(random_seed):
    """Initializes the GPU settings and sets the random seed."""
    
    # Device settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    return device


def initialize_wandb(config, wandb_config):
    """Initializes wandb with the given config."""
    
    run = wandb.init(name=f"{config.model_name}_seed{config.random_seed}",
                    project=wandb_config.project,
                    entity=wandb_config.entity,
                    tags=[f"seed_{config.random_seed}", config.dataset, config.model_name, config.data_file],
                    config=config)
    
    print(f"Model {config.model_name}, "
          f"training type: {config.training_type}, "
          f"random seed: {config.random_seed}, "
          f"epochs: {config.epochs}, "
          f"learning rate: {config.learning_rate}")
    
    return run


def get_model_checkpoint_name(config):
    """Returns the model checkpoint name based on the config."""
    
    model_name = f"{config.model_name}_{config.model_tag}" if config.model_tag else config.model_name
    model_checkpoint_name = f"{model_name}_{config.training_type}_seed_{config.random_seed}_dataset_{config.dataset}_files_{config.data_file[:10]}"
    
    return model_checkpoint_name


def count_model_params(model):
    """Calculates number of parameters of the given model. Complex-valued weights count as two weights (for imaginary
    and real part)."""
    
    params = []
    for p in model.parameters():
        if p.requires_grad:
            if torch.is_complex(p):
                params.append(2 * p.numel())
            else:
                params.append(p.numel())
    return sum(params)


def exclude_params_from_weight_decay(model,
                                     exclude=["bias", "filter_bias", "norm", "query_pos", "modulation_weight", "B"],
                                     verbose=False):
    """Excludes the given parameters from the weight decay."""
    
    named_parameters = model.named_parameters()
    decay_parameters = []
    decay_parameters_names = []
    no_decay_parameters = []
    no_decay_parameters_names = []

    for name, param in named_parameters:
        if not any(ex in name for ex in exclude):
            decay_parameters_names.append(name)
            decay_parameters.append(param)
        else:
            no_decay_parameters_names.append(name)
            no_decay_parameters.append(param)

    if verbose:
        print("Exclude from weight decay:", no_decay_parameters_names)
        print("Weight decay for:", decay_parameters_names)

    grouped_parameters = [
        {"params": decay_parameters},
        {"params": no_decay_parameters, "weight_decay": 0.0}
    ]
    return grouped_parameters


def get_optimizer_scheduler_loss(model, config, train_loader, loss_dim=-2):
    """Returns the optimizer, scheduler, and loss for the given model and config.
    
    Args:
        model (torch.nn.Module): The model whose parameters will be optimized.
        config (object): Configuration object containing optimizer, scheduler, and loss function settings.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data, used to determine the number of steps per epoch.
        loss_dim (int, optional): Dimension over which to compute the relative L2 loss. Defaults to -2.
    
    Returns:
        tuple: A tuple containing:
            - optimizer (torch.optim.Optimizer): Configured optimizer.
            - scheduler (torch.optim.lr_scheduler._LRScheduler): Configured learning rate scheduler.
            - loss_fn (torch.nn.Module): Primary loss function.
            - rel_l2_loss_fn (torch.nn.Module): Relative L2 loss function (always returned for evaluation).
    
    Raises:
        ValueError: If an unsupported optimizer, scheduler, or loss function is specified in the config.
    """
    
    # Get optimizer
    if config.optimizer == "adam":
        # we have to exclude the bias and weights from norms
        grouped_parameters = exclude_params_from_weight_decay(model)
        optimizer = torch.optim.Adam(grouped_parameters, lr=config.learning_rate, weight_decay=1e-5)
    elif config.optimizer == "adamw":
        # we have to exclude the bias and weights from norms
        grouped_parameters = exclude_params_from_weight_decay(model, exclude=["bias", "norm", "query_pos", "B"])
        optimizer = torch.optim.AdamW(grouped_parameters, lr=config.learning_rate, weight_decay=1e-4)
    else:
        raise ValueError("Optimizer not supported!")
    
    # Get scheduler
    if config.scheduler == "one-cycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=config.learning_rate,
                                                        pct_start=config.scheduler_warmup_fraction,
                                                        div_factor=1e2, final_div_factor=1e3,
                                                        total_steps=epochs * len(train_loader))
    elif config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs * len(train_loader))
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)
    elif config.scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=-1)
    else:
        raise ValueError("Scheduler not supported!")
    
    # Get loss functions
    if config.loss_fn == "mse":
        loss_fn = nn.MSELoss(reduction="mean")
    elif config.loss_fn == "l1":
        loss_fn = nn.L1Loss(reduction="mean")
    elif config.loss_fn == "rel_l2":
        loss_fn = RelL2Loss(dim=loss_dim, reduction="sum")
    else:
        raise ValueError("Loss function not supported!")
    
    # Get rel. L2 loss functions
    rel_l2_loss_fn = RelL2Loss(dim=loss_dim, reduction="sum")

    return optimizer, scheduler, loss_fn, rel_l2_loss_fn
