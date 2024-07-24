from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from callbacks import (
    LogGradientNorm,
    LogParameter,
    CheckpointEveryNSteps,
)
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import os
import torch


def init_train():
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    try:
        torch.set_float32_matmul_precision("medium")
    except:
        pass


def strtobool(val):
    """Convert a string representation of truth to True or False."""
    val_lower = val.lower()
    if val_lower in ("yes", "y", "true", "t", "1"):
        return True
    elif val_lower in ("no", "n", "false", "f", "0"):
        return False
    else:
        raise ValueError(f"Invalid truth value: {val}")


def get_checkpoint_callback(save_model_dir, save_model_name) -> ModelCheckpoint:
    return CheckpointEveryNSteps(
        checkpoint_dir=save_model_dir, checkpoint_name=save_model_name
    )


def get_logger(logger_type, run_name=None, project=None, entity=None, log_args={}):
    if logger_type == "wandb":
        if entity is None:
            raise ValueError("entity must be provided when using wandb logger")
        logger = WandbLogger(
            name=run_name,
            project=project,
            entity=entity,
            config=vars(log_args),
        )
    elif logger_type == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=f"tb_logs/{run_name}",
            name=run_name,
        )
    else:
        logger = None
    return logger


def get_callbacks(
    save_model_dir,
    save_model_name,
    save_interval=None,
    log_gradient_norm=False,
    log_parameter=False,
):
    callbacks = []
    if save_interval is not None:
        callbacks.append(
            CheckpointEveryNSteps(
                save_step_frequency=save_interval,
                checkpoint_dir=save_model_dir,
                checkpoint_name=save_model_name,
            )
        )
    if log_gradient_norm:
        callbacks.append(LogGradientNorm())
    if log_parameter:
        callbacks.append(LogParameter())
    return callbacks


def get_strategy(strategy, find_unused_parameters=True):
    if strategy == "ddp":
        return DDPStrategy(find_unused_parameters=find_unused_parameters)
    else:
        return strategy
