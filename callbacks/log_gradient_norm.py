from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch.optim import Optimizer
import torch.distributed as dist
import torch


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def calculate_gradient_norm(model):
    local_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            local_norm += param.grad.data.norm(2).item() ** 2
    local_norm = local_norm**0.5
    if is_distributed():
        global_norm_tensor = torch.tensor(local_norm**2, device=torch.device("cuda"))
        dist.all_reduce(global_norm_tensor, op=dist.ReduceOp.SUM)
        global_norm = global_norm_tensor.item() ** 0.5
    else:
        global_norm = local_norm
    return global_norm


class LogGradientNorm(Callback):

    def __init__(self, log_freq=500):
        self.log_freq = log_freq

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer
    ) -> None:
        if trainer.global_step % self.log_freq == 0:
            grad_norm = calculate_gradient_norm(pl_module)
            if hasattr(trainer.logger.experiment, "log"):
                # For Weights & Biases
                trainer.logger.experiment.log({"gradient_norm": grad_norm, "global_step": trainer.global_step})
            elif hasattr(trainer.logger.experiment, "add_scalar"):
                # For TensorBoard
                trainer.logger.experiment.add_scalar("gradient_norm", grad_norm, trainer.global_step)
