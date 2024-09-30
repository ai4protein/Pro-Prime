from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch.optim import Optimizer
import wandb


class LogParameter(Callback):

    def __init__(self, log_freq=500):
        self.log_freq = log_freq

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer
    ) -> None:
        if trainer.global_step % self.log_freq == 0:
            for name, param in pl_module.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if hasattr(trainer.logger.experiment, "log"):
                        trainer.logger.experiment.log(
                            {f"parameters/{name}": wandb.Histogram(param.data.cpu())},
                            step=trainer.global_step,
                        )