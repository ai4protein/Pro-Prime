import os
from lightning import Trainer
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.callbacks import Callback


class CheckpointEveryNSteps(Callback):
    """
    Save a checkpoint every N steps
    """

    def __init__(
        self,
        save_step_frequency,
        checkpoint_dir,
        checkpoint_name,
    ):
        self.save_step_frequency = save_step_frequency
        self.checkpoint_name = checkpoint_name
        self.checkpoint_dir = checkpoint_dir

    def on_train_batch_end(self, trainer: Trainer, *args, **kwargs):
        """Check if we should save a checkpoint after every train batch"""
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0 and global_step > 0:
            filename = f"{self.checkpoint_name}_{epoch=}_{global_step=}.ckpt"
            ckpt_path = os.path.join(self.checkpoint_dir, filename)
            rank_zero_info(f"Save checkpoint at {ckpt_path}")
            trainer.save_checkpoint(ckpt_path)
