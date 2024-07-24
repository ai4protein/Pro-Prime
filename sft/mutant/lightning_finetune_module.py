from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers import AutoModel
import torch
from transformers import get_scheduler
from typing import Any
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score, PearsonCorrCoef, SpearmanCorrCoef


class LightningForFineTuning(LightningModule):

    def __init__(
        self,
        model_path=None,
        tokenizer=None,
        lr=None,
        adam_epsilon=1e-8,
        adam_beta1=0.9,
        adam_beta2=0.999,
        warmup_steps=None,
        warmup_max_steps=None,
        weight_dacay=0.01,
        scheduler_type="linear",
        eval_determinstic=False,
        args=None,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.lr = lr
        self.adam_epsilon = adam_epsilon
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.weight_decay = weight_dacay
        self.scheduler_type = scheduler_type
        self.warmup_max_steps = warmup_max_steps
        self.warmup_steps = warmup_steps
        self.eval_determinstic = eval_determinstic
        self.tokenizer = tokenizer
        self.args = args
        
        self.valid_metrics = torch.nn.ModuleDict({
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "r2": R2Score(),
            "pearson": PearsonCorrCoef(),
            "spearman": SpearmanCorrCoef(),
        })
        
        self.test_metrics = torch.nn.ModuleDict({
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "r2": R2Score(),
            "pearson": PearsonCorrCoef(),
            "spearman": SpearmanCorrCoef(),
        })
    
        self.save_hyperparameters(ignore=["config", "ignore"])

    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        values = batch["values"]
        output = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            values=values,
        )
        loss = output.loss
        self.log(
            "train/loss",
            loss.detach(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=self.trainer.num_devices > 1,
        )         
        self.log(
            "train/lr",
            self.optimizers().param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=False,
            rank_zero_only=True,
        )
        self.log(
            "train/step",
            self.global_step,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
            rank_zero_only=True,
        )
        return loss

    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        values = batch["values"]
        output = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            values=values,
        )
        loss = output.loss
        self.log(
            "val/loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.trainer.num_devices > 1,
        )
        for k, v in self.valid_metrics.items():
            v(output.predicted_values, values.reshape(-1, 1))
            self.log(
                f"val/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=self.trainer.num_devices > 1,
            )
        return loss

    def test_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        values = batch["values"]
        output = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            values=values,
        )
        loss = output.loss
        self.log(
            "test/loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.trainer.num_devices > 1,
        )
        for k, v in self.valid_metrics.items():
            v(output.predicted_values, values.reshape(-1, 1))
            self.log(
                f"test/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=self.trainer.num_devices > 1,
            )
        return loss

    def configure_optimizers(self) -> Any:
        for param in self.model.pro_prime.parameters():
            param.requires_grad = False
        
        no_decay = [
            "bias",
            "LayerNorm.weight",
        ]  # no decay for bias and LayerNorm.weight

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            eps=self.adam_epsilon,
            betas=(self.adam_beta1, self.adam_beta2),
        )
        scheduler = get_scheduler(
            self.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.warmup_max_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
