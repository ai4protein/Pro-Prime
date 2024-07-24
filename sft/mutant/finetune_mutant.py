import argparse
from lightning import seed_everything
from lightning.pytorch.trainer import Trainer
from sft.mutant.lightning_finetune_module import LightningForFineTuning
from sft.mutant.data_module import MutantDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import AutoTokenizer
from sft.train_utils import (
    get_logger,
    get_callbacks,
    get_strategy,
    init_train,
    strtobool,
)
import torch
import gc

init_train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_gradient_norm",
        type=strtobool,
        default="False",
        help="Watch model gradients",
        required=False,
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default="AI4Protein/ProPrime_650M"
    )
    parser.add_argument("--model_path", type=str, default="AI4Protein/ProPrime_650M")
    # Parameters for data
    parser.add_argument("--fasta_file", type=str)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--valid_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int)

    # Parameters for training
    parser.add_argument("--accumulate_grad_batches", default=32, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--adam_beta1", default=0.9, type=float)
    parser.add_argument("--adam_beta2", default=0.999, type=float)
    parser.add_argument("--adam_epsilon", default=1e-07, type=float)
    parser.add_argument("--max_steps", default=1000000, type=int)
    parser.add_argument("--warmup_max_steps", default=None, type=int)
    parser.add_argument("--max_epochs", default=1000, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--gradient_clip_value", default=1.0, type=float)
    parser.add_argument("--gradient_clip_algorithm", default="norm", type=str)
    parser.add_argument("--precision", default="32", type=str)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--scheduler_type", default="linear", type=str)

    # Parameters for validation and checkpoint
    parser.add_argument("--monitor", default="val/pearson", type=str)
    parser.add_argument("--save_model_dir", default="checkpoint", type=str)
    parser.add_argument("--save_model_name", required=True, type=str)

    # Parameters for logging
    parser.add_argument("--log_steps", default=5, type=int)
    parser.add_argument("--logger", default="tensorboard", type=str)
    parser.add_argument("--logger_project", default="openesm", type=str)
    parser.add_argument("--logger_run_name", default="openesm", type=str)
    parser.add_argument("--wandb_entity", default=None, type=str)

    # parameters for distributed training
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--nodes", default=1, type=int)
    parser.add_argument("--accelerator", default="gpu", type=str)
    parser.add_argument("--strategy", default="ddp", type=str)

    # parameters for resume training
    parser.add_argument("--trainer_ckpt", default=None, type=str)
    args = parser.parse_args()

    if args.max_epochs is None or args.max_epochs < 0:
        args.max_epochs = 1000000
    if args.max_steps is None and args.max_epochs is None:
        raise ValueError("max_steps and max_epochs can not be None at the same time")

    if args.warmup_max_steps is None or (args.warmup_max_steps < args.max_steps):
        print("warmup_max_steps should be larger than max_steps")
        print("Set warmup_max_steps to max_steps")
        args.warmup_max_steps = args.max_steps

    return args


def get_trainer(args):
    callbacks = get_callbacks(
        save_model_dir=args.save_model_dir,
        save_model_name=args.save_model_name,
        log_gradient_norm=args.log_gradient_norm,
        log_parameter=False,
    )
    model_checkpoint = ModelCheckpoint(
        monitor=args.monitor,
        dirpath=args.save_model_dir,
        filename=args.save_model_name,
        save_top_k=1,
        mode="max",
    )
    callbacks.append(model_checkpoint)
    logger = get_logger(
        logger_type=args.logger,
        run_name=args.logger_run_name,
        project=args.logger_project,
        entity=args.wandb_entity,
        log_args=args,
    )
    strategy = get_strategy(args.strategy)

    trainer = Trainer(
        strategy=strategy,
        accelerator=args.accelerator,
        devices=args.devices,
        log_every_n_steps=args.log_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_value,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        logger=logger,
        precision=args.precision,
        num_nodes=args.nodes,
        use_distributed_sampler=False,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        reload_dataloaders_every_n_epochs=1,
    )

    return trainer


def main():
    args = parse_args()
    if args.seed is not None:
        seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )
    model = LightningForFineTuning(
        model_path=args.model_path,
        tokenizer=tokenizer,
        lr=args.lr,
        adam_epsilon=args.adam_epsilon,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        warmup_steps=args.warmup_steps,
        warmup_max_steps=args.warmup_max_steps,
        weight_dacay=args.weight_decay,
        scheduler_type=args.scheduler_type,
        eval_determinstic=False,
        args=args,
    )
    trainer = get_trainer(args)
    datamodule = MutantDataModule(
        fasta_file=args.fasta_file,
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        tokenizer=tokenizer,
        seed=args.seed,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.trainer_ckpt)
    model = LightningForFineTuning.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, args=args
    )
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
