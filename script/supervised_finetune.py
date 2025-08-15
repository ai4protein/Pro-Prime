from argparse import ArgumentParser
from pathlib import Path
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer
import lightning as L
from torchmetrics.regression import SpearmanCorrCoef
import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--fasta", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--valid_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--prediction_csv", type=str, required=False)
    parser.add_argument("--output_csv", type=str, required=False)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--model_path", type=str, default="AI4Protein/Prime_690M")
    parser.add_argument("--normalize", type=str, default="zscore")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--skip_train", action="store_true", default=False)
    parser.add_argument("--checkpoint", type=str, required=False)
    args = parser.parse_args()
    return args

def read_seq(fasta_path: str):
    for record in SeqIO.parse(fasta_path, "fasta"):
        return str(record.seq)
    
def read_csv(csv_path: str, wt_seq: str):
    df = pd.read_csv(csv_path, nrows=20)
    mutated_seqs = []
    for mutant in df["mutant"]:
        mutated_seq = wt_seq
        for sub in mutant.split(":"):
            wt_aa, pos, mut_aa = sub[0], int(sub[1:-1]) - 1, sub[-1]
            mutated_seq = mutated_seq[:pos] + mut_aa + mutated_seq[pos+1:]
        mutated_seqs.append(mutated_seq)
    df["mutated_seq"] = mutated_seqs
    return df

class SequenceRegressionDataset(Dataset):
    
    def __init__(self, sequences, values):
        super().__init__()
        self.sequences = sequences
        self.values = values
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {"sequence": self.sequences[idx], "value": self.values[idx]}

class SequenceDataset(Dataset):
    
    def __init__(self, sequences):
        super().__init__()
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {"sequence": self.sequences[idx]}


class PrimeFT(nn.Module):
    
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.contact_head = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.out = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, 1)
        )
        
    def forward(self, sequences, values: torch.Tensor=None):
        inputs = self.tokenizer(sequences, return_tensors="pt").to("cuda")
        values = values.to("cuda") if values is not None else None
        outputs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        hidden_states = outputs.sequence_hidden_states
        predictions = self.out(hidden_states)
        if values is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions, values.unsqueeze(-1))
            return predictions, loss
        return predictions, None
    
class LitPrimeFT(L.LightningModule):
    
    def __init__(self, model_path, lr=1e-4):
        super().__init__()
        self.model = PrimeFT(model_path)
        self.train_spearman = SpearmanCorrCoef()
        self.valid_spearman = SpearmanCorrCoef()
        self.test_spearman = SpearmanCorrCoef()
        self.lr = lr
        self.save_hyperparameters()
        
    def training_step(self, batch, *args, **kwargs):
        sequences = batch["sequence"]
        values = batch["value"].to(self.device) if batch["value"] is not None else None
        predictions, loss = self.model(sequences, values)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.train_spearman(predictions.squeeze(-1), values)
        self.log("train/spearman", self.train_spearman, on_epoch=True, prog_bar=True, on_step=False)
        return loss
    
    def validation_step(self, batch, *args, **kwargs):
        sequences = batch["sequence"]
        values = batch["value"].to(self.device) if batch["value"] is not None else None
        predictions, loss = self.model(sequences, values)
        self.log("valid/loss", loss, on_epoch=True, prog_bar=True)
        self.valid_spearman(predictions.squeeze(-1), values)
        self.log("valid_spearman", self.valid_spearman, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, *args, **kwargs):
        sequences = batch["sequence"]
        values = batch["value"].to(self.device) if batch["value"] is not None else None
        predictions, loss = self.model(sequences, values)
        self.test_spearman(predictions.squeeze(-1), values)
        self.log("test/spearman", self.test_spearman, on_epoch=True, prog_bar=True)
        
    def predict_step(self, batch, *args, **kwargs):
        sequences = batch["sequence"]
        predictions, loss = self.model(sequences, None)
        return predictions.squeeze(-1)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
    
def main():
    args = parse_args()
    if args.checkpoint is not None:
        model = LitPrimeFT.load_from_checkpoint(args.checkpoint)
    else:
        model = LitPrimeFT(args.model_path)
    wt_seq = read_seq(args.fasta)
    if args.skip_train:
        pass
    else:
        print(f"{Path(args.fasta).name} WT sequence: {wt_seq}")
        train_df = read_csv(args.train_csv, wt_seq)
        valid_df = read_csv(args.valid_csv, wt_seq)
        test_df = read_csv(args.test_csv, wt_seq)
        if args.normalize == "zscore":
            train_mean = train_df["score"].mean()
            train_std = train_df["score"].std()
            train_df["score"] = (train_df["score"] - train_mean) / train_std
            valid_df["score"] = (valid_df["score"] - train_mean) / train_std
            test_df["score"] = (test_df["score"] - train_mean) / train_std
        train_dataset = SequenceRegressionDataset(train_df["mutated_seq"], train_df["score"].astype(np.float32))
        valid_dataset = SequenceRegressionDataset(valid_df["mutated_seq"], valid_df["score"].astype(np.float32))
        test_dataset = SequenceRegressionDataset(test_df["mutated_seq"], test_df["score"].astype(np.float32))
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)
        trainer = L.Trainer(
            max_epochs=args.max_epochs, 
            accelerator="gpu",
            devices=1,
            # enable_checkpointing=False,
            logger=False,
            callbacks=[
                EarlyStopping(monitor="valid_spearman", patience=args.patience, mode="max"),
                ModelCheckpoint(
                    monitor="valid_spearman", 
                    mode="max", 
                    save_top_k=1,
                    save_last=True,
                    filename=f"{Path(args.fasta).stem}" + "-{epoch}-{valid_spearman:.4f}",
                )
            ],
        )
        trainer.fit(model, train_dataloader, valid_dataloader)
        trainer.test(model, test_dataloader)
    
    if args.prediction_csv is not None:
        prediction_df = read_csv(args.prediction_csv, wt_seq)
        prediction_dataset = SequenceDataset(prediction_df["mutated_seq"])
        prediction_dataloader = DataLoader(prediction_dataset, batch_size=args.eval_batch_size, shuffle=False)
        predictions = trainer.predict(model, prediction_dataloader)
        predictions = torch.cat(predictions, dim=0)
        predictions = predictions.cpu().numpy()
        predictions = predictions * train_std + train_mean
        prediction_df["score"] = predictions
        prediction_df.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    main()
