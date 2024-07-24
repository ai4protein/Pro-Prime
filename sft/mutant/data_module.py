import torch.distributed
import torch
from torch.utils.data import DataLoader
import torch
from lightning import LightningDataModule
from Bio import SeqIO
import pandas as pd
from copy import deepcopy


def read_seq(fasta_file):
    for record in SeqIO.parse(fasta_file, "fasta"):
        return str(record.seq)


class Collator:

    def __init__(
        self,
        tokenizer,
    ):
        self.tokenizer = tokenizer

    def collate(self, batch):
        seqs = [item["sequence"] for item in batch]
        values = torch.tensor([item["value"] for item in batch], dtype=torch.float32)
        input_dict = self.tokenizer(seqs, padding=True, return_tensors="pt")
        input_ids = input_dict["input_ids"]
        attention_mask = input_dict["attention_mask"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "values": values,
        }

    def __call__(self, batch):
        return self.collate(batch)


class MutantDataModule(LightningDataModule):

    def __init__(
        self,
        fasta_file=None,
        train_file=None,
        valid_file=None,
        test_file=None,
        tokenizer=None,
        seed=333,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=16,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.fasta_file = fasta_file
        self.sequence = read_seq(fasta_file)
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.seed = seed
        self.collator = Collator(tokenizer)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

    def build_dataset(self, sequence, df):
        sequences = []
        for mutant in df["mutant"]:
            mseq = deepcopy(sequence)
            for sub_mutant in mutant.split(":"):
                wt, pos, mut = sub_mutant[0], int(sub_mutant[1:-1]) - 1, sub_mutant[-1]
                assert mseq[pos] == wt
                mseq = mseq[:pos] + mut + mseq[pos + 1 :]
            sequences.append(mseq)
        scores = df["score"].values
        return [
            {"sequence": seq, "value": score} for seq, score in zip(sequences, scores)
        ]

    def setup(self, stage=None):
        if stage == "fit":
            self.train_df = pd.read_csv(self.train_file)
            self.valid_df = pd.read_csv(self.valid_file)
            self._trainset = self.build_dataset(self.sequence, self.train_df)
            self._validset = self.build_dataset(self.sequence, self.valid_df)
        elif stage == "test":
            self.test_df = pd.read_csv(self.test_file)
            self._testset = self.build_dataset(self.sequence, self.test_df)
        else:
            pass

    def train_dataloader(self):
        return DataLoader(
            self._trainset,
            collate_fn=self.collator,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._validset,
            collate_fn=self.collator,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self._testset,
            collate_fn=self.collator,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
