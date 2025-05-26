from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.data.data_collator import DataCollatorForLanguageModeling
import lightning as L
import torch
from torch.utils.data import DataLoader
from Bio import SeqIO
from argparse import ArgumentParser

def load_fasta(fasta_path):
    sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences

class FastaDataset(torch.utils.data.Dataset):
    
    def __init__(self, fasta_path, tokenizer, max_length=2048):
        super().__init__()
        self.sequences = load_fasta(fasta_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

class ModelWrapper(L.LightningModule):
    
    def __init__(self, model, tokenizer, max_length=2048, train_fasta_path="data/train.fasta", val_fasta_path="data/val.fasta"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_fasta_path = train_fasta_path
        self.val_fasta_path = val_fasta_path
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )
        
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        return optimizer
    
    def train_dataloader(self):
        dataset = FastaDataset(self.train_fasta_path, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=16, num_workers=4, collate_fn=self.data_collator)
        return dataloader
    
    def val_dataloader(self):
        dataset = FastaDataset(self.val_fasta_path, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=16, num_workers=4, collate_fn=self.data_collator)
        return dataloader
    

def main():
    parser = ArgumentParser()
    parser.add_argument("--train_fasta_path", type=str, default="data/train.fasta")
    parser.add_argument("--val_fasta_path", type=str, default="data/val.fasta")
    parser.add_argument("--output_dir", type=str, default="models/finetuned_model")
    parser.add_argument("--model_path", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--max_epochs", type=int, default=3)
    args = parser.parse_args()
    model = AutoModelForMaskedLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained("AI4Protein/ProPrime")
    model_wrapper = ModelWrapper(
        model, 
        tokenizer, 
        train_fasta_path=args.train_fasta_path, 
        val_fasta_path=args.val_fasta_path
    )
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        strategy="auto",
        devices="auto",
        accelerator="auto",
        precision="bf16-mixed",
    )
    trainer.fit(model_wrapper)
    trainer.save_checkpoint(args.output_dir)

if __name__ == "__main__":
    main()
