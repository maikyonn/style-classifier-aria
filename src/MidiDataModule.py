import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torch
from .MidiStyleDataset import MidiStyleDataset

class MidiDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, max_len=1024, num_workers=8, pin_memory=True, seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        
        self.loader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': True if self.num_workers > 0 else False,
            'prefetch_factor': 2,
            'drop_last': True,
        }

    def setup(self, stage=None):
        # Create full dataset
        full_dataset = MidiStyleDataset(self.data_dir, self.max_len)
        self.tokenizer = full_dataset.get_tokenizer()
        self.pad_token = self.tokenizer.pad_id
        
        # Calculate lengths for splits (80-10-10 split)
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        
        # Use generator for reproducibility
        generator = torch.Generator().manual_seed(self.seed)
        
        # Create splits
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=generator
        )
        
        print(f"\nDataset split sizes:")
        print(f"Train: {len(self.train_dataset)}")
        print(f"Validation: {len(self.val_dataset)}")
        print(f"Test: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            **self.loader_kwargs
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            **self.loader_kwargs
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            **self.loader_kwargs
        )

    def get_tokenizer(self):
        """Returns the tokenizer used by the dataset."""
        return self.tokenizer