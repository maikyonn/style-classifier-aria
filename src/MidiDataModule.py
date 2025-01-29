# src/MidiDataModule.py

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import os
import random
from .MidiStyleDataset import MidiStyleDataset
from aria.tokenizer import AbsTokenizer

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
        # 1. Collect all MIDI and label files
        midi_folder = os.path.join(self.data_dir, 'midi')
        labels_folder = os.path.join(self.data_dir, 'labels')
        
        midi_files = sorted([
            os.path.join(midi_folder, f) for f in os.listdir(midi_folder) 
            if f.endswith('.mid')
        ])
        label_files = sorted([
            os.path.join(labels_folder, f) for f in os.listdir(labels_folder) 
            if f.endswith('.txt')
        ])

        if len(midi_files) != len(label_files):
            raise ValueError("Mismatch between number of MIDI and label files.")
        
        # 2. Create a list of (midi_file, label_file) pairs
        file_pairs = list(zip(midi_files, label_files))
        
        # 3. Shuffle and split
        random.seed(self.seed)
        random.shuffle(file_pairs)

        total_size = len(file_pairs)
        train_size = int(0.9 * total_size)  # Increased to 90% for train
        val_size = total_size - train_size   # Remaining 10% for validation

        train_pairs = file_pairs[:train_size]
        val_pairs = file_pairs[train_size:]

        print(f"\nFile-level split sizes:")
        print(f"Train: {len(train_pairs)}")
        print(f"Validation: {len(val_pairs)}")

        # 4. Create a single tokenizer for all splits
        self.tokenizer = AbsTokenizer()
        self.tokenizer.add_tokens_to_vocab(["A", "B", "C", "D"])
        self.pad_token = self.tokenizer.pad_id

        # 5. Instantiate datasets (each does its own chunking)
        self.train_dataset = MidiStyleDataset(train_pairs, self.tokenizer, max_len=self.max_len)
        self.val_dataset = MidiStyleDataset(val_pairs, self.tokenizer, max_len=self.max_len)

        print(f"Train dataset length (chunks): {len(self.train_dataset)}")
        print(f"Val dataset length (chunks):   {len(self.val_dataset)}\n")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.loader_kwargs)

    def get_tokenizer(self):
        return self.tokenizer
