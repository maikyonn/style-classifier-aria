#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training Script for MIDI Style Transformer Model

This script handles the training and inference of a MIDI style transformer model.
It includes custom loss functions, data loading, model setup, training loops, 
checkpointing, logging, and result visualization.

Dependencies:
- torch
- accelerate
- tqdm
- matplotlib
- numpy
- other custom modules (aria.tokenizer, src.midi_load_utils, etc.)

Ensure that all custom modules are accessible in your PYTHONPATH.

Usage:
    python train.py --dataset_dir <path_to_dataset>
"""

import os
import csv
import argparse
from datetime import datetime
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import random_split, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import accelerate
import wandb
from typing import Optional

from aria.tokenizer import AbsTokenizer, Tokenizer
from aria.data.midi import MidiDict
from src.midi_load_utils import load_midi_and_tokenize_multi, build_dataset, chunk_sequences
from aria.config import load_model_config
from src.model import ModelConfig, TransformerLM

# Initialize Accelerator
accelerator = accelerate.Accelerator()

# Set device
device = accelerator.device

# Ensure checkpoints directory exists
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===========================
# Dataset Definition
# ===========================

class MidiStyleDataset(Dataset):
    def __init__(self, data_dir, max_len=1024):
        self.tokenizer = AbsTokenizer()
        self.tokenizer.add_tokens_to_vocab(["A", "B", "C", "D"])
        self.midi_sequences, self.style_sequences = build_dataset(data_dir, self.tokenizer)
        self.max_len = max_len
        self.pad_token = self.tokenizer.encode(["<P>"])[0]

        # Break sequences into chunks of max_len using the chunk_sequences function
        self.midi_sequences = chunk_sequences(self.midi_sequences, self.max_len, self.pad_token)
        self.style_sequences = chunk_sequences(self.style_sequences, self.max_len, self.pad_token)

    def __len__(self):
        return len(self.midi_sequences)
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def __getitem__(self, idx):
        _mid = self.midi_sequences[idx]
        _sty = self.style_sequences[idx]
        midi_seq = torch.tensor(_mid, dtype=torch.long)
        style_seq = torch.tensor(_sty, dtype=torch.long)

        return midi_seq, style_seq

def build_dataloaders(dataset, batch_size=16, train_split=0.8):
    # Calculate the split sizes
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader

# ===========================
# Custom Loss Functions
# ===========================

class ChangePenaltyLoss(nn.Module):
    def __init__(self, ignore_index, alpha=10.0, max_changes=3):
        super().__init__()
        self.base_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.alpha = alpha
        self.max_changes = max_changes

    def forward(self, logits, labels):
        """
        logits: (batch_size, vocab_size, seq_len)
        labels: (batch_size, seq_len)
        """
        # Compute base cross-entropy loss
        base_loss = self.base_loss(logits, labels)  # (batch_size, seq_len)

        # Compute token changes 
        token_changes = (labels[:, 1:] != labels[:, :-1]).float()  # (batch_size, seq_len-1)

        # Count number of changes
        change_count = token_changes.sum(dim=1)  # (batch_size)

        # Penalize only when changes exceed max_changes
        change_penalty = torch.clamp(change_count - self.max_changes, min=0)  # (batch_size)

        # Combine losses
        combined_loss = base_loss.mean() + self.alpha * change_penalty.mean()

        return combined_loss

# ===========================
# Main Training Function
# ===========================

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop

def main(args):
    # ---------------------------
    # Setup and Configuration
    # ---------------------------
    
    # Load dataset and tokenizer
    dataset = MidiStyleDataset(data_dir=args.dataset_dir, max_len=1024)
    tokenizer = dataset.get_tokenizer()
    PAD_ID = tokenizer.pad_id

    # Load model configuration and initialize model
    model_config_dict = load_model_config("small")
    model_config = ModelConfig(**model_config_dict)
    model_config.set_vocab_size(tokenizer.vocab_size)
    model = TransformerLM(model_config)

    # Initialize DataLoader
    train_dataloader, val_dataloader = build_dataloaders(dataset, batch_size=16, train_split=0.8)

    # Initialize loss function
    loss_fn = ChangePenaltyLoss(ignore_index=PAD_ID, alpha=100, max_changes=3)

    # Set number of epochs
    epochs = 20

    # Initialize optimizer and scheduler
    optimizer, scheduler = get_optim(
        model,
        num_epochs=epochs,
        steps_per_epoch=len(train_dataloader),
    )

    # Prepare model, optimizer, and dataloaders with accelerator
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # Prepare CSV file for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"training_log_{timestamp}.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])

    # Initialize TensorBoard writer (optional)
    # writer = SummaryWriter(log_dir=f"runs/{timestamp}")

    # Add wandb initialization after the setup section
    run = wandb.init(
        project="midi-style-transformer",
        config={
            "learning_rate": 1e-4,
            "epochs": 20,
            "batch_size": 16,
            "model_config": model_config_dict,
            "max_len": 1024,
        }
    )

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=1e-4)

    # ---------------------------
    # Training Loop
    # ---------------------------

    for epoch in range(epochs):
        # Training Phase
        model.train()
        total_train_loss = 0
        train_progress_bar = tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}/{epochs}")

        for batch in train_progress_bar:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)  # (batch_size, seq_len, vocab_size)
            logits = logits.transpose(1, 2)  # (batch_size, vocab_size, seq_len)

            loss = loss_fn(logits, labels)
            
            # Backward pass
            accelerator.backward(loss)
            
            # Optimizer and scheduler steps
            optimizer.step()
            scheduler.step()
            
            # Accumulate loss
            total_train_loss += loss.item()
            
            # Update progress bar
            train_progress_bar.set_postfix({'Train Loss': total_train_loss / (train_progress_bar.n + 1)})

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validation Phase
        model.eval()
        total_val_loss = 0
        val_progress_bar = tqdm(val_dataloader, desc=f"Val Epoch {epoch+1}/{epochs}")

        with torch.no_grad():
            for batch in val_progress_bar:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                logits = model(inputs)
                logits = logits.transpose(1, 2)

                loss = loss_fn(logits, labels)
                total_val_loss += loss.item()

                val_progress_bar.set_postfix({'Val Loss': total_val_loss / (val_progress_bar.n + 1)})

        avg_val_loss = total_val_loss / len(val_dataloader)

        # Log metrics to wandb
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch": epoch + 1
        })

        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        # Logging
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        with open(csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([epoch+1, avg_train_loss, avg_val_loss])

        # Optionally log to TensorBoard
        # writer.add_scalar('Loss/Train', avg_train_loss, epoch+1)
        # writer.add_scalar('Loss/Val', avg_val_loss, epoch+1)

        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt")
        accelerator.save_state(checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    # Cleanup
    wandb.finish()
    print("Training complete. Log saved to", csv_filename)

    

# ===========================
# Optimizer and Scheduler Setup
# ===========================

def get_optim(model, num_epochs, steps_per_epoch, lr=1e-4, weight_decay=1e-5):
    """
    Initialize optimizer and learning rate scheduler.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = num_epochs * steps_per_epoch
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='linear'
    )
    return optimizer, scheduler

# ===========================
# Entry Point
# ===========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MIDI Style Transformer Model')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--wandb_project', type=str, default="midi-style-transformer", help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity/username')
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode="online"  # Use "disabled" for testing without logging
    )
    
    main(args)
