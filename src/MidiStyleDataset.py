# src/MidiStyleDataset.py

import torch
from torch.utils.data import Dataset
from aria.tokenizer import AbsTokenizer
from src.midi_load_utils import build_dataset, chunk_sequences

class MidiStyleDataset(Dataset):
    def __init__(self, data_dir, max_len=1024):
        self.tokenizer = AbsTokenizer()
        print("Loading tokenizer and adding tokens...")
        self.tokenizer.add_tokens_to_vocab(["A", "B", "C", "D"])
        self.pad_token = self.tokenizer.pad_id

        # Define label to ID mapping
        self.label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

        print("Building dataset from MIDI files...")
        self.midi_sequences, self.style_sequences = build_dataset(data_dir, self.tokenizer)

        print("Chunking sequences...")
        self.midi_sequences = chunk_sequences(self.midi_sequences, max_len, self.pad_token)
        self.style_sequences = chunk_sequences(self.style_sequences, max_len, self.pad_token)

    def __len__(self):
        return len(self.midi_sequences)
    
    def get_tokenizer(self):
        return self.tokenizer
    
        
    def __getitem__(self, idx):
        midi = torch.tensor(self.midi_sequences[idx], dtype=torch.long)
        style_tokens = self.style_sequences[idx]

        # Convert token IDs back to tokens (assuming tokenizer can do this)
        style_texts = self.tokenizer.decode(style_tokens)
        
        # Split into individual tokens if needed (depends on tokenizer implementation)
        # Assuming style_tokens are a list of single-character tokens
        style_labels = [self.label_map.get(token, -100) for token in style_texts]
        
        style_labels = torch.tensor(style_labels, dtype=torch.long)
        
        # Validate label range, allowing for padding label (-100)
        valid_labels = (style_labels >= 0) & (style_labels < len(self.label_map))
        padding_labels = (style_labels == -100)
        is_valid = torch.all(valid_labels | padding_labels)
        
        if not is_valid:
            raise ValueError(f"Invalid label found at index {idx}. Labels must be between 0 and {len(self.label_map)-1}, or -100 for padding.")
        
        return midi, style_labels