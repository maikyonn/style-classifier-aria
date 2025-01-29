# src/MidiStyleDataset.py

import torch
from torch.utils.data import Dataset
from aria.tokenizer import AbsTokenizer
from src.midi_load_utils import process_file_pair, chunk_sequences

class MidiStyleDataset(Dataset):
    def __init__(self, file_pairs, tokenizer, max_len=1024):
        """
        file_pairs: list of (midi_file_path, label_file_path)
        tokenizer: a shared AbsTokenizer object
        max_len: chunk size
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_token = self.tokenizer.pad_id
        
        # Define label to ID mapping
        self.label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

        # Build sequences from the file pairs
        midi_seqs, style_seqs = self._build_sequences(file_pairs)

        # Chunk them
        self.midi_sequences = chunk_sequences(midi_seqs, max_len=max_len, padding_value=self.pad_token, stride=512)
        self.style_sequences = chunk_sequences(style_seqs, max_len=max_len, padding_value=self.pad_token, stride=512)

        # Sanity check: lengths should match after chunking
        if len(self.midi_sequences) != len(self.style_sequences):
            raise ValueError("Mismatch in number of MIDI vs style chunks.")

    def _build_sequences(self, file_pairs):
        """
        Build the raw MIDI and style sequences (token lists) from each file pair,
        skipping pairs that fail or have length mismatches.
        """
        midi_sequences = []
        style_sequences = []
        
        for midi_file, label_file in file_pairs:
            result = process_file_pair((midi_file, label_file), self.tokenizer)
            if result is None:
                # indicates an error or mismatch
                continue
            midi_seq, style_seq = result
            # We only add the sequence if lengths match
            if len(midi_seq) == len(style_seq):
                midi_sequences.append(midi_seq)
                style_sequences.append(style_seq)
        return midi_sequences, style_sequences
    
    def get_tokenizer(self):
        return self.tokenizer

    def __len__(self):
        return len(self.midi_sequences)

    def __getitem__(self, idx):
        # Return a single chunk from the chunked arrays
        midi = torch.tensor(self.midi_sequences[idx], dtype=torch.long)
        style_tokens = self.style_sequences[idx]

        # Convert token IDs back to tokens
        style_texts = self.tokenizer.decode(style_tokens)
        
        # Map tokens to label IDs
        style_labels = [self.label_map.get(token, -100) for token in style_texts]
        style_labels = torch.tensor(style_labels, dtype=torch.long)

        # Quick validation
        valid_labels = (style_labels >= 0) & (style_labels < len(self.label_map))
        padding_labels = (style_labels == -100)
        is_valid = torch.all(valid_labels | padding_labels)
        if not is_valid:
            raise ValueError(f"Invalid label found at index {idx}.")

        return midi, style_labels
