import torch
from torch.utils.data import Dataset
from aria.tokenizer import AbsTokenizer
from src.midi_load_utils import build_dataset, chunk_sequences

class MidiStyleDataset(Dataset):
    def __init__(self, data_dir, max_len=1024):
        self.tokenizer = AbsTokenizer()
        print("Loading tokenizer and adding tokens...")
        self.tokenizer.add_tokens_to_vocab(["A", "B", "C", "D"])
        self.pad_token = self.tokenizer.encode(["<P>"])[0]

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
        return (torch.tensor(self.midi_sequences[idx], dtype=torch.long),
                torch.tensor(self.style_sequences[idx], dtype=torch.long))