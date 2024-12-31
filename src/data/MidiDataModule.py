import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from .MidiStyleDataset import MidiStyleDataset

class MidiDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, max_len=1024, num_workers=8, pin_memory=True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.loader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': True if self.num_workers > 0 else False,
            'prefetch_factor': 2,
            'drop_last': True,
        }

    def setup(self, stage=None):
        dataset = MidiStyleDataset(self.data_dir, self.max_len)
        self.tokenizer = dataset.get_tokenizer()
        self.pad_token = self.tokenizer.pad_id
        
        train_size = int(0.9 * len(dataset))
        self.train_dataset, self.val_dataset = random_split(
            dataset, 
            [train_size, len(dataset) - train_size]
        )
        
        print(f"\nDataset split: {len(self.train_dataset)} train, {len(self.val_dataset)} val")

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
    def get_tokenizer(self):
        """Returns the tokenizer used by the dataset."""
        return self.tokenizer