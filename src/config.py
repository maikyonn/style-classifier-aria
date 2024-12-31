from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    n_classes: int = 4
    lr: float = 2e-5
    tokenizer_name: str = 'bert-base-uncased'
    max_length: int = 512
    dropout_rate: float = 0.3
    hidden_size: Optional[int] = None  # Will be set from BERT model

@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 20
    devices: int = 4
    num_workers: int = 32
    num_nodes: int = 1
    node_rank: int = 0
    precision: int = 16
    early_stopping_patience: int = 5
    save_top_k: int = 5

@dataclass
class DataConfig:
    train_ratio: float = 0.9
    max_seq_length: int = 1024
    style_tokens: list = None

    def __post_init__(self):
        if self.style_tokens is None:
            self.style_tokens = ["A", "B", "C", "D"] 