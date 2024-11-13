import os
import argparse
from datetime import datetime
import json
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

# Set CUDA optimization for Tensor Cores
torch.set_float32_matmul_precision('high')

from src.MidiStyleTransformer import MidiStyleTransformer
from src.MidiDataModule import MidiDataModule
from src.model import ModelConfig
from src.MidiStyleDataset import MidiStyleDataset

def load_model_config(config_path: str, vocab_size: int) -> ModelConfig:
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    config_dict['vocab_size'] = vocab_size
    return ModelConfig(**config_dict)

def main(args):
    # Create run directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name if args.run_name else f"run_{timestamp}"
    run_dir = Path(args.runs_dir) / run_name
    checkpoint_dir = run_dir / 'checkpoints'
    log_dir = run_dir / 'logs'
    
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    # Save configuration
    with open(run_dir / 'config.json', 'w') as f:
        json.dump({**vars(args), 'timestamp': timestamp}, f, indent=4)
    
    # Setup data and model
    datamodule = MidiDataModule(args.dataset_dir, args.batch_size, num_workers=args.num_workers)
    datamodule.setup()
    
    model_config = load_model_config(args.model_config, datamodule.tokenizer.vocab_size)
    model = MidiStyleTransformer(model_config, args.learning_rate)
    
    # Enhanced WandB logger setup
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        save_dir=log_dir,
        version=timestamp,
        config={
            # Track hyperparameters
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'accumulate_grad_batches': args.accumulate_grad_batches,
            'model_config': model_config.__dict__,
            'num_workers': args.num_workers,
            'precision': '16-mixed',
            'optimizer': 'AdamW',
            'scheduler': 'OneCycleLR',
        },
        tags=[
            f'gpus_{args.devices}',
            'distributed_training',
            run_name
        ],
        log_model=True
    )
    
    # Initialize trainer with updated logger
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=args.devices,
        strategy=DDPStrategy(
            find_unused_parameters=False,
            process_group_backend="nccl",
            start_method='spawn'
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename='midi-{epoch:02d}-{val_loss:.2f}',
                monitor='val/loss',
                mode='min',
                save_top_k=3,
                save_last=True
            ),
            EarlyStopping(
                monitor='val/loss',
                patience=5,
                mode='min'
            )
        ],
        logger=wandb_logger,
        precision='16-mixed',
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=50,
        sync_batchnorm=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MIDI Style Transformer Model')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--wandb_project', type=str, default="midi-style-transformer")
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--accumulate_grad_batches', type=int, default=4)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--runs_dir', type=str, default='runs')
    parser.add_argument('--model_config', type=str, default='config/models/small.json')
    parser.add_argument('--devices', type=int, default=4)
    
    args = parser.parse_args()
    main(args)