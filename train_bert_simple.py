# train.py

import os
import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

from transformers import BertModel
from transformers.optimization import get_cosine_schedule_with_warmup

from src.MidiDataModule import MidiDataModule
from torchmetrics import Accuracy

# Set wandb environment variables to use local directories
os.environ["WANDB_CACHE_DIR"] = "./wandb-cache"
os.environ["WANDB_DIR"] = "./wandb-local"
os.makedirs("./wandb-cache", exist_ok=True)
os.makedirs("./wandb-local", exist_ok=True)

# Set float32 matmul precision to medium for better performance
torch.set_float32_matmul_precision('medium')

class MidiClassifier(pl.LightningModule):
    def __init__(
        self,
        n_classes: int = 4,
        lr: float = 2e-5,
        max_length: int = 512,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Placeholders for custom tokenizer/pad
        self.tokenizer = None
        self.pad_id = None

        # BERT model
        self.model = BertModel.from_pretrained('bert-large-uncased')
        
        # Reset BERT model weights
        def init_weights(module):
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, torch.nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        self.model.apply(init_weights)
        # Ensure BERT is in training mode
        self.model.train()
        
        # Explicitly unfreeze BERT parameters (though they should be unfrozen by default)
        for param in self.model.parameters():
            param.requires_grad = True

        # Dropouts
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.dropout2 = torch.nn.Dropout(dropout_rate)

        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(self.model.config.hidden_size, n_classes)
        )

        self.lr = lr
        self.max_length = max_length

        # Metrics
        self.train_accuracy = Accuracy(task='multiclass', num_classes=n_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=n_classes)

    def on_fit_start(self):
        """Hook called by PyTorch Lightning right before training starts."""
        if self.trainer and self.trainer.datamodule:
            self.tokenizer = self.trainer.datamodule.get_tokenizer()
            self.pad_id = self.tokenizer.pad_id
        else:
            raise RuntimeError("No datamodule found. Cannot retrieve tokenizer.")

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = self.dropout1(outputs.last_hidden_state)
        logits = self.classifier(hidden_states)
        logits = self.dropout2(logits)
        return logits

    def training_step(self, batch, batch_idx):
        midi_sequences, style_sequences = batch

        attention_mask = (midi_sequences != self.pad_id).long()
        logits = self(input_ids=midi_sequences, attention_mask=attention_mask)

        logits = logits.view(-1, self.hparams.n_classes)
        style_labels = style_sequences.view(-1)

        loss = F.cross_entropy(logits, style_labels, ignore_index=-100)

        # Update training accuracy only on valid labels
        preds = torch.argmax(logits, dim=-1)
        valid_mask = (style_labels != -100)
        self.train_accuracy(preds[valid_mask], style_labels[valid_mask])

        # Log metrics with standardized keys
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_loss_step', loss, on_step=True, prog_bar=False, sync_dist=True)
        self.log('train_loss_epoch', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('step', self.global_step, on_step=True, on_epoch=False, prog_bar=False)
        self.log('epoch', self.current_epoch, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        midi_sequences, style_sequences = batch
        batch_size = midi_sequences.shape[0]

        attention_mask = (midi_sequences != self.pad_id).long()
        logits = self(midi_sequences, attention_mask)
        logits_flat = logits.view(-1, self.hparams.n_classes)
        labels_flat = style_sequences.view(-1)

        mask = (labels_flat != -100)
        valid_logits = logits_flat[mask]
        valid_labels = labels_flat[mask]

        if len(valid_labels) > 0:
            loss = F.cross_entropy(valid_logits, valid_labels)
            # Update validation metric logging
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                     batch_size=batch_size)
            return loss
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        if self.trainer is not None:
            total_steps = self.trainer.estimated_stepping_batches
        else:
            total_steps = 10000

        warmup_steps = int(0.1 * total_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val_loss"
            }
        }


def main(args):
    workers_per_gpu = max(1, args.num_workers // args.devices)

    data_module = MidiDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_len=args.max_length,
        num_workers=workers_per_gpu,
        pin_memory=True
    )
    data_module.setup()

    model = MidiClassifier(
        n_classes=args.n_classes,
        lr=args.lr,
        max_length=args.max_length,
        dropout_rate=args.dropout_rate
    )
    print(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='checkpoint-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=50,
        save_last=True,
        every_n_epochs=1,
        auto_insert_metric_name=False
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
        min_delta=1e-3,
        verbose=True
    )

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=f"{args.wandb_name}-dropout-{args.dropout_rate}-lr-{args.lr}" if args.wandb_name else f"dropout-{args.dropout_rate}-lr-{args.lr}",
        log_model=True,
        save_dir="./wandb-local",
        version=None,
        job_type="train"
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.devices if torch.cuda.is_available() else None,
        num_nodes=args.num_nodes,
        strategy='ddp_find_unused_parameters_true',
        precision=16,
        enable_progress_bar=True,
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=1  # Run validation every epoch
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MIDI Token Classifier')

    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--devices', type=int, default=1, help='Number of GPUs to use per node')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of dataloader workers')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--n_classes', type=int, default=4, help='Number of classification classes')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes to use')
    parser.add_argument('--node_rank', type=int, default=0, help='Rank of this node')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for regularization')
    parser.add_argument('--wandb_project', type=str, default='midi-style-classifier', help='WandB project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='WandB run name')

    args = parser.parse_args()

    global_batch_size = args.batch_size * args.devices * args.num_nodes
    print(f"Global batch size: {global_batch_size}")

    main(args)
