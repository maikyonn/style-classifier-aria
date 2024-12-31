# train.py

import os
import argparse
import pdb
import torch
import torch.nn.functional as F
import csv
from pathlib import Path
from itertools import groupby
from collections import deque

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

# We'll use a learning rate scheduler from Hugging Face Transformers
from transformers import BertModel
from transformers.optimization import get_cosine_schedule_with_warmup

from src.MidiDataModule import MidiDataModule
from torchmetrics import Accuracy
from src.midi_load_utils import prepare_midi_for_inference

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
        test_midi_path: str = 'datasets/varied-forms-4k-collapsed-test/labels/00554_midi.mid'
    ):
        super().__init__()
        self.save_hyperparameters()

        # Placeholders for custom tokenizer/pad
        self.tokenizer = None
        self.pad_id = None

        # BERT model
        self.model = BertModel.from_pretrained('bert-base-uncased')

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

        # Log file for test predictions
        self.test_predictions_path = Path('test_midi_predictions.txt')
        if not self.test_predictions_path.exists():
            with open(self.test_predictions_path, 'w') as f:
                f.write("MIDI Style Predictions Log\n")
                f.write("=" * 50 + "\n\n")

        self.test_midi_path = test_midi_path
        self.test_style_path = 'datasets/varied-forms-4k-collapsed-test/labels/00554_style.txt'

        # Sample storage
        self.num_train_samples = 5
        self.stored_train_samples = []
        self.stored_train_labels = []

        # Loss smoothing window
        self.loss_window_size = 20
        self.recent_losses = deque(maxlen=self.loss_window_size)

        self.prev_val_loss = None

    def on_fit_start(self):
        """Hook called by PyTorch Lightning right before training starts."""
        if self.trainer and self.trainer.datamodule:
            self.tokenizer = self.trainer.datamodule.get_tokenizer()
            self.pad_id = self.tokenizer.pad_id
        else:
            raise RuntimeError("No datamodule found. Cannot retrieve tokenizer.")

    @staticmethod
    def condense_sequence(sequence):
        """Condense repeated tokens into count format."""
        if not sequence:
            return ""
        return ", ".join(f"{len(list(group))}x{item}" 
                         for item, group in groupby(sequence))

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
        """Standard training_step with average loss logging and LR logging."""
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

        # Smooth the training loss over a deque
        self.recent_losses.append(loss.detach())
        trailing_loss = torch.mean(torch.stack(list(self.recent_losses)))

        # Get current learning rate from your first optimizer
        optimizers = self.optimizers(use_pl_optimizer=False)
        if isinstance(optimizers, list):
            optimizer = optimizers[0]
        else:
            optimizer = optimizers  # Could be a single optimizer, not a list

        current_lr = optimizer.param_groups[0]["lr"]

        # Log metrics
        self.log('train_loss', trailing_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('lr', current_lr, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        if hasattr(self, 'prev_val_loss') and self.prev_val_loss is not None:
            self.log('prev_val_loss', self.prev_val_loss, on_step=True, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        """Standard validation_step that logs val_loss and updates val_accuracy."""
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
            preds = torch.argmax(valid_logits, dim=-1)
            self.val_accuracy.update(preds, valid_labels)

            # Log validation loss here. val_accuracy is logged in on_validation_epoch_end.
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                     batch_size=batch_size)
            return loss
        return None

    def on_train_batch_start(self, batch, batch_idx):
        # Optionally store examples for logging after validation
        if len(self.stored_train_samples) < self.num_train_samples and batch_idx == 0:
            midi_seq, style_seq = batch
            samples_to_store = min(self.num_train_samples - len(self.stored_train_samples),
                                   len(midi_seq))
            for i in range(samples_to_store):
                self.stored_train_samples.append(midi_seq[i].clone())
                self.stored_train_labels.append(style_seq[i].clone())

    def on_validation_epoch_end(self):
        """Compute and log val_acc, then log samples and test MIDI evaluations."""
        val_acc = self.val_accuracy.compute()
        self.log('val_acc_epoch', val_acc, prog_bar=True)
        self.val_accuracy.reset()

        if self.trainer.is_global_zero:
            val_loss = self.trainer.callback_metrics.get('val_loss')
            try:
                tokenizer = self.trainer.datamodule.get_tokenizer()

                with open(self.test_predictions_path, 'a') as f:
                    f.write(f"\nEpoch {self.current_epoch}:\n")
                    f.write("-" * 30 + "\n")
                    if val_loss is not None:
                        f.write(f"Validation Loss: {val_loss:.4f}\n\n")

                    f.write("Test MIDI File Results:\n")
                    test_predictions = self._process_and_evaluate(
                        self.test_midi_path,
                        self.test_style_path,
                        tokenizer
                    )
                    f.write(f"Test MIDI Accuracy: {test_predictions['accuracy']:.4f}\n")
                    f.write(f"Predicted Sequence: {test_predictions['pred_seq']}\n")
                    f.write(f"True Sequence: {test_predictions['true_seq']}\n\n")

                    # Log stored training samples
                    if self.stored_train_samples:
                        f.write("\nTraining Sample Results:\n")
                        f.write("-" * 30 + "\n")
                        label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
                        for idx, (train_sample, train_label) in enumerate(zip(self.stored_train_samples, 
                                                                              self.stored_train_labels)):
                            f.write(f"\nTraining Sample {idx + 1}:\n")
                            with torch.no_grad():
                                train_logits = self(
                                    train_sample.unsqueeze(0).to(self.device),
                                    torch.ones_like(train_sample.unsqueeze(0)).to(self.device)
                                )
                                train_preds = torch.argmax(train_logits.squeeze(0), dim=-1)

                            pred_styles = [label_map[p.item()] for p in train_preds if p != -100]
                            true_styles = [label_map[l.item()] for l in train_label if l != -100]
                            condensed_pred = self.condense_sequence(pred_styles)
                            condensed_true = self.condense_sequence(true_styles)

                            correct = sum(p == t for p, t in zip(pred_styles, true_styles))
                            accuracy = correct / len(true_styles) if true_styles else 0.0

                            f.write(f"Accuracy: {accuracy:.4f}\n")
                            f.write(f"Predicted: {condensed_pred}\n")
                            f.write(f"True: {condensed_true}\n")

                        # Average training sample accuracy
                        total_acc = 0.0
                        for s, l_true in zip(self.stored_train_samples, self.stored_train_labels):
                            with torch.no_grad():
                                logits_s = self(
                                    s.unsqueeze(0).to(self.device),
                                    torch.ones_like(s.unsqueeze(0)).to(self.device)
                                )
                                preds_s = torch.argmax(logits_s.squeeze(0), dim=-1)
                            pred_styles_s = [label_map[p.item()] for p in preds_s if p != -100]
                            true_styles_s = [label_map[l.item()] for l in l_true if l != -100]
                            if true_styles_s:
                                correct_s = sum(p == t for p, t in zip(pred_styles_s, true_styles_s))
                                total_acc += correct_s / len(true_styles_s)
                        avg_train_accuracy = total_acc / len(self.stored_train_samples)
                        f.write(f"\nAverage Training Sample Accuracy: {avg_train_accuracy:.4f}\n")

                    f.write("\n")

            except Exception as e:
                print(f"Error during validation epoch end: {str(e)}")

        # Store the current validation loss for logging in training steps
        if 'val_loss' in self.trainer.callback_metrics:
            self.prev_val_loss = self.trainer.callback_metrics['val_loss'].detach().clone()

    def _process_and_evaluate(self, midi_path, style_path, tokenizer):
        chunked_sequences, stride = prepare_midi_for_inference(
            midi_path,
            max_len=self.max_length,
            tokenizer=tokenizer
        )

        with open(style_path, 'r') as f:
            true_style_sequence = f.read().strip()
        true_labels = list(true_style_sequence)

        predictions_per_position = [[] for _ in range(len(true_labels))]

        for i, chunk in enumerate(chunked_sequences):
            chunk_tensor = torch.tensor(chunk).unsqueeze(0)
            attention_mask = (chunk_tensor != tokenizer.pad_id).long()

            with torch.no_grad():
                if self.device.type == 'cuda':
                    chunk_tensor = chunk_tensor.cuda()
                    attention_mask = attention_mask.cuda()

                logits = self(chunk_tensor, attention_mask)
                predictions = torch.argmax(logits.squeeze(0), dim=-1)
                predictions = predictions.cpu().numpy()

                for pos, pred in enumerate(predictions):
                    if i + pos < len(predictions_per_position):
                        predictions_per_position[i + pos].append(pred)

        all_predictions = []
        for pos_predictions in predictions_per_position:
            if pos_predictions:
                pred = max(set(pos_predictions), key=pos_predictions.count)
                all_predictions.append(pred)

        label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        predicted_styles = [label_map[pred] for pred in all_predictions]

        min_len = min(len(predicted_styles), len(true_labels))
        predicted_styles = predicted_styles[:min_len]
        true_labels = true_labels[:min_len]

        condensed_pred = self.condense_sequence(predicted_styles)
        condensed_true = self.condense_sequence(true_labels)

        correct = sum(p == t for p, t in zip(predicted_styles, true_labels))
        accuracy = correct / len(true_labels) if true_labels else 0.0

        return {
            'accuracy': accuracy,
            'pred_seq': condensed_pred,
            'true_seq': condensed_true
        }

    def configure_optimizers(self):
        """
        Define optimizer and LR scheduler:
          - AdamW as the optimizer
          - Cosine schedule with linear warmup
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # We'll compute the total training steps inside the trainer. 
        # This is the recommended approach for dynamic scheduling.
        if self.trainer is not None:
            # total number of steps (batches) in one epoch across all GPUs
            # multiplied by the desired number of epochs
            total_steps = self.trainer.estimated_stepping_batches
        else:
            # fallback if trainer is not initialized
            total_steps = 10000

        # Warmup: e.g., 10% of total steps
        warmup_steps = int(0.1 * total_steps)

        # Create a cosine schedule with warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Return both the optimizer and the scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",      # call the scheduler every step
                "frequency": 1,
                "monitor": "val_loss"    # not strictly required for this scheduler
            }
        }


def main(args):
    # pl.seed_everything(42)

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

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='checkpoint-{epoch:02d}-{val_acc:.2f}',
        monitor='val_acc_epoch',
        mode='max',
        save_top_k=50,
        save_last=True,
        every_n_epochs=1,
        auto_insert_metric_name=False
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        min_delta=1e-4,
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
        reload_dataloaders_every_n_epochs=1
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MIDI Token Classifier')

    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--devices', type=int, default=4, help='Number of GPUs to use per node')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of dataloader workers')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--n_classes', type=int, default=4, help='Number of classification classes')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes to use')
    parser.add_argument('--node_rank', type=int, default=0, help='Rank of this node')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate for regularization')
    parser.add_argument('--wandb_project', type=str, default='midi-style-classifier', help='WandB project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='WandB run name')

    args = parser.parse_args()

    global_batch_size = args.batch_size * args.devices * args.num_nodes
    print(f"Global batch size: {global_batch_size}")

    main(args)
