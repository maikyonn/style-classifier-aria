import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from collections import deque

from .bert_classifier import BertStyleClassifier
from ..utils.metrics import calculate_sequence_accuracy

class MidiClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = BertStyleClassifier(config)
        self.recent_losses = deque(maxlen=20)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def _shared_step(self, batch, stage):
        midi_sequences, style_sequences = batch
        batch_size = midi_sequences.shape[0]
        
        accuracies = []
        all_logits = []
        all_labels = []
        
        for i in range(batch_size):
            acc, logits, labels = self._process_single_sequence(
                midi_sequences[i], style_sequences[i]
            )
            accuracies.append(acc)
            all_logits.append(logits)
            all_labels.append(labels)
        
        batch_accuracy = torch.tensor(accuracies, device=self.device).mean()
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.stack(all_labels)
        
        logits_flat = all_logits.view(-1, self.config.n_classes)
        labels_flat = all_labels.view(-1)
        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
        
        return loss, batch_accuracy

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, "train")
        self.recent_losses.append(loss.detach())
        trailing_loss = torch.mean(torch.stack(list(self.recent_losses)))
        
        self.log('train_loss', trailing_loss, on_step=True, on_epoch=True, 
                prog_bar=True, sync_dist=True, add_dataloader_idx=False)
        self.log('train_acc', acc, on_step=False, on_epoch=True, 
                prog_bar=True, sync_dist=True, add_dataloader_idx=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, "val")
        self.log('val_loss', loss, on_step=False, on_epoch=True, 
                prog_bar=True, sync_dist=True, add_dataloader_idx=False)
        self.log('val_acc', acc, on_step=False, on_epoch=True, 
                prog_bar=True, sync_dist=True, add_dataloader_idx=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.lr)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict 