from .model import TransformerLM
import torch
import pytorch_lightning as pl

class MidiStyleTransformer(pl.LightningModule):
    def __init__(self, model_config, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerLM(model_config)
        self.learning_rate = learning_rate
        self.val_loss = float('inf')

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        logits = logits.transpose(1, 2)
        return torch.nn.functional.cross_entropy(
            logits, labels, 
            ignore_index=self.trainer.datamodule.pad_token
        )

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        
        self.log('loss', loss, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log('val_loss', self.val_loss, prog_bar=True, sync_dist=True, rank_zero_only=True)
        
        self.log_dict({
            'train/loss': loss,
            'train/learning_rate': self.trainer.optimizers[0].param_groups[0]['lr'],
            'train/global_step': float(self.global_step),
        }, sync_dist=True, rank_zero_only=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.val_loss = loss.item()
        
        self.log_dict({
            'val/loss': loss,
            'val/epoch': float(self.current_epoch),
        }, sync_dist=True, on_epoch=True, rank_zero_only=True)
        
        return loss
    
    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        self.log_dict({
            'epoch/train_loss': metrics.get('train/loss', 0.0),
            'epoch/val_loss': metrics.get('val/loss', 0.0),
            'epoch/learning_rate': self.trainer.optimizers[0].param_groups[0]['lr'],
            'epoch/number': float(self.current_epoch),
        }, sync_dist=True, rank_zero_only=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy='linear',
            cycle_momentum=True,
            div_factor=25.0,
            final_div_factor=10000.0
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }