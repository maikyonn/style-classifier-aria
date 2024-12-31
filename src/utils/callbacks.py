from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from typing import List

def get_callbacks(config) -> List:
    """Get training callbacks."""
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='checkpoint-{epoch:02d}-{val_acc:.2f}',
        monitor='val_acc',
        mode='max',
        save_top_k=5,
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

    return [checkpoint_callback, early_stopping] 