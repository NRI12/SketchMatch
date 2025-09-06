import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import hydra
from omegaconf import DictConfig
from src.data.datamodule import SketchDataModule
from src.lightning_modules.sketch_retrieval import SketchRetrievalModule

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    
    datamodule = SketchDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.data.image_size
    )
    
    model = SketchRetrievalModule(
        sketch_backbone=cfg.model.sketch_backbone,
        photo_backbone=cfg.model.photo_backbone,
        embedding_dim=cfg.model.embedding_dim,
        learning_rate=cfg.model.learning_rate,
        temperature=cfg.model.temperature,
        weight_decay=cfg.model.weight_decay
    )
    
    callbacks = [
        ModelCheckpoint(
            monitor=cfg.callbacks.model_checkpoint.monitor,
            mode=cfg.callbacks.model_checkpoint.mode,
            save_top_k=cfg.callbacks.model_checkpoint.save_top_k,
            filename=cfg.callbacks.model_checkpoint.filename
        ),
        EarlyStopping(
            monitor=cfg.callbacks.early_stopping.monitor,
            patience=cfg.callbacks.early_stopping.patience,
            mode=cfg.callbacks.early_stopping.mode
        )
    ]
    
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=callbacks
    )
    
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()