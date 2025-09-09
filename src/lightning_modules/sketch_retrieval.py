import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from ..models.dual_encoder import DualEncoder
from ..models.losses import CLIPStyleInfoNCE

class SketchRetrievalModule(pl.LightningModule):
    def __init__(
        self,
        sketch_backbone="resnet18",
        photo_backbone="resnet18",
        embedding_dim=128,
        learning_rate=1e-3,
        temperature=0.07,
        weight_decay=1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = DualEncoder(
            sketch_backbone=sketch_backbone,
            photo_backbone=photo_backbone,
            embedding_dim=embedding_dim
        )
        
        self.loss_fn = CLIPStyleInfoNCE(temperature=temperature)
        self.accuracy = Accuracy(task="binary", num_classes=2)
        
    def forward(self, sketch, photo):
        return self.model(sketch, photo)
    
    def training_step(self, batch, batch_idx):
        sketch = batch['sketch']
        photo = batch['positive_photo']
        
        sketch_embed = self.model.encode_sketch(sketch)
        photo_embed = self.model.encode_photo(photo)
        
        # CLIP-style loss với batch-all negatives
        loss = self.loss_fn(sketch_embed, photo_embed)
        
        with torch.no_grad():
            sim_matrix = F.cosine_similarity(
                sketch_embed.unsqueeze(1), 
                photo_embed.unsqueeze(0), 
                dim=2
            )
            pred_indices = sim_matrix.argmax(dim=1)
            correct = pred_indices == torch.arange(len(sketch), device=sketch.device)
            acc = correct.float().mean()
        
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/accuracy', acc, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        sketch = batch['sketch']
        photo = batch['positive_photo']

        sketch_embed = self.model.encode_sketch(sketch)
        photo_embed = self.model.encode_photo(photo)

        loss = self.loss_fn(sketch_embed, photo_embed)

        sim_matrix = F.cosine_similarity(
            sketch_embed.unsqueeze(1), 
            photo_embed.unsqueeze(0), 
            dim=2
        )
        pred_indices = sim_matrix.argmax(dim=1)
        correct = pred_indices == torch.arange(len(sketch), device=sketch.device)
        acc = correct.float().mean()

        self.log('val/loss', loss, prog_bar=True)
        self.log('val/accuracy', acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss"
            }
        }