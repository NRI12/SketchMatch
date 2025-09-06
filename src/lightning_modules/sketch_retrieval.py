import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from ..models.dual_encoder import DualEncoder
from ..models.losses import InfoNCELoss

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
        
        self.loss_fn = InfoNCELoss(temperature=temperature)
        self.accuracy = Accuracy(task="binary", num_classes=2)
        
    def forward(self, sketch, photo):
        return self.model(sketch, photo)
    
    def training_step(self, batch, batch_idx):
        sketch = batch['sketch']
        positive_photo = batch['positive_photo']
        negative_photo = batch['negative_photo']
        
        sketch_embed = self.model.encode_sketch(sketch)
        pos_embed = self.model.encode_photo(positive_photo)
        neg_embed = self.model.encode_photo(negative_photo)
        
        loss = self.loss_fn(sketch_embed, pos_embed, neg_embed)
        
        pos_sim = F.cosine_similarity(sketch_embed, pos_embed, dim=1)
        neg_sim = F.cosine_similarity(sketch_embed, neg_embed, dim=1)
        
        predictions = (pos_sim > neg_sim).float()
        targets = torch.ones_like(predictions)
        acc = self.accuracy(predictions, targets)
        
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/accuracy', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        sketch = batch['sketch']
        positive_photo = batch['positive_photo']
        negative_photo = batch['negative_photo']
        
        sketch_embed = self.model.encode_sketch(sketch)
        pos_embed = self.model.encode_photo(positive_photo)
        neg_embed = self.model.encode_photo(negative_photo)
        
        loss = self.loss_fn(sketch_embed, pos_embed, neg_embed)
        
        pos_sim = F.cosine_similarity(sketch_embed, pos_embed, dim=1)
        neg_sim = F.cosine_similarity(sketch_embed, neg_embed, dim=1)
        
        predictions = (pos_sim > neg_sim).float()
        targets = torch.ones_like(predictions)
        acc = self.accuracy(predictions, targets)
        
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