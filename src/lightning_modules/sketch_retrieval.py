import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.retrieval import RetrievalRecall
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
        
        # Recall metrics cho training và validation
        self.train_metrics = MetricCollection({
            'recall@1': RetrievalRecall(k=1),
            'recall@5': RetrievalRecall(k=5),
        })
        
        self.val_metrics = MetricCollection({
            'recall@1': RetrievalRecall(k=1),
            'recall@5': RetrievalRecall(k=5),
            'recall@10': RetrievalRecall(k=10),
        })
        
    def forward(self, sketch, photo):
        return self.model(sketch, photo)
    
    def compute_recall_metrics(self, sketch_embed, photo_embed, metrics):
        batch_size = sketch_embed.size(0)
        
        sim_matrix = torch.matmul(sketch_embed, photo_embed.T)
        targets = torch.arange(batch_size, device=sketch_embed.device)
        

        all_preds = []
        all_targets = []
        all_indexes = []
        
        for i in range(batch_size):
            # For i-th sketch, similarities with all photos
            similarities = sim_matrix[i]  # [batch_size]
            
            target_binary = torch.zeros(batch_size, device=sketch_embed.device)
            target_binary[i] = 1.0
            
            all_preds.append(similarities)
            all_targets.append(target_binary)
            all_indexes.extend([i] * batch_size)
        
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        indexes = torch.tensor(all_indexes, device=sketch_embed.device)
        
        return metrics(preds, targets, indexes)
    
    def training_step(self, batch, batch_idx):
        sketch = batch['sketch']
        photo = batch['positive_photo']
        
        sketch_embed = self.model.encode_sketch(sketch)
        photo_embed = self.model.encode_photo(photo)
        
        # CLIP-style loss
        loss = self.loss_fn(sketch_embed, photo_embed)
        
        recall_results = self.compute_recall_metrics(sketch_embed, photo_embed, self.train_metrics)
        
        # Log metrics
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/recall@1', recall_results['recall@1'], prog_bar=True)
        self.log('train/recall@5', recall_results['recall@5'], prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        sketch = batch['sketch']
        photo = batch['positive_photo']

        sketch_embed = self.model.encode_sketch(sketch)
        photo_embed = self.model.encode_photo(photo)

        loss = self.loss_fn(sketch_embed, photo_embed)

        recall_results = self.compute_recall_metrics(sketch_embed, photo_embed, self.val_metrics)

        # Log metrics
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/recall@1', recall_results['recall@1'], prog_bar=True)
        self.log('val/recall@5', recall_results['recall@5'], prog_bar=True)
        self.log('val/recall@10', recall_results['recall@10'], prog_bar=True)

        return loss
    
    def on_train_epoch_end(self):
        # Reset metrics at epoch end
        self.train_metrics.reset()
    
    def on_validation_epoch_end(self):
        # Reset metrics at epoch end
        self.val_metrics.reset()

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