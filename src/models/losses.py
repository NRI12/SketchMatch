import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPStyleInfoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, sketch_embeddings, photo_embeddings):
        sketch_embeddings = F.normalize(sketch_embeddings, dim=1)
        photo_embeddings = F.normalize(photo_embeddings, dim=1)
        
        # Tính similarity matrix [batch, batch]
        logits = torch.matmul(sketch_embeddings, photo_embeddings.T) / self.temperature
        
        # Ground truth: đường chéo (i-th sketch → i-th photo)
        batch_size = sketch_embeddings.size(0)
        labels = torch.arange(batch_size, device=sketch_embeddings.device)
        
        loss_s2p = F.cross_entropy(logits, labels)  # sketch → photo
        loss_p2s = F.cross_entropy(logits.T, labels)  # photo → sketch
        
        return (loss_s2p + loss_p2s) / 2