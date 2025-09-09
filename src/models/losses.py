import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, sketch_embed, positive_embed, negative_embed):
        batch_size = sketch_embed.size(0)
        
        positive_sim = F.cosine_similarity(sketch_embed, positive_embed, dim=1) / self.temperature
        negative_sim = F.cosine_similarity(sketch_embed, negative_embed, dim=1) / self.temperature
        
        logits = torch.stack([positive_sim, negative_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=sketch_embed.device)
        
        return F.cross_entropy(logits, labels)

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        
    def forward(self, sketch_embed, positive_embed, negative_embed):
        return self.triplet_loss(sketch_embed, positive_embed, negative_embed)

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