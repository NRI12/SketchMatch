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