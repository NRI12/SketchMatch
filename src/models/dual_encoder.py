import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return F.normalize(self.net(x), dim=1)
class DualEncoder(nn.Module):
    def __init__(
        self,#
        sketch_backbone="resnet50",
        photo_backbone="resnet50", 
        embedding_dim=128,
        pretrained=True
    ):
        super().__init__()
        
        self.sketch_encoder = timm.create_model(
            sketch_backbone,
            pretrained=pretrained, 
            num_classes=0
        )
        
        self.photo_encoder = timm.create_model(
            photo_backbone,
            pretrained=pretrained,
            num_classes=0
        )
        
        feat_dim = self.sketch_encoder.num_features
        self.sketch_projector = ProjectionHead(feat_dim, 512, embedding_dim)
        self.photo_projector = ProjectionHead(feat_dim, 512, embedding_dim)

    def encode_sketch(self, sketch):
        features = self.sketch_encoder(sketch)
        return self.sketch_projector(features)
    def encode_photo(self, photo):
        features = self.photo_encoder(photo)
        return self.photo_projector(features)
    def forward(self, sketch, photo):
        sketch_embed = self.encode_sketch(sketch)
        photo_embed = self.encode_photo(photo)
        return sketch_embed, photo_embed
