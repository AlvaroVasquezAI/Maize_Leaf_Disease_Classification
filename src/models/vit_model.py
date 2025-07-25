import torch
import torch.nn as nn
from transformers import ViTForImageClassification

class MaizeViTModel(nn.Module):
    def __init__(self, num_classes=6, pretrained_model='google/vit-base-patch16-224', 
                 hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1):
        super(MaizeViTModel, self).__init__()
        
        self.vit = ViTForImageClassification.from_pretrained(
            pretrained_model,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )
        
        self.vit.config.use_cache = False
        
    def load_from_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.vit.load_state_dict(checkpoint['state_dict'])
        else:
            self.vit.load_state_dict(checkpoint)
        return self
        
    def forward(self, **inputs):
        return self.vit(**inputs)
    
    def freeze_backbone(self):
        """Freeze all parameters except classifier"""
        for param in self.vit.vit.parameters():
            param.requires_grad = False
        
    def unfreeze_backbone(self):
        """Unfreeze all parameters"""
        for param in self.vit.parameters():
            param.requires_grad = True

    def gradual_unfreeze(self, epoch):
        """Gradually unfreeze layers"""
        if epoch == 0:
            # Freeze all except classifier
            self.freeze_backbone()
        elif epoch == 2:
            # Unfreeze last few layers
            for param in self.vit.vit.encoder.layer[-2:].parameters():
                param.requires_grad = True
        elif epoch == 4:
            # Unfreeze all layers
            self.unfreeze_backbone()