import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from torchvision import transforms

class MaizeLeafDataset(Dataset):
    def __init__(self, csv_file, feature_extractor, transform=None, train=True):
        self.data = pd.read_csv(csv_file)
        self.feature_extractor = feature_extractor
        self.train = train
        
        # Define augmentations for training
        # In dataset.py, enhance augmentations
        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(30),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=30,
                translate=(0.2, 0.2),
                scale=(0.8, 1.2)
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
        ])
        
        # Get unique classes and create class mapping
        self.classes = sorted(self.data['Class'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Verify all images exist
        self._verify_images()
        
    def _verify_images(self):
        valid_indices = []
        for idx, row in self.data.iterrows():
            if os.path.exists(row['image_path']):
                try:
                    with Image.open(row['image_path']) as img:
                        valid_indices.append(idx)
                except:
                    print(f"Warning: Couldn't open image {row['image_path']}")
            else:
                print(f"Warning: Image not found {row['image_path']}")
        
        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        label = self.data.iloc[idx]['Class_ID']
        
        # Load and convert image
        image = Image.open(img_path).convert('RGB')
        
        # Apply augmentation if training
        if self.train:
            image = self.train_transforms(image)
            
        # Apply feature extraction
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        inputs['labels'] = torch.tensor(label)
        
        return inputs
    
    def get_class_names(self):
        return self.classes
    
    def get_image_info(self, idx):
        """Return additional information about an image"""
        return {
            'path': self.data.iloc[idx]['image_path'],
            'class': self.data.iloc[idx]['Class'],
            'size': self.data.iloc[idx]['size'],
            'format': self.data.iloc[idx]['format']
        }