import os

import cv2
import pandas as pd
import torch
import torchvision.transforms as transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_path, transform=None):
        self.image_dir = image_dir
        self.data = pd.read_csv(label_path)
        
        if transform is None:
            transform = transforms.Compose([
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                transforms.Resize((224, 224))
            ])
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        
        img_path = os.path.join(self.image_dir, data[0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.FloatTensor(img).permute(2, 0, 1) / 255
        
        class_id = data[1]
        
        x_est, y_est, w_est, h_est = data[2:6]
        x_gt, y_gt, w_gt, h_gt = data[6:]
        
        if self.transform:
            img = self.transform(img)
            
        return {
            'image': img,
            'class_id': torch.LongTensor([class_id]),
            'est_bbox': torch.FloatTensor([x_est, y_est, w_est, h_est]),
            'gt_bbox': torch.FloatTensor([x_gt, y_gt, w_gt, h_gt])
        }