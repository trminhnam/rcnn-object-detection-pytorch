from collections import OrderedDict

import torch
import torch.nn as nn


class RCNN(nn.Module):
    def __init__(self, backbone, classes):
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)

        self.classifier = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(0.2)),
            ('classifier_fc', nn.Linear(25088, len(classes)))
        ]))
        
        self.rpn = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(0.2)),
            ('rpn_fc', nn.Linear(25088, 4))
        ]))
        
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for i, c in enumerate(classes)}
        
    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
            features = features.flatten(start_dim=1)
            
        class_id = self.classifier(features)
        box = self.rpn(features)
        
        return class_id, box
    
    def save(self, path):
        # save classifier and rpn only
        save_dict = {
            'classifier': self.classifier.state_dict(),
            'rpn': self.rpn.state_dict(),
            'classes': self.classes,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class
        }
        torch.save(save_dict, path)

    def load(self, path):
        # load classifier and rpn only
        checkpoint = torch.load(path)
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.rpn.load_state_dict(checkpoint['rpn'])
        self.classes = checkpoint['classes']
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = checkpoint['idx_to_class']