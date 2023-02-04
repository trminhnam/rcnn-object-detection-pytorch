import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm

train_classifier_config = {
    'epochs': 1,
    'learning_rate': 1e-3,
    'bbox_reg': False,
    'weight_decay': 0.0005
}

fine_tuning_config = {
    'epochs': 1,
    'learning_rate': 1e-3,
    'bbox_reg': True,
}

def training(model, train_loader, train_config, device):
    clf_criterion = nn.CrossEntropyLoss()
    box_criterion = nn.MSELoss()
    
    # Optimizer for classification and regression
    optimizer = optim.Adam(
        model.classifier.parameters() if not train_config.get('bbox_reg', False) else model.parameters(),
        lr=train_config.get('learning_rate', 0.001),
        weight_decay=train_config.get('weight_decay', 0.0005)
    )
    
    for epoch in range(train_config.get('epochs', 1)):
        model.train()  # set model to train mode
        print(f'Epoch {epoch + 1}/{train_config["epochs"]}')
        clf_losses = []
        box_losses = []
        pbar = tqdm(train_loader, position=0, leave=True, total=len(train_loader))
        
        for step, data in enumerate(pbar):
            data = {k: v.to(device) for k, v in data.items()}
            labels = data['class_id'].squeeze().long()
            
            optimizer.zero_grad()
            preds, bbox = model(data['image'])
            
            clf_loss = clf_criterion(preds, labels)
            loss = clf_loss
            clf_losses.append(clf_loss.item())
            
            # bbox reg
            if train_config.get('bbox_reg', False):
                p_x, p_y, p_w, p_h = data['est_bbox'].to(device).split(1, dim=1)
                g_x, g_y, g_w, g_h = data['gt_bbox'].to(device).split(1, dim=1)
                
                bbox_ans = torch.cat([(g_x - p_x) / p_w, (g_y - p_y) / p_h, torch.log(g_w / p_w), torch.log(g_h / p_h)], dim=1)
                bbox_ans = bbox_ans.float().to(device)
                
                mask = (data['class_id'] != 0).reshape(len(data['class_id']), 1).float().to(device)
                bbox = bbox * mask
                bbox_ans = bbox_ans * mask
                
                bbox_loss = box_criterion(bbox, bbox_ans)
                loss += bbox_loss
                box_losses.append(bbox_loss.item())
                pbar.set_description(f"Cls Loss: {clf_loss.item():.4f} | Bbox Loss: {bbox_loss.item():.4f}")
            else:
                pbar.set_description(f"Cls Loss: {clf_loss.item():.4f}")
            pbar.update()
            
            loss.backward()
            optimizer.step()
        
        print(f'Avg Cls Loss: {np.mean(clf_losses):.4f}')
        if train_config.get('bbox_reg', False):
            print(f'Avg Bbox Loss: {np.mean(box_losses):.4f}')
        print('#' * 50, end='\n\n')