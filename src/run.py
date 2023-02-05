# add current directory to path
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

import torch
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import region_proposal, non_max_suppression, plot_one_box

def refine_bbox(bbox, delta):
    x, y, w, h = bbox
    dx, dy, dw, dh = delta
    
    x = x + dx * w
    y = y + dy * h
    w = w * torch.exp(dw)
    h = h * torch.exp(dh)
    
    return torch.stack([x, y, w, h])

def detect(model, img, transform, classes, device, threshold=0.8, nms_threshold=0.5, save_dir=None):
    model.eval()
    img, regions, bboxes = region_proposal(img)
    clone = img.copy()
    print(f'[INFO] Found {len(regions)} regions')
    
    labels = {}
    
    dataloader = torch.utils.data.DataLoader(np.array(regions, dtype=np.float32), batch_size=32, shuffle=False)
    pbar = tqdm(dataloader, total=len(dataloader))
    for batch_idx, batch in enumerate(pbar):
        regions = transform(torch.FloatTensor(batch).permute(0, 3, 1, 2) / 255).to(device)
        
        with torch.no_grad():
            score, bbox = model(regions)
        score = score.softmax(axis=1).cpu().detach().numpy()
        bbox = bbox.cpu().detach()
        
        pred = np.argmax(score, axis=1)
        # print(f'pred: {pred}')
        # print(f'score:', end=' ')
        # for i in range(len(pred)):
        #     print(f'{score[i][pred[i]]:.2f}', end=' ')
        # print(f'pred.shape: {pred.shape}')
        # print(f'score.shape: {score.shape}')
        # print(f'bbox.shape: {bbox.shape}')
        # print(score)
        
        pred = pred.tolist()
        for j in range(len(pred)):
            class_idx = pred[j]
            if pred[j] != 0 and score[j][pred[j]] > threshold:
                print(f'[INFO] Detected {classes[class_idx]} with confidence {score[j][class_idx]:.2f}')
                region_bbox = bboxes[batch_idx * 32 + j]
                delta = bbox[j]
                refined_bbox = refine_bbox(region_bbox, delta)
                
                clone = img.copy()
                x, y, w, h = refined_bbox.numpy()
                x, y, w, h = np.clip(x, 0, img.shape[1]), np.clip(y, 0, img.shape[0]), np.clip(w, 0, img.shape[1]), np.clip(h, 0, img.shape[0])
                x, y, w, h = int(x), int(y), int(w), int(h)
                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.putText(img, classes[pred[i]], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # plt.imshow(img)
                # plt.show()
                # x, y, w, h = region_bbox
                box_xyxy = [x, y, x + w, y + h]
                L = labels.get(classes[class_idx], [])
                L.append((box_xyxy, score[j][class_idx]))
                labels[classes[class_idx]] = L

        # non maximum suppression
        for class_name in labels:
            boxes = np.array([box for box, _ in labels[class_name]])
            scores = np.array([score for _, score in labels[class_name]])
            boxes, scores = non_max_suppression(boxes, scores, overlapThresh=nms_threshold)
            # labels[class_name] = list(zip(boxes, scores))
            for (x_min, y_min, x_max, y_max), score in zip(boxes, scores):
                plot_one_box((x_min, y_min, x_max, y_max), clone, label=f"{class_name} {score:.2f}")

    # plot_one_box((x, y, x + w, y + h), clone, label=f'{classes[pred[j]]} {score[j][pred[j]]:.2f}')
    if save_dir is not None:
        cv2.imwrite(os.path.join(save_dir, 'result.jpg'), clone)
    plt.imshow(img)
    plt.show()
    plt.imshow(clone)
    plt.show()
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model.pth')
    parser.add_argument('--img', type=str, default='data/test.jpg')
    parser.add_argument('--classes', type=str, default='data/classes.txt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='data')
    args = parser.parse_args()
    
    # load model
    model = torch.load(args.model)
    model = model.to(args.device)
    
    # load image
    img = cv2.imread(args.img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # load classes
    with open(args.classes, 'r') as f:
        classes = f.read().splitlines()
    
    # transform
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    detect(model, img, transform, classes, args.device, save_dir=args.save_dir)