import random
import os
from tqdm.auto import tqdm
import pandas as pd
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from utils import parse_annotation, calculate_IoU, region_proposal

def parse_annotation(xml_path, classes):
    """Parse the xml file and return the image and bounding boxes

    Args:
        xml_path (str): path to the xml file

    Returns:
        bboxes (list): list of bounding boxes in the form (x_min, y_min, x_max, y_max, class_id)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    bboxes = []
    for member in root.findall('object'):
        class_name = member.find('name').text
        if class_name not in classes:
            continue
        class_id = classes.index(class_name)
        bndbox = member.find('bndbox')
        x_min = int(bndbox.find('xmin').text)
        y_min = int(bndbox.find('ymin').text)
        x_max = int(bndbox.find('xmax').text)
        y_max = int(bndbox.find('ymax').text)
        bboxes.append([x_min, y_min, x_max, y_max, class_id])
    
    return bboxes

def region_extraction(annotation_dir, image_dir, region_dir, label_path='dataset/regions.csv', max_objects=5):
    df = pd.DataFrame(columns=['image_name', 'class_id', 'x_est', 'y_est', 'w_est', 'h_est', 'x_gt', 'y_gt', 'w_gt', 'h_gt'])

    if not os.path.exists(region_dir):
        os.makedirs(region_dir)

    dirs = os.listdir(annotation_dir)
    pbar = tqdm(dirs, desc="Parsing annotations", total=len(dirs), leave=True, position=0)

    for annotation_file in pbar:
        try:
            train_images = []
            train_labels = []
            
            gtvalues = parse_annotation(os.path.join(annotation_dir, annotation_file))
            
            image_name = annotation_file.split(".")[0]+".jpg"
            img, regions, bboxes = region_proposal(os.path.join(image_dir, image_name))
            
            obj_counter = 0
            bg_counter = 0
            is_object = False
            
            for i, (x, y, w, h) in enumerate(bboxes[:2000]):
                region = regions[i]
                est_bbox_xywh = (x, y, w, h)
                
                for i, gtval in enumerate(gtvalues):
                    x_gt_min, y_gt_min, x_gt_max, y_gt_max, class_id = gtval
                    x_gt, y_gt, w_gt, h_gt = x_gt_min, y_gt_min, x_gt_max-x_gt_min, y_gt_max-y_gt_min
                    gt_bbox_xywh = (x_gt, y_gt, w_gt, h_gt)
                    
                    iou = calculate_IoU((x_gt_min, y_gt_min, x_gt_max, y_gt_max), (x, y, x+w, y+h))
                    is_object = True if iou > 0.7 else False
                    
                    if is_object and obj_counter < max_objects:
                        
                        train_images.append(region)
                        train_labels.append((class_id, est_bbox_xywh, gt_bbox_xywh))
                        obj_counter += 1
                        
                if is_object == False and (bg_counter < 0.3 * max_objects or bg_counter == 0):
                    train_images.append(region)
                    train_labels.append((0, est_bbox_xywh, gt_bbox_xywh))
                    bg_counter += 1
                    
                if obj_counter >= max_objects and bg_counter >= 0.3 * max_objects:
                    break
            
            for j, (img, label) in enumerate(zip(train_images, train_labels)):
                class_id, est_bbox_xywh, gt_bbox_xywh = label
                x_est, y_est, w_est, h_est = est_bbox_xywh
                x_gt, y_gt, w_gt, h_gt = gt_bbox_xywh
                
                img_name = f'{image_name.split(".")[0]}_{j}.jpg'
                cv2.imwrite(os.path.join(region_dir, img_name), img)
                tmp = {
                    'image_name': img_name, 
                    'class_id': class_id, 
                    'x_est': x_est, 'y_est': y_est, 'w_est': w_est, 'h_est': h_est, 
                    'x_gt': x_gt, 'y_gt': y_gt, 'w_gt': w_gt, 'h_gt': h_gt
                    }
                df.loc[len(df)] = tmp
                
        except Exception as e:
            print(e)
            print(f'Error in {annotation_file}')
            # continue
            continue

    df.to_csv(label_path, index=False)