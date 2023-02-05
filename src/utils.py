import cv2
import numpy as np
import random

def calculate_IoU(bbox1, bbox2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        bbox1 (tuple): (x_min, y_min, x_max, y_max) of the first bounding box, (x_min, y_min) is the top left corner of the bounding box, (x_max, y_max) is the bottom right corner of the bounding box (x_min <= x_max, y_min <= y_max)
        bbox2 (tuple): (x_min, y_min, x_max, y_max) of the second bounding box (the same format as bbox1)

    Returns:
        float: IoU of the two bounding boxes
    """
    
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    x_tl = max(x1_min, x2_min)
    y_tl = max(y1_min, y2_min)
    x_br = min(x1_max, x2_max)
    y_br = min(y1_max, y2_max)
    
    if x_br < x_tl or y_br < y_tl:
        return 0.0

    intersection = max(0, x_br - x_tl) * max(0, y_br - y_tl)

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    return intersection / float(union)

def selective_search(image, mode="fast"):
    """Perform selective search on the image
Args:
        image (np.array): image to perform selective search on (h x w x 3)
        mode (str): "fast" or "quality", "fast" is faster but less accurate (default: "fast")

    Returns:
        rects (list): list of rectangles (x, y, w, h) that were selected by selective search
    """
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    if mode == "quality":
        ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    return rects

def region_proposal(img, mode='fast'):
    """Perform region proposal on the image

    Args:
        img (np.array or str): image to perform region proposal on
        mode (str): 'fast' or 'quality' (default: 'fast')

    Returns:
        img (np.array): image that was passed in
        regions (list): list of regions (np.array) that were selected by selective search
        bboxes (list): list of bounding boxes in the form (x, y, w, h)
    """
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    
    rects = selective_search(img, mode)
    
    regions = []
    bboxes = []
    for x, y, w, h in rects:
        # skip small regions
        if h < height * 0.1 or w < width * 0.1:
            continue
        
        region = img[y:y+h, x:x+w]
        region = cv2.resize(region, (224, 224), interpolation=cv2.INTER_CUBIC)
        regions.append(region)
        bboxes.append((x, y, w, h))
    
    return np.array(img), np.array(regions), bboxes

def non_max_suppression(boxes, probs=None, overlapThresh=0.2):
    """Perform non-maximum suppression on the bounding boxes. 
        Reference: https://github.com/PyImageSearch/imutils/blob/master/imutils/object_detection.py

    Args:
        boxes (np.array): bounding boxes in the form (x1, y1, x2, y2)
        probs (np.array, optional): probabilities associated with each bounding box. Defaults to None.
        overlapThresh (float, optional): threshold to determine when boxes overlap too much. Defaults to 0.3.
        
    Returns:
        list: a list of bounding boxes that were selected by non-maximum suppression
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    
    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
        
    # initialize the list of picked indexes
    pick = []
    
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2
    
    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs
        
    # sort the indexes
    idxs = np.argsort(idxs)
    
    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        
        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
        
    # return only the bounding boxes that were picked
    return boxes[pick].astype("int"), probs[pick].astype("float")

def plot_one_box(x, image, color=None, label=None, line_thickness=None):
    """Plots one bounding box on image img in YOLO style

    Args:
        x (list): bounding box in the form (x1, y1, x2, y2)
        image (np.array): image to plot on
        color (list, optional): color of the bounding box. Defaults to None.
        label (str, optional): label of the bounding box. Defaults to None.
        line_thickness (int, optional): thickness of the bounding box. Defaults to None.
    """
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    text_color = (0, 0, 0) if np.mean(color) > 150 else (255, 255, 255)
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3, text_color, thickness=tf, lineType=cv2.LINE_AA)