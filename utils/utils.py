import pandas as pd
import os
import json
import colorsys
import random
import numpy as np
import cv2
import imageio
from matplotlib import patches
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from datetime import datetime
import zlib
import base64
from mrcnn.model import MaskRCNN
from mrcnn.utils import resize_image
import xml.etree.ElementTree as ET


# ===============================
# INLINES
# ===============================

def get_image_local_path(sample_id: str, directory_path: str) -> str:
    return os.path.join(directory_path, "JPEGImages", sample_id + '.jpg')

def get_image_local_path_B(sample_id: str, directory_path: str) -> str:
    return os.path.join(directory_path, "Dataset_2", sample_id + '.bmp')

def get_mask_local_path_B(sample_id: str, directory_path: str) -> str:
    return os.path.join(directory_path, "Dataset_2", sample_id + '.png')

def get_xml_local_path(sample_id: str, directory_path: str) -> str:
    return os.path.join(directory_path, "Annotations", sample_id + '.xml')


# ===============================
# INFERENCE MODE
# ===============================

def filter_resuLt_by_score(result: dict, thres: float = 0.9) -> dict:
    """Filter the inference results by the given confidence."""
    filters = (result['scores'] >= thres)
    fitered_result = {}
    fitered_result['rois'] = result['rois'][filters]
    fitered_result['masks'] = []
    for i in range(len(filters)):
        if filters[i]:
            fitered_result['masks'].append(result['masks'][..., i])
    fitered_result['masks'] = np.stack(fitered_result['masks'], axis=-1)
    fitered_result['class_ids'] = result['class_ids'][filters]
    fitered_result['scores'] = result['scores'][filters]
    return fitered_result


# ===============================
# EXPOLORE DATA
# ===============================

def visualize_sample(sample_id: str, directory_path: str) -> None:
    """Render the image and annotations for a given sample id. """
    image = read_image(get_image_local_path(sample_id, directory_path))
    annotations, image_shape = load_xml(get_xml_local_path(sample_id, directory_path))
    assert image.shape == image_shape
    masks, class_names = convert_annotations(annotations, image_shape)
    _, ax = plt.subplots(1, figsize=(20, 20))
    colors = random_colors(len(class_names))
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(f"Sample_id: {sample_id}, shape {image.shape}")
    print(image.shape, masks.shape)
    masked_image = image.astype(np.uint32).copy()

    for i, class_name in enumerate(class_names):
        mask = masks[..., i]
        ret, thresh = cv2.threshold(mask.astype(np.uint8), 1, 255, 0)
        cnt, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = np.squeeze(cnt[0], axis=1)
        p = Polygon(cnt,
                    facecolor="none", edgecolor=colors[i], alpha=0.7, linestyle="dashed", 
                    linewidth=5, fill=False)
        ax.add_patch(p)

        # Label
        ax.text(cnt[0,0] + 16, cnt[0,1] + 16, class_name,
                color=colors[i], size=20, backgroundcolor="none")

    ax.imshow(masked_image.astype(np.uint8))
    plt.show()
    return

def visualize_sample_B(df: pd.DataFrame, sample_id: str, IMAGE_DIR: str) -> None:
    WBC_types = df[df["sample_id"] == sample_id].WBC_types.iloc[0]
    print(sample_id, WBC_types)
    image = read_image(get_image_local_path_B(sample_id, IMAGE_DIR))
    mask = read_image(get_mask_local_path_B(sample_id, IMAGE_DIR))
    masks = mask[..., 0:1]
    class_names = WBC_types
    _, ax = plt.subplots(1, figsize=(20, 20))
    colors = random_colors(len(class_names))
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(f"Sample_id: {sample_id}, shape {image.shape}, WBC type {WBC_types}")
    print(image.shape, masks.shape)
    masked_image = image.astype(np.uint32).copy()
    
    for i, class_name in enumerate(class_names):
        mask = masks[..., i]
        ret, thresh = cv2.threshold(mask.astype(np.uint8), 1, 255, 0)
        cnt, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = np.squeeze(cnt[0], axis=1)
        p = Polygon(cnt,
                    facecolor="none", edgecolor=colors[i], alpha=0.7, linestyle="dashed", 
                    linewidth=5, fill=False)
        ax.add_patch(p)

        # Label
        ax.text(cnt[0,0] + 16, cnt[0,1] + 16, class_name,
                color=colors[i], size=20, backgroundcolor="none")

    ax.imshow(masked_image.astype(np.uint8))
    plt.show()

# ===============================
# EVALUATION FUNCTIONS
# ===============================


def get_precision_recall(df: pd.DataFrame) -> (list, list):
    """Calculate P/R pairs for a range of confidence_levels"""
    precisions = [0]
    recalls = [1]
    for confidence_level in np.arange(0, 1.0, 0.01):
        tp = len(df[(df.pred_class == df.gt_class) &
                 (df.pred_score >= confidence_level)])
        fn = len(df[((df.pred_class < 0) |
                 (df.pred_score < confidence_level)) &
                 (df.gt_class > -1)])
        fp = len(df[((df.pred_class > -1) &
                 (df.pred_score >= confidence_level)) &
                 (df.gt_class < 0)])
        precisions.append(tp/(tp+fp+1e-5))
        recalls.append(tp/(tp+fn+1e-5))
    precisions.append(1)
    recalls.append(0)
    return precisions, recalls


def calculate_ap_from_pr(precisions: list, recalls: list) -> float:
    """Calculate Average Percision from P-R pairs"""
    AP = 0
    for i in range(len(precisions) - 1):
        AP += (recalls[i] - recalls[i+1]) * (precisions[i] + precisions[i+1]) / 2
    return AP


def get_object_matches(result: dict,
                       pred_matches: list,
                       gt_matches: list,
                       gt_class_id: list,
                       image_id: int,
                       IoUs: list) -> list:
    """Sort and merge the matched objects between gt and pred."""
    IoUs = np.max(IoUs, axis=1)
    assert(len(IoUs) == len(pred_matches))
    object_matches = []
    for pred_score, pred_match, pred_class_id, IoU in zip(result["scores"], pred_matches, result["class_ids"], IoUs):
        pred_class = pred_class_id
        gt_class = gt_class_id[int(pred_match)] if pred_match > -1 else -1

        object_matches.append({"image_id": image_id,
                               "pred_class": int(pred_class),
                               "gt_class": int(gt_class),
                               "pred_score": pred_score,
                               "highest_mIoU": IoU})

    for gt_match, gt_class in zip(gt_matches, gt_class_id):
        if gt_match == -1:
            object_matches.append({"image_id": image_id,
                                   "pred_class": -1,
                                   "gt_class": int(gt_class),
                                   "pred_score": 0,
                                   "highest_mIoU": 0})
    return object_matches


# ===============================
# MISCELLANEOUS FUNCTIONS
# ===============================
def correct_class_label(label, WBC_type):
    if label == "WBC":
        if len(WBC_type) == 1:
            return WBC_type[0]
    return label

def read_imageio(image_path: str):
    """Load image by imageio library."""
    image = imageio.imread(image_path)
    return image


def read_image(image_path: str):
    """Load image by opencv library."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def random_colors(N: int, bright=True) -> list:
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def draw_rectangle(mask, bbox):
    """Helper function for rendering."""
    y1, x1, y2, x2 = bbox[1], bbox[0], bbox[3], bbox[2]
    contour = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
    contour = np.array(contour, dtype=np.int32)
    cv2.drawContours(mask, [contour], -1, (255), -1)
    return mask

def draw_rectangle2(mask, bbox):
    """Helper function for rendering."""
    x1, y1, x2, y2 = bbox[1], bbox[0], bbox[3], bbox[2]
    contour = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
    contour = np.array(contour, dtype=np.int32)
    cv2.drawContours(mask, [contour], -1, (255), -1)
    return mask


def draw_polygon(mask, contour):
    """Helper function for rendering."""
    contour = np.array(contour, dtype=np.int32)
    cv2.drawContours(mask, [contour], -1, (255), -1)
    return mask

def load_xml(annotation_path):
    tree = ET.parse(annotation_path)
    ret = []
    height, width, depth = None, None, None
    for elem in tree.iter():
        if 'filename' in elem.tag:
            # double check
            # assert elem.text in annotation_path
            # No double checking due to transiting to new dataset format
            pass
        if 'size' in elem.tag:
            for attr in list(elem):
                if 'width' in attr.tag:
                    width = int(attr.text)
                elif 'height' in attr.tag:
                    height = int(attr.text)
                elif 'depth' in attr.tag:
                    depth = int(attr.text)
        if 'object' in elem.tag or 'part' in elem.tag:
            for attr in list(elem):
                if 'name' in attr.tag:
                    name = attr.text
                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            x1 = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            y1 = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            x2 = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            y2 = int(round(float(dim.text)))
                    ret.append({
                        "class_label": name,
                        "bbox": [x1, y1, x2, y2]
                    })
    return ret, (height, width, depth)


def convert_annotations(annotations, image_shape):
    """Convert masks from polygon to binary mask for training."""
    masks = []
    class_names = []
    for obj in annotations:
        class_name = obj["class_label"]
        mask = np.zeros(image_shape[:-1], dtype=np.int8)
        mask = draw_rectangle(mask, obj["bbox"])
        masks.append(mask)
        class_names.append(class_name)
    masks = np.stack(masks, axis=2)
    return masks.astype(np.uint8), class_names

def convert_bbox_to_masks(bbox, mask_shape):
    masks = []
    for b in bbox:
        mask = np.zeros(mask_shape, dtype=np.int8)
        mask = draw_rectangle2(mask, b)
        masks.append(mask)
    return np.stack(masks, axis=2).astype(np.uint8)
