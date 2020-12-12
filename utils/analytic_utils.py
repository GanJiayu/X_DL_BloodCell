from credentials.blob_credentials import facts_sas_token, facts_container, workspace_sas_token, workspace_container
from azure.storage.blob import ContainerClient, BlobClient
import pandas as pd
import os
import json
import colorsys
import random
import numpy as np
import cv2
from tqdm import tqdm
import imageio
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime
import zlib
import base64
from utils.utils import *


def compute_objects(sample_id, directory_path):
    with open(get_poly_local_path_2(sample_id, directory_path)) as f:
        annotation_poly = json.load(f)
    objects = []
    for instance in annotation_poly['objects']:
        objects.append(instance['classTitle'])
    return objects


def get_poly_local_path_2(sample_id, directory_path):
    return os.path.join(directory_path, sample_id + '_poly.json')


def visualize_sample_analytic_2(sample_id, directory_path):
    """Render the image and annotations for a given sample id in analytic set."""
    image = cv2.imread(get_image_local_path(sample_id, directory_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with open(get_people_local_path(sample_id, directory_path)) as f:
        annotation_people = json.load(f)
    with open(get_poly_local_path_2(sample_id, directory_path)) as f:
        annotation_poly = json.load(f)
    _, ax = plt.subplots(1, figsize=(20, 20))
    print(f"{len(annotation_people['objects'])} peoples + {len(annotation_poly['objects'])} polygons")
    colors = random_colors(len(annotation_people["objects"]) +
                           len(annotation_poly["objects"]))
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(f"Sample_id: {sample_id}, shape {image.shape}")

    masked_image = image.astype(np.uint32).copy()

    for i, instance in enumerate(annotation_people["objects"]):
        if instance["geometryType"] == "rectangle":
            y1, x1, y2, x2 = instance["points"]["exterior"][0][1], \
                             instance["points"]["exterior"][0][0], \
                             instance["points"]["exterior"][1][1], \
                             instance["points"]["exterior"][1][0]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=colors[i], facecolor='none')
            ax.add_patch(p)

        # Label
        ax.text(x1, y1 + 8, instance["classTitle"],
                color='w', size=11, backgroundcolor="none")

    for j, instance in enumerate(annotation_poly["objects"]):
        if instance["geometryType"] == "bitmap":
            mask = base64_2_mask(instance["bitmap"]["data"])
            x1, y1 = instance["bitmap"]["origin"][0], \
                instance["bitmap"]["origin"][1]
            contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            cnt = np.squeeze(contours[0], axis=1)

            # Depends on whether the mini-mask was adpoted
            # cnt += [x1, y1]

            p = Polygon(cnt, facecolor="none",
                        edgecolor=colors[len(annotation_people["objects"])+j],
                        linewidth=None, fill=True)
            p.set_fill(True)
            ax.add_patch(p)

        # Label
        ax.text(x1, y1 + 8, instance["classTitle"],
                color='w', size=11, backgroundcolor="none")

    ax.imshow(masked_image.astype(np.uint8))
    plt.show()
    return


# Calculate the number of formworks/rebars in each 3 zone
def overlap(sample_id, directory_path, image1, image2, image3):
    with open(get_poly_local_path_2(sample_id, directory_path)) as f:
        annotation_poly = json.load(f)
    dis_formwork = [0, 0, 0]
    dis_rebars = [0, 0, 0]
    assert image1.shape == image2.shape == image3.shape
    blank = np.zeros(image1.shape)
    classes = ["Vertical_formwork", "Horizontal_formwork", "Rebars"]
    for j, instance in enumerate(annotation_poly["objects"]):
        if instance['classTitle'] in classes:
            title = instance['classTitle']
            mask = base64_2_mask(instance["bitmap"]["data"])
            contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            cnt = np.squeeze(contours[0], axis=1)
            imagef = cv2.drawContours(blank.copy(), [cnt], -1, 1, cv2.FILLED)
            farea = np.sum(imagef)
            intersection1 = cv2.bitwise_and(image1, imagef)
            intersection2 = cv2.bitwise_and(image2, imagef)
            intersection3 = cv2.bitwise_and(image3, imagef)
            if title == 'Rebars':
                # Set overlapped threshold 0.9
                if np.sum(intersection1)/farea > 0.9:
                    dis_rebars[0] += 1
                if np.sum(intersection2)/farea > 0.9:
                    dis_rebars[1] += 1
                if np.sum(intersection3)/farea > 0.9:
                    dis_rebars[2] += 1
            else:
                if np.sum(intersection1)/farea > 0.9:
                    dis_formwork[0] += 1
                if np.sum(intersection2)/farea > 0.9:
                    dis_formwork[1] += 1
                if np.sum(intersection3)/farea > 0.9:
                    dis_formwork[2] += 1
    return dis_formwork, dis_rebars


# Detect appearance of pump hose in three areas
# Here we found is more feasiable to only have 2 areas
def detect_pump(sample_id, directory_path):
    with open(get_poly_local_path_2(sample_id, directory_path)) as f:
        annotation_poly = json.load(f)
    dis_pump = [0, 0, 0]
    x = 0
    y = 0
    for j, instance in enumerate(annotation_poly["objects"]):
        if instance['classTitle'] in ["Concrete_pump_hose"]:
            mask = base64_2_mask(instance["bitmap"]["data"])
            contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            try:
                cnt = np.squeeze(contours[0], axis=1)
                # Find the lowest point of pump hose
                for i, row in enumerate(cnt):
                    if row[1] > y:
                        y = row[1]
                        x = row[0]
                    else:
                        pass
                # Set threshold of left part and right part to 740
                if x < 740:
                    dis_pump[0] += 1
                if x < 740:
                    dis_pump[1] += 1
                if x > 740:
                    dis_pump[2] += 1
            except:
                pass
            break
    return dis_pump


# Detect how many workers are located in each area
def detect_people(sample_id, directory_path, area1, area2, area3):
    with open(get_people_local_path(sample_id, directory_path)) as f:
        annotation_people = json.load(f)
    dis_peo = [0, 0, 0]
    for j, instance in enumerate(annotation_people["objects"]):
        people = instance
        # Get central point of the bottom line of the bounding boxs for workers
        yc = max(people["points"]["exterior"][0][1],
                 people["points"]["exterior"][1][1]) / 2
        xc = (people["points"]["exterior"][0][0] +
              people["points"]["exterior"][1][0]) / 2
        if cv2.pointPolygonTest(area1, (xc, yc), False) == 1:
            dis_peo[0] += 1
        if cv2.pointPolygonTest(area2, (xc, yc), False) == 1:
            dis_peo[1] += 1
        if cv2.pointPolygonTest(area3, (xc, yc), False) == 1:
            dis_peo[2] += 1
    return dis_peo


def convert(x):
    return datetime.strptime(x, "%Y-%m-%d-%H-%M-%S")


# Define conditions of current stage
def if_concreting(formwork, pump, people):
    if formwork > 0 and pump > 0 and people > 0:
        result = 1
    else:
        result = 0
    return result