import os
import sys
import argparse
import json
from mrcnn import utils
import mrcnn.model as modellib
from utils.dataset import BloodCellDataset
from utils.configs import BloodCellConfig
import keras
import tensorflow as tf
assert(tf.__version__ == "1.15.2")
assert(keras.__version__ == "2.2.4")

# Define working directory
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "eleven_images")

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

IMAGE_COLLECTION_PATH = r'./image_collection_bis.json'


def train(steps: int = 400,
          lr: float = 0.001,
          init_with: str = "coco",
          model_branch: str = None,
          checkpoint_epoch: int = None,
          head_epochs: int = 10,
          all_epochs: int = 100
          ) -> int:
    if not os.path.exists(IMAGE_COLLECTION_PATH):
        raise Exception(f"IC_PATH @ {IMAGE_COLLECTION_PATH} not found.")
    else:
        with open(IMAGE_COLLECTION_PATH, 'r') as outfile:
            dataset = json.load(outfile)
        train_sample_id = dataset["train"]
        val_sample_id = dataset["val"]

    print(f"Number of samples in training set: {len(train_sample_id)}")
    print(f"Number of samples in valid set: {len(val_sample_id)}")

    dataset_train = BloodCellDataset(IMAGE_DIR, train_sample_id)
    dataset_train.load_kernel()
    dataset_val = BloodCellDataset(IMAGE_DIR, val_sample_id)
    dataset_val.load_kernel()

    config = BloodCellConfig()
    config.STEPS_PER_EPOCH = steps
    config.LEARNING_RATE = lr
    config.NAME = "BloodCell"
    config.display()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    if init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        model_path = os.path.join(MODEL_DIR, model_branch,
                                  f"mask_rcnn_BloodCell_{checkpoint_epoch:04d}.h5")
        # Load the last model you trained and continue training
        model.load_weights(model_path, by_name=True)

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=head_epochs,
                layers='heads')

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=all_epochs,
                layers="all")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a Mask R-CNN')
    parser.add_argument('--steps', type=int, default=400,
                        help='Number of training steps per epoch.')
    parser.add_argument('--head_epochs', type=int, default=2,
                        help='Number of epochs to retrain the FC layers.')
    parser.add_argument('--all_epochs', type=int, default=0,
                        help='Number of epochs to retrain all layers.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate to train all layers.')
    parser.add_argument('--init', type=str, default="coco",
                        help='coco or last.')
    parser.add_argument('--branch', type=str,
                        help='model directory name in logs folder.')
    parser.add_argument('--cp', type=int,
                        help='Number of checkpoint of the trained model.')
    args = parser.parse_args()

    train(steps=args.steps,
          lr=args.lr,
          init_with=args.init,
          model_branch=args.branch,
          checkpoint_epoch=args.cp,
          head_epochs=args.head_epochs,
          all_epochs=args.all_epochs
          )
