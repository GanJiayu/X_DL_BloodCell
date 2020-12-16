from mrcnn import utils as mrcnn_utils
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
from utils.utils import get_image_local_path, get_image_local_path_B, \
        read_image, convert_annotations, load_xml, get_xml_local_path, \
        correct_class_label, get_mask_local_path_B


class BloodCellDataset(mrcnn_utils.Dataset):
    """Derived class of mrcnn Dataset to handle blood cell images."""

    def __init__(self, directory_path, sample_ids, verbose=False):
        super().__init__()
        self.DIR_PATH = directory_path
        self.sample_ids = sample_ids
        self.verbose = verbose
        self.source = "BloodCell"
        self.class_names_preset = np.array(['BG',
                                            'RBC',
                                            'WBC',
                                            'Platelets'])

    def _get_id_of_class(self, clabel):
        if clabel not in self.class_names_preset:
            raise Exception(f'Class {clabel} not present in dataset classes: {self.class_names_preset}')
        return np.where(self.class_names_preset == clabel)[0][0]

    def load_image(self, image_id, zero_image=False):
        """ Load the images during training. """
        info = self.image_info[image_id]
        image_id = info['id']
        image_path = get_image_local_path(image_id, self.DIR_PATH)
        image = read_image(image_path)
        return image

    def load_mask(self, image_id):
        """ Load the masks during training. """
        info = self.image_info[image_id]
        image_id = info['id'] 
        mask_path = get_xml_local_path(image_id, self.DIR_PATH)
        annotations, image_shape = load_xml(mask_path)
        masks, class_labels = convert_annotations(annotations, image_shape)
        class_ids = np.array([self._get_id_of_class(c) for c in class_labels], dtype=np.int32)
        return masks, class_ids

    def load_kernel(self) -> None:
        """ Load the metadate before training. """
        self.classes = set()
        for sample_id in tqdm(self.sample_ids):
            try:
                image_path = get_image_local_path(sample_id, self.DIR_PATH)
                image = read_image(image_path)
                # print(f"image shape {image.shape}")
                annotation_path = get_xml_local_path(sample_id, self.DIR_PATH)
                annotations, image_shape = load_xml(annotation_path)
                _, class_labels = convert_annotations(annotations, image_shape)
                if (len(class_labels) == 0):
                    print(f'Skip sample {sample_id}, no gt_objects')
                    continue
                for class_label in class_labels:
                    self.classes.update({class_label})
                self.add_image(source=self.source,
                               image_id=sample_id,
                               sub_class_label="BloodCell",
                               path=image_path,
                               image_size=image.shape,
                            )
            except Exception as e:
                print(f'EXCP: {e} during handling {sample_id}')
                continue
        for i, c in enumerate(list(self.class_names_preset[1:])):
            self.add_class(self.source, i + 1, c)
        self.prepare()
        return

class BloodCellDatasetFromDataFrame(mrcnn_utils.Dataset):
    """Derived class of mrcnn Dataset to handle blood cell images. load from a given dataframe"""

    def __init__(self, df, directory_path_A=None,  directory_path_B=None, verbose=False):
        super().__init__()
        self.DIR_PATH_A = directory_path_A
        self.DIR_PATH_B = directory_path_B
        self.df = df
        self.verbose = verbose
        self.source = "Blood_cell"
        self.origin = ["A", "B"]
        self.class_names_preset = np.array(['BG',
                                            'RBC',
                                            'WBC', # deprecated
                                            'Platelets',
                                            "neutrophil".upper(),
                                            "lymphocyte".upper(),
                                            "monocyte".upper(),
                                            "eosinophil".upper(),
                                            "basophil".upper()
                                            ])

    def _get_id_of_class(self, clabel):
        if clabel not in self.class_names_preset:
            raise Exception(f'Class {clabel} not present in dataset classes: {self.class_names_preset}')
        return np.where(self.class_names_preset == clabel)[0][0]

    def load_image(self, image_id, zero_image=False):
        """ Load the images during training. """
        info = self.image_info[image_id]
        image_id = info['id']
        origin = info['origin']
        if origin == "A":
            image_path = get_image_local_path(image_id, self.DIR_PATH_A)
        elif origin == "B":
            image_path = get_image_local_path_B(image_id, self.DIR_PATH_B)
        image = read_image(image_path)
        return image

    def load_mask(self, image_id):
        """ Load the masks during training. """
        info = self.image_info[image_id]
        image_id = info['id'] 
        origin = info['origin']
        WBC_type = info['sub_class_label']
        if origin == "A":
            mask_path = get_xml_local_path(image_id, self.DIR_PATH_A)
            annotations, image_shape = load_xml(mask_path)
            masks, class_labels = convert_annotations(annotations, image_shape)
            class_labels = [correct_class_label(l, WBC_type) for l in class_labels]
        elif origin == "B":
            mask_path = get_mask_local_path_B(image_id, self.DIR_PATH_B)
            mask = read_image(mask_path)
            masks = mask[..., 0:1]
            class_labels = info["sub_class_label"]
        
        class_ids = np.array([self._get_id_of_class(c) for c in class_labels], dtype=np.int32)
        return masks, class_ids

    def load_kernel(self) -> None:
        """ Load the metadate before training. """
        self.classes = set()
        for i, row in self.df.iterrows():
            try:
                sample_id = row["sample_id"]
                WBC_type = row["WBC_types"]
                origin = row["origin"]
                if origin == "A":
                    image_path = get_image_local_path(sample_id, self.DIR_PATH_A)
                    image = read_image(image_path)
                    annotation_path = get_xml_local_path(sample_id, self.DIR_PATH_A)
                    annotations, image_shape = load_xml(annotation_path)
                    masks, class_labels = convert_annotations(annotations, image_shape)
                    class_labels = [correct_class_label(l, WBC_type) for l in class_labels]
                elif origin == "B":
                    image_path = get_image_local_path_B(sample_id, self.DIR_PATH_B)
                    image = read_image(image_path)
                    mask = read_image(get_mask_local_path_B(sample_id, self.DIR_PATH_B))
                    masks = mask[..., 0:1]
                    class_labels = WBC_type
                # print(class_labels)
                if (len(class_labels) == 0):
                    print(f'Skip sample {sample_id}, no gt_objects')
                    continue
                for class_label in class_labels:
                    self.classes.update({class_label})
                self.add_image(source=self.source,
                               origin=origin,
                               image_id=sample_id,
                               sub_class_label=WBC_type,
                               path=image_path,
                               image_size=image.shape,
                            )
            except Exception as e:
                print(f'EXCP: {e} during handling {sample_id}')
                continue
        for i, c in enumerate(list(self.class_names_preset[1:])):
            self.add_class(self.source, i + 1, c)
        self.prepare()
        return
