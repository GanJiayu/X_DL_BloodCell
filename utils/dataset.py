from mrcnn import utils as mrcnn_utils
from tqdm import tqdm
import numpy as np
import json
from utils.utils import get_image_local_path, read_image, get_xml_local_path, load_xml
from utils.utils import convert_annotations


class BloodCellDataset(mrcnn_utils.Dataset):
    """Derived class of mrcnn Dataset to handle eleven images."""

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
                masks, class_labels = convert_annotations(annotations, image_shape)
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


class BloodCellExternalDataset(BloodCellDataset):
    """Derived class of mrcnn Dataset to handle test set images."""

    def __init__(self, directory_path, sample_ids, verbose=False):
        super().__init__(directory_path, sample_ids, verbose)

    def load_mask(self, image_id):
        raise NotImplementedError("No ground truth annotations for samples in the test set.")

    def load_kernel(self) -> None:
        """ Load the dataset before inference. """
        self.classes = set()
        for sample_id in tqdm(self.sample_ids):
            try:
                image_path = get_image_local_path(sample_id, self.DIR_PATH)
                image = read_image(image_path)

                self.add_image(
                    source=self.source,
                    image_id=sample_id,
                    sub_class_label="ConstructionSite",
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
