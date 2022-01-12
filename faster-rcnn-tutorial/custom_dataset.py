import os
import pathlib
from multiprocessing import Pool
from typing import List, Dict
import numpy as np
import json
import cv2

import torch
from skimage.color import rgba2rgb
from skimage.io import imread
from torchvision.ops import box_convert
from detection.transformations import ComposeDouble, ComposeSingle, map_class_to_int
from detection.utils import read_json


# https://github.com/TannerGilbert/Object-Detection-and-Image-Segmentation-with-Detectron2
def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename

        annos = v["regions"]
        boxes, labels = [], []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]

            boxes.append([np.min(px), np.min(py), np.max(px), np.max(py)])
            labels.append('balloon')
        record["annotations"] = {'boxes': boxes, 'labels': labels}
        dataset_dicts.append(record)
    return dataset_dicts


class ObjectDetectionDataSet(torch.utils.data.Dataset):
    """
    Builds a dataset with images and their respective targets.
    A target is expected to be a json file
    and should contain at least a 'boxes' and a 'labels' key.
    inputs and targets are expected to be a list of pathlib.Path objects.

    In case your labels are strings, you can use mapping (a dict) to int-encode them.
    Returns a dict with the following keys: 'x', 'x_name', 'y', 'y_name'
    """

    def __init__(self,
                 data_path: str,
                 transform: ComposeDouble = None,
                 mapping: Dict = None
                 ):
        self.data_path = data_path
        self.dataset_dict = get_balloon_dicts(data_path)
        self.transform = transform
        self.mapping = mapping

    def __len__(self):
        return len(self.dataset_dict)

    def __getitem__(self,
                    index: int):
        record = self.dataset_dict[index]

        # Load input
        x = imread(record["file_name"])
        img_name = os.path.basename(record["file_name"])
        
        # From RGBA to RGB
        if x.shape[-1] == 4:
            x = rgba2rgb(x)
        
        # Label Mapping
        y = record["annotations"]
        if self.mapping:
            labels = map_class_to_int(y['labels'], mapping=self.mapping)
        else:
            labels = y['labels']

        # Create target, should be converted to np.ndarrays
        target = {'boxes': np.array(y['boxes']),
                  'labels': np.array(labels)}

        if self.transform is not None:
            x, target = self.transform(x, target)  # returns np.ndarrays

        # Typecasting
        x = torch.from_numpy(x).type(torch.float32)
        target = {key: torch.from_numpy(value).type(torch.int64) for key, value in target.items()}

        return {'x': x, 'y': target, 'x_name': img_name, 'y_name': img_name}



class ObjectDetectionDatasetSingle(torch.utils.data.Dataset):
    """
    Builds a dataset with images.
    inputs is expected to be a list of pathlib.Path objects.

    Returns a dict with the following keys: 'x', 'x_name'
    """

    def __init__(self,
                 inputs: List[pathlib.Path],
                 transform: ComposeSingle = None,
                 use_cache: bool = False,
                 ):
        self.inputs = inputs
        self.transform = transform
        self.use_cache = use_cache

        if self.use_cache:
            # Use multiprocessing to load images and targets into RAM
            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, inputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            x = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]

            # Load input and target
            x = self.read_images(input_ID)

        # From RGBA to RGB
        if x.shape[-1] == 4:
            x = rgba2rgb(x)

        # Preprocessing
        if self.transform is not None:
            x = self.transform(x)  # returns a np.ndarray

        # Typecasting
        x = torch.from_numpy(x).type(torch.float32)

        return {'x': x, 'x_name': self.inputs[index].name}

    @staticmethod
    def read_images(inp):
        return imread(inp)

class ObjectDetectionDataSetDouble(torch.utils.data.Dataset):
    """
    Builds a dataset with images and their respective targets.
    A target is expected to be a json file
    and should contain at least a 'boxes' and a 'labels' key.
    inputs and targets are expected to be a list of pathlib.Path objects.
    In case your labels are strings, you can use mapping (a dict) to int-encode them.
    Returns a dict with the following keys: 'x', 'x_name', 'y', 'y_name'
    """

    def __init__(self,
                 inputs: List[pathlib.Path],
                 targets: List[pathlib.Path],
                 transform: ComposeDouble = None,
                 use_cache: bool = False,
                 convert_to_format: str = None,
                 mapping: Dict = None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.use_cache = use_cache
        self.convert_to_format = convert_to_format
        self.mapping = mapping

        if self.use_cache:
            # Use multiprocessing to load images and targets into RAM
            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, zip(inputs, targets))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x, y = self.read_images(input_ID, target_ID)

        # From RGBA to RGB
        if x.shape[-1] == 4:
            x = rgba2rgb(x)

        # Read boxes
        try:
            boxes = torch.from_numpy(y['boxes']).to(torch.float32)
        except TypeError:
            boxes = torch.tensor(y['boxes']).to(torch.float32)
            
        # Read scores
        if 'scores' in y.keys():
            try:
                scores = torch.from_numpy(y['scores']).to(torch.float32)
            except TypeError:
                scores = torch.tensor(y['scores']).to(torch.float32)            

        # Label Mapping
        if self.mapping:
            labels = map_class_to_int(y['labels'], mapping=self.mapping)
        else:
            labels = y['labels']

        # Read labels
        try:
            labels = torch.from_numpy(labels).to(torch.int64)
        except TypeError:
            labels = torch.tensor(labels).to(torch.int64)


        # Create target
        target = {'boxes': boxes,
                  'labels': labels}

        if 'scores' in y.keys():
            target['scores'] = scores

        # Preprocessing
        target = {key: value.numpy() for key, value in target.items()}  # all tensors should be converted to np.ndarrays

        if self.transform is not None:
            x, target = self.transform(x, target)  # returns np.ndarrays

        # Typecasting
        x = torch.from_numpy(x).type(torch.float32)
        target = {key: torch.from_numpy(value).type(torch.int64) for key, value in target.items()}

        return {'x': x, 'y': target, 'x_name': self.inputs[index].name, 'y_name': self.targets[index].name}

    @staticmethod
    def read_images(inp, tar):
        return imread(inp), read_json(tar)