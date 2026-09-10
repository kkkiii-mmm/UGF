#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Single-class ("smoke") VOC-format dataset loader, shared by every
smoke-detection domain registered in `paths_route.py` (e.g. hpwrennemo2aihub,
hpwrennemo2boreal). Unlike CityDataset/FoggyDataset/etc., a single class here
serves all such domains: which physical directory/split it reads is resolved
purely from `PATH_ROUTES`, keyed the same way `cvpods.data.build` resolves it
(`dataset_name.split("_")[0].upper()`), so adding a new domain only requires a
new `paths_route.py` entry - no new dataset class.
"""

import copy
import os
import os.path as osp
import xml.etree.ElementTree as ET
import megfile

import numpy as np

import torch

from cvpods.structures import BoxMode

from ..base_dataset import BaseDataset
from ..detection_utils import (
    annotations_to_instances,
    check_image_size,
    filter_empty_instances,
    read_image
)
from ..registry import DATASETS, PATH_ROUTES

CLASS_NAMES = ("smoke",)


@DATASETS.register()
class SmokeDataset(BaseDataset):
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(SmokeDataset, self).__init__(cfg, dataset_name, transforms, is_train)

        route_key = dataset_name.split("_")[0].upper()
        image_root, split = PATH_ROUTES.get(route_key)["voc"][self.name]
        self.image_root = osp.join(self.data_root, image_root) \
            if "://" not in image_root else image_root
        self.split = split

        self.meta = self._get_metadata()
        self.dataset_dicts = self._load_annotations()

        # fmt: off
        self.data_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.filter_empty = cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        self.proposal_files = cfg.DATASETS.PROPOSAL_FILES_TRAIN
        # fmt: on

        if is_train:
            self.dataset_dicts = self._filter_annotations(
                filter_empty=self.filter_empty,
                min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
                if self.keypoint_on else 0,
                proposal_files=self.proposal_files if self.load_proposals else None,
            )
            self._set_group_flag()

        self.eval_with_gt = cfg.TEST.get("WITH_GT", False)
        self.keypoint_hflip_indices = None

    def __getitem__(self, index):
        """Load data, apply transforms, convert to Instances."""
        dataset_dict = copy.deepcopy(self.dataset_dicts[index])

        image = read_image(dataset_dict["file_name"], format=self.data_format)
        check_image_size(dataset_dict, image)

        if "annotations" in dataset_dict:
            annotations = dataset_dict.pop("annotations")
            annotations = [
                ann for ann in annotations if ann.get("iscrowd", 0) == 0]
        else:
            annotations = None

        image, annotations = self._apply_transforms(image, annotations)

        if annotations is not None:
            image_shape = image.shape[:2]  # h, w
            instances = annotations_to_instances(
                annotations, image_shape, mask_format=self.mask_format
            )
            dataset_dict["instances"] = filter_empty_instances(instances)

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))

        return dataset_dict

    def __len__(self):
        return len(self.dataset_dicts)

    def _get_metadata(self):
        meta = {
            "thing_classes": CLASS_NAMES,
            "evaluator_type": "pascal_voc",
            "dirname": self.image_root,
            "split": self.split,
            "year": 2007,
        }
        return meta

    @staticmethod
    def _read_fileids(path):
        # Some ImageSets/*.txt files here were exported with a leading BOM;
        # stripping "﻿" clears it regardless of how `smart_open` handled
        # the encoding. Read line-by-line rather than whitespace-tokenizing,
        # since a handful of file ids are hashes but most are descriptive
        # (underscore-separated, not space-separated).
        with megfile.smart_open(path, "r") as f:
            return [line.strip().lstrip("﻿") for line in f if line.strip()]

    def _load_annotations(self):
        """
        Load Pascal VOC-format single-class ("smoke") detection annotations.

        Args:
            dirname: contains "Annotations", "ImageSets", "JPEGImages"
            split (str): one of "train", "val", "unlabeled_train"
        """
        dirname = self.image_root
        split = self.split

        fileids = self._read_fileids(
            megfile.smart_path_join(dirname, "ImageSets", "Main", split + ".txt"))

        dicts = []
        for fileid in fileids:
            anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
            tree = ET.parse(anno_file)

            filename_node = tree.find("./filename")
            if filename_node is not None and filename_node.text:
                jpeg_file = os.path.join(dirname, "JPEGImages", filename_node.text)
            else:
                # fall back to guessing the extension if `<filename>` is absent
                jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")
                if not os.path.exists(jpeg_file):
                    jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpeg")

            r = {
                "file_name": jpeg_file,
                "image_id": fileid,
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
            }
            instances = []

            for obj in tree.findall("object"):
                cls = obj.find("name").text
                if cls not in CLASS_NAMES:
                    continue
                bbox = obj.find("bndbox")
                bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
                # Original annotations are integers in the range [1, W or H]
                # Assuming they mean 1-based pixel indices (inclusive),
                # a box with annotation (xmin=1, xmax=W) covers the whole image.
                # In coordinate space this is represented by (xmin=0, xmax=W)
                bbox[0] -= 1.0
                bbox[1] -= 1.0
                instances.append({
                    "category_id": CLASS_NAMES.index(cls),
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS
                })
            r["annotations"] = instances
            dicts.append(r)

        return dicts
