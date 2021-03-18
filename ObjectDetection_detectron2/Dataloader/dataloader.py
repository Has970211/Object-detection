import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import cv2
import random
import os
import json
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode

class dataloader(object):
    def __init__(self, data_dir, json_dir, img_dir):
        self.data_dir=data_dir
        self.json_dir=json_dir
        self.img_dir=img_dir

    def get_dataset_dicts(self):
        json_file = os.path.join(self.data_dir, self.json_dir)   #json_dir_train="results/merged/annotations/merged.json", json_dir_val="annotations/merged.json"
        with open(json_file) as f:
            imgs_anns = json.load(f)

        dataset_dicts = []
        for idx, v in enumerate(imgs_anns['images']):
            record = {}
            
            filename = os.path.join(self.data_dir, self.img_dir, v["file_name"])  #img_dir_train="results/merged/images", img_dir_val="images/image_set"
            src = cv2.imread(filename)
            img_height, img_width = src.shape[:2]
            image_id = idx + 1
            json_height, json_width = v['height'], v['width']
            
            record["file_name"] = filename
            record["image_id"] = image_id
            record["height"] = json_height
            record["width"] = json_width

            if img_height == json_width and img_width == json_height:
                print("turned")
                image = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE) 
                cv2.imwrite(filename, image)
        
            annos = list(filter(lambda img: img['image_id'] == image_id, imgs_anns['annotations']))
            objs = []
            for anno in annos:
                iscrowd = anno['iscrowd']
                category_id = anno['category_id'] - 1
                bbox = anno['bbox']
                segmentation = anno['segmentation']
                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": segmentation,
                    "category_id": category_id,
                    "iscrowd": iscrowd
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts

