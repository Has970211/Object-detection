import cv2
import os
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from PIL import Image
#from ObjectDetection_detectron2.Dataloader.dataloader import dataloader
from ObjectDetection_detectron2.CREATED.CREATE_D import CreateD

class test(object):
    def __init__(self, output_dir, img_folder, threshold_scr):
       self.output_dir= output_dir
       self.img_folder=img_folder
       self.threshold_scr = threshold_scr

    def call(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7 
        cfg.OUTPUT_DIR=self.output_dir
        
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold_scr  #set a custom testing threshold
        predictor = DefaultPredictor(cfg)
        
        PATH = os.path.join(self.img_folder, 'OUTPUT_RESULTS', 'Images')
        if not os.path.exists(PATH):
            os.mkdir(PATH)
            
        PathJ = os.path.join(self.img_folder, 'OUTPUT_RESULTS', 'JsonFile')
        if not os.path.exists(PathJ):
            os.mkdir(PathJ)
           
        dict_main={}
        for filename in os.listdir(self.img_folder):
            im = cv2.imread(os.path.join(self.img_folder, filename))
            rgb_image = im[:, :, ::-1]
            outputs = predictor(rgb_image) # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

            dict1=CreateD(outputs)
            dict_main[str(filename)]=dict1
            
            v = Visualizer(rgb_image,  MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            img = Image.fromarray(np.uint8(v.get_image()[:, :, ::-1]))
            img.save(os.path.join(PATH, filename))

        
        file = os.path.join(self.img_folder, 'OUTPUT_RESULTS', 'JsonFile', 'sample.json')
        try:
            open(file, 'a').close()
        except OSError:
            print('Failed creating the file')
        else:
            print('File created')
        with open(file, "r+") as outfile:
            json.dump(dict_main, outfile, default=lambda o: o.__dict__, indent=4)





