import detectron2
import cv2
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from ObjectDetection_detectron2.Dataloader.dataloader import dataloader

class test(object):
    def __init__(self, output_dir, data_dir,json_dir,img_dir, threshold_scr):
       self.output_dir= output_dir
       self.data_dir=data_dir
       self.json_dir=json_dir
       self.img_dir=img_dir
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

        for d in ['val']:
            DatasetCatalog.register("dataset_val", lambda d=d: dataloader.get_dataset_dicts(self.data_dir, self.json_dir, self.img_dir))
            MetadataCatalog.get("dataset_val").set(thing_classes=['gauge', 'valve', 'isolation', 'tank', 'idtag', 'pump', 'gaugeD'])
        custom_metadata = MetadataCatalog.get("dataset_val")
        dataset_val=dataloader.get_dataset_dicts(self.data_dir+d,self.json_dir, img_dir)
 
        for d in random.sample(dataset_val, 16):
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            v = Visualizer(im[:, :, ::-1],
                        metadata=custom_metadata,
                        scale=0.5,
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )

            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow(out.get_image()[:, :, ::-1])








