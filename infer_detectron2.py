import os

import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode

from utils import save_preview_input_gt_output

cfg = get_cfg()
cfg.OUTPUT_DIR = "/home/kevinmf94/output500x500"
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

im = cv2.imread('images/Gava_Viladecans/Images/image_3.tif')[600:900, 600:900]
mask = cv2.imread('images/Gava_Viladecans/Masks/image_3_mask.tif', -1).astype(np.uint8)[600:900, 600:900]
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

MetadataCatalog.get("data_test").set(thing_classes=["asbestos", "streets", "builds", "greenspaces"])
metadata = MetadataCatalog.get("data_test")
v = Visualizer(im[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE_BW)

out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
save_preview_input_gt_output(f'out_infer.jpg', im, mask, out.get_image()[:, :, ::-1])
