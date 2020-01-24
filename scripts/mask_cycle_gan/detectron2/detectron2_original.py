import torch, torchvision
print(torch.__version__)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import os, glob

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer as FullVisualizer
from detectron2_repo_local.detectron2.utils.visualizer import Visualizer as MaskVisualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode

num = 1
num_for_input_and_output = str(num)
input_name = "input_" + num_for_input_and_output + ".jpg"
im = cv2.imread("./input_image/" + input_name)
cv2.imshow("input", im)

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) if you're not running a model in detectron2's core library
cfg.merge_from_file("./detectron2_repo_local/configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
cfg.MODEL.WEIGHTS = "./detectron2_repo_local/configs/model/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.pkl"

predictor = DefaultPredictor(cfg)
outputs = predictor(im)

print(outputs["instances"].pred_classes)

# full prediction
v_full = FullVisualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v_full = v_full.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow('output_image_full', v_full.get_image()[:, :, ::-1])
output_name = "output_" + num_for_input_and_output + ".jpg"
cv2.imwrite("./output_image_full/" + output_name, v_full.get_image()[:, :, ::-1])

# mask prediction
mask_base = np.zeros((im.shape[0], im.shape[1], im.shape[2]), np.uint8)
v_mask = MaskVisualizer(mask_base, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2, instance_mode=ColorMode.SEGMENTATION)
v_mask = v_mask.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow('output_image_mask', v_mask.get_image()[:, :, ::-1])

# white mask prediction
v_mask_gray = cv2.cvtColor(v_mask.get_image()[:, :, ::-1], cv2.COLOR_RGB2GRAY)
threshold = 0.0000000001
ret, v_mask_thresh = cv2.threshold(v_mask_gray, threshold, 255, cv2.THRESH_BINARY)
cv2.imshow('output_image_binary_mask', v_mask_thresh)
cv2.imwrite("./output_image_binary_mask/" + output_name, v_mask_thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()