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
import sys
import threading

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer as FullVisualizer
from detectron2_repo_local.detectron2.utils.visualizer import Visualizer as MaskVisualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode


class MaskRCNN(object):
    def __init__(self):
        # load mask-structure
        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            "./detectron2_repo_local/configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = \
            "./detectron2_repo_local/configs/model/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.pkl"
        self.predictor = DefaultPredictor(self.cfg)

        # set params for binary mask
        self.threshold = 0.0000000001

        # set threading
        self.lock = threading.Lock()

    def main(self):
        # get directory path in input_image
        directory_path = './input_image'
        directories = os.listdir(directory_path)

        for directory in directories:
            # create dir for outputs
            self.mkdir(directory)

            # get file path in directory
            file_path = './input_image/{}/*.jpg'.format(directory)
            files = glob.glob(file_path)

            for file in files:
                img = cv2.imread(file)
                outputs = self.predictor(img)
                output_name = os.path.basename(file)

                self.full_prediction(img, outputs, directory, output_name)
                self.mask_prediction(img, outputs, directory, output_name)
                self.binary_mask_prediction(img, outputs, directory, output_name)

                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

    def full_prediction(self, img, outputs, directory, output_name):
        v_full = FullVisualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.0)
        v_full = v_full.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imshow('output_image_full', v_full.get_image()[:, :, ::-1])
        output_folder_path = "./output_image_with_full/{}/{}".format(directory, output_name)
        cv2.imwrite(output_folder_path, v_full.get_image()[:, :, ::-1])

    def mask_prediction(self, img, outputs, directory, output_name):
        v_mask = MaskVisualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                                scale=1.0, instance_mode=ColorMode.SEGMENTATION)
        v_mask, is_in_person = v_mask.draw_instance_predictions(outputs["instances"].to("cpu"), output_name)

        if is_in_person == 1:
            # cv2.imshow('output_image_mask', v_mask.get_image()[:, :, ::-1])
            output_folder_path = "./output_image_with_binary_mask/{}/{}".format(directory, output_name)
            cv2.imwrite(output_folder_path, v_mask.get_image()[:, :, ::-1])

            # create new_input for gan
            new_input_folder_path = './new_input_image/{}/{}'.format(directory, output_name)
            cv2.imwrite(new_input_folder_path, img)

        elif is_in_person == -1:
            print('{} is disregarded'.format(output_name))

    def binary_mask_prediction(self, img, outputs, directory, output_name):
        mask_base = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8)
        v_binary_mask = MaskVisualizer(mask_base, MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                                       scale=1.0, instance_mode=ColorMode.SEGMENTATION)
        v_binary_mask, is_in_person = v_binary_mask.draw_instance_predictions(outputs["instances"].to("cpu"), output_name)

        if is_in_person == 1:
            v_mask_gray = cv2.cvtColor(v_binary_mask.get_image()[:, :, ::-1], cv2.COLOR_RGB2GRAY)
            ret, v_binary_mask_thresh = cv2.threshold(v_mask_gray, self.threshold, 255, cv2.THRESH_BINARY)
            # cv2.imshow('output_image_binary_mask', v_binary_mask_thresh)
            output_binary_mask_folder_path = "./output_image_binary_mask/{}/{}".format(directory, output_name)
            cv2.imwrite(output_binary_mask_folder_path, v_binary_mask_thresh)

            img_mask = cv2.cvtColor(v_binary_mask_thresh, cv2.COLOR_GRAY2BGR)
            img_with_mask = cv2.bitwise_and(img, img_mask)
            output_mask_folder_path = "./output_image_mask/{}/2.png".format(directory)
            cv2.imwrite(output_mask_folder_path, img_with_mask)
            # cv2.imshow('output_image__mask', img_with_mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        elif is_in_person == -1:
            print('{} is disregarded'.format(output_name))

    @staticmethod
    def mkdir(directory):
        # check if output folders exist or not
        if not os.path.exists('./output_image_with_full'):
            os.mkdir('./output_image_with_full')
        if not os.path.exists('./output_image_with_binary_mask'):
            os.mkdir('./output_image_with_binary_mask')
        if not os.path.exists('./new_input_image'):
            os.mkdir('./new_input_image')
        if not os.path.exists('./output_image_binary_mask'):
            os.mkdir('./output_image_binary_mask')
        if not os.path.exists('./output_image_mask'):
            os.mkdir('./output_image_mask')

        os.mkdir('./output_image_with_full/{}'.format(directory))
        os.mkdir('./output_image_with_binary_mask/{}'.format(directory))
        os.mkdir('./new_input_image/{}'.format(directory))
        os.mkdir('./output_image_binary_mask/{}'.format(directory))
        os.mkdir('./output_image_mask/{}'.format(directory))


if __name__ == '__main__':
    mask = MaskRCNN()
    mask.main()






