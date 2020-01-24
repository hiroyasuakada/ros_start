from PIL import Image
from pathlib import Path
import os, glob  # manipulate file or directory
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse


class DataArrangement(object):
    def __init__(self, dir_name_for_main, dir_name_for_overlay_mask_and_binary_mask,
                 img_num_for_main, img_num_for_overlay_mask_and_binary_mask):

        self.main = './test_quantitative/'\
                    + str(dir_name_for_main)  # No Nan
        self.overlay_mask = './qualitative_mask/'\
                            + str(dir_name_for_overlay_mask_and_binary_mask)
        self.overlay_binary_mask = './qualitative_binary_mask/'\
                                   + str(dir_name_for_overlay_mask_and_binary_mask)
        self.dir_name_for_main = dir_name_for_main
        self.dir_name_for_overlay_mask_and_binary_mask = dir_name_for_overlay_mask_and_binary_mask
        self.img_num_for_main = img_num_for_main
        self.img_num_for_mask_and_binary_mask = img_num_for_overlay_mask_and_binary_mask

    def overlay(self):
        self.mkdir_1()
        self.mkdir_2(self.dir_name_for_main)

        file_path_main = self.main + '/' + str(self.img_num_for_main) + '.jpg'
        file_path_mask = self.overlay_mask + '/' + str(self.img_num_for_mask_and_binary_mask) + '.jpg'
        file_path_binary_mask = self.overlay_binary_mask + '/' + str(self.img_num_for_mask_and_binary_mask) + '.jpg'

        img_main = cv2.imread(file_path_main)
        img_mask = cv2.imread(file_path_mask)
        img_binary_mask = cv2.imread(file_path_binary_mask)

        img_binary_mask_revert = cv2.bitwise_not(img_binary_mask)
        dst = cv2.bitwise_and(img_main, img_binary_mask_revert)
        img_main_with_mask = cv2.addWeighted(dst, 1, img_mask, 1, 0)
        img_main_with_binary_mask = cv2.addWeighted(img_main, 1, img_binary_mask, 1, 0)

        cv2.imwrite('./test_quantitative_as_gt/{}/{}'
                    .format(self.dir_name_for_main, self.img_num_for_main)
                    + '.jpg', img_main)
        cv2.imwrite('./test_quantitative_by_hand/{}/{}'
                    .format(self.dir_name_for_main, self.img_num_for_main)
                    + '.jpg', img_main_with_mask)
        cv2.imwrite('./test_quantitative_with_binary_mask_by_hand/{}/{}'
                    .format(self.dir_name_for_main, self.img_num_for_main)
                    + '.jpg', img_main_with_binary_mask)

    @staticmethod
    def mkdir_1():
        if not os.path.exists('./test_quantitative_as_gt'):
            os.mkdir('./test_quantitative_as_gt')
        if not os.path.exists('./test_quantitative_with_binary_mask_by_hand'):
            os.mkdir('./test_quantitative_with_binary_mask_by_hand')
        if not os.path.exists('./test_quantitative_by_hand'):
            os.mkdir('./test_quantitative_by_hand')

    @staticmethod
    def mkdir_2(dir_name_for_main):
        if not os.path.exists('./test_quantitative_as_gt/{}'.format(dir_name_for_main)):
            os.mkdir('./test_quantitative_as_gt/{}'.format(dir_name_for_main))
        if not os.path.exists('./test_quantitative_with_binary_mask_by_hand/{}'.format(dir_name_for_main)):
            os.mkdir('./test_quantitative_with_binary_mask_by_hand/{}'.format(ir_name_for_main))
        if not os.path.exists('./test_quantitative_by_hand/{}'.format(ir_name_for_main)):
            os.mkdir('./test_quantitative_by_hand/{}'.format(ir_name_for_main))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('arg1', help='dir_name_for_main')  # for example, 2019_11_1314_double_ridge_04
    parser.add_argument('arg2', help='dir_name_for_overlay_mask_and_binary_mask')  # for example, 2019_11_1314_tracking_double_ridge_05
    parser.add_argument('arg3', help='img_num_for_main')
    parser.add_argument('arg4', help='img_num_for_overlay_mask_and_binary_mask')

    args = parser.parse_args()

    gen_data = DataArrangement(args.arg1, args.arg2, args.arg3, args.arg4)
    gen_data.overlay()





# class DataArrangement(object):
#     def __init__(self, dir_name_for_main, dir_name_for_overlay_mask, dir_name_for_overlay_binary_mask
#                  ):
#
#         self.main_1 = './test_quantitative/2019_11_1314_double_ridge_04'  # No Nan
#         self.overlay_mask_1 = './data_from_detectron2/test/qualitative_mask/2019_11_1314_tracking_double_ridge_05'
#         self.overlay_binary_mask_1 = './data_from_detectron2/test/qualitative_binary_mask/2019_11_1314_tracking_double_ridge_05'
#         self.count_main_1 = 5  # start at this num
#         self.count_mask_1 = 70
#
#     def overlay(self):
#         self.mkdir()
#         vacant_count = 0
#         loop_count = 0
#         files_main = glob.glob(self.main_1 + '/*.jpg')
#         num_main = len(files_main) - self.count_main_1
#         files_mask = glob.glob(self.main_1 + '/*.jpg')
#         num_mask = len(files_mask) - self.count_mask_1
#
#         num_min = None
#         num_max = None
#
#         for i in range(2500):
#             print(str(loop_count + vacant_count))
#
#             file_path_main = self.main_1 + '/' + str(self.count_main_1 + loop_count) + '.jpg'
#             file_path_mask = self.overlay_mask_1 + '/' + str(self.count_mask_1 + loop_count + vacant_count) + '.jpg'
#             file_path_binary_mask = self.overlay_binary_mask_1 + '/' + str(
#                 self.count_mask_1 + loop_count + vacant_count) + '.jpg'
#
#             if not os.path.exists(file_path_main):
#                 break
#
#             if not os.path.exists(file_path_mask):
#                 vacant_count += 1
#                 continue
#
#             img_main = cv2.imread(file_path_main)
#             img_mask = cv2.imread(file_path_mask)
#             img_binary_mask = cv2.imread(file_path_binary_mask)
#
#             img_binary_mask_revert = cv2.bitwise_not(img_binary_mask)
#             dst = cv2.bitwise_and(img_main, img_binary_mask_revert)
#             img_main_with_mask = cv2.addWeighted(dst, 1, img_mask, 1, 0)
#             img_main_with_binary_mask = cv2.addWeighted(img_main, 1, img_binary_mask, 1, 0)
#
#             cv2.imwrite('./test_quantitative_as_gt/{}'.format(self.count_main_1 + loop_count)
#                         + '.jpg', img_main)
#             cv2.imwrite('./test_quantitative_with_mask_by_hand/{}'.format(self.count_main_1 + loop_count)
#                         + '.jpg', img_main_with_mask)
#             cv2.imwrite('./test_quantitative_with_binary_mask_by_hand/{}'.format(self.count_main_1 + loop_count)
#                         + '.jpg', img_main_with_binary_mask)
#
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#
#             loop_count += 1