# generate data from bag images

from PIL import Image
from pathlib import Path
import os, glob  # manipulate file or directory
import numpy as np
import cv2
import matplotlib.pyplot as plt


class DataArrangement(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

        self.prepared_data_directories = ['qualitative_binary_mask',
                                          'qualitative_mask',
                                          'test_qualitative',
                                          'test_qualitative_with_binary_mask',
                                          'test_quantitative',
                                          'test_quantitative_as_gt',
                                          'test_quantitative_by_hand',
                                          'test_quantitative_with_binary_mask_by_hand',
                                          'train_not_tracking',
                                          'train_tracking',
                                          'train_tracking_with_binary_mask'
                                          ]

        self.resize_data_directories = ['trainA', 'trainB']  # trainA: tracking (mask or not), trainB: not_tracking
        # self.target_dir = ['double', 'single', 'out_to_in', 'in_to_out']
        self.dict_for_mask_or_not = None
        self.X_not_tracking = []
        self.X_not_tracking_i = []
        self.Y_tracking = []
        self.Y_tracking_i = []
        self.iteration = 1
        self.directory_name = None
        self.new_csv_name = None

    def resize_data(self):
        for prepared_data in self.prepared_data_directories:
            print(prepared_data)  # not_tracking or tracking
            self.mkdir_1(self.height, self.width, prepared_data)
            # path = Path(__file__).parent
            # path /= '../../dataset_for_gan/{}'.format(prepared_data)
            path = './180_320/dataset_for_cyclegan_by_csv/{}'.format(prepared_data)
            directories = os.listdir(path)

            for directory in directories:

                # if directory.find(self.target_dir[0]) != -1\
                #         or directory.find(self.target_dir[1]) != -1\
                #         or directory.find(self.target_dir[2]) != -1\
                #         or directory.find(self.target_dir[3]) != -1:

                self.mkdir_2(self.height, self.width, prepared_data, directory)
                files = glob.glob(str(path + '/{}/*.jpg'.format(directory)))

                for file in files:
                    get_file_name = os.path.basename(file)
                    output_name = '{}_{}'.format(directory, get_file_name)

                    image = Image.open(file)
                    image = image.convert('RGB')
                    image = image.resize((self.width, self.height))  # 108, 192 / 180, 320
                    data = np.asarray(image)

                    print(output_name)

                    bgr_image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

                    cv2.imwrite('./{}_{}/dataset_for_cyclegan_by_csv/{}/{}'
                                .format(self.height, self.width, prepared_data, directory)
                                + '/' + get_file_name, bgr_image)

    @staticmethod
    def mkdir_1(height, width, directory):
        if not os.path.exists('./{}_{}/dataset_for_cyclegan_by_csv/{}'.format(height, width, directory)):
            os.mkdir('./{}_{}/dataset_for_cyclegan_by_csv/{}'.format(height, width, directory))

    @staticmethod
    def mkdir_2(height, width, directory_parent, directory):
        if not os.path.exists('./{}_{}/dataset_for_cyclegan_by_csv/{}/{}'.format(height, width, directory_parent, directory)):
            os.mkdir('./{}_{}/dataset_for_cyclegan_by_csv/{}/{}'.format(height, width, directory_parent, directory))



########################################################################################################################

                        # if prepared_data == 'train_not_tracking':
                        #     cv2.imwrite('./{}_{}/with_binary_mask_4_situation/trainB'.format(self.height, self.width) +
                        #                 '/' + output_name, bgr_image)
                        #     cv2.imwrite('./{}_{}/without_mask_4_situation/trainB'.format(self.height, self.width) +
                        #                 '/' + output_name, bgr_image)
                        #
                        # elif prepared_data == 'train_tracking':
                        #     cv2.imwrite('./{}_{}/without_mask_4_situation/trainA'.format(self.height, self.width) +
                        #                 '/' + output_name, bgr_image)
                        #
                        # elif prepared_data == 'train_tracking_with_binary_mask':
                        #     cv2.imwrite('./{}_{}/with_binary_mask_4_situation/trainA'.format(self.height, self.width) +
                        #                 '/' + output_name, bgr_image)

                        # if prepared_data == 'test_qualitative':
                        #     cv2.imwrite('./{}_{}/without_mask_4_situation/test_qualitative'.format(self.height, self.width) +
                        #                 '/' + output_name, bgr_image)
                        #
                        # elif prepared_data == 'test_qualitative_with_binary_mask':
                        #     cv2.imwrite('./{}_{}/with_binary_mask_4_situation/test_qualitative_with_binary_mask'.format(self.height, self.width) +
                        #                 '/' + output_name, bgr_image)
                        #
                        # elif prepared_data == 'test_quantitative_as_gt':
                        #     cv2.imwrite('./{}_{}/with_binary_mask_4_situation/test_quantitative_as_gt'.format(self.height, self.width) +
                        #                 '/' + output_name, bgr_image)
                        #     cv2.imwrite('./{}_{}/without_mask_4_situation/test_quantitative_as_gt'.format(self.height, self.width) +
                        #                 '/' + output_name, bgr_image)
                        #
                        # elif prepared_data == 'test_quantitative_by_hand':
                        #     cv2.imwrite('./{}_{}/without_mask_4_situation/test_quantitative_by_hand'.format(self.height, self.width) +
                        #                 '/' + output_name, bgr_image)
                        #
                        # elif prepared_data == 'test_quantitative_with_binary_mask_by_hand':
                        #     cv2.imwrite('./{}_{}/with_binary_mask_4_situation/test_quantitative_with_binary_mask_by_hand'.format(self.height, self.width) +
                        #                 '/' + output_name, bgr_image)



    def load_resized_data_for_gan(self, mask=True):
        if mask is True:
            self.dict_for_mask_or_not = 'with_mask'
        else:
            self.dict_for_mask_or_not = 'without_mask'

        for i, directory in enumerate(self.resize_data_directories):
            file_path = './{}_{}/{}/{}'.format(self.height, self.width, self.dict_for_mask_or_not, directory)
            files = glob.glob(file_path + '/*.jpg')

            for j, file in enumerate(files):
                data = cv2.imread(file)

                if directory == 'trainA':  # trainA: tracking (mask or not)
                    self.Y_tracking.append(data)
                elif directory == 'trainB':  # trainB: not_tracking
                    self.X_not_tracking.append(data)

                if int(j + 1) % 5000 == 0:
                    print('finished image processing for ' + str(5000 * self.iteration))
                    self.iteration += 1

        return np.array(self.X_not_tracking), np.array(self.Y_tracking)

    # def load_resized_data_for_lstm_gan(self):
    #
    #     return np.array(self.X_not_tracking), np.array(self.Y_tracking), \
    #            np.array(self.X_not_tracking_i), np.array(self.Y_tracking_i)
    #     pass


if __name__ == '__main__':
    DA = DataArrangement(108, 192)  # (height, width) = (160, 320) or (80, 160) or (360, 360) in this case,
    DA.resize_data()
    print("finished_resize_data")
    # X_not_tracking, Y_tracking = DA.load_resized_data_for_gan(mask=True)
    # print('X_not_tracking_raw: {}, Y_tracking_raw: {}'.format(X_not_tracking, Y_tracking))
    # print('shape of X_not_tracking_raw: {}, shape of Y_tracking_raw: {}'.format(X_not_tracking.shape, Y_tracking.shape))
