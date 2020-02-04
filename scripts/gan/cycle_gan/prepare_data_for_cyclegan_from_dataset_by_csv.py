# generate data from bag images

from PIL import Image
from pathlib import Path
import os, glob  # manipulate file or directory
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt


class DataArrangement(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

        # self.prepared_data_directories = ['train_not_tracking',
        #                                   'train_tracking',
        #                                   'train_tracking_with_binary_mask',
        #                                   'test_qualitative',
        #                                   'test_qualitative_with_binary_mask',
        #                                   'test_quantitative',
        #                                   'test_quantitative_as_gt',
        #                                   'test_quantitative_by_hand',
        #                                   'test_quantitative_with_binary_mask_by_hand',
        #                                   'qualitative_binary_mask',
        #                                   'qualitative_mask']
        self.prepared_data_directories = ['tracking_binary_mask',
                                          'tracking_mask'
                                          ]

        # self.prepared_data_directories = ['not_tracking_raw', 'tracking_raw_mask', 'test_raw_mask', 'tracking_raw', 'test_raw']
        self.resize_data_directories = ['trainA', 'trainB']  # trainA: tracking (mask or not), trainB: not_tracking
        self.dict_for_mask_or_not = None
        self.X_not_tracking = []
        self.X_not_tracking_i = []
        self.Y_tracking = []
        self.Y_tracking_i = []
        self.iteration = 1
        self.dir_name = None
        self.directory_name = None
        self.new_csv_name = None

    def resize_data(self):
        for prepared_data in self.prepared_data_directories:
            print(prepared_data)  # not_tracking or tracking
            path = Path(__file__).parent
            path /= '../../dataset_for_gan/{}'.format(prepared_data)
            directories = os.listdir(path)

            for directory in directories:
                print(directory)
                files = glob.glob(str(path.resolve()) + '/{}/*.jpg'.format(directory))

                # create dir
                self.dir_name = './{}_{}/dataset_for_cyclegan_by_csv/{}' \
                            .format(self.height, self.width, prepared_data)
                if not os.path.exists(self.dir_name):
                    os.mkdir(self.dir_name)

                # load csv
                path_csv = Path(__file__).parent
                path_csv /= '../../dataset_for_gan/enc_theta_dx_csv/{}.csv'.format(directory)

                df = pd.read_csv(path_csv, header=None)
                df_new = df[df[0] == 1]
                img_of_1 = df_new[1].values

                # print(df_new)
                # print(img_of_1)
                # print(len(img_of_1))

                self.new_csv_name = \
                    './{}_{}/dataset_for_cyclegan_by_csv/enc_theta_dx_of_1_csv/{}.csv'.format(self.height, self.width, directory)
                # if os.path.exists(self.new_csv_name) is False:
                #     df_new.to_csv(self.new_csv_name, header=False, index=False)

                for file in files:
                    get_file_name = os.path.basename(file)
                    name_comparison = get_file_name.strip('.jpg')

                    if int(name_comparison) in img_of_1:
                        image = Image.open(file)
                        image = image.convert('RGB')
                        image = image.resize((self.width, self.height))  # 360, 640 or 180, 320
                        data = np.asarray(image)

                        print('{}, {}'.format(get_file_name, data.shape))

                        bgr_image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

                        self.directory_name = './{}_{}/dataset_for_cyclegan_by_csv/{}/{}' \
                            .format(self.height, self.width, prepared_data, directory)

                        if not os.path.exists(self.directory_name):
                            os.mkdir(self.directory_name)
                        cv2.imwrite(self.directory_name + '/' + get_file_name, bgr_image)

                        # if prepared_data == 'train_not_tracking':
                        #     self.directory_name = './{}_{}_lstm_cyclegan/with_mask/trainB/{}'.format(self.height, self.width, directory)
                        #
                        # elif prepared_data == 'train_tracking_mask':
                        #     self.directory_name = './{}_{}_lstm_cyclegan/with_mask/trainA/{}'.format(self.height, self.width, directory)
                        #
                        # elif prepared_data == 'test_qualitative':
                        #     self.directory_name = './{}_{}_lstm_cyclegan/with_mask/test_qualitative/{}'.format(self.height, self.width, directory)
                        #
                        # elif prepared_data == 'test_qualitative_mask':
                        #     self.directory_name = './{}_{}_lstm_cyclegan/with_mask/test_qualitative_mask/{}'.format(self.height, self.width, directory)
                        #
                        # elif prepared_data == 'test_quantitative_as_gt':
                        #     self.directory_name = './{}_{}_lstm_cyclegan/with_mask/test_quantitative_as_gt/{}'.format(self.height, self.width, directory)
                        #
                        # elif prepared_data == 'test_quantitative_mask_by_hand':
                        #     self.directory_name = './{}_{}_lstm_cyclegan/with_mask/test_quantitative_mask_by_hand/{}'.format(self.height, self.width, directory)
                        #
                        # if not os.path.exists(self.directory_name):
                        #     os.mkdir(self.directory_name)
                        # cv2.imwrite(self.directory_name + '/' + get_file_name, bgr_image)

    def load_resized_data_for_gan(self, mask=True):
        if mask is True:
            self.dict_for_mask_or_not = 'with_mask'
        else:
            self.dict_for_mask_or_not = 'without_mask'

        for i, directory in enumerate(self.resize_data_directories):
            file_path = './{}_{}/{}/{}/*'.format(self.height, self.width, self.dict_for_mask_or_not, directory)
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
    DA = DataArrangement(180, 320)  # (height, width) = (160, 320)
    DA.resize_data()
    print('data_processing_stopped')
    # X_not_tracking, Y_tracking = DA.load_resized_data_for_gan(mask=True)
    # print('X_not_tracking_raw: {}, Y_tracking_raw: {}'.format(X_not_tracking, Y_tracking))
    # print('shape of X_not_tracking_raw: {}, shape of Y_tracking_raw: {}'.format(X_not_tracking.shape, Y_tracking.shape))
