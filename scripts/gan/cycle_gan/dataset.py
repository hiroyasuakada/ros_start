import os, glob
import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import time
import cv2
from natsort import natsorted
# import pandas as pd
# import seaborn as sns


class UnalignedDataset(torch.utils.data.Dataset):
    def __init__(self, is_train):
        super(UnalignedDataset, self).__init__()

        # make path to data
        root_dir = os.path.join('128_256', 'without_mask_4_situation_by_csv')

        if is_train:
            dir_A = os.path.join(root_dir, 'trainA')
            dir_B = os.path.join(root_dir, 'trainB')
            dir_A_mask = os.path.join(root_dir, 'tracking_binary_mask')

            # dir_A = os.path.join(root_dir, 'trainA_jn')
            # dir_B = os.path.join(root_dir, 'trainB_jn')
            # dir_A_mask = os.path.join(root_dir, 'trainA_mask_jn')
        else:
            dir_A = os.path.join(root_dir, 'test_quantitative_by_hand')
            dir_B = os.path.join(root_dir, 'test_quantitative_by_hand')
            dir_A_mask = os.path.join(root_dir, 'test_quantitative_binary_mask')

            # dir_A = os.path.join(root_dir, 'testA')
            # dir_B = os.path.join(root_dir, 'testB')
            # dir_A_mask = os.path.join(root_dir, 'testA_mask')

        self.image_paths_A = self._make_dataset(dir_A)
        self.image_paths_B = self._make_dataset(dir_B)
        self.image_paths_A_mask = self._make_dataset(dir_A_mask)

        self.size_A = len(self.image_paths_A)
        self.size_B = len(self.image_paths_B)
        self.size_A_mask = len(self.image_paths_A_mask)

        self.transform = self._make_transform(is_train)

    # get tensor data
    def __getitem__(self, index):
        index_A = index % self.size_A  # due to the different num of each data A, B
        path_A = self.image_paths_A[index_A]

        index_A_mask = index % self.size_A_mask
        path_A_mask = self.image_paths_A_mask[index_A_mask]

        # クラスBの画像はランダムに選択
        index_B = random.randint(0, self.size_B - 1)
        path_B = self.image_paths_B[index_B]

        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')
        img_A_mask = Image.open(path_A_mask).convert('RGB')

        # データ拡張
        A = self.transform(img_A)
        B = self.transform(img_B)
        A_mask = self.transform(img_A_mask)

        return {'A': A, 'B': B, 'A_mask': A_mask}

        # return {'A': A, 'B': B, 'A_mask': A_mask,
        #         'path_A': path_A, 'path_B': path_B, 'path_A_mask': path_A_mask}

    def __len__(self):
        len = min(self.size_A, self.size_B)
        print(len)
        return len   # self.size_A_mask

    @staticmethod
    def _make_dataset(dir):
        images = []
        for fname in os.listdir(dir):
            if fname.endswith('.jpg'):
                path = os.path.join(dir, fname)
                images.append(path)
        sorted(images)
        return images

    @staticmethod
    def _make_transform(is_train):
        transforms_list = []
#         transforms_list.append(transforms.Resize((load_size, load_size), Image.BICUBIC))
#         transforms_list.append(transforms.RandomCrop(fine_size))
#         if is_train:
#             transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # [0, 1] => [-1, 1]
        return transforms.Compose(transforms_list)

    #################################################################################################################


class LSTMDataset(torch.utils.data.Dataset):
    def __init__(self, _batch_size, _window_size, _step_size, is_train):
        super(LSTMDataset, self).__init__()
        self.batch_size = _batch_size
        self.window_size = _window_size
        self.step_size = _step_size

        # make path to data
        root_dir = os.path.join('128_256', 'without_mask_4_situation_by_csv_lstm')

        if is_train:
            dir_A = os.path.join(root_dir, 'trainA')
            dir_B = os.path.join(root_dir, 'trainB')
            dir_A_mask = os.path.join(root_dir, 'tracking_binary_mask')

        else:
            dir_A = os.path.join(root_dir, 'test_quantitative_by_hand')
            dir_B = os.path.join(root_dir, 'test_quantitative_by_hand')  # has no effect
            dir_A_mask = os.path.join(root_dir, 'test_quantitative_binary_mask')

        self.image_paths_A = self._make_dataset(dir_A, self.window_size, self.step_size)
        self.image_paths_B = self._make_dataset(dir_B, self.window_size, self.step_size)
        self.image_paths_A_mask = self._make_dataset(dir_A_mask, self.window_size, self.step_size)

        self.num_img = self.window_size / self.step_size

        self.size_A = len(self.image_paths_A)
        self.size_B = len(self.image_paths_B)
        self.size_A_mask = len(self.image_paths_A_mask)

        self.transform = self._make_transform(is_train)

    def __len__(self):
        len = min(self.size_A, self.size_B) - min(self.size_A, self.size_B) % int(self.batch_size)
        print('len: ' + str(len))
        return len   # self.size_A_mask

    def __getitem__(self, index):
        index_A = index % self.size_A
        path_A = self.image_paths_A[index_A]  # at the point, [3, 4, 5]

        index_A_mask = index % self.size_A_mask
        path_A_mask = self.image_paths_A_mask[index_A_mask]
        
        index_B = random.randint(0, self.size_B - 1)
        # index_B = index % self.size_B
        path_B = self.image_paths_B[index_B]

        # process the sequential images with transform function
        A = self.get_sequential_imgs(path_A, self.num_img)
        B = self.get_sequential_imgs(path_B, self.num_img)
        A_mask = self.get_sequential_imgs(path_A_mask, self.num_img)

        list_path_A = '-'.join(path_A)
        list_path_B = '-'.join(path_B)
        list_path_A_mask = '-'.join(path_A_mask)

        # return {'A': A, 'B': B,
        #         'path_A': list_path_A, 'path_B': list_path_B}

        return {'A': A, 'B': B, 'A_mask': A_mask,
                'path_A': list_path_A, 'path_B': list_path_B, 'path_A_mask': list_path_A_mask}

    def get_sequential_imgs(self, path, num_img):
        # img_transformed = [[0] for i in range(self.window_size)]
        img_transformed = []
        for i in range(int(num_img)):
            img = Image.open(path[i]).convert('RGB')
            img_transformed.append(self.transform(img))

        return torch.stack(img_transformed)

    @staticmethod
    def _make_dataset(dir, _window_size, _step_size):
        directories = os.listdir(dir)
        image_paths = []

        for directory in directories:
            files = glob.glob(os.path.join(dir, directory, '*.jpg'))
            if len(files) > _window_size:
                files = natsorted(files)
                num_sequential_imgs = (len(files) - _window_size) + 1  # (all - ws) / s + 1
                for i in range(num_sequential_imgs):
                    sequential_imgs = files[i: i + _window_size: _step_size]  # 0 ~ 6 (not including No68)
                    assert len(sequential_imgs) == (_window_size / _step_size)
                    image_paths.append(sequential_imgs)
        print(len(image_paths))
        return image_paths

    @staticmethod
    def _make_transform(is_train):
        transforms_list = []
        #         transforms_list.append(transforms.Resize((load_size, load_size), Image.BICUBIC))
        #         transforms_list.append(transforms.RandomCrop(fine_size))
        #         if is_train:
        #             transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # [0, 1] => [-1, 1]
        return transforms.Compose(transforms_list)

####################################################################################################################


class ActionConditionedLSTMDataset(torch.utils.data.Dataset):
    def __init__(self, _batch_size, _window_size, _step_size, is_train):
        super(ActionConditionedLSTMDataset, self).__init__()
        self.batch_size = _batch_size
        self.window_size = _window_size
        self.step_size = _step_size

        # make path to data
        root_dir = os.path.join('128_256', 'without_mask_4_situation_by_csv_lstm')

        if is_train:
            dir_A = os.path.join(root_dir, 'trainA')
            dir_B = os.path.join(root_dir, 'trainB')
            dir_A_mask = os.path.join(root_dir, 'tracking_binary_mask')
            dir_enc = os.path.join(root_dir, 'enc_theta_dx_csv')

        else:
            dir_A = os.path.join(root_dir, 'test_quantitative_by_hand')
            dir_B = os.path.join(root_dir, 'test_quantitative_by_hand')  # has no effect
            dir_A_mask = os.path.join(root_dir, 'test_quantitative_binary_mask')
            dir_enc = os.path.join(root_dir, 'enc_theta_dx_csv')

        self.image_paths_A, self.enc_value_theta_A, self.enc_value_x_A, self.enc_value_y_A = \
            self._make_dataset(dir_A, dir_enc, self.window_size, self.step_size)
        self.image_paths_B, self.enc_value_theta_B, self.enc_value_x_B, self.enc_value_y_B = \
            self._make_dataset(dir_B, dir_enc, self.window_size, self.step_size)
        self.image_paths_A_mask, _, _, _ = \
            self._make_dataset(dir_A_mask, dir_enc, self.window_size, self.step_size, mode_enc=False)

        self.num_img = self.window_size / self.step_size

        self.size_A = len(self.image_paths_A)
        self.size_B = len(self.image_paths_B)
        self.size_A_mask = len(self.image_paths_A_mask)

        self.transform = self._make_transform(is_train)

    def __len__(self):
        len = min(self.size_A, self.size_B) - min(self.size_A, self.size_B) % int(self.batch_size)
        print('len: ' + str(len))
        return len  # self.size_A_mask

    def __getitem__(self, index):
        index_A = index % self.size_A
        path_A = self.image_paths_A[index_A]  # at the point, [3, 4, 5]
        theta_A = torch.stack(list(self.enc_value_theta_A[index_A]))
        x_A = torch.stack(list(self.enc_value_x_A[index_A]))
        y_A = torch.stack(list(self.enc_value_y_A[index_A]))

        index_A_mask = index % self.size_A_mask
        path_A_mask = self.image_paths_A_mask[index_A_mask]

        index_B = random.randint(0, self.size_B - 1)
        # index_B = index % self.size_B
        path_B = self.image_paths_B[index_B]
        theta_B = self.enc_value_theta_B[index_B]
        x_B = self.enc_value_x_B[index_B]
        y_B = self.enc_value_y_B[index_B]

        # process the sequential images with transform function
        A = self.get_sequential_imgs(path_A, self.num_img)
        B = self.get_sequential_imgs(path_B, self.num_img)
        A_mask = self.get_sequential_imgs(path_A_mask, self.num_img)

        list_path_A = '-'.join(path_A)
        list_path_B = '-'.join(path_B)
        list_path_A_mask = '-'.join(path_A_mask)

        # return {'A': A, 'B': B,
        #         'path_A': list_path_A, 'path_B': list_path_B}

        return {'A': A,
                'path_A': list_path_A,
                'theta_A': theta_A,
                'x_A': x_A,
                'y_A': y_A,

                'B': B,
                'path_B': list_path_B,
                'theta_B': theta_B,
                'x_B': x_B,
                'y_B': y_B,

                'A_mask': A_mask,
                'path_A_mask': list_path_A_mask
                }

    def get_sequential_imgs(self, path, num_img):
        # img_transformed = [[0] for i in range(self.window_size)]
        img_transformed = []
        for i in range(int(num_img)):
            img = Image.open(path[i]).convert('RGB')
            img_transformed.append(self.transform(img))

        return torch.stack(img_transformed)

    @staticmethod
    def _make_dataset(dir, dir_enc, _window_size, _step_size, mode_enc=True):
        directories = os.listdir(dir)
        image_paths = []
        enc_value_theta_full = []
        enc_value_x_full = []
        enc_value_y_full = []
        # ###################
        # theta = []
        # x = []
        # y = []
        # ###################

        for directory in directories:
            files = glob.glob(os.path.join(dir, directory, '*.jpg'))
            if len(files) > _window_size:
                files = natsorted(files)
                num_sequential_imgs = (len(files) - _window_size) + 1  # (all - ws) / s + 1

                if mode_enc is True:
                    df = pd.read_csv(os.path.join(dir_enc, directory + '.csv'), header=None)
                    print(directory)

                for i in range(num_sequential_imgs):
                    sequential_imgs = files[i: i + _window_size: _step_size]  # 0 ~ 6 (not including No68)
                    assert len(sequential_imgs) == (_window_size / _step_size)
                    image_paths.append(sequential_imgs)

                    if mode_enc is True:
                        enc_value_theta = []
                        enc_value_x = []
                        enc_value_y = []

                        for j in range(int(_window_size / _step_size)):
                            if j == 0:
                                # print(int(os.path.basename(sequential_imgs[j]).strip('.jpg')))
                                enc_value_theta.append(torch.tensor(0, dtype=torch.double))
                                enc_value_x.append(torch.tensor(0, dtype=torch.double))
                                enc_value_y.append(torch.tensor(0, dtype=torch.double))
                            else:
                                previous_file_name = int(os.path.basename(sequential_imgs[j - 1]).strip('.jpg'))
                                current_file_name = int(os.path.basename(sequential_imgs[j]).strip('.jpg'))

                                df_part = df[previous_file_name: current_file_name]

                                enc_value_theta.append(torch.tensor(sum(df_part[2]), dtype=torch.double))
                                enc_value_x.append(torch.tensor(sum(df_part[3]), dtype=torch.double))
                                enc_value_y.append(torch.tensor(sum(df_part[4]), dtype=torch.double))

                                # theta.append(sum(df_part[2]))
                                # x.append(sum(df_part[3]))
                                # y.append(sum(df_part[4]))

                        enc_value_theta_full.append(enc_value_theta)
                        enc_value_x_full.append(enc_value_x)
                        enc_value_y_full.append(enc_value_y)

        # print(round(sum(theta) / len(theta), 5))
        # print(round(sum(x) / len(x), 5))
        # print(round(sum(y) / len(y), 5))
        # print('=====')
        #
        # print(round(max(theta), 6))
        # print(round(max(x), 6))
        # print(round(max(y), 6))
        # print('=====')
        #
        # print(round(min(theta), 6))
        # print(round(min(x), 6))
        # print(round(min(y), 6))

        print(len(image_paths))
        # print(len(enc_value_theta_full))
        # print(len(enc_value_x_full))
        # print(len(enc_value_y_full))
        #
        # x = input()

        return image_paths, enc_value_theta_full, enc_value_x_full, enc_value_y_full

    @staticmethod
    def _make_transform(is_train):
        transforms_list = []
        #         transforms_list.append(transforms.Resize((load_size, load_size), Image.BICUBIC))
        #         transforms_list.append(transforms.RandomCrop(fine_size))
        #         if is_train:
        #             transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # [0, 1] => [-1, 1]
        return transforms.Compose(transforms_list)


####################################################################################################################


if __name__ == '__main__':

    batch_size = 2
    window_size = 48
    step_size = 8

    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    print('device {}'.format(device))

    # train_dataset = ActionConditionedLSTMDataset(batch_size, window_size, step_size, is_train=True)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #
    # for batch_idx, data in enumerate(train_loader):
    #     print('===========================================================')
    #     print('batch_idx: ' + str(batch_idx))
    #     # print(data['A'])
    #     print(data['A'].shape)  # torch.Size([4, 8, 3, 128, 256])
    #     print(data['A'][0].shape)  # torch.Size([8, 3, 128, 256])
    #     print(data['path_A'])
    #     print(data['path_A'][0])
    #
    #     print(data['x_A'])
    #     print(len(data['x_A']))
    #     print('===========================================================')
    #
    #     if batch_idx == 0:
    #         x = input()

    # data = iter(train_loader).next()
    # print('===========================================================')
    # print(data['A'].shape)
    # print(data['A'][0].shape)
    # # print(data['B'])
    # print(data['path_A'])  # torch.Size([4, 8, 3, 128, 256])
    # print(data['path_A'][0])  # torch.Size([8, 3, 128, 256])
    # # print(data['path_B'])
    # print('=====================')

    ####################################################################################################################
    # check mask
    ####################################################################################################################

    train_dataset = LSTMDataset(batch_size, window_size, step_size, is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #
    data = iter(train_loader).next()
    print('===========================================================')
    # print(data['A'].shape)
    # print(data['A'][0].shape)
    # print(data['B'])
    # print(data['path_A'])  # torch.Size([4, 8, 3, 128, 256])
    # print(data['path_A'][0])  # torch.Size([8, 3, 128, 256])
    data['A'] = data['A'].to(device)
    print('A: ' + str(data['A'].shape))
    print('===========================================================')

    input_1 = torch.tensor([0.1, 0.1, 0.1], requires_grad=False).to(device)

    real = data['A'].view(int(batch_size * window_size / step_size), 3, 128, 256).to(device)
    real_mask = data['A_mask'].view(int(batch_size * window_size / step_size), 3, 128, 256).to(device)

    print(real.shape)
    print(real_mask.shape)

    # [-1,1] => [0, 1]
    real_A = 0.5 * (real + 1)
    # fake_B = 0.5 * (fake + 1)
    real_A_mask = 0.5 * (real_mask + 1)

    # transpose axis
    real_A = real_A.permute(0, 2, 3, 1)
    # fake_B = fake_B.permute(0, 2, 3, 1)
    real_A_mask = real_A_mask.permute(0, 2, 3, 1)

    target_with_mask = torch.where(real_A_mask[:, :, :] > input_1, real_A_mask * 0, real_A).to(device)
    # fake_with_mask = torch.where(real_A_mask[:, :, :] > input_1, real_A_mask * 0, fake_B).to(self.device)

    # real_A = real_A.permute(0, 3, 1, 2)
    #
    # real_A = real_A.view(int(batch_size), int(window_size / step_size), 3, 128, 256)
    #
    # print(((real_A * 2 - 1) == data['A']).all())

    real_A = real_A[9].to('cpu').detach().clone().numpy() * 255
    real_A_mask = real_A_mask[9].to('cpu').detach().clone().numpy() * 255
    target_with_mask = target_with_mask[9].to('cpu').detach().clone().numpy() * 255

    print(real_A.shape)

    real_A = cv2.cvtColor(real_A, cv2.COLOR_RGB2BGR)
    real_A_mask = cv2.cvtColor(real_A_mask, cv2.COLOR_RGB2BGR)
    target_with_mask = cv2.cvtColor(target_with_mask, cv2.COLOR_RGB2BGR)

    cv2.imwrite('test_A.jpg', real_A)
    cv2.imwrite('test_A_mask.jpg', real_A_mask)
    cv2.imwrite('test_target.jpg', target_with_mask)
