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

        print('path_A')
        print(path_A)
        print('path_A')

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

        # return {'A': A, 'B': B, 'A_mask': A_mask}

        return {'A': A, 'B': B, 'A_mask': A_mask,
                'path_A': [path_A], 'path_B': path_B, 'path_A_mask': path_A_mask}

    def __len__(self):
        return max(self.size_A, self.size_B, self.size_A_mask)

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
    def __init__(self, _window_size, _step_size, is_train, is_condition):
        super(LSTMDataset, self).__init__()
        self.window_size = _window_size
        self.step_size = _step_size

        # make path to data
        root_dir = os.path.join('128_256', 'without_mask_4_situation_by_csv_lstm')

        if is_train:
            dir_A = os.path.join(root_dir, 'trainA')
            dir_B = os.path.join(root_dir, 'trainB')
            dir_A_mask = os.path.join(root_dir, 'tracking_binary_mask')

            if is_condition:
                dir_enc = os.path.join(root_dir, 'enc_theta_dx_of_1_csv')

        else:
            dir_A = os.path.join(root_dir, 'test_quantitative_by_hand')
            dir_B = os.path.join(root_dir, 'test_quantitative_by_hand')  # has no effect
            dir_A_mask = os.path.join(root_dir, 'test_quantitative_binary_mask')

            if is_condition:
                dir_enc = os.path.join(root_dir, 'enc_theta_dx_of_1_csv')

        _, self.image_paths_A = self._make_dataset(dir_A, self.window_size, self.step_size)
        _, self.image_paths_B = self._make_dataset(dir_B, self.window_size, self.step_size)
        _, self.image_paths_A_mask = self._make_dataset(dir_A_mask, self.window_size, self.step_size)

        self.num_img = self.window_size / self.step_size

        self.size_A = len(self.image_paths_A)
        self.size_B = len(self.image_paths_B)
        self.size_A_mask = len(self.image_paths_A_mask)

        self.transform = self._make_transform(is_train)

    def __len__(self):
        return max(self.size_A, self.size_B, self.size_A_mask)

    def __getitem__(self, index):
        index_A = index % self.size_A
        path_A = self.image_paths_A[index_A]  # at the point, [3, 4, 5]

        print(path_A)
        print('=')
        print(path_A[0])
        print(self.size_A)

        index_A_mask = index % self.size_A_mask
        path_A_mask = self.image_paths_A_mask[index_A_mask]

        index_B = index % self.size_B
        path_B = self.image_paths_B[index_B]
        # index_B = random.randint(0, self.size_B - 1)

        # process the sequential images with transform function
        A = self.get_sequential_imgs(path_A, self.num_img)
        B = self.get_sequential_imgs(path_B, self.num_img)
        A_mask = self.get_sequential_imgs(path_A_mask, self.num_img)

        list_path_A = '-'.join(path_A)
        list_path_B = '-'.join(path_B)
        list_path_A_mask = '-'.join(path_A_mask)

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
        directory_name = []

        for directory in directories:
            files = glob.glob(os.path.join(dir, directory, '*.jpg'))
            files = natsorted(files)
            num_sequential_imgs = (len(files) - _window_size) + 1  # (all - ws) / s + 1
            for i in range(num_sequential_imgs):
                sequential_imgs = files[i: i + _window_size: _step_size]  # 0 ~ 8 (not including No.8)
                image_paths.append(sequential_imgs)
                directory_name.append(directory)

        return directory_name, image_paths

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

    batch_size = 4
    window_size = 48
    step_size = 6
    train_dataset = LSTMDataset(window_size, step_size, is_train=True, is_condition=False)
    # train_dataset = UnalignedDataset(is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # data = iter(train_loader).next()
    # print('===========================================================')
    # print(data['A'].shape)
    # print(data['A'][0].shape)
    # # print(data['B'])
    # print(data['path_A'])  # torch.Size([4, 8, 3, 128, 256])
    # print(data['path_A'][0])  # torch.Size([8, 3, 128, 256])
    # # print(data['path_B'])
    # print('=====================')

    for batch_idx, data in enumerate(train_loader):
        print(str(batch_idx) + '===========================================================')
        print(data['A'].shape)
        print(data['A'][0].shape)
        # print(data['B'])
        print(data['path_A'])  # torch.Size([4, 8, 3, 128, 256])
        print(data['path_A'][0])  # torch.Size([8, 3, 128, 256])
        # print(data['path_B'])
        print('=====================')

        if batch_idx == 0:
            x = input()