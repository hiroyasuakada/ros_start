import os
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

##################################################################
from dataset import LSTMDataset
from model_base_2 import LSTMGenerator_A, LSTMGenerator_B, \
    LSTMDiscriminator_A, LSTMDiscriminator_B, \
    individualDiscriminator_A, individualDiscriminator_B
from model_lstm_cyclegan import LSTMCycleGAN
##################################################################


def test(log_dir, device, lr, beta1, lambda_idt, lambda_A, lambda_B, lambda_mask,
         batch_size_test, window_size, step_size,
         num_epoch, num_epoch_resume, save_epoch_freq, test_loader, epoch_label):
    model = LSTMCycleGAN(log_dir=log_dir, device=device, lr=lr, beta1=beta1, lambda_idt=lambda_idt,
                         lambda_A=lambda_A, lambda_B=lambda_B, lambda_mask=lambda_mask,
                         batch_size=batch_size_test, window_size=window_size, step_size=step_size, mode_train=False)
    model.log_dir = log_dir
    model.load(epoch_label)

    time_list = []

    for batch_idx, data in enumerate(test_loader):
        t1 = time.perf_counter()

        # generate images
        fake_B = model.netG_A(data['A'].to(device))

        # transpose axis
        # real_A = data['A'].permute(0, 1, 3, 4, 2)
        fake_B = fake_B.data.permute(0, 1, 3, 4, 2)

        # [-1,1] => [0, 1]
        # real_A = 0.5 * (real_A + 1) * 255
        fake_B = 0.5 * (fake_B + 1) * 255

        # tensor to array
        device2 = torch.device('cpu')
        # real_A = real_A.to(device2)
        # real_A = real_A.detach().clone().numpy()
        fake_B = fake_B.to(device2)
        fake_B = fake_B.detach().clone().numpy()

        # if not os.path.exists('./{}/real_A'.format(log_dir)):
        #     os.mkdir('./{}/real_A'.format(log_dir))
        if not os.path.exists('./{}/fake_B'.format(log_dir)):
            os.mkdir('./{}/fake_B'.format(log_dir))

        for i in range(fake_B.shape[0]):
            file_name_all = data['path_A'][i].split('-')
            
            for s in range(fake_B.shape[1]):
                folder_name = file_name_all[int(window_size / step_size) - 1].split('/')[3]

                # if not os.path.exists('./{}/real_A/{}'.format(log_dir, folder_name)):
                #     os.mkdir('./{}/real_A/{}'.format(log_dir, folder_name))
                if not os.path.exists('./{}/fake_B/{}'.format(log_dir, folder_name)):
                    os.mkdir('./{}/fake_B/{}'.format(log_dir, folder_name))

                file_name_last = file_name_all[int(window_size / step_size) - 1].split('/')[4]

                # if not os.path.exists('./{}/real_A/{}/{}'.format(log_dir, folder_name, file_name_last)):
                #     os.mkdir('./{}/real_A/{}/{}'.format(log_dir, folder_name, file_name_last))
                if not os.path.exists('./{}/fake_B/{}/{}'.format(log_dir, folder_name, file_name_last)):
                    os.mkdir('./{}/fake_B/{}/{}'.format(log_dir, folder_name, file_name_last))

                file_name = file_name_all[s].split('/')[4]

                print(file_name)

                # save_path_real_A = './{}/real_A/{}/{}/'.format(log_dir, folder_name, file_name_last) + file_name
                save_path_fake_B = './{}/fake_B/{}/{}/'.format(log_dir, folder_name, file_name_last) + file_name

                # real_A_id_i_s = real_A[i][s]
                fake_B_id_i_s = fake_B[i][s]

                # real_A_id_i_s = cv2.cvtColor(real_A_id_i_s, cv2.COLOR_RGB2BGR)
                fake_B_id_i_s = cv2.cvtColor(fake_B_id_i_s, cv2.COLOR_RGB2BGR)

                # cv2.imwrite(save_path_real_A, real_A_id_i_s)
                cv2.imwrite(save_path_fake_B, fake_B_id_i_s)

        t2 = time.perf_counter()
        get_processing_time = t2 - t1
        time_list.append(get_processing_time)

        if batch_idx % 10 == 0:
            print('batch: {} / elapsed_time: {} sec'.format(batch_idx, sum(time_list)))
            time_list = []


if __name__ == '__main__':

    # random seeds
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    # image
    height = 128
    width = 256

    # training details
    window_size = 48
    step_size = 8
    batch_size = 4
    lr = 0.0002  # initial learning rate for adam
    beta1 = 0.5  # momentum term of adam

    num_epoch = 100
    num_epoch_resume = 0
    save_epoch_freq = 5

    # weights of loss function
    lambda_idt = 5
    lambda_A = 10.0
    lambda_B = 10.0
    lambda_mask = 0.0

    # files, dirs
    log_dir = 'logs_lstm_cyclegan'

    # gpu
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    print('device {}'.format(device))

    # test detail ###################################################################
    batch_size_test = 1
    window_size_test = 6
    step_size_test = 1
    #################################################################################

    # dataset
    test_dataset = LSTMDataset(window_size_test, step_size_test, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

    # test
    epoch_label = 'epoch10'

    test(log_dir, device, lr, beta1, lambda_idt, lambda_A, lambda_B, lambda_mask,
         batch_size_test, window_size, step_size,
         num_epoch, num_epoch_resume, save_epoch_freq, test_loader, epoch_label)



