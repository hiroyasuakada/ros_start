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

from model_base import Generator, Discriminator, LSTMGenerator_A, LSTMGenerator_B, \
    Conv3dGenerator, Conv3dDiscriminator, FrameDiscriminator, Conv2dConv3dGenerator


class Conv3dCycleGAN(object):

    def __init__(self, log_dir='logs_conv3d_cyclegan', device='cuda:0', lr=0.0002, beta1=0.5, lambda_idt=5, lambda_A=10.0,
                 lambda_B=10.0, lambda_mask=10.0, batch_size=4, window_size=48, step_size=8, mode_train=True):
        self.lr = lr
        self.beta1 = beta1
        self.device = device
        self.batch_size = batch_size
        self.window_size = window_size
        self.step_size = step_size
        print('loss_mask: {}'.format(lambda_mask))
        if mode_train:
            self.gpu_ids = [0, 1]  # 0, 1, 2
        else:
            self.gpu_ids = [0]

        self.netG_A = Conv2dConv3dGenerator(self.batch_size, self.window_size, self.step_size).to(self.device)
        self.netG_B = Conv2dConv3dGenerator(self.batch_size, self.window_size, self.step_size).to(self.device)
        self.netD_A_seq = Conv3dDiscriminator().to(self.device)
        self.netD_B_seq = Conv3dDiscriminator().to(self.device)
        self.netD_A_frame = FrameDiscriminator(self.batch_size, self.window_size, self.step_size).to(self.device)
        self.netD_B_frame = FrameDiscriminator(self.batch_size, self.window_size, self.step_size).to(self.device)

        print(torch.cuda.is_available())

        # multi-GPUs
        self.netG_A = torch.nn.DataParallel(self.netG_A, self.gpu_ids)
        self.netG_B = torch.nn.DataParallel(self.netG_B, self.gpu_ids)
        self.netD_A_seq = torch.nn.DataParallel(self.netD_A_seq, self.gpu_ids)
        self.netD_B_seq = torch.nn.DataParallel(self.netD_B_seq, self.gpu_ids)
        self.netD_A_frame = torch.nn.DataParallel(self.netD_A_frame, self.gpu_ids)
        self.netD_B_frame = torch.nn.DataParallel(self.netD_B_frame, self.gpu_ids)

        self.seq_fake_A_pool = ImagePool(50)
        self.seq_fake_B_pool = ImagePool(50)

        # targetが本物か偽物かで代わるのでオリジナルのGANLossクラスを作成
        self.criterionGAN = GANLoss(self.device)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionMask = MASKLoss(self.device)

        # weights of loss function
        self.lambda_idt = lambda_idt
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_seq_A = lambda_A  # ########################################### 暫定
        self.lambda_seq_B = lambda_B  # ########################################### 暫定
        self.lambda_mask = lambda_mask

        # Generatorは2つのパラメータを同時に更新
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
            lr=self.lr,
            betas=(self.beta1, 0.999))
        self.optimizer_D_A_seq = torch.optim.Adam(self.netD_A_seq.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizer_D_B_seq = torch.optim.Adam(self.netD_B_seq.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizer_D_A_frame = torch.optim.Adam(self.netD_A_frame.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizer_D_B_frame = torch.optim.Adam(self.netD_B_frame.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D_A_seq)
        self.optimizers.append(self.optimizer_D_B_seq)
        self.optimizers.append(self.optimizer_D_A_frame)
        self.optimizers.append(self.optimizer_D_B_frame)

        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def set_input(self, input):
        # self.real_A = input['A']
        # self.real_B = input['B']
        # self.real_A_mask = input['A_mask']

        self.seq_real_A = input['A'].to(self.device)
        self.seq_real_B = input['B'].to(self.device)
        self.seq_real_A_mask = input['A_mask'].to(self.device)

    def backward_G(self, seq_real_A, seq_real_B, seq_real_A_mask):  # seq_real_A_mask
        # LSTMGeneratorに関連するlossと勾配計算処理
        ################################################################################################################
        # at sequence level

        # sequence loss D_A(G_A(A))
        seq_fake_B = self.netG_A(seq_real_A)

        pred_seq_fake_B = self.netD_A_seq(seq_fake_B)
        loss_G_A_seq = self.criterionGAN(pred_seq_fake_B, True)

        # sequence loss D_B(G_A(B))
        seq_fake_A = self.netG_B(seq_real_B)

        pred_seq_fake_A = self.netD_B_seq(seq_fake_A)
        loss_G_B_seq = self.criterionGAN(pred_seq_fake_A, True)

        # forward sequence cycle loss
        # seq_real_A = > seq_fake_B = > seq_rec_Aが元のseq_real_Aに近いほどよい
        seq_rec_A = self.netG_B(seq_fake_B)
        loss_cycle_A_seq = self.criterionCycle(seq_rec_A, seq_real_A) * self.lambda_seq_A

        # backward sequence cycle loss
        seq_rec_B = self.netG_A(seq_fake_A)
        loss_cycle_B_seq = self.criterionCycle(seq_rec_B, seq_real_B) * self.lambda_seq_B

        # sequence identity loss
        # TODO: idt_Aの命名はよくない気がする idt_Bの方が適切では？
        seq_idt_A = self.netG_A(seq_real_B)
        loss_idt_A_seq = self.criterionIdt(seq_idt_A, seq_real_B) * self.lambda_idt

        seq_idt_B = self.netG_B(seq_real_A)
        loss_idt_B_seq = self.criterionIdt(seq_idt_B, seq_real_A) * self.lambda_idt

        ################################################################################################################
        # at frame level

        # sequence to individuals

        pred_fake_B = self.netD_A_frame(seq_fake_B)
        loss_G_A_frame = self.criterionGAN(pred_fake_B, True)

        pred_fake_A = self.netD_B_frame(seq_fake_A)
        loss_G_B_frame = self.criterionGAN(pred_fake_A, True)

        ################################################################################################################
        # # mse for mase as a new loss function / [int(batch_size), int(window_size / step_size), 3, 128, 256]

        if self.lambda_mask == 0:
            loss_mask = torch.tensor(0).to(self.device)
        else:
            real_A = seq_real_A.view(int(self.batch_size * self.window_size / self.step_size), 3, 128, 256)
            fake_B = seq_fake_B.view(int(self.batch_size * self.window_size / self.step_size), 3, 128, 256)
            real_A_mask = seq_real_A_mask.view(int(self.batch_size * self.window_size / self.step_size), 3, 128, 256)

            loss_mask = self.criterionMask(real_A, fake_B, real_A_mask) * self.lambda_mask

        ################################################################################################################
        # combined loss
        loss_G = loss_G_A_seq + loss_G_B_seq + loss_cycle_A_seq + loss_cycle_B_seq + \
                 loss_idt_A_seq + loss_idt_B_seq + loss_G_A_frame + loss_G_B_frame + loss_mask  # + loss_mask
        loss_G.backward()

        # 次のDiscriminatorの更新でfake画像が必要なので一緒に返す
        return loss_G_A_seq.data, loss_G_B_seq.data, loss_cycle_A_seq.data, loss_cycle_B_seq.data, \
               loss_idt_A_seq.data, loss_idt_B_seq.data, loss_G_A_frame.data, loss_G_A_frame.data, \
               seq_fake_A.data, seq_fake_B.data, loss_mask.data  # loss_mask.data

    def backward_D_A(self, seq_real_B, seq_fake_B):
        # ドメインAから生成したfake_Bが本物か偽物か見分ける
        ################################################################################################################
        # at sequence level

        # fake_Bを直接使わずに過去に生成した偽画像から新しくランダムサンプリングしている？
        seq_fake_B = self.seq_fake_B_pool.query(seq_fake_B)

        # 本物画像を入れたときは本物と認識するほうがよい
        pred_seq_real = self.netD_A_seq(seq_real_B)
        loss_D_seq_real = self.criterionGAN(pred_seq_real, True)

        # ドメインAから生成した偽物画像を入れたときは偽物と認識するほうがよい
        # fake_Bを生成したGeneratorまで勾配が伝搬しないようにdetach()する
        pred_seq_fake = self.netD_A_seq(seq_fake_B.detach())
        loss_D_seq_fake = self.criterionGAN(pred_seq_fake, False)

        # combined loss
        loss_D_A_seq = (loss_D_seq_real + loss_D_seq_fake) * 0.5

        ################################################################################################################
        # at frame level

        # 本物画像を入れたときは本物と認識するほうがよい  sequence to individuals
        pred_real = self.netD_A_frame(seq_real_B)
        loss_D_real = self.criterionGAN(pred_real, True)

        # ドメインAから生成した偽物画像を入れたときは偽物と認識するほうがよい
        # fake_Bを生成したGeneratorまで勾配が伝搬しないようにdetach()する
        pred_fake = self.netD_A_frame(seq_fake_B.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # combined loss
        loss_D_A_frame = (loss_D_real + loss_D_fake) * 0.5

        ################################################################################################################
        # combined all losses
        loss_D_A_all = loss_D_A_seq + loss_D_A_frame
        loss_D_A_all.backward()

        return loss_D_A_seq.data, loss_D_A_frame.data

    def backward_D_B(self, seq_real_A, seq_fake_A):
        # ドメインBから生成したfake_Aが本物か偽物か見分ける
        ################################################################################################################
        # at sequence level

        seq_fake_A = self.seq_fake_A_pool.query(seq_fake_A)

        # 本物画像を入れたときは本物と認識するほうがよい
        pred_seq_real = self.netD_B_seq(seq_real_A)
        loss_D_seq_real = self.criterionGAN(pred_seq_real, True)

        # 偽物画像を入れたときは偽物と認識するほうがよいdpkg エラー 修復
        pred_seq_fake = self.netD_B_seq(seq_fake_A.detach())
        loss_D_seq_fake = self.criterionGAN(pred_seq_fake, False)

        # combined loss
        loss_D_B_seq = (loss_D_seq_real + loss_D_seq_fake) * 0.5

        ################################################################################################################
        # at frame level

        # 本物画像を入れたときは本物と認識するほうがよい
        pred_real = self.netD_B_frame(seq_real_A)
        loss_D_real = self.criterionGAN(pred_real, True)

        # 偽物画像を入れたときは偽物と認識するほうがよいdpkg エラー 修復
        pred_fake = self.netD_B_frame(seq_fake_A.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # combined loss
        loss_D_B_frame = (loss_D_real + loss_D_fake) * 0.5

        ################################################################################################################
        # combined all losses
        loss_D_B_all = loss_D_B_seq + loss_D_B_frame
        loss_D_B_all.backward()

        return loss_D_B_seq.data, loss_D_B_frame.data

    def optimize(self):

        # update Generator (G_A and G_B)
        self.optimizer_G.zero_grad()
        loss_G_A_seq, loss_G_B_seq, loss_cycle_A_seq, loss_cycle_B_seq, \
        loss_idt_A_seq, loss_idt_B_seq, loss_G_A_frame, loss_G_B_frame, \
        seq_fake_A, seq_fake_B, loss_mask \
            = self.backward_G(self.seq_real_A, self.seq_real_B, self.seq_real_A_mask)  # loss_mask & # self.seq_real_A_mask
        self.optimizer_G.step()

        # update D_A
        self.optimizer_D_A_seq.zero_grad()
        loss_D_A_seq, loss_D_A_frame = self.backward_D_A(self.seq_real_B, seq_fake_B)
        self.optimizer_D_A_seq.step()

        # update D_B
        self.optimizer_D_B_seq.zero_grad()
        loss_D_B_seq, loss_D_B_frame = self.backward_D_B(self.seq_real_A, seq_fake_A)
        self.optimizer_D_B_seq.step()

        ret_loss = [loss_G_A_seq, loss_G_B_seq,
                    loss_cycle_A_seq, loss_cycle_B_seq,
                    loss_idt_A_seq, loss_idt_B_seq,
                    loss_G_A_frame, loss_G_B_frame,

                    loss_D_A_seq, loss_D_B_seq,
                    loss_D_A_frame, loss_D_B_frame,

                    loss_mask
                    ]  # loss_mask

        return np.array(ret_loss)

    def train(self, data_loader):
        running_loss = np.array([0.0, 0.0,
                                 0.0, 0.0,
                                 0.0, 0.0,
                                 0.0, 0.0,
                                 0.0, 0.0,
                                 0.0, 0.0,
                                 0.0
                                 ])

        time_list = []
        for batch_idx, data in enumerate(data_loader):

            t1 = time.perf_counter()
            self.set_input(data)
            losses = self.optimize()
            losses = losses.astype(np.float32)
            running_loss += losses

            t2 = time.perf_counter()
            get_processing_time = t2 - t1
            time_list.append(get_processing_time)

            if batch_idx % 500 == 0:
                print('batch: {} / elapsed_time: {} sec'.format(batch_idx, sum(time_list)))
                time_list = []

        running_loss /= len(data_loader)
        return running_loss

    def save_network(self, network, network_label, epoch_label):
        save_filename = '{}_net_{}.pth'.format(epoch_label, network_label)
        save_path = os.path.join(self.log_dir, save_filename)

        # GPUで動いている場合はCPUに移してから保存
        # これやっておけばCPUでモデルをロードしやすくなる？
        torch.save(network.cpu().state_dict(), save_path)
        # GPUに戻す
        network.to(self.device)

        #         torch.save({'epoch': epoch_label,
        #                     'model_state_dict': network.cpu().state_dict(),
        #                     'optimizer_state_dict': optimizer.cpu().state_dict(),
        #                     'loss': loss}, save_path)
        #         # GPUに戻す
        #         network.to(device)

        # if network is self.netG_A or network is self.netD_A_seq or network is self.netD_A_frame:
        #     network.to('cuda:0')
        # else:
        #     network.to('cuda:1')

    def load_network(self, network, network_label, epoch_label):
        load_filename = '{}_net_{}.pth'.format(epoch_label, network_label)
        load_path = os.path.join(self.log_dir, load_filename)
        network.load_state_dict(torch.load(load_path))

    #         network = torch.nn.DataParallel(network, self.gpu_ids)

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label)
        self.save_network(self.netD_A_seq, 'D_A_seq', label)
        self.save_network(self.netD_A_frame, 'D_A_frame', label)
        self.save_network(self.netG_B, 'G_B', label)
        self.save_network(self.netD_B_seq, 'D_B_seq', label)
        self.save_network(self.netD_B_frame, 'D_B_frame', label)

    def load(self, label):
        self.load_network(self.netG_A, 'G_A', label)
        self.load_network(self.netD_A_seq, 'D_A_seq', label)
        self.load_network(self.netD_A_frame, 'D_A_frame', label)
        self.load_network(self.netG_B, 'G_B', label)
        self.load_network(self.netD_B_seq, 'D_B_seq', label)
        self.load_network(self.netD_B_frame, 'D_B_frame', label)


class ImagePool(object):
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        # プールを使わないときはそのまま返す
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            # バッチの次元を削除して3Dテンソルに
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


class GANLoss(nn.Module):

    def __init__(self, device):
        super(GANLoss, self).__init__()
        self.device = device
        self.real_label_var = None
        self.fake_label_var = None
        self.loss = nn.MSELoss()

    #         self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    #         self.cuda = torch.cuda.is_available()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            # 高速化のため？
            # varがNoneのままか形状が違うときに作り直す
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = torch.ones(input.size()).to(self.device)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = torch.zeros(input.size()).to(self.device)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class MASKLoss(nn.Module):
    def __init__(self, device):
        super(MASKLoss, self).__init__()
        self.device = device
        self.loss = nn.MSELoss()

    def get_img_with_mask(self, real, fake, real_mask):
        input_1 = torch.tensor([0.1, 0.1, 0.1], requires_grad=False).to(self.device)

        # [-1,1] => [0, 1]
        real_A = 0.5 * (real + 1)
        fake_B = 0.5 * (fake + 1)
        real_A_mask = 0.5 * (real_mask + 1)

        # transpose axis
        real_A = real_A.permute(0, 2, 3, 1)
        fake_B = fake_B.permute(0, 2, 3, 1)
        real_A_mask = real_A_mask.permute(0, 2, 3, 1)

        target_with_mask = torch.where(real_A_mask[:, :, :] > input_1, real_A_mask * 0, real_A).to(self.device)
        fake_with_mask = torch.where(real_A_mask[:, :, :] > input_1, real_A_mask * 0, fake_B).to(self.device)

        return fake_with_mask, target_with_mask

    def __call__(self, real, fake, real_mask):
        fake_with_mask, target_with_mask = self.get_img_with_mask(real, fake, real_mask)
        return self.loss(fake_with_mask, target_with_mask)
