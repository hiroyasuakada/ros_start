import os
import numpy as np
from tensorflow import keras
import PIL
from matplotlib import pyplot as plt

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.utils.np_utils import to_categorical
from conditional_dcgan import ConditionalDCGAN


# Normalize image from [0, 255] to [-1, 1]
def normalize(X):
    return (X - 127.5) / 127.5


# Denormalize from [-1, 1] to [0, 255]
def denormalize(X):
    return (X + 1.0) * 127.5


def train(latent_dim, height, width, channels, num_class):
    # load mnist dataset
    (X_train, Y_train), (_, _) = keras.datasets.mnist.load_data()
    Y_train = to_categorical(Y_train, num_class)  # convert data into one-hot vectors
    # X_train = X_train.reshape((X_train.shape[0], ) + (height, width, channels)).astype('float32')
    X_train = X_train.astype('float32')
    X_train = normalize(X_train)
    X_train = X_train[:, :, :, None]

    epochs = 15  # 50
    batch_size = 128
    iterations = int(X_train.shape[0] // batch_size)
    dcgan = ConditionalDCGAN(latent_dim, height, width, channels, num_class)
    discriminator_loss_for_epoch = []
    generator_loss_for_epoch = []

    for epoch in range(epochs):
        discriminator_loss_for_iteration = []
        generator_loss_for_iteration = []
        conditions = None

        for iteration in range(iterations):
            real_images = X_train[iteration * batch_size: (iteration + 1) * batch_size]
            conditions = Y_train[iteration * batch_size: (iteration + 1) * batch_size]
            d_loss, g_loss = dcgan.train(real_images, conditions, batch_size)
            discriminator_loss_for_iteration.append(d_loss)
            generator_loss_for_iteration.append(g_loss)

            # show the progress of learning based on iteration
            if (iteration + 1) % iterations == 0:
                print('{} / {}'.format(iteration + 1, iterations))
                print('discriminator loss: {:.2f}'.format(d_loss))
                print('generator loss: {:.2f}'.format(g_loss))
                print()

        discriminator_loss_for_epoch.append(np.mean(discriminator_loss_for_iteration))
        generator_loss_for_epoch.append(np.mean(generator_loss_for_iteration))
        print('epoch' + str(epoch))
        print('discriminator loss: {:.2f}'.format(discriminator_loss_for_epoch[epoch]))
        print('generator loss: {:.2f}'.format(generator_loss_for_epoch[epoch]))

        with open('loss.txt', 'a') as f:
            f.write(str(discriminator_loss_for_epoch) + ',' + str(generator_loss_for_epoch) + '\r')

        if epoch % 5 == 0:
            dcgan.save_weights('gan' + '_epoch' + str(epoch) + '.h5')
            random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
            generated_images = dcgan.predict(random_latent_vectors, conditions)
            # save generated images
            for i, generated_image in enumerate(generated_images):
                img = denormalize(generated_image)
                img = image.array_to_img(img, scale=False)
                condition = np.argmax(conditions[1])
                img.save(os.path.join('generated_images', str(epoch) + '_' + str(condition) + '.png'))
        # print('epoch' + str(epoch) + ' end')
        print()

    plt.figure()
    plt.plot(range(epochs), discriminator_loss_for_epoch, 'b', label='discriminator loss')
    plt.plot(range(epochs), generator_loss_for_epoch, 'g', label='generator loss')
    plt.title('generator and discriminator losses')
    plt.legend()
    plt.show()


def predict(latent_dim, height, width, channels, num_class):
    dcgan = ConditionalDCGAN(latent_dim, height, width, channels, num_class)
    dcgan.load_weights('gan_epoch30.h5')  # load weights after 50 times learning
    # dcgan.load_weights('gan_epoch2.h5')  # load weights after 1 times learning
    for num in range(num_class):
        for id in range(10):
            random_latent_vectors = np.random.normal(size=(1, latent_dim))
            # create conditions like [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] = 2
            condition = np.zeros((1, num_class), dtype=np.float32)
            condition[0, num] = 1
            generated_images = dcgan.predict(random_latent_vectors, condition)
            img = image.array_to_img(denormalize(generated_images[0]), scale=False)
            img.save(os.path.join('generated_images', str(num) + '_' + str(id) + '.png'))


if __name__ == '__main__':
    _latent_dim = 100
    _height = 28
    _width = 28
    _channels = 1
    _num_class = 10
    train(_latent_dim, _height, _width, _channels, _num_class)
    predict(_latent_dim, _height, _width, _channels, _num_class)









