import numpy as np
import matplotlib.pyplot as plt
import os

from keras.datasets import mnist
# from keras.utils import np_utils  # to_categorical

from skimage import transform


class DataStore:
    def __init__(self):
        # read the original MNIST data: test and train

        (self.X_train, self.y_train), (self.X_test, self.y_test) =\
            mnist.load_data()

        # obtain the data for degraded resolution

        print('self.X_train.dtype:', self.X_train.dtype,
              'self.X_train.shape:', self.X_train.shape)

        self.train28x28 = self.X_train  # alias
        self.train14x14 = None
        self.train7x7 = None
        self.test28x28 = self.X_test    # alias
        self.test14x14 = None
        self.test7x7 = None

        self.fname_train14x14 = 'train14x14.npy'
        self.fname_train7x7 = 'train7x7.npy'

        self.fname_test14x14 = 'test14x14.npy'
        self.fname_test7x7 = 'test7x7.npy'

        if (os.path.exists(self.fname_train14x14) and
                os.path.exists(self.fname_train7x7) and
                os.path.exists(self.fname_test14x14) and
                os.path.exists(self.fname_test7x7)):
            print('--> read degraded resolution pictures from disk')

            self.train14x14 = np.load(self.fname_train14x14)
            self.train7x7 = np.load(self.fname_train7x7)
            self.test14x14 = np.load(self.fname_test14x14)
            self.test7x7 = np.load(self.fname_test7x7)
        else:
            print(
                '--> create degraded resolution pictures')

            self.train14x14 = np.empty((self.X_train.shape[0], 14, 14),
                                  dtype=self.X_train.dtype)
            self.train7x7 = np.empty((self.X_train.shape[0], 7, 7),
                                     dtype=self.X_train.dtype)

            for ipic in range(self.X_train.shape[0]):
                self.train14x14[ipic, :, :] =\
                    transform.resize(self.X_train[ipic],
                                     (14, 14),
                                     mode='constant',
                                     preserve_range=True)
                self.train7x7[ipic, :, :] =\
                    transform.resize(self.train14x14[ipic],
                                     (7, 7),
                                     mode='constant',
                                     preserve_range=True)

            self.test14x14 = np.empty((self.X_test.shape[0], 14, 14),
                                      dtype=self.X_test.dtype)
            self.test7x7 = np.empty((self.X_test.shape[0], 7, 7),
                                    dtype=self.X_test.dtype)

            for ipic in range(self.X_test.shape[0]):
                self.test14x14[ipic, :, :] =\
                    transform.resize(self.X_test[ipic],
                                     (14, 14),
                                     mode='constant',
                                     preserve_range=True)
                self.test7x7[ipic, :, :] =\
                    transform.resize(self.test14x14[ipic],
                                     (7, 7),
                                     mode='constant',
                                     preserve_range=True)

            np.save(self.fname_train14x14, self.train14x14)
            np.save(self.fname_train7x7, self.train7x7)
            np.save(self.fname_test14x14, self.test14x14)
            np.save(self.fname_test7x7, self.test7x7)

    def plotTrain(self):
        """Plots the first three digits from the train set
        """
        N = 3
        print('plot', N, 'pictures')

        fig_res = plt.figure()
        for ipic in range(N):
            plt.subplot(3, N, 0 * N + ipic + 1)
            plt.tight_layout()
            plt.imshow(self.train28x28[ipic], cmap='gray', interpolation='none')
            plt.title("true digit: {}".format(self.y_train[ipic]))

        for ipic in range(N):
            plt.subplot(3, N, 1 * N + ipic + 1)
            plt.tight_layout()
            plt.imshow(self.train14x14[ipic], cmap='gray', interpolation='none')
            plt.title("true digit: {}".format(self.y_train[ipic]))

        for ipic in range(N):
            plt.subplot(3, N, 2 * N + ipic + 1)
            plt.tight_layout()
            plt.imshow(self.train7x7[ipic], cmap='gray', interpolation='none')
            plt.title("true digit: {}".format(self.y_train[ipic]))

        plt.show()
