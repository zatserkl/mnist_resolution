import numpy as np  # np.max
import matplotlib.pyplot as plt
import os

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils    # to_categorical


class ModelRes:
    def __init__(self, resolution, x_train, y_train,
                 x_test, y_test):
        """resolution and data are inherently connected
        """
        self.resolution = resolution

        self.result_dir = 'result'
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

        self.result_fname = 'keras_mnist' + str(self.resolution) + '.h5'
        self.model_path = os.path.join(self.result_dir, self.result_fname)

        # 1) reshape data to (m, n)
        # 2) normalize data
        self.x_train = x_train.reshape(x_train.shape[0], -1) / 255
        self.x_train = self.x_train.astype('float32')
        self.x_test = x_test.reshape(x_test.shape[0], -1) / 255
        self.x_test = self.x_test.astype('float32')

        # encode the labels like 3 --> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        n_classes = 10
        self.y_train = np_utils.to_categorical(y_train, n_classes)
        self.y_test = np_utils.to_categorical(y_test, n_classes)

        # Model: sequential

        # hidden layer, activation: relu
        n_nodes = 256
        self.model = Sequential()
        self.model.add(Dense(n_nodes,
                             input_shape=(self.resolution*self.resolution,)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        # output layer, activation: softmax
        self.model.add(Dense(n_classes))

        """AZ: I got an error here (vesion incompatibility) about extra
        parameter for function softmax I removed 'axis' in return statement
        in def softmax in
        ~/anaconda/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py

        def softmax(x, axis=-1):
            # AZ: omitted docstring here

            # return tf.nn.softmax(x, axis=axis)	# AZ
            return tf.nn.softmax(x)
        """
        self.model.add(Activation('softmax'))

        # compile the model
        self.model.compile(loss='categorical_crossentropy',
                           metrics=['accuracy'], optimizer='adam')

    def train(self):
        """Train the model.
        Currently parameters batch_size and the number of epochs are hardcoded.
        """
        print('.. Training model for resolution {}x{}'.format(
            self.resolution, self.resolution
        ))

        print('self.x_train.shape', self.x_train.shape)
        print('self.y_train.shape', self.y_train.shape)

        history = None
        print('model file:', self.model_path)
        if not os.path.exists(self.model_path):
            # NB: batch_size is irrelevant in our case, use single thread
            history = self.model.fit(self.x_train, self.y_train,
                                     batch_size=128, epochs=10,
                                     verbose=2,
                                     validation_data=(self.x_test, self.y_test))

            self.model.save(self.model_path)
        else:
            print('read the model from the disk')
            self.model = load_model(self.model_path)

        if history:
            # plotting the metrics
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy ' +
                      str(self.resolution) + 'x' + str(self.resolution))
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='lower right')

            plt.subplot(2, 1, 2)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss ' +
                      str(self.resolution) + 'x' + str(self.resolution))

            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper right')

            plt.tight_layout()

        plt.figure()
        predicted_proba = self.model.predict_proba(self.x_test)
        p_max = np.max(predicted_proba, axis=1)
        plt.hist(p_max, bins=40)  # ignore histogram content
        plt.title('test model: max probability, {}x{}'.format(
            self.resolution, self.resolution))

        # # performance on test sample: NB: use self.model
        # mnist_model = load_model(self.model_path)
        #
        # plt.figure()
        # predicted_proba = mnist_model.predict_proba(self.x_test)
        # p_max = np.max(predicted_proba, axis=1)
        # plt.hist(p_max, bins=40)  # ignore histogram content
        # plt.title('test mnist_model: max probability, {}x{}'.format(
        #     self.resolution, self.resolution))

        # plt.figure()
        # predicted_proba = self.model.predict_proba(self.x_train)
        # p_max = np.max(predicted_proba, axis=1)
        # plt.hist(p_max, bins=40)  # ignore histogram content
        # plt.title('train: max probability, {}x{}'.format(
        #     self.resolution, self.resolution))
