import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras import optimizers

from octconv import OctConv2D

class TestKerasTrain(unittest.TestCase):

    def test_simple_mnist(self):
        #Dataset
        (trainx, trainy), (testx, testy) = mnist.load_data()

        trainx = trainx[:, :, :, np.newaxis]
        testx = testx[:, :, :, np.newaxis]
        trainy = utils.to_categorical(trainy)
        testy = utils.to_categorical(testy)
        
        train_gen = ImageDataGenerator(rescale=1/255,
                                       height_shift_range=0.2,
                                       width_shift_range=0.2,
                                      )
        test_gen = ImageDataGenerator(rescale=1/255)

        #Model
        inputs = layers.Input((28, 28, 1))

        net = OctConv2D(16, 3, padding='same', activation='relu')(inputs)
        net = OctConv2D(32, 3, padding='same', activation='relu', strides=2)(net)
        net = OctConv2D(64, 3, padding='same', activation='relu')(net)
        net = layers.Concatenate()([layers.AveragePooling2D(2)(net[0]), net[1]])
        net = layers.Conv2D(64, 1, activation='relu')(net)
        net = layers.Flatten()(net)
        net = layers.Dense(10, activation='softmax')(net)

        model = models.Model(inputs, net)
        model.summary()

        model.compile(optimizer=optimizers.SGD(lr=0.01, decay = 1e-6, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['acc'],
                     )

        model.fit_generator(train_gen.flow(trainx, trainy, batch_size=16, shuffle=True),
                            steps_per_epoch=100,
                            validation_data=test_gen.flow(testx, testy, batch_size=16, shuffle=False),
                            validation_steps=50,
                            epochs=5)
