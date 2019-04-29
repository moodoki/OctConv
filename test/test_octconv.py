import unittest
import tensorflow as tf
from octconv import OctConv1D, OctConv2D, OctConv3D

class TestOctConv1D(unittest.TestCase):

    def test_simpledeepnet_creation(self):
        x = tf.placeholder(tf.float32, [None, 1024, 2])
        net_h, net_l = OctConv1D(filters=10, 
                                 kernel_size=3,
                                 padding='same',
                                 alpha = 0.5
                                )(x)
        self.assertTrue(net_h.shape.as_list() == [None, 1024, 5])
        self.assertTrue(net_l.shape.as_list() == [None, 512, 5])

    def test_simpledeepnet_creation_fixedbatch(self):
        x = tf.placeholder(tf.float32, [32, 1024, 2])
        net_h, net_l = OctConv1D(filters=10, 
                                 kernel_size=3,
                                 padding='same',
                                 alpha = 0.5
                                )(x)
        self.assertTrue(net_h.shape, tf.TensorShape([32, 1024, 5]))
        self.assertTrue(net_l.shape, tf.TensorShape([32, 512, 5]))


class TestOctConv2D(unittest.TestCase):

    def test_simpledeepnet_creation(self):
        x = tf.placeholder(tf.float32, [None, 64, 64, 2])
        net_h, net_l = OctConv2D(filters=10, 
                                 kernel_size=3,
                                 padding='same',
                                 alpha = 0.5
                                )(x)
        self.assertTrue(net_h.shape.as_list() == [None, 64, 64, 5])
        self.assertTrue(net_l.shape.as_list() == [None, 32, 32, 5])

    def test_simpledeepnet_creation_fixedbatch(self):
        x = tf.placeholder(tf.float32, [32, 64, 64, 2])
        net_h, net_l = OctConv2D(filters=10, 
                                 kernel_size=3,
                                 padding='same',
                                 alpha = 0.5
                                )(x)
        self.assertTrue(net_h.shape, tf.TensorShape([32, 64, 64, 5]))
        self.assertTrue(net_l.shape, tf.TensorShape([32, 32, 32, 5]))

class TestOctConv3D(unittest.TestCase):

    def test_simpledeepnet_creation(self):
        x = tf.placeholder(tf.float32, [None, 64, 64, 64, 2])
        net_h, net_l = OctConv3D(filters=10, 
                                 kernel_size=3,
                                 padding='same',
                                 alpha = 0.5
                                )(x)
        self.assertTrue(net_h.shape.as_list() == [None, 64, 64, 64, 5])
        self.assertTrue(net_l.shape.as_list() == [None, 32, 32, 32, 5])

    def test_simpledeepnet_creation_fixedbatch(self):
        x = tf.placeholder(tf.float32, [32, 64, 64, 64, 2])
        net_h, net_l = OctConv3D(filters=10, 
                                 kernel_size=3,
                                 padding='same',
                                 alpha = 0.5
                                )(x)
        self.assertTrue(net_h.shape, tf.TensorShape([32, 64, 64, 64, 5]))
        self.assertTrue(net_l.shape, tf.TensorShape([32, 32, 32, 32, 5]))

