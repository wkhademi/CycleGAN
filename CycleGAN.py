import numpy as np
import tensorflow as tf


class CycleGAN():
    def __init__(self, opt, is_training=True):
        self.opt = opt
        self.is_training = is_training

        # create placeholders for real and generated training/test data
        self.realA = tf.placeholder(tf.float32, shape=(None, self.opt.image_size, self.opt.image_size, 3) 'Real Set A')
        self.realB = tf.placeholder(tf.float32, shape=(None, self.opt.image_size, self.opt.image_size, 3) 'Real Set B')
        self.fakeA = tf.placeholder(tf.float32, shape=(None, self.opt.image_size, self.opt.image_size, 3) 'Fake Set A')
        self.fakeB = tf.placeholder(tf.float32, shape=(None, self.opt.image_size, self.opt.image_size, 3) 'Fake Set B')

    def set_input(self):
        pass

    def build_model(self):
        pass

    def G_loss(self):
        pass

    def D_loss(self):
        pass

    def cycle_consistency_loss(self):
        pass
