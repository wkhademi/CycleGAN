import numpy as np
import tensorflow as tf


class CycleGAN():
    def __init__(self, opt):
        self.opt = opt

        # create placeholders for real and generated training/test data
        self.realA = tf.placeholder(tf.float32, shape=(None, self.opt.image_size, self.opt.image_size, 3) 'Real Set A')
        self.realB = tf.placeholder(tf.float32, shape=(None, self.opt.image_size, self.opt.image_size, 3) 'Real Set B')
        self.fakeA = tf.placeholder(tf.float32, shape=(None, self.opt.image_size, self.opt.image_size, 3) 'Fake Set A')
        self.fakeB = tf.placeholder(tf.float32, shape=(None, self.opt.image_size, self.opt.image_size, 3) 'Fake Set B')


    def build_model():
        pass

    def G_loss():
        pass

    def D_loss():
        pass

    def cycle_consistency_loss():
        pass
