import numpy as np
import tensorflow as tf

class Generator():
    def __init__(self, opt, is_training, name=None):
        self.opt = opt
        self.is_training = is_training
        self.name = name

    def __call__(self):
        pass
