import os
import sys
import argparse
import tensorboard
import numpy as np
import tensorflow as tf
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--data_A', type=str, default='',
                    help='Path to the first set of images.')
parser.add_argument('--data_B', type=str, default='',
                    help='Path to the second set of images.')
parser.add_argument('--direction', type=str, default='AtoB',
                    help='AtoB or BtoA')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Default batch size is 1.')
parser.add_argument('--image_size', type=int, default=256,
                    help='Default image size is 256x256.')
parser.add_argument('--learning_rate', type=float, default=2e-4,
                    help='Default learning rate is 2e-4.')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='Moment term for adam. Default is 0.5')
parser.add_argument('--niter', type=int, default=100,
                    help='# of iterations at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100,
                    help='# of iterations to linearly decay learning rate to zero')
parser.add_argument('--norm_type', type=str, default='instance',
                    help='Either instance norm or batch norm. Default is instance norm.')
parser.add_argument('--lambda_A', type=float, default=10.0,
                    help='weight for forward cycle loss (A -> B -> A)')
parser.add_argument('--lambda_B', type=float, default=10.0,
                    help='weight for backward cycle loss (B -> A -> B)')
parser.add_argument('--ngf', type=int, default=64,
                    help='# of gen filters in first conv layer. Default is 64.')
parser.add_argument('--ndf', type=int, default=64,
                    help='# of discrim filters in first conv layer. Default is 64.')
parser.add_argument('--pool_size', type=int, default=50,
                    help='the size of image buffer that stores previously generated images')
parser.add_argument('--use_lsgan', type=bool, default=True,
                    help='Use least square GAN or vanilla GAN. Default is LSGAN.')
parser.add_argument('load_model', type=str, default=None,
                    help='Load a model to continue training where you left off.')
opt = parser.parse_args()


def train():
    if opt.load_model is not None:
        checkpoint = 'checkpoints/' + opt.load_model
    else:
        checkpoint_name = datetime.now().strftime(%m%d%Y-%H%M)
        checkpoint = 'checkpoints/{}'.format(checkpoint_name)

        try:
            os.makedirs(checkpoint)
        except os.error:
            print("Failed to make new checkpoint directory.")
            sys.exit(1)


    with tf.Session() as sess:
        if opt.load_model is not None:
            pass
        else:
            sess.run(tf.global_variables_initializer())
            step = 0


if __name__ == '__main__':
    train()
