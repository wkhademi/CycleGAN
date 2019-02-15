import os
import sys
import argparse
import tensorboard
import numpy as np
import tensorflow as tf
from datetime import datetime


parser = argparse.ArgumentParser()
# dataset arguments
parser.add_argument('--data_A', type=str, default='',
                    help='Path to the first set of images.')
parser.add_argument('--data_B', type=str, default='',
                    help='Path to the second set of images.')
parser.add_argument('--in_channels', type=int, default=3,
                    help='# of channels for input images')
parser.add_argument('--out_channels', type=int, default=3,
                    help='# of channels for output images')
parser.add_argument('--direction', type=str, default='AtoB',
                    help='AtoB or BtoA')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Default batch size is 1.')
parser.add_argument('--load_size', type=int, default=286,
                    help='Default size to load in an image.')
parser.add_argument('--crop_size', type=int, default=256,
                    help='Size to crop an image to.')
parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                    help='Augmentation to be performed when loading in an image. [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
parser.add_argument('--flip', type=bool, default=True,
                    help='Flip images during augmentation.')
# training and model arguments
parser.add_argument('--learning_rate', type=float, default=2e-4,
                    help='Default learning rate is 2e-4.')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='Moment term for adam. Default is 0.5')
parser.add_argument('--niter', type=int, default=100,
                    help='# of iterations at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100,
                    help='# of iterations to linearly decay learning rate to zero')
parser.add_argument('--lr_policy', type=str, default='linear',
                    help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', type=int, default=50,
                    help='multiply by a gamma every lr_decay_iters iterations')
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
parser.add_argument('--netG', type=str, default='resnet_9blocks',
                    help='Specify generator architecture. [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
parser.add_argument('--pool_size', type=int, default=50,
                    help='the size of image buffer that stores previously generated images')
parser.add_argument('--gan_mode', type=str, default='lsgan',
                    help='Use least square GAN or vanilla GAN. Default is LSGAN.')
parser.add_argument('--load_model', type=str, default=None,
                    help='Load a model to continue training where you left off.')
parser.add_argument('--epoch', type=int, default=1,
                    help='Epoch to start training on in case loading model in')
parser.add_argument('--display_frequency', type=int, default=100,
                    help='The number of epochs to train GAN on before printing loss')
parser.add_argument('--checkpoint_frequency', type=int, default=1000,
                    help='The number of epochs to train GAN on before saving a checkpoint')
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

    CycleGAN = CycleGAN(opt, is_training=True)

    # set the direction data will go [AtoB | BtoA]
    CycleGAN.set_input()

    # build the CycleGAN model
    model = CycleGAN.build_model()

    with tf.Session() as sess:
        if opt.load_model is not None:
            pass
        else:
            sess.run(tf.global_variables_initializer())
            step = 0

        for epoch in range(opt.epoch, opt.niter + opt.niter_decay + 1):
            for batch in range(model.num_batches):
                step += 1

            # display the losses of the Generators and Discriminators
            if step % opt.display_frequency == 0:
                pass

            # save a checkpoint of the model to the /checkpoints directory
            if epoch % opt.checkpoint_frequency == 0:
                pass


if __name__ == '__main__':
    train()
