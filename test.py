import os
import sys
import utils
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from CycleGAN import CycleGAN
from DataLoader import DataLoader


parser = argparse.ArgumentParser()
# dataset arguments
parser.add_argument('--data_A', type=str, default='./data/apple2orange/testA',
                    help='Path to the first set of images.')
parser.add_argument('--data_B', type=str, default='./data/apple2orange/testB',
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
parser.add_argument('--preprocess', type=str, default=None,
                    help='Augmentation to be performed when loading in an image. [resize_and_crop | crop | scale_width | scale_width_and_crop | None]')
parser.add_argument('--flip', type=bool, default=False,
                    help='Flip images during augmentation.')
# testing and model arguments
parser.add_argument('--norm_type', type=str, default='instance',
                    help='Type of normalization. [instance | batch]')
parser.add_argument('--init_type', type=str, default='normal',
                    help='Type of initialization of weights [normal | xavier | orthogonal]')
parser.add_argument('--init_gain', type=int, default=0.02,
                    help='Scaling factor for normal, xavier, and orthogonal')
parser.add_argument('--dropout', type=bool, default=False,
                    help='Whether or not to include dropout in generator')
parser.add_argument('--netG', type=str, default='resnet_9blocks',
                    help='Specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
parser.add_argument('--ngf', type=int, default=64,
                    help='# of gen filters in first conv layer. Default is 64.')
parser.add_argument('--load_model', type=str, default=None,
                    help='Load a model to test generating samples with.')
parser.add_argument('--num_samples', type=int, default=32,
                    help='Number of samples you would like to generate.')
parser.add_argument('--sample_directoy', type=str, default='./samples/apple2orange/',
                    help='Directory in which samples will be saved to.')
opt = parser.parse_args()


def test():
    if opt.load_model is not None:
        checkpoint = 'checkpoints/' + opt.load_model
    else:
        print("Must load in a model to test on.")
        sys.exit(1)

    # create an iterator for datasets
    dataloader = iter(DataLoader(opt))

    graph = tf.Graph()
    with graph.as_default():
        # build CycleGAN graph
        cyclegan = CycleGAN(opt, is_training=False)
        cyclegan.build_model()

        # get real and fake data
        fakeA, fakeB =  cyclegan.generate()

        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        # get latest checkpoint of loaded model
        ckpt = tf.train.latest_checkpoint(checkpoint)
        saver.restore(sess, ckpt)

        if opt.direction is 'AtoB': # map image from Domain A to Domain B
            samples_dir = os.path.expanduser(os.path.join(opt.sample_directoy, opt.direction))
            fakeImg = fakeB
        elif opt.direction is 'BtoA': # map image from Domain B to Domain A
            samples_dir = os.path.expanduser(os.path.join(opt.sample_directoy, opt.direction))
            fakeImg = fakeA
        else:
            print('Invalid direction')
            sys.exit(1)

        # generate new images and save them to the `samples` directory
        for idx in range(opt.num_samples):
            realA, realB = next(dataloader)

            generated_image = sess.run(fakeImg, feed_dict={cyclegan.realA: realA,
                                                           cyclegan.realB: realB})

            image_name = 'sample' + str(idx) + '.jpg'
            utils.save_image(generated_image, os.path.join(samples_dir, image_name))


if __name__ == '__main__':
    test()
