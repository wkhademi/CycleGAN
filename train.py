import os
import sys
import argparse
import tensorboard
import numpy as np
import tensorflow as tf
from datetime import datetime
from ImagePool import ImagePool
from CycleGAN import CycleGAN


parser = argparse.ArgumentParser()
# dataset arguments
parser.add_argument('--data_A', type=str, default='./data/apple2orange/trainA',
                    help='Path to the first set of images.')
parser.add_argument('--data_B', type=str, default='./data/apple2orange/trainB',
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
parser.add_argument('--niter', type=int, default=100000,
                    help='# of steps at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100000,
                    help='# of steps to linearly decay learning rate to zero')
parser.add_argument('--norm_type', type=str, default='instance',
                    help='Type of normalization. [instance | batch]')
parser.add_argument('--init_type', type=str, default='normal',
                    help='Type of initialization of weights [normal | xavier | orthogonal]')
parser.add_argument('--init_gain', type=float, default=0.2,
                    help='Scaling factor for normal, xavier, and orthogonal')
parser.add_argument('--dropout', type=bool, default=False,
                    help='Whether or not to include dropout in generator')
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
parser.add_argument('--netD', type=str, default='basic',
                    help='Specify discriminator architecture [basic | n_layers | pixel]. Basic model is 70x70 PatchGAN')
parser.add_argument('--n_layers', type=int, default=3,
                    help='# of layers for discriminator. Only used if netD==n_layers')
parser.add_argument('--pool_size', type=int, default=50,
                    help='the size of image buffer that stores previously generated images')
parser.add_argument('--gan_mode', type=str, default='lsgan',
                    help='Use least square GAN or vanilla GAN. Default is LSGAN.')
parser.add_argument('--load_model', type=str, default=None,
                    help='Load a model to continue training where you left off.')
parser.add_argument('--epoch', type=int, default=1,
                    help='Epoch to start training on in case loading model in')
parser.add_argument('--display_frequency', type=int, default=100,
                    help='The number of steps to train GAN on before printing loss')
parser.add_argument('--checkpoint_frequency', type=int, default=1000,
                    help='The number of steps to train GAN on before saving a checkpoint')
opt = parser.parse_args()


def train():
    if opt.load_model is not None:
        checkpoint = 'checkpoints/' + opt.load_model
    else:
        checkpoint_name = datetime.now().strftime("%d%m%Y-%H%M")
        checkpoint = 'checkpoints/{}'.format(checkpoint_name)

        try:
            os.makedirs(checkpoint)
        except os.error:
            print("Failed to make new checkpoint directory.")
            sys.exit(1)

    # build the CycleGAN graph
    cyclegan = CycleGAN(opt, is_training=True)
    cyclegan.build_model()
    realX, realY, fakeX, fakeY = cyclegan.get_data()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        Gen_loss, D_Y_loss, D_X_loss = cyclegan.get_losses(realX, realY, fakeX, fakeY)
        Gen_opt, D_Y_opt, D_X_opt = cyclegan.get_optimizers(Gen_loss, D_Y_loss, D_X_loss)

    # create image pools for holding previously generated images
    fakeX_pool = ImagePool(opt.pool_size)
    fakeY_pool = ImagePool(opt.pool_size)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        if opt.load_model is not None: # restore graph and variables
            ckpt = tf.train.get_checkpoint_state(checkpoint)
            meta_graph_path = ckpt.model_checkpoint_path + '.meta'
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoint))
            start_step = int(meta_graph_path.split("-")[2].split(".")[0])
        else:
            sess.run(tf.global_variables_initializer())
            start_step = 1

        # generate fake images
        fakeX_imgs, fakeY_imgs = sess.run([fakeX, fakeY])

        try:
            for step in range(start_step, opt.niter + opt.niter_decay + 1):
                # calculate losses for the generators and discriminators and minimize them
                Gen_loss_val, D_Y_loss_val, D_X_loss_val, \
                fakeX_imgs, fakeY_imgs, _, _, _ = sess.run([Gen_loss, D_Y_loss, D_X_loss, fakeX,
                                                               fakeY, Gen_opt, D_Y_opt, D_X_opt],
                                                              feed_dict={cyclegan.poolX: fakeX_pool.query(fakeX_imgs),
                                                                         cyclegan.poolY: fakeY_pool.query(fakeY_imgs)})

                # display the losses of the Generators and Discriminators
                if step % opt.display_frequency == 0:
                    print('Step {}:'.format(step))
                    print('Gen_loss: {}'.format(Gen_loss_val))
                    print('D_Y_loss: {}'.format(D_Y_loss_val))
                    print('D_X_loss: {}'.format(D_X_loss_val))

                # save a checkpoint of the model to the `checkpoints` directory
                if step % opt.checkpoint_frequency == 0:
                    save_path = saver.save(sess, checkpoint + '/model.ckpt', global_step=step)
                    print("Model saved as {}".format(save_path))

        except KeyboardInterrupt: # save training before exiting
            print("Saving models training progress to the `checkpoints` directory...")
            save_path = saver.save(sess, checkpoint + '/model.ckpt', global_step=step)
            print("Model saved as {}".format(save_path))
            sys.exit(0)


if __name__ == '__main__':
    train()
