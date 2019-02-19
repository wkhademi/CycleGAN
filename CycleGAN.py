import numpy as np
import tensorflow as tf
from DataLoader import DataLoader
from generator import Generator
from discriminator import Discriminator


class CycleGAN():
    def __init__(self, opt, is_training=True):
        self.opt = opt
        self.is_training = is_training

        self.poolX = tf.placeholder(dtype=tf.float32,
                                    shape=(None, self.opt.crop_size, self.opt.crop_size, self.opt.in_channels),
                                    name='poolX')
        self.poolY = tf.placeholder(dtype=tf.float32,
                                    shape=(None, self.opt.crop_size, self.opt.crop_size, self.opt.out_channels),
                                    name='poolY')

    def set_input(self):
        """
            Set the inputs based on the direction of CycleGAN [AtoB | BtoA].
        """
        pass

    def build_model(self):
        """
            Build the forward pass of the graph for CycleGAN
        """
        # create an iterator for our datasets
        self.dataloader = iter(DataLoader(self.opt))

        # build the Generator graphs for each GAN
        self.G = Generator(self.opt.in_channels, self.opt.out_channels, self.opt.netG,
                           self.opt.ngf, self.opt.norm_type, self.opt.init_type,
                           self.opt.init_gain, self.is_training, name='G')
        self.F = Generator(self.opt.out_channels, self.opt.in_channels, self.opt.netG,
                           self.opt.ngf, self.opt.norm_type, self.opt.init_type,
                           self.opt.init_gain, self.is_training, name='F')

        # build the Discriminator graphs for each GAN only if in training phase
        if self.is_training:
            self.D_X = Discriminator(self.opt.in_channels, self.opt.netD, self.opt.n_layers, self.opt.ndf,
                                     self.opt.norm_type, self.opt.init_type, self.opt.init_gain,
                                     self.is_training, self.opt.gan_mode, name='D_X')
            self.D_Y = Discriminator(self.opt.out_channels, self.opt.netD, self.opt.n_layers, self.opt.ndf,
                                     self.opt.norm_type, self.opt.init_type, self.opt.init_gain,
                                     self.is_training, self.opt.gan_mode, name='D_Y')

    def get_data(self):
        """
            Get real and fake data batches to be used for training CycleGAN

            Returns:
                realX: A batch of real images from Domain X
                realY: A batch of real images from Domain Y
                fakeX: A batch of generated images replicating Domain X
                fakeY: A batch of generated images replication Domain Y
        """
        # get a batch of real images
        realX, realY = next(self.dataloader)

        # generate fake images
        fakeY = self.G(realX)
        fakeX = self.F(realY)

        return realX, realY, fakeX, fakeY

    def get_losses(self, realX, realY, fakeX, fakeY):
        """
            Build the loss part of the graph for CycleGAN

            Args:
                realX: A batch of real images from Domain X
                realY: A batch of real images from Domain Y
                fakeX: A batch of generated images replicating Domain X
                fakeY: A batch of generated images replication Domain Y

            Returns:
                G_loss: Generator mapping X->Y loss
                D_Y_loss: Discriminator for Domain Y images loss
                F_loss: Generator mapping Y->X loss
                D_X_loss: Discriminator for Domain X images loss
        """
        # calculate cycle cycle consistency loss
        cc_loss = self.cycle_consistency_loss(self.G, self.F, fakeX, fakeY, realX, realY)

        # generator losses
        G_loss = self.G_loss(self.D_Y, fakeY) + cc_loss
        F_loss = self.G_loss(self.D_X, fakeX) + cc_loss

        # discriminator losses
        D_X_loss = self.D_loss(self.D_X, realX, self.poolX)
        D_Y_loss = self.D_loss(self.D_Y, realY, self.poolY)

        return G_loss, D_Y_loss, F_loss, D_X_loss

    def get_optimizers(self, G_loss, D_Y_loss, F_loss, D_X_loss):
        """
            Build the optimizer part of the graph out for CycleGAN

            Args:
                G_loss: Generator mapping X->Y loss
                D_Y_loss: Discriminator for Domain Y images loss
                F_loss: Generator mapping Y->X loss
                D_X_loss: Discriminator for Domain X images loss

            Returns:
                G_opt: Optimizer for generator mapping X->Y
                D_Y_opt: Optimizer for discriminator for Domain Y
                F_opt: Optimizer for generator mapping Y->X
                D_X_opt: Optimizer for discriminator for Domain X
        """
        def make_optimizer(loss, variables, name='Adam'):
            """
                Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
                and a linearly decaying rate that goes to zero over the next 100k steps
            """
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.opt.learning_rate
            end_learning_rate = 0.0
            start_decay_step = self.opt.niter
            decay_steps = self.opt.niter_decay
            beta1 = self.opt.beta1
            learning_rate = (tf.where(tf.greater_equal(global_step, start_decay_step),
                                      tf.train.polynomial_decay(starter_learning_rate,
                                                                global_step-start_decay_step,
                                                                decay_steps, end_learning_rate,
                                                                power=1.0),
                                      starter_learning_rate))

            learning_step = (tf.train.AdamOptimizer(learning_rate, beta1=beta1,
                                                    name=name).minimize(loss, global_step=global_step,
                                                                        var_list=variables))

            return learning_step

        G_opt = make_optimizer(G_loss, self.G.variables, name='Adam_G')
        D_Y_opt = make_optimizer(D_Y_loss, self.D_Y.variables, name='Adam_D_Y')
        F_opt = make_optimizer(F_loss, self.F.variables, name='Adam_F')
        D_X_opt = make_optimizer(D_X_loss, self.D_X.variables, name='Adam_D_X')

        return G_opt, D_Y_opt, F_opt, D_X_opt

    def G_loss(self, D, fake, real_label=1.0, epsilon=1e-12):
        """
            Find the generator loss using either least squared error or negative log likelihood.

            Args:
                D: Discriminator model
                fake: Fake images generated by the Generator model
                real_label: The value assigned to a real label (fake label would be 0.0)
                epsilon: Small value to ensure log of zero is not taken

            Returns:
                generator_loss: The loss of the Generator model
        """
        if self.opt.gan_mode is 'lsgan': # least squared error
            generator_loss = tf.reduce_mean(tf.squared_difference(D(fake), real_label))
        elif self.opt.gan_mode is 'vanilla': # negative log likelihood
            generator_loss = -1 * tf.reduce_mean(tf.log(D(fake) + epsilon))

        return generator_loss

    def D_loss(self, D, real, fake, real_label=1.0, epsilon=1e-12):
        """
            Find the discriminator loss using either least squared error or negative log likelihood.

            Args:
                D: Discriminator model
                real: Real images from the dataset
                fake: Fake images generated by the Generator model
                real_label: The value assigned to a real label (fake label would be 0.0)
                epsilon: Small value to ensure log of zero is not taken

            Returns:
                discriminator_loss: The loss of the Discriminator model
        """
        if self.opt.gan_mode is 'lsgan': # least squared error
            discriminator_loss = tf.reduce_mean(tf.squared_difference(D(real), real_label)) + \
                                 tf.reduce_mean(tf.square(D(fake)))
        elif self.opt.gan_mode is 'vanilla': # negative log likelihood
            discriminator_loss = -1 * (tf.reduce_mean(tf.log(D(real) + epsilon)) + \
                                       tf.reduce_mean(tf.log((1 - D(fake)) + epsilon)))

        discriminator_loss = discriminator_loss * 0.5

        return discriminator_loss

    def cycle_consistency_loss(self, G, F, fakeX, fakeY, realX, realY):
        """
            Find the forward loss and backward loss of our GANS.

            Use L1 norm for calculating error between mapping and real.

            Args:
                G: Generator that maps Domain X -> Domain Y
                F: Generator that maps Domain Y -> Domain X
                fakeX: Fake images from Domain X generated by F
                fakeY: Fake images from Domain Y generated by G
                realX: Real images from Domain X
                realY: Real images from Domain Y

            Returns:
                loss: The cycle consistency loss of our network
        """
        forward_loss = tf.reduce_mean(tf.abs(F(fakeY) - realX))
        backward_loss = tf.reduce_mean(tf.abs(G(fakeX) - realY))
        loss = (self.opt.lambda_A * forward_loss) + (self.opt.lambda_B * backward_loss)
        return loss
