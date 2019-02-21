import numpy as np
import tensorflow as tf
from generator import Generator
from discriminator import Discriminator


class CycleGAN():
    def __init__(self, opt, is_training=True):
        self.opt = opt
        self.is_training = is_training

        self.realA = tf.placeholder(dtype=tf.float32,
                                    shape=(None, self.opt.crop_size, self.opt.crop_size, self.opt.in_channels),
                                    name='realA')
        self.realB = tf.placeholder(dtype=tf.float32,
                                    shape=(None, self.opt.crop_size, self.opt.crop_size, self.opt.out_channels),
                                    name='realB')
        self.poolA = tf.placeholder(dtype=tf.float32,
                                    shape=(None, self.opt.crop_size, self.opt.crop_size, self.opt.in_channels),
                                    name='poolA')
        self.poolB = tf.placeholder(dtype=tf.float32,
                                    shape=(None, self.opt.crop_size, self.opt.crop_size, self.opt.out_channels),
                                    name='poolB')

    def set_input(self):
        """
            Set the inputs based on the direction of CycleGAN [AtoB | BtoA].
        """
        pass

    def build_model(self):
        """
            Build the forward pass of the graph for CycleGAN
        """
        # build the Generator graphs for each GAN
        self.G = Generator(self.opt.in_channels, self.opt.out_channels, self.opt.netG,
                           self.opt.ngf, self.opt.norm_type, self.opt.init_type,
                           self.opt.init_gain, self.is_training, name='G')
        self.F = Generator(self.opt.out_channels, self.opt.in_channels, self.opt.netG,
                           self.opt.ngf, self.opt.norm_type, self.opt.init_type,
                           self.opt.init_gain, self.is_training, name='F')

        # build the Discriminator graphs for each GAN only if in training phase
        if self.is_training:
            self.D_A = Discriminator(self.opt.in_channels, self.opt.netD, self.opt.n_layers, self.opt.ndf,
                                     self.opt.norm_type, self.opt.init_type, self.opt.init_gain,
                                     self.is_training, self.opt.gan_mode, name='D_A')
            self.D_B = Discriminator(self.opt.out_channels, self.opt.netD, self.opt.n_layers, self.opt.ndf,
                                     self.opt.norm_type, self.opt.init_type, self.opt.init_gain,
                                     self.is_training, self.opt.gan_mode, name='D_B')

    def generate(self):
        """
            Generate fake data batches to be used for training CycleGAN

            Returns:
                fakeA: A batch of generated images replicating Domain A
                fakeB: A batch of generated images replication Domain B
        """
        # generate fake images
        fakeB = self.G(self.realA)
        fakeA = self.F(self.realB)

        tf.summary.image('generated/A', fakeA)
        tf.summary.image('generated/B', fakeB)

        return fakeA, fakeB

    def get_losses(self, fakeA, fakeB):
        """
            Build the loss part of the graph for CycleGAN

            Args:
                fakeA: A batch of generated images replicating Domain A
                fakeB: A batch of generated images replication Domain B

            Returns:
                Gen_loss: Generators combined loss
                D_B_loss: Discriminator for Domain B images loss
                D_A_loss: Discriminator for Domain A images loss
        """
        # calculate cycle cycle consistency loss
        cc_loss = self.cycle_consistency_loss(self.G, self.F, fakeA, fakeB, self.realA, self.realB)

        # generator losses
        Gen_loss = self.G_loss(self.D_B, fakeB) + self.G_loss(self.D_A, fakeA) + cc_loss

        # discriminator losses
        D_A_loss = self.D_loss(self.D_A, self.realA, self.poolA)
        D_B_loss = self.D_loss(self.D_B, self.realB, self.poolB)

        tf.summary.scalar('loss/Gen', Gen_loss)
        tf.summary.scalar('loss/D_A', D_A_loss)
        tf.summary.scalar('loss/D_B', D_B_loss)

        return Gen_loss, D_B_loss, D_A_loss

    def get_optimizers(self, Gen_loss, D_B_loss, D_A_loss):
        """
            Build the optimizer part of the graph out for CycleGAN

            Args:
                Gen_loss: Generators combined loss
                D_B_loss: Discriminator for Domain B images loss
                D_A_loss: Discriminator for Domain A images loss

            Returns:
                Gen_opt: Optimizer for generators
                D_B_opt: Optimizer for discriminator for Domain B
                D_A_opt: Optimizer for discriminator for Domain A
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

        Gen_opt = make_optimizer(Gen_loss, self.G.variables + self.F.variables, name='Adam_Gen')
        D_B_opt = make_optimizer(D_B_loss, self.D_B.variables, name='Adam_D_B')
        D_A_opt = make_optimizer(D_A_loss, self.D_A.variables, name='Adam_D_A')

        return Gen_opt, D_B_opt, D_A_opt

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

    def cycle_consistency_loss(self, G, F, fakeA, fakeB, realA, realB):
        """
            Find the forward loss and backward loss of our GANS.

            Use L1 norm for calculating error between mapping and real.

            Args:
                G: Generator that maps Domain A -> Domain B
                F: Generator that maps Domain B -> Domain A
                fakeA: Fake images from Domain A generated by F
                fakeB: Fake images from Domain B generated by G
                realA: Real images from Domain A
                realB: Real images from Domain B

            Returns:
                loss: The cycle consistency loss of our network
        """
        recA = F(fakeB)
        recB = G(fakeA)

        tf.summary.image('reconstructed/A', recA)
        tf.summary.image('reconstructed/B', recB)

        forward_loss = tf.reduce_mean(tf.abs(recA - realA))
        backward_loss = tf.reduce_mean(tf.abs(recB - realB))
        loss = (self.opt.lambda_A * forward_loss) + (self.opt.lambda_B * backward_loss)
        return loss
