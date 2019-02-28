import sys
import ops
import numpy as np
import tensorflow as tf

class Generator():
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 netG='resnet_9blocks',
                 ngf=64,
                 norm_type='instance',
                 init_type='normal',
                 init_gain=1.0,
                 dropout=False,
                 is_training=True,
                 name=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.netG = netG
        self.ngf = ngf
        self.norm_type = norm_type
        self.init_type = init_type
        self.init_gain = init_gain
        self.dropout = dropout
        self.is_training = is_training
        self.name = name
        self.reuse = False

    def __call__(self, input):
        with tf.variable_scope(self.name):
            if self.netG is 'resnet_9blocks':
                output = self.resnet_generator(input, self.in_channels, self.out_channels, self.ngf,
                                               self.norm_type, self.init_type, self.init_gain, self.dropout,
                                               self.is_training, n_blocks=9)
            elif self.netG is 'resnet_6blocks':
                output = self.resnet_generator(input, self.in_channels, self.out_channels, self.ngf,
                                               self.norm_type, self.init_type, self.init_gain, self.dropout,
                                               self.is_training, n_blocks=6)
            elif self.netG is 'unet_256':
                print("Haven't implemented yet...")
                sys.exit(1)
            elif self.netG is 'unet_128':
                print("Haven't implemented yet...")
                sys.exit(1)
            else:
                print("Invalid generator architecture.")
                sys.exit(1)

        # set reuse to True for next call
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output

    def resnet_generator(self,
                         input,
                         in_channels=3,
                         out_channels=3,
                         ngf=64,
                         norm_type='instance',
                         init_type='normal',
                         init_gain=1.0,
                         dropout=False,
                         is_training=True,
                         n_blocks=6):
        """
            Resnet-based generator that contains Resnet blocks in between some downsampling and upsampling layers.

            Args:
                input:
                in_channels:
                out_channels:
                ngf:
                norm_type:
                init_type:
                init_gain:
                dropout:
                is_training:
                n_blocks:

            Returns:
                c7s1_3: Final output of the Resnet-based generator
        """
        # 7x7 convolution-instance norm-relu layer with 64 filters and stride 1
        c7s1_64 = ops.conv(input, in_channels=in_channels, out_channels=ngf, filter_size=7,
                           stride=1, padding_type='VALID', weight_init_type=init_type, weight_init_gain=init_gain,
                           norm_type=norm_type, is_training=is_training, scope='c7s1-64', reuse=self.reuse)

        # 3x3 convolution-instance norm-relu layer with 128 filters and stride 2
        d128 = ops.conv(c7s1_64, in_channels=ngf, out_channels=2*ngf, filter_size=3,
                        stride=2, weight_init_type=init_type, weight_init_gain=init_gain,
                        norm_type=norm_type, is_training=is_training, scope='d128', reuse=self.reuse)

        # 3x3 convolution-instance norm-relu layer with 256 filters and stride 2
        d256 = ops.conv(d128, in_channels=2*ngf, out_channels=4*ngf, filter_size=3,
                        stride=2, weight_init_type=init_type, weight_init_gain=init_gain,
                        norm_type=norm_type, is_training=is_training, scope='d256', reuse=self.reuse)

        r256 = d256

        # Resnet blocks with 256 filters
        for idx in range(n_blocks):
            # residual block that contains two 3x3 convolution layers with 256 filters and stride 1
            r256 = ops.resnet_block(r256, in_channels=4*ngf, out_channels=4*ngf, filter_size=3,
                                    stride=1, norm_type=norm_type, is_training=is_training, dropout=dropout,
                                    scope='r256-'+str(idx), reuse=self.reuse)

        # 3x3 fractional strided convolution-instance norm-relu layer with 128 filters and stride 1/2
        u128 = ops.uk(r256, in_channels=4*ngf, out_channels=2*ngf, out_shape=2*r256.get_shape().as_list()[1],
                      filter_size=3, stride=2, weight_init_type=init_type, weight_init_gain=init_gain,
                      norm_type=norm_type, is_training=is_training, scope='u128', reuse=self.reuse)

        # 3x3 fractional strided convolution-instance norm-relu layer with 64 filters and stride 1/2
        u64 = ops.uk(u128, in_channels=2*ngf, out_channels=ngf, out_shape=2*u128.get_shape().as_list()[1],
                      filter_size=3, stride=2, weight_init_type=init_type, weight_init_gain=init_gain,
                      norm_type=norm_type, is_training=is_training, scope='u64', reuse=self.reuse)

        # 7x7 convolution-instance norm-relu layer with 3 filters and stride 1
        c7s1_3 = ops.conv(u64, in_channels=ngf, out_channels=out_channels, filter_size=7,
                          stride=1, padding_type='VALID', weight_init_type=init_type, weight_init_gain=init_gain,
                          norm_type=None, activation_type='tanh', is_training=is_training, scope='c7s1-3',
                          reuse=self.reuse)

        return c7s1_3

    def unet_generator(self):
        """
            Unet-based generator
        """
        pass
