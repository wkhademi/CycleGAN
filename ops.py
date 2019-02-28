import numpy as np
import tensorflow as tf

"""
    Operation names taken straight from the appendix in the CycleGAN paper.
"""

def conv(input,
         in_channels,
         out_channels,
         filter_size,
         stride,
         padding_type='SAME',
         weight_init_type='normal',
         weight_init_gain=1.0,
         bias_const=0.0,
         use_bias=False,
         norm_type='instance',
         activation_type='ReLU',
         slope=0.2,
         is_training=True,
         scope=None,
         reuse=False):
    """
        Convolution-Normalization-Activation layer.

        Args:
            input:
            in_channels:
            out_channels:
            filter_size:
            stride:
            padding_type:
            weight_init_type:
            weight_init_gain:
            bias_const:
            use_bias:
            norm_type:
            activation_type:
            is_training:
            scope:
            reuse:

        Returns:
            layer:
    """
    with tf.variable_scope(scope, reuse=reuse):
        # weight initialization
        weights = __weights_init(filter_size, in_channels, out_channels,
                                 init_type=weight_init_type, init_gain=weight_init_gain)

        if padding_type is 'VALID': # add reflection padding to input
            padding = __fixed_padding(filter_size)
            padded_input = tf.pad(input, padding, 'REFLECT')
        elif padding_type is 'SAME':
            padded_input = input

        layer = tf.nn.conv2d(padded_input, weights, strides=[1, stride, stride, 1],
                             padding=padding_type)

        if use_bias:
            biases = __biases_init(out_channels, constant=bias_const)
            layer = tf.nn.bias_add(layer, biases)

        # instance, batch, or no normalization
        layer = __normalization(layer, is_training, norm_type=norm_type)

        # relu, leaky relu, or no activation
        layer = __activation_fn(layer, slope=slope, activation_type=activation_type)

    return layer


def resnet_block(input,
                 in_channels,
                 out_channels,
                 filter_size=3,
                 stride=1,
                 norm_type='instance',
                 activation_type='ReLU',
                 is_training=True,
                 dropout=False,
                 scope=None,
                 reuse=False):
    """
        Residual block that contains two 3x3 convolution layers with the same number
        of filters on both layer.

        Args:
            input:
            in_channels:
            out_channels:
            filter_size:
            stride:
            norm_type:
            activation_type:
            is_training:
            dropout:
            scope:
            reuse:

        Returns:
            layer:
    """
    with tf.variable_scope(scope, reuse=reuse):
        conv1 = conv(input, in_channels, out_channels, filter_size=filter_size, stride=stride,
                     padding_type='VALID', norm_type=norm_type, activation_type=activation_type,
                     is_training=is_training, scope='res_conv1', reuse=reuse)

        if dropout:
            conv1 = tf.nn.dropout(conv1, keep_prob=0.5)

        conv2 = conv(conv1, in_channels, out_channels, filter_size=filter_size, stride=stride,
                     padding_type='VALID', norm_type=norm_type, activation_type=None,
                     is_training=is_training, scope='res_conv2', reuse=reuse)

        layer = input + conv2

    return layer


def uk(input,
       in_channels,
       out_channels,
       out_shape,
       filter_size=3,
       stride=2,
       weight_init_type='normal',
       weight_init_gain=1.0,
       norm_type='instance',
       activation_type='ReLU',
       is_training=True,
       scope=None,
       reuse=False):
    """
        3x3 Fractional-Strided-Convolution-InstanceNorm-ReLU layer.

        Args:
            inputs:
            in_channels:
            out_channels:
            out_shape:
            filter_size:
            stride:
            weight_init_type:
            weight_init_gain:
            bias_const:
            norm_type:
            activation_type:
            is_training:
            scope:
            reuse:

        Returns:
            layer:
    """
    with tf.variable_scope(scope, reuse=reuse):
        batch_size = tf.shape(input)[0]

        # weight initialization
        #weights = bilinear_upsample_filter(filter_size, stride, in_channels, out_channels)
        weights = __weights_init(filter_size, out_channels, in_channels,
                                 init_type=weight_init_type, init_gain=weight_init_gain)

        layer = tf.nn.conv2d_transpose(input, weights,
                                       output_shape=[batch_size, out_shape, out_shape, out_channels],
                                       strides=[1, stride, stride, 1], padding='SAME')

        # instance, batch, or no normalization
        layer = __normalization(layer, is_training=is_training, norm_type=norm_type)

        # relu, leaky relu, or no activation
        layer = __activation_fn(layer, activation_type=activation_type)

    return layer


def __normalization(inputs,
                    is_training=True,
                    norm_type='instance'):
    """
        Normalization to be applied to layer.

        Args:
            inputs: Input tensor
            is_training: Whether in training or testing phase
            norm_type: Type of normalization to apply to inputs

        Returns:
            norm: Normalized inputs
    """
    if norm_type is 'batch':
        norm = tf.contrib.layers.batch_norm(inputs, is_training=is_training)
    elif norm_type is 'instance':
        norm = tf.contrib.layers.instance_norm(inputs)
    else:
        norm = inputs

    return norm


def __activation_fn(inputs,
                    slope=0.2,
                    activation_type='ReLU'):
    """
        Non-linear activation to be applied to layer.

        Args:
            inputs: Input tensor
            slope: Scalar value for Leaky ReLU
            activation_type: Type of activation function

        Returns:
            activation: Inputs that have had a non-linear activation applied to them
    """
    if activation_type is 'ReLU':
        activation = tf.nn.relu(inputs, name='relu')
    elif activation_type is 'LeakyReLU':
        activation= tf.nn.leaky_relu(inputs, alpha=slope, name='leakyrelu')
    elif activation_type is 'tanh':
        activation = tf.nn.tanh(inputs, name='tanh')
    elif activation_type is 'sigmoid':
        activation = tf.nn.sigmoid(inputs, name='sigmoid')
    else:
        activation = inputs

    return activation


def __weights_init(size,
                   in_channels,
                   out_channels,
                   init_type='normal',
                   init_gain=1.0):
    """
        Initialize weights given a specific initialization type.

        Args:
            size: Size of filter matrix
            in: # of channels for input
            out: # of channels desired for output
            init_type: Type of weight initialization
            init_gain: Scaling factor for weight initialization

        Returns:
            weights: Weight tensor
    """
    if init_type is 'normal':
        init = tf.initializers.truncated_normal(stddev=init_gain)
    elif init_type is 'xavier':
        init = tf.initializers.glorot_normal()
    elif init_type is 'orthogonal':
        init = tf.initializers.orthogonal(gain=init_gain)

    weights = tf.get_variable("weights", shape=[size, size, in_channels, out_channels],
                              dtype=tf.float32, initializer=init)

    return weights


def __biases_init(size,
                  constant=0.0):
    """
        Initialize biases to a given constant.

        Args:
            size: Size of the bias vector
            constant: A constant value to initialize the entry in the bias vector to

        Returns:
            biases: Bias vector
    """
    biases = tf.get_variable("biases", shape=[size], dtype=tf.float32,
                             initializer=tf.constant_initializer(constant))

    return biases


def __fixed_padding(filter_size):
    """
        Calculate padding needed to keep input from being downsampled.

        Args:
            filter_size: Size of filter to be convolved with input

        Returns:
            padding: Padding size needed to keep input from being downsampled
    """
    pad_total = filter_size - 1
    pad = pad_total // 2
    padding = [[0,0], [pad, pad], [pad, pad], [0, 0]]

    return padding


def bilinear_upsample_filter(ksize,
                             factor,
                             in_channels,
                             out_channels):
    """
        Create a 2D bilinear kernel used to upsample.
        Arguments:
            ksize: The desired size of bilinear filter
            factor: The amount you want to scale the input up by
            in_channels: The number of input channels
            out_channels: The desired number of output channels
        Returns:
            weights: A Numpy array containing the set of filters used during upsampling
    """
    if (ksize % 2 == 1):
        center = factor - 1
    else:
        center = factor - 0.5

    # bilinear filter
    og = np.ogrid[:ksize, :ksize]
    kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

    # apply bilinear filter to proper kernel size
    weights = np.zeros((in_channels, out_channels, ksize, ksize), dtype=np.float32)
    weights[:, :, :, :] = kernel
    weights = np.transpose(weights, (2, 3, 0, 1))

    return weights
