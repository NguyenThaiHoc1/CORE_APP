from tensorflow.keras.layers import Conv2D, BatchNormalization, ZeroPadding2D
from tensorflow.keras.layers import ReLU


def generator_branch(inputs, conv_dim=64, repeat_num=6, input_nc=3):
    """
    https://stackoverflow.com/questions/68088889/how-to-add-instancenormalization-on-tensorflow-keras
    https://stackoverflow.com/questions/62112123/how-to-give-padding-6-in-tensorflow-2-0
    https://stackoverflow.com/questions/37659538/custom-padding-for-convolutions-in-tensorflow
    :param inputs:
    :param conv_dim:
    :param repeat_num:
    :param input_nc:
    :return:
    """
    layers_branch = []
    layers_branch.append(ZeroPadding2D(padding=(3, 3)))
    layers_branch.append(Conv2D(filters=conv_dim,
                                kernel_size=(7, 7),
                                strides=(1, 1),
                                padding='valid',
                                use_bias=False,
                                activation=None, name='conv'))
    layers_branch.append(BatchNormalization(axis=[0, 1]))
    layers_branch.append(ReLU())

    layers_branch.append(Conv2D(filters=conv_dim * 2,
                                kernel_size=(4, 4),
                                strides=2,
                                padding=1,
                                use_bias=False,
                                activation=None, name='conv1'))
    layers_branch.append(BatchNormalization(axis=[0, 1]))
    layers_branch.append(ReLU())