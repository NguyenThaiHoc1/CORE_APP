import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, Input
from tensorflow.python.training import py_checkpoint_reader

from core.config import settings


def residual_block(inputs, output_channel, name):
    layer = Conv2D(filters=output_channel, kernel_size=(3, 3), padding='same',
                   activation=LeakyReLU(alpha=0.2), name=name + '_conv1')(inputs)

    layer = Conv2D(filters=output_channel, kernel_size=(3, 3), padding='same',
                   activation=None, name=name + '_conv2')(layer)

    return layer + inputs


def architecture_unet(layer):
    with tf.name_scope('generator'):
        layer_0 = Conv2D(filters=32, kernel_size=(7, 7), padding='same',
                         activation=LeakyReLU(alpha=0.2), name='conv')(layer)

        layer_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding='same',
                         activation=LeakyReLU(alpha=0.2), name='conv_1')(layer_0)
        layer_1 = Conv2D(filters=32 * 2, kernel_size=(3, 3), padding='same',
                         activation=LeakyReLU(alpha=0.2), name='conv_2')(layer_1)

        layer_2 = Conv2D(filters=32 * 2, kernel_size=(3, 3), strides=2, padding='same',
                         activation=LeakyReLU(alpha=0.2), name='conv_3')(layer_1)
        layer_2 = Conv2D(filters=32 * 4, kernel_size=(3, 3), padding='same',
                         activation=LeakyReLU(alpha=0.2), name='conv_4')(layer_2)

        for index in range(4):
            layer_2 = residual_block(layer_2, output_channel=32 * 4, name='block_{}'.format(index))

        layer_2 = Conv2D(filters=32 * 2, kernel_size=(3, 3), padding='same',
                         activation=LeakyReLU(alpha=0.2), name='conv_5')(layer_2)

        # reshape layer 3
        h1, w1 = tf.shape(layer_2)[1], tf.shape(layer_2)[2]
        layer_3 = tf.image.resize(layer_2, (h1 * 2, w1 * 2))
        layer_3 = Conv2D(filters=32 * 2, kernel_size=(3, 3), padding='same',
                         activation=LeakyReLU(alpha=0.2), name='conv_6')(layer_3 + layer_1)
        layer_3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                         activation=LeakyReLU(alpha=0.2), name='conv_7')(layer_3)

        # reshape layer 4
        h2, w2 = tf.shape(layer_3)[1], tf.shape(layer_3)[2]
        layer_4 = tf.image.resize(layer_3, (h2 * 2, w2 * 2))
        layer_4 = Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                         activation=LeakyReLU(alpha=0.2), name='conv_8')(layer_4 + layer_0)
        layer_4 = Conv2D(filters=3, kernel_size=(7, 7), padding='same',
                         activation=None, name='conv_9')(layer_4)
        return layer_4


def create_model_unet(input_layer, name_model='unet_architecture', show=True):
    output = architecture_unet(input_layer)
    model = Model(input_layer, output, name=name_model)
    if show:
        model.summary()
    return model


def matching_layer(name_layer, tensor, model_path_tf1=settings.MODEL_PATH):
    ck_path = tf.train.latest_checkpoint(model_path_tf1)
    reader = py_checkpoint_reader.NewCheckpointReader(ck_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    list_numpy_array = []
    for key, value in sorted(var_to_shape_map.items()):
        split_string = key.lower().split('/')
        layer_weight_name = '_'.join(name for name in split_string[1:-1])
        if layer_weight_name == name_layer.lower():
            value_layer = reader.get_tensor(key)
            list_numpy_array.append(value_layer)

    if len(list_numpy_array) > 0:
        list_numpy_array.reverse()
        tensor.set_weights(weights=list_numpy_array)


def loading_weights(model_architecture):
    for layer in model_architecture.layers:
        matching_layer(name_layer=layer.name, tensor=layer)


if __name__ == '__main__':
    input_layer = Input(shape=(280, 384, 3))
    model = create_model_unet(input_layer, show=True)