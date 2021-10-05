import tensorflow as tf
from tensorflow.keras import Model

from core.architecture.model import create_model_unet
from core.utilis.utlis import guided_filter


class UNETModel(Model):
    def __init__(self, input_tensor):
        super(UNETModel, self).__init__()
        self.backbone = create_model_unet(input_tensor, show=False)

    def call(self, inputs, training=True, mask=None):
        prelogits = self.backbone(inputs)
        if not training:
            decode_model = guided_filter(x=inputs, y=prelogits, r=1, eps=5e-3)
            output = (tf.squeeze(decode_model) + 1) * 127.5
            output = tf.cast(tf.clip_by_value(output, 0, 255), tf.uint8)
            return output
        return prelogits


if __name__ == '__main__':
    from tensorflow.keras.layers import Input
    from core.config import settings

    input_tensor = Input(shape=(None, None, 3))
    model = UNETModel(input_tensor)
    model.backbone.load_weights(settings.MODEL_FILE_TF2)
    # model.backbone.save_weights(str(settings.MODEL_PATH_TF2) + '/weights.h5')
