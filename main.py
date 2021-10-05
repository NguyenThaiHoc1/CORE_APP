import os
import time

import cv2
from tensorflow.keras.layers import Input

from core.architecture.unet import UNETModel
from core.config import settings
from core.utilis.utlis import process_image

if __name__ == '__main__':
    input_tensor = Input(shape=(None, None, 3))
    model = UNETModel(input_tensor)
    model.backbone.load_weights(settings.MODEL_FILE_TF2)

    name_list = os.listdir(settings.SOURCE_FRAME)
    for name in name_list:
        start_time = time.time()
        try:
            load_path = os.path.join(settings.SOURCE_FRAME, name)
            save_path = os.path.join(settings.CARTOON_IMAGE, name)
            image = process_image(image_path=load_path)
            output = model(image, training=False)
            cv2.imwrite(save_path, output.numpy())
        except TypeError as e:
            print('cartoonize {} failed'.format(load_path))
        print(f'End time: {time.time() - start_time}')
