import cv2
import numpy as np
import tensorflow as tf


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image


def tf_box_filter(x, r):
    k_size = int(2 * r + 1)
    ch = x.get_shape().as_list()[-1]
    weight = 1 / (k_size ** 2)
    box_kernel = weight * np.ones((k_size, k_size, ch, 1))
    box_kernel = np.array(box_kernel).astype(np.float32)
    output = tf.nn.depthwise_conv2d(x, box_kernel, [1, 1, 1, 1], 'SAME')
    return output


def guided_filter(x, y, r, eps=1e-2):
    x_shape = tf.shape(x)
    N = tf_box_filter(tf.ones((1, x_shape[1], x_shape[2], 1), dtype=x.dtype), r)
    mean_x = tf_box_filter(x, r) / N
    mean_y = tf_box_filter(y, r) / N
    cov_xy = tf_box_filter(x * y, r) / N - mean_x * mean_y
    var_x = tf_box_filter(x * x, r) / N - mean_x * mean_x
    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x
    mean_A = tf_box_filter(A, r) / N
    mean_b = tf_box_filter(b, r) / N
    output = mean_A * x + mean_b
    return output


def process_image(image_path):
    image = cv2.imread(image_path)
    image = resize_crop(image)
    batch_image = image.astype(np.float32) / 127.5 - 1
    batch_image = np.expand_dims(batch_image, axis=0)
    return batch_image