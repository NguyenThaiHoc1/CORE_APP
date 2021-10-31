# -*- coding: utf-8 -*-

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

import argparse
import glob
import os

import cv2
import numpy as np
from imageio import imread, imsave

parser = argparse.ArgumentParser()
parser.add_argument('--no_makeup', type=str, default='./data/adh.jpg',
                    help='path to the no_makeup image')
args = parser.parse_args()


def preprocess(img):
    return (img / 255. - 0.5) * 2


def deprocess(img):
    return (img + 1) / 2


batch_size = 1
img_size = 256
no_makeup = cv2.resize(imread(args.no_makeup, as_gray=False, pilmode="RGB"), (img_size, img_size)) # ,
X_img = np.expand_dims(preprocess(no_makeup), 0)
makeups = glob.glob(os.path.join('core_makeup', 'imgs', 'makeup', '*.*'))
result = np.ones((2 * img_size, (len(makeups) + 1) * img_size, 3))
result[img_size: 2 * img_size, :img_size] = no_makeup / 255.

tf.disable_eager_execution()

tf.reset_default_graph()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph(os.path.join('core_makeup', 'model', 'model.meta'))
saver.restore(sess, tf.train.latest_checkpoint('core_makeup/model/'))

graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')

for i in range(len(makeups)):
    makeup = cv2.resize(imread(makeups[i]), (img_size, img_size))
    Y_img = np.expand_dims(preprocess(makeup), 0)
    Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
    Xs_ = deprocess(Xs_)
    result[:img_size, (i + 1) * img_size: (i + 2) * img_size] = makeup / 255.
    result[img_size: 2 * img_size, (i + 1) * img_size: (i + 2) * img_size] = Xs_[0]

imsave('./result_makeup/result.jpg', result)
