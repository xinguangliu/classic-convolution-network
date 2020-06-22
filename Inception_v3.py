from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Activation,Dense,Input,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D
from keras.layers import GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'


def conv2d_bn(x,
              filters,              # 通道数
              num_row,              # filter size
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def InceptionV3(input_shape=[299,299,3],
                classes=1000):


    img_input = Input(shape=input_shape)                        # 299,299,3  input

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')     # 149,149,32
    x = conv2d_bn(x, 32, 3, 3, padding='valid')                             # 147,147,32
    x = conv2d_bn(x, 64, 3, 3)                                              # 147,147,64
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)                             # 73,73,64

    x = conv2d_bn(x, 80, 1, 1, padding='valid')                             # 73,73,80
    x = conv2d_bn(x, 192, 3, 3, padding='valid')                            # 71,71,192
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)                             # 35,35,192

    #--------------------------------#
    #   Block1 35x35
    #--------------------------------#
    # Block1 part1
    # 35 x 35 x 192 -> 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)                      # 35,35,64

    branch5x5 = conv2d_bn(x, 48, 1, 1)                      # 35,35,48
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)              # 35,35,64

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)                   # 35,35,64
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)        # 35,35,96
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)        # 35,35,96

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)       # 35,35,192
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)          # 35,35,32
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed0')                                      # 35,35,256

    # Block1 part2
    # 35 x 35 x 256 -> 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)                      # 35,35,64

    branch5x5 = conv2d_bn(x, 48, 1, 1)                      # 35,35,48
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)              # 35,35,64

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)                   # 35,35,64
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)        # 35,35,96
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)        # 35,35,96

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)   # 35,35,256
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)                              # 35,35,64
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed1')                                      # 35,35,288

    # Block1 part3
    # 35 x 35 x 288 -> 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)                      # 35,35,64

    branch5x5 = conv2d_bn(x, 48, 1, 1)                      # 35,35,48
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)              # 35,35,64

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)                   # 35,35,64
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)        # 35,35,96
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)        # 35,35,96

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)   # 35,35,288
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)                              # 35,35,64
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed2')                                                          # 35,35,288

    #--------------------------------#
    #   Block2 17x17
    #--------------------------------#
    # Block2 part1
    # 35 x 35 x 288 -> 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')        # 17,17,384

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)                                       # 35,35,64
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)                            # 35,35,96
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')                # 17,17,96

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)                       # 17,17,288
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed3')          # 17,17,768

    # Block2 part2
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)                     # 17,17,192

    branch7x7 = conv2d_bn(x, 128, 1, 1)                     # 17,17,128
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)             # 17,17,128
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)             # 17,17,192

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)                  # 17,17,128
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)       # 17,17,128
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)       # 17,17,128
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)       # 17,17,128
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)       # 17,17,192

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)   # 17,17,768
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)                             # 17,17,192
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed4')                                                          # 17,17,768

    # Block2 part3 and part4
    # 17 x 17 x 768 -> 17 x 17 x 768 -> 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)                 # 17,17,192

        branch7x7 = conv2d_bn(x, 160, 1, 1)                 # 17,17,160
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)         # 17,17,160
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)         # 17,17,192

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)              # 17,17,160
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)   # 17,17,160
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)   # 17,17,160
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)   # 17,17,160
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)   # 17,17,192

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)      # 17,17,768
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)     # 17,17,192
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=3,
            name='mixed' + str(5 + i))                      # 17,17,768

    # Block2 part5
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)                     # 17,17,192

    branch7x7 = conv2d_bn(x, 192, 1, 1)                     # 17,17,192
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)             # 17,17,192
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)             # 17,17,192

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)                  # 17,17,192
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)       # 17,17,192
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)       # 17,17,192
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)       # 17,17,192
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)       # 17,17,192

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)   # 17,17,768
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)                             # 17,17,192
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed7')                                                          # 17,17,768

    #--------------------------------#
    #   Block3 8x8
    #--------------------------------#
    # Block3 part1
    # 17 x 17 x 768 -> 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)                                     # 17,17,192
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')                  # 8,8,320

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)                                   # 17,17,192
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)                         # 17,17,192
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)                         # 17,17,192
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')            # 8,8,192

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)                   # 8,8,768
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed8')       # 8,8,1280

    # Block3 part2 part3
    # 8 x 8 x 1280 -> 8 x 8 x 2048 -> 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)                                 # 8,8,320

        branch3x3 = conv2d_bn(x, 384, 1, 1)                                 # 8,8,384
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)                       # 8,8,384
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)                       # 8,8,384
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))    # 8,8,768

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)                              # 8,8,448
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)                   # 8,8,384
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)                 # 8,8,384
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)                 # 8,8,384
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=3)                       # 8,8,768

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)                      # 8,8,1280
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)                     # 8,8,192
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=3,
            name='mixed' + str(9 + i))                                      # 8,8,2048
    
    # 平均池化后全连接。
    x = GlobalAveragePooling2D(name='avg_pool')(x)                          # 2048
    x = Dense(classes, activation='softmax', name='predictions')(x)         # 列别个数


    inputs = img_input

    model = Model(inputs, x, name='inception_v3')

    return model

# 转换到[-1, 1]
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.             # 转换到[-1, 1]
    return x


if __name__ == '__main__':
    model = InceptionV3()

    print(model.summary())

    # weights_path = get_file('inception_v3_weights_tf_dim_ordering_tf_kernels.h5',WEIGHTS_PATH,cache_subdir='models',md5_hash='9a0d58056eeedaa3f26cb7ebd46da564')

    # model.load_weights(weights_path)
    # img_path = 'elephant.jpg'
    # img = image.load_img(img_path, target_size=(299, 299))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)

    # x = preprocess_input(x)

    # preds = model.predict(x)
    # print('Predicted:', decode_predictions(preds))
