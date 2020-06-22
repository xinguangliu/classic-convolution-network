import tensorflow as tf
from tensorflow import keras
from keras import Model,Sequential
from keras.layers import Flatten, Dense, Conv2D, GlobalAveragePooling2D
from keras.layers import Input, MaxPooling2D, GlobalMaxPooling2D
from keras.utils.data_utils import get_file
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'


def VGG16(num_classes):
    # 224，224，3
    image_input = Input(shape = (224,224,3))
    # 第一个卷积部分
    # 112，112，64
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same',name = 'block1_conv1')(image_input)    # 224，224，64
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same', name = 'block1_conv2')(x)             # 224，224，64
    x = MaxPooling2D((2,2), strides = (2,2), name = 'block1_pool')(x)                               # 112，112，64

    # 第二个卷积部分
    # 56,56,128
    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv1')(x)             # 112，112，128
    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv2')(x)             # 112，112，128
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block2_pool')(x)                                 # 56，56，128

    # 第三个卷积部分
    # 28,28,256
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv1')(x)             # 56，56，256
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv2')(x)             # 56，56，256
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv3')(x)             # 56，56，256
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block3_pool')(x)                                 # 28，28，256

    # 第四个卷积部分
    # 14,14,512
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv1')(x)            # 28，28，512
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv2')(x)            # 28，28，512
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv3')(x)            # 28，28，512
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block4_pool')(x)                                 # 14，14，512

    # 第五个卷积部分
    # 7,7,512
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv1')(x)            # 14，14，512
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv2')(x)            # 14，14，512
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv3')(x)            # 14，14，512
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block5_pool')(x)                                 # 7，7，512
    # 提取特征

    # 分类部分
    # 7x7x512
    # 25088
    x = Flatten(name='flatten')(x)                                                                  # 7x7x512=25088
    x = Dense(4096, activation='relu', name='fc1')(x)                                               # 4096
    x = Dense(4096, activation='relu', name='fc2')(x)                                               # 4096
    x = Dense(num_classes, activation='softmax', name='predictions')(x)                             # 目标类别数

    # Model API
    model = Model(image_input,x,name = 'vgg16')

    return model

if __name__ == '__main__':
    model = VGG16(1000)     # 参数为目标类别个数

    print(model.summary())

    # # 在线下载权重文件
    # weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
    #                                 WEIGHTS_PATH,
    #                                 cache_subdir='models')
    # model.load_weights(weights_path)

    # # 测试图片
    # img_path = 'elephant.jpg'
    # img = image.load_img(img_path, target_size=(224, 224))      # 载入图片，并调整图片尺寸

    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)       # 在最前面添加一个维度
    # x = preprocess_input(x)             # 预处理
    # print('Input image shape:', x.shape)

    # preds = model.predict(x)
    # print('Predicted:', decode_predictions(preds))

