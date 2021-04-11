import keras.backend as K
import tensorflow as tf
from keras.models import Model
from keras.layers import *
from utils.tools import LC

dr=0.6
def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = ZeroPadding2D((0, 2))(input_tensor)
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a', init='glorot_uniform')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Dropout(dr)(x)

    x = ZeroPadding2D((0, 2))(x)
    x = Conv2D(filters2, kernel_size,
               padding='valid', name=conv_name_base + '2b', init='glorot_uniform')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Dropout(dr)(x)

    #x = ZeroPadding2D((0, 2), data_format="channels_first")(x)
    x = Conv2D(filters3, (1, 3), name=conv_name_base + '2c', init='glorot_uniform')(x)  # (1, 1)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Dropout(dr)(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    input_tensor_padding = ZeroPadding2D((0, 2))(input_tensor)
    x = Conv2D(filters1, (1, 1),  # (1, 1)
               name=conv_name_base + '2a',padding='valid', init='glorot_uniform')(input_tensor_padding)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Dropout(dr)(x)

    # 这里必须为same,要不然就会出现下面拼接的时候尺寸不对
    x = ZeroPadding2D((0, 2))(x)
    x = Conv2D(filters2, kernel_size, padding='valid',
               name=conv_name_base + '2b', init='glorot_uniform')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Dropout(dr)(x)

    #x = ZeroPadding2D((0, 2), data_format="channels_first")(x)
    x = Conv2D(filters3, (1, 3), name=conv_name_base + '2c', padding='valid', init='glorot_uniform')(x)  # (1, 1)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Dropout(dr)(x)

    shortcut = Conv2D(filters3, (1, 1),   # (1, 1)
                      name=conv_name_base + '1', init='glorot_uniform')(input_tensor_padding)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Dropout(dr)(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def residual_stack_complex(x, kernel_size, filters):
    filters1, filters2, filters3 = filters

    def residual_unit(y):
        shortcut_unit = y
        # 1x1 conv linear
        y = ZeroPadding2D((0, 2))(y)
        y = Convolution2D(filters2, kernel_size, padding='valid', activation='relu',
                          kernel_initializer='glorot_uniform')(y)
        y = BatchNormalization()(y)
        y = Dropout(dr)(y)
        y = ZeroPadding2D((0, 2))(y)
        y = Convolution2D(filters3, kernel_size, padding='valid', activation='linear',
                          kernel_initializer='glorot_uniform')(y)
        y = BatchNormalization()(y)
        y = Dropout(dr)(y)
        # add batch normalization
        y = add([shortcut_unit, y])
        return y

    x = ZeroPadding2D((1, 2))(x)
    x = Convolution2D(filters1, (2, 3), padding='valid', activation='linear', kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Permute((3, 2, 1))(x)
    x = Lambda(LC)(x)
    x = Permute((3, 2, 1))(x)
    x = Activation('relu')(x)
    x = Dropout(dr)(x)
    x = residual_unit(x)
    x = residual_unit(x)
    # maxpool for down sampling
    # x = MaxPooling2D((1,2),data_format="channels_first")(x)
    return x