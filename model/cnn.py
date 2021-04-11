from keras.layers import *
from keras.models import *
from utils.tools import LC

def cnn(classes , in_shp=[2,128], weights=None,**kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')
    dr = 0.6

    _in_ = Input(shape=in_shp)
    input_x = Reshape(in_shp + [1])(_in_)
    # input_x_padding = ZeroPadding2D((0, 2))(input_x)
    #加强特征
    x = Convolution2D(20, (2, 8), padding='same', activation='relu', name="conv11",
                              kernel_initializer='glorot_uniform')(input_x)
    x = BatchNormalization()(x)
    # locnet = Permute((3,2,1))(locnet)
    # locnet = Lambda(LC)(locnet)
    # locnet = Permute((3,2,1))(locnet)
    # locnet = Activation('relu')(locnet)
    x = Dropout(dr)(x)
    #######baseline classifier#######
    x = ZeroPadding2D((1, 2))(x)
    x = Conv2D(kernel_initializer="glorot_uniform", activation="linear", padding="valid", name="conv21", filters=128,
               kernel_size=(2, 16))(x)
    x = BatchNormalization()(x)
    x = Permute((3, 2, 1))(x)
    x = Lambda(LC)(x)
    x = Permute((3, 2, 1))(x)
    x = Activation('relu')(x)
    x = Dropout(dr)(x)

    x = ZeroPadding2D((0, 2))(x)
    x = Conv2D(kernel_initializer="glorot_uniform", activation="relu", padding="valid", name="conv22", filters=64,
               kernel_size=(2, 8))(x)
    x = BatchNormalization()(x)
    x = Dropout(dr)(x)
    #x = MaxPooling2D(pool_size=(1, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu', init='he_normal', name="dense1")(x)
    x = BatchNormalization()(x)
    x = Dropout(dr)(x)
    x = Dense(128, activation='relu', init='he_normal', name="dense2")(x)
    x = BatchNormalization()(x)
    x = Dropout(dr)(x)
    x = Dense(len(classes), init='he_normal', name="dense3")(x)
    x = Activation('softmax')(x)
    _out_ = Reshape([len(classes)])(x)

    model = Model(inputs=_in_, outputs=_out_)

    return model