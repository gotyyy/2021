from keras.layers import *
from keras.models import *
from layers.attention import *
from utils.tools import LC

def rtn_da(classes , in_shp=[2,128], weights=None,**kwargs):

    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')
    dr = 0.6
    _in_ = Input(shape=in_shp)
    input_x = Reshape(in_shp + [1])(_in_)
    # input_x_padding = ZeroPadding2D((0, 2))(input_x)
    input_stn = Convolution2D(20, (2, 8), padding='same', activation='relu', name="conv11",
                              kernel_initializer='glorot_uniform')(input_x)
    input_stn = BatchNormalization()(input_stn)
    # locnet = Permute((3,2,1))(locnet)
    # locnet = Lambda(LC)(locnet)
    # locnet = Permute((3,2,1))(locnet)
    # locnet = Activation('relu')(locnet)
    input_stn = Dropout(dr)(input_stn)
    ##############STN start:##################
    #pam = PAM()(input_stn)
    #pam = Convolution2D(20, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(pam)
    #pam = BatchNormalization()(pam)
    #pam = Activation('relu')(pam)
    #pam = Dropout(dr)(pam)
    locnet_size = list(np.shape(input_stn))
    input_dim = int(locnet_size[-1] * locnet_size[-2])
    timesteps = int(locnet_size[-3])
    locnet = Flatten()(input_stn)
    locnet = Reshape((1, input_dim * timesteps))(locnet)
    lstm_out = LSTM(32, return_sequences=True)(locnet)
    lstm_out = LSTM(64, return_sequences=True)(lstm_out)
    locnet = Dense(64, activation='relu')(lstm_out)
    locnet = Dropout(dr)(locnet)
    weights = get_initial_weights(64)
    locnet = Dense(6, weights=weights)(locnet)
    rtn_out = BilinearInterpolation((2, 128))([input_stn, locnet])
    #######position attention module######################
    pam = PAM()(rtn_out)
    pam = Convolution2D(20, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(pam)
    pam = BatchNormalization()(pam)
    pam = Activation('relu')(pam)
    pam = Dropout(dr)(pam)
    #######channel attention module#######################
    #cam = CAM()(rtn_out)
    #cam = Convolution2D(20, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(cam)
    #cam = BatchNormalization()(cam)
    #cam = Activation('relu')(cam)
    #cam = Dropout(dr)(cam)
    ########attention fusion########################
    #attention_sum = add([cam, pam])
    #attention_sum = Dropout(dr)(attention_sum)
    #############classifier###############
    x = ZeroPadding2D((1, 2))(pam)
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
    # x = MaxPooling2D(pool_size= (1,2))(x)
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
