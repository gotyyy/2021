import os,random
import numpy as np
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras.backend as K
from keras.utils import np_utils
import keras
import pickle, random, time
import math
from dataloader.data_2016 import  *
from model.cnn import *
from model.rtn_lstm import  *
from model.rtn_da import  *
from model.rtn_cbam import  *
from utils.tools import  *


#读取数据
#X_train,Y_train,X_test,Y_test,classes,snrs,test_SNRs  = data_load("../dataset/2016.04C.multisnr.pkl")
X_train,Y_train,X_test,Y_test,classes,snrs,test_SNRs  = data_load("2016.04C.multisnr.pkl")

#classes= ['8PSK','AM-DSB','AM-SSB','BPSK','CPFSK','GFSK','PAM4','QAM16','QAM64','QPSK','WBFM']

#超参数初始化
epochs = 100
batch_size = 512

K.clear_session()
#加载模型
model_name = 'rtn_ca'
#model_name = 'cnn'
#model_name = 'rtn_cbam'
#model_name = 'rtn_da'
model= rtn_1(classes , in_shp=list(X_train.shape[1:]))
#model= cnn(classes , in_shp=list(X_train.shape[1:]))
#model= rtn_cbam(classes , in_shp=list(X_train.shape[1:]))
#model= rtn_da(classes , in_shp=list(X_train.shape[1:]))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()

#训练
start = time.time()
#weight_file = '../weights/'+ model_name+ '.wts.h5'
weight_file = 'weights/'+ model_name+ '.wts.h5'
history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=epochs,
    # show_accuracy=False,
    verbose=2,
    validation_data=(X_test, Y_test),
    class_weight='auto',
    callbacks = [
        keras.callbacks.ModelCheckpoint(weight_file, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    ])
model.load_weights(weight_file)


#计算训练时间
end = time.time()
duration = end - start
print('Training time = ' + str(round(duration/60,5)) + 'minutes')

#model_path = '../models/'+model_name+ '.h5'
model_path = 'models/'+model_name+ '.h5'
model.save(model_path)

#画训练历史
show_history(history,model_name)

#评估网络
#score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
#print(score)

#预测阶段
test_Y_hat = model.predict(X_test , batch_size=batch_size)

#结果分析
#整体准确率&comfusion matric
plot_cm_all(X_test, Y_test, test_Y_hat, model_name,classes)

#每snr的准确率&最低最高Snr下的comfusion matrics
calculate_accuracy_each_snr(X_test, Y_test, model, model_name, classes, snrs, test_SNRs)


