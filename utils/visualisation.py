import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
from dataloader.data_2016 import *
from keras.models import *

#X_train_AP,Y_train,z_train,X_test_AP,Y_test,z_test,classes, snrs,test_SNRs=data_load("2016.04C.multisnr.pkl")
#X_train_AP,Y_train,z_train,X_test_AP,Y_test,z_test,classes, snrs,test_SNRs=data_load("../dataset/2016.04C.multisnr.pkl")

classes=['8PSK',
 'AM-DSB',
 'AM-SSB',
 'BPSK',
 'CPFSK',
 'GFSK',
 'PAM4',
 'QAM16',
 'QAM64',
 'QPSK',
 'WBFM']

def get_labels_single(y):
    b = np.argwhere(y == 1)
    label = classes[b[0][0]]
    return label


def get_labels(y):
    labels = []
    for i in y:
        b = np.argwhere(i == 1)
        label_index = b[0][0]
        label = classes[label_index]
        labels.append(label)
    return labels


# 某SNR下的信号索引,返回（原索引，调制类别）数组
def input_snr_index(y_inputs, z_inputs, snr_target,z_train):
    index_group = []
    for i_snr in range(len(z_inputs)):
        if z_train[i_snr] == snr_target:
            index = i_snr
            index_group.append(index)
    labels_get = []
    for i_type in y_inputs[index_group[:]]:
        b = np.argwhere(i_type == 1)
        label_get = b[0][0]
        labels_get.append(label_get)

    result_get = np.vstack((index_group, labels_get)).transpose((1, 0))
    return result_get


# 获取某一SNR下前几种不同的信号的索引,输入二维数组，返回list
def input_modelation_index(inputs):
    resultList = []
    resultIndex = []
    for index in range(len(inputs)):
        if not inputs[index, 1] in resultList:
            resultList.append(inputs[index, 1])
            resultIndex.append(inputs[index, 0])

    return resultIndex


#输入信号可视化函数(subplots子图分布代码还需改善)
def input_visual_iq(x_datas,y_labels,z_snrs):
    x_inputs = x_datas
    labels = get_labels(y_labels)
    Zsnr = z_snrs
    t = np.arange(0,128)
    figs,axes = plt.subplots(3, 4,sharex=True,sharey=True,figsize=(10,5))
    for ax, x_input, lbl, zsnr in zip(axes.flat, x_inputs, labels, Zsnr):
        ax.set_title(lbl+str(zsnr)+"dB")
        ax.plot(t, x_input[0,:], label='I')
        ax.plot(t, x_input[1,:], label='Q')
        ax.legend()
    plt.tight_layout()
    plt.savefig('vis/input_' + lbl+str(zsnr) + 'dB.jpg')
    plt.show()

#输入信号可视化函数(subplots子图分布代码还需改善)
def input_visual_AP(x_datas,y_labels,z_snrs):
    x_inputs = x_datas
    labels = get_labels(y_labels)
    Zsnr = z_snrs
    t = np.arange(0,128)
    figs,axes = plt.subplots(3, 4,sharex=True,sharey=True,figsize=(22,8))
    for ax, x_input, lbl, zsnr in zip(axes.flat, x_inputs, labels, Zsnr):
        ax.set_title(lbl+str(zsnr)+"dB")
        ax.plot(t, x_input[0,:], label='A')
        ax.plot(t, x_input[1,:], label='P')
        ax.legend()
    plt.tight_layout()
    plt.savefig('vis/input_' + lbl+str(zsnr) + 'dB.jpg')
    plt.show()


# 卷积层输出可视化
def conv_outputvisual(model, layer_index, x, x_index,y_train,z_train):
    # this is the placeholder for the input
    input_data = model.input
    # this is the placeholder for the conv output
    out_conv = model.get_layer(index=layer_index).output
    # get the intermediate layer model
    intermediate_layer_model = Model(inputs=input_data, outputs=out_conv)
    # get the output of intermediate layer model
    intermediate_output = intermediate_layer_model.predict(np.expand_dims(x, axis=0))
    minmax = MinMaxScaler()

    # 绘图
    fea_try = intermediate_output.transpose((3, 1, 2, 0)).squeeze()
    if fea_try.ndim == 2:
        feature_inputs = np.expand_dims(fea_try, axis=1)
    else:
        feature_inputs = fea_try  # (filters,1or2,t)

    filters = feature_inputs.shape[0]
    filters_n = list(np.arange(filters))
    # for i in range(filters):
    # feature_inputs[i] = minmax.fit_transform(feature_inputs[i])

    t = np.arange(0, len(feature_inputs[0, 0, :]))
    x_inputs = feature_inputs
    lbl = get_labels_single(y_train[x_index])
    zsnr = z_train[x_index]

    figs,axes = plt.subplots(int(np.ceil(filters//8))+1, 8,sharex=True,sharey=True,figsize=(16,1.5*(int(np.ceil(filters//8))+1)))
    plt.suptitle(out_conv.name + '_' + lbl + str(zsnr) + "dB", fontsize=15)
    if fea_try.ndim == 2:
        for ax, x_input, filter_n in zip(axes.flat, x_inputs, filters_n):
            ax.set_title("filter_" + str(filter_n), fontsize=8)
            ax.plot(t, x_input[0, :])
    else:
        for ax, x_input, filter_n in zip(axes.flat, x_inputs, filters_n):
            ax.set_title("filter_" + str(filter_n), fontsize=8)
            ax.plot(t, x_input[0, :])
            ax.plot(t, x_input[1, :])

    plt.tight_layout()
    plt.savefig('vis/' + model.layers[layer_index].name + '_' + lbl + str(zsnr) + 'dB.jpg')
    plt.show()
    return

# 可视化滤波器
def kernelvisual(model, layer_target=1, num_iterate=100):
    # 图像尺寸和通道
    img_height, img_width = K.int_shape(model.input)[1:3]   #, num_channels
    num_out = K.int_shape(model.layers[layer_target].output)[-1]

    plt.suptitle('[%s] convnet filters visualizing' % model.layers[layer_target].name)

    print('第%d层有%d个通道' % (layer_target, num_out))
    for i_kernal in range(num_out):
        input_img = model.input
        # 构建一个损耗函数，使所考虑的层的第n个滤波器的激活最大化，-1层softmax层
        if layer_target == -1:
            loss = K.mean(model.output[:, i_kernal])
        else:
            loss = K.mean(model.layers[layer_target].output[:, :, :, i_kernal])  # m*28*28*128
        # 计算图像对损失函数的梯度
        grads = K.gradients(loss, input_img)[0]
        # 效用函数通过其L2范数标准化张量
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        # 此函数返回给定输入图像的损耗和梯度
        iterate = K.function([input_img], [loss, grads])
        # 从带有一些随机噪声的灰色图像开始
        np.random.seed(0)
        # 随机图像
        # input_img_data = np.random.randint(0, 255, (1, img_height, img_width, num_channels)) # 随机
        # input_img_data = np.zeros((1, img_height, img_width, num_channels)) # 零值
        input_img_data = np.random.random((1, img_height, img_width)) * 20 + 128.  # 随机灰度
        input_img_data = np.array(input_img_data, dtype=float)
        failed = False
        # 运行梯度上升
        print('####################################', i_kernal + 1)
        loss_value_pre = 0
        # 运行梯度上升num_iterate步
        for i in range(num_iterate):
            loss_value, grads_value = iterate([input_img_data])
            if i % int(num_iterate / 5) == 0:
                print('Iteration %d/%d, loss: %f' % (i, num_iterate, loss_value))
                print('Mean grad: %f' % np.mean(grads_value))
                if all(np.abs(grads_val) < 0.000001 for grads_val in grads_value.flatten()):
                    failed = True
                    print('Failed')
                    break
                if loss_value_pre != 0 and loss_value_pre > loss_value:
                    break
                if loss_value_pre == 0:
                    loss_value_pre = loss_value
                # if loss_value > 0.99:
                #  break
            input_img_data += grads_value * 1  # e-3
        img_re = deprocess_image(input_img_data[0])
        #img_re = input_img_data[0]
        #if num_channels == 1:
            #img_re = np.reshape(img_re, (img_height, img_width))
        #else:
            #img_re = np.reshape(img_re, (img_height, img_width, num_channels))
        img_re = np.reshape(img_re, (img_height, img_width))
        t = np.arange(0, img_width)
        ax = plt.subplot(np.ceil(np.sqrt(num_out)), np.ceil(np.sqrt(num_out)), i_kernal + 1)
        ax.plot(t, img_re[0, :])
        ax.plot(t, img_re[1, :])
        #plt.imshow(img_re)  # , cmap='gray'
        #plt.figure(figsize=(16,10))
        plt.axis('off')

    #plt.figure(figsize=(10,8))
    plt.tight_layout()
    plt.savefig('vis/' + model.layers[layer_target].name+ 'convnet filters visualizing.jpg')
    plt.show()
    return

# 将浮点图像转换成有效图像
def deprocess_image(x):
    # 对张量进行规范化
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    # 转化到RGB数组
    #x *= 255
    #x = np.clip(x, 0, 255).astype('uint8')
    return x

# 卷积层输出可视化
def conv_outputvisual(model, layer_index, x, x_index,y_train,z_train):
    # this is the placeholder for the input
    input_data = model.input
    # this is the placeholder for the conv output
    out_conv = model.get_layer(index=layer_index).output
    # get the intermediate layer model
    intermediate_layer_model = Model(inputs=input_data, outputs=out_conv)
    # get the output of intermediate layer model
    intermediate_output = intermediate_layer_model.predict(np.expand_dims(x, axis=0))
    minmax = MinMaxScaler()

    # 绘图
    fea_try = intermediate_output.transpose((3, 1, 2, 0)).squeeze()
    if fea_try.ndim == 2:
        feature_inputs = np.expand_dims(fea_try, axis=1)
    else:
        feature_inputs = fea_try  # (filters,1or2,t)

    filters = feature_inputs.shape[0]
    # for i in range(filters):
    # feature_inputs[i] = minmax.fit_transform(feature_inputs[i])

    t = np.arange(0, len(feature_inputs[0, 0, :]))
    x_inputs = feature_inputs
    lbl = get_labels_single(y_train[x_index])
    zsnr = z_train[x_index]

    plt.suptitle(out_conv.name + '_' + lbl + str(zsnr) + "dB", fontsize=15)
    for i_filter in range(filters):
        ax = plt.subplot(np.ceil(np.sqrt(filters)), np.ceil(np.sqrt(filters)), i_filter + 1)

        if fea_try.ndim == 2:
            ax.set_title("filter_" + str(i_filter+1), fontsize=8)
            ax.plot(t, x_inputs[i_filter, 0, :])
        else:
            ax.set_title("filter_" + str(i_filter+1), fontsize=8)
            ax.plot(t, x_inputs[i_filter, 0, :])
            ax.plot(t, x_inputs[i_filter, 1, :])

    plt.tight_layout()
    plt.savefig('vis/' + model.layers[layer_index].name + '_' + lbl + str(zsnr) + 'dB.jpg')
    plt.show()
    return