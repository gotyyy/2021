import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K

# Define the linear combination
def LC(x):
    y = K.constant([0, 1, 0, -1, 0, 1],shape=[2,3])
    return K.dot(x,K.transpose(y))

def show_history(history,name):
    plt.figure(figsize = (7,7))
    plt.plot(history.epoch, history.history['loss'], label=name+":Training Error + Loss")
    plt.plot(history.epoch, history.history['val_loss'], label=name+":Validation Error")
    plt.xlabel('Epoch')
    plt.ylabel('% Error')
    plt.legend()
    plt.savefig('figures/history_'+name+'.jpg')
    plt.show()

# Create a function to plot the confusion matrices
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[],save_filename=None):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_filename is not None:
        plt.savefig(save_filename)
    plt.show()

def plot_cm_all(X,Y_test, test_Y_hat,name, classes):
    #test_Y_hat = model.predict(X, batch_size=batch_size)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,X.shape[0]):
        j = list(Y_test[i,:]).index(1)
        k = int(np.argmax(test_Y_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Overall Accuracy - "+ name +": ", cor / (cor+ncor))
    acc = 1.0*cor/(cor+ncor)
    plt.figure()
    plot_confusion_matrix(confnorm, labels=classes)
    plt.savefig('figures/Confusion_'+name+'.jpg',transparent = True, bbox_inches = 'tight', pad_inches = 0.01)


def calculate_accuracy_each_snr(X, Y, model, name, classes, snrs, test_SNRs, plot_acc_snr=True):
    acc_cnn = []
    for snr in snrs:

        # extract classes @ SNR
        #test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
        # print(test_SNRs)
        test_X_i = X[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y[np.where(np.array(test_SNRs) == snr)]

        # estimate classes
        test_Y_i_hat = model.predict(test_X_i)
        conf = np.zeros([len(classes), len(classes)])
        confnorm = np.zeros([len(classes), len(classes)])
        for i in range(0, test_X_i.shape[0]):
            j = list(test_Y_i[i, :]).index(1)
            k = int(np.argmax(test_Y_i_hat[i, :]))
            conf[j, k] = conf[j, k] + 1
        for i in range(0, len(classes)):
            confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
        if snr == -20 or snr == 18:
            plt.figure()
            plot_confusion_matrix(confnorm, labels=classes)
            plt.savefig('figures/Confusion_' + name + '_' + str(snr) + '.jpg', transparent=True,
                        bbox_inches='tight', pad_inches=0.01)

        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        # print ("Overall Accuracy: ", cor / (cor+ncor))
        acc_cnn.append(1.0 * cor / (cor + ncor))
    print(acc_cnn)

    if plot_acc_snr:
        plt.figure()
        plot_acc_each_snr(acc_cnn, name)
        plt.savefig('figures/Classification_Accuracy_All_' + name + '.jpg', transparent=True, bbox_inches='tight',
                    pad_inches=0.01)

def plot_acc_each_snr(acc,name):
    plt.plot(range(-20,20,2),np.array(acc)*100, label = name, color = 'green')
    plt.grid()
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy (%)')
    plt.legend()