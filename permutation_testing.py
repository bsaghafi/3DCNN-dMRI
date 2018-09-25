# Purpose: performs permutation testing to evaluate the statistical significance of the overall model
# Inputs: delta FA maps (Post-Pre) and their classification labels, the 3DCNN model, the accuracy of the model on the test data
# Outputs: permutation scores, p-value, Probability density function (histogram)
# Date: 01/15/2018
# Author: Behrouz Saghafi
import numpy as np

np.random.seed(2016)

import os
import glob
import datetime
import time
import timeit
import warnings
import theano
warnings.filterwarnings("ignore")

# starting a timer
start = timeit.default_timer()

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers.advanced_activations import ELU,LeakyReLU, PReLU
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2,l1
from keras.wrappers.scikit_learn import KerasClassifier
from keras import __version__ as keras_version
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss, accuracy_score, recall_score, roc_auc_score, f1_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import scipy.io as sio
## Loading Train Data


def get_binary_metrics(expected_labels, our_labels):

   # sensitivity
   recall = recall_score(expected_labels, our_labels)
   print("%s: %.2f%%" % ('Sensitivity', recall * 100))
   # print '=========================='

   # specificity
   cm = confusion_matrix(expected_labels, our_labels)
   tn, fp, fn, tp = cm.ravel()
   specificity = tn / float(tn + fp)
   print("%s: %.2f%%" % ('Specificity', specificity * 100))
   print cm


   # roc_auc_score
   roc = roc_auc_score(expected_labels, our_labels)
   print("%s: %.2f%%" % ('ROC_AUC sore', roc * 100))

   # f1 score
   f1score = f1_score(expected_labels, our_labels)
   print("%s: %.2f%%" % ('F1 Score', f1score * 100))
   # print '=========================='

   accuracy = accuracy_score(expected_labels, our_labels)
   print("%s: %.2f%%" % ('Accuracy', accuracy * 100))
   # print '=========================='

   kappa=cohen_kappa_score(expected_labels, our_labels)
   print("%s: %.2f%%" % ('Kappa', kappa * 100))
   print '=========================='

   return recall, specificity, roc, f1score, accuracy, kappa


# X_train, X_test, y_train, y_test = load_data()
def load_data():
    X=np.load('/project/bioinformatics/DLLab/Behrouz/dev/ITaKL/delta_FA_CD_new_24_36.npy')
    y=np.load('/project/bioinformatics/DLLab/Behrouz/dev/ITaKL/rwe_cp_new_24_36.npy')
    return X,y

# Reshaping the Data

def read_and_normalize_data():
    X, y = load_data()

    # Xd = np.reshape(X, (X.shape[0], -1))
    # # scaler = MinMaxScaler(feature_range=(-1, 1))
    # # scaler = Normalizer(norm='l2')
    # scaler = StandardScaler()  # ZM1V transformation
    # Xd = scaler.fit_transform(Xd)
    # X = np.reshape(Xd, np.shape(X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7, stratify=y)

    print('Convert to numpy...')
    train_data = np.array(X_train, dtype=np.float64)
    # print('Normalize Train data...')
    # traind=np.reshape(train_data,(train_data.shape[0],-1))
    # # scaler=MinMaxScaler(feature_range=(-1, 1))
    # scaler = Normalizer(norm='l2')
    # # scaler = StandardScaler() #  ZM1V transformation
    # traind = scaler.fit_transform(traind)
    # train_data=np.reshape(traind,np.shape(train_data))
    # # least=np.amin(train_data)
    # # most=np.amax(train_data)
    # # print 'least is ',least,', most is ',most
    # # train_data = (train_data-least) / (most-least)
    # # print 'smallest train value is ', np.amin(train_data), ', largest train value is ', np.amax(train_data)
    # #Scikitlearn way:
    # # train_data=MinMaxScaler().fit_transform(train_data)
    # # train_data = StandardScaler().fit_transform(train_data)
    # Xmax=np.amax(train_data,axis=0)
    # Xmin=np.amin(train_data,axis=0)
    # Xmax_reptrain=[]
    # Xmin_reptrain=[]
    # for i in range(train_data.shape[0]):
    #     Xmax_reptrain.append(Xmax)
    #     Xmin_reptrain.append(Xmin)
    # Xmax_reptrain = np.array(Xmax_reptrain, dtype=np.float64)
    # Xmin_reptrain= np.array(Xmin_reptrain, dtype=np.float64)
    # # eps = np.finfo(float).eps  # 2.22e-16
    # train_data_scaled=train_data
    # train_data_scaled [Xmax_reptrain!=Xmin_reptrain]= (train_data[Xmax_reptrain!=Xmin_reptrain] - Xmin_reptrain[Xmax_reptrain!=Xmin_reptrain]) / (Xmax_reptrain[Xmax_reptrain!=Xmin_reptrain] - Xmin_reptrain[Xmax_reptrain!=Xmin_reptrain])
    # train_data=train_data_scaled
    print np.amax(train_data)
    print np.amin(train_data)



    print('Reshaping Train Data...')
    train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2], train_data.shape[3])
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')

    y_train = np.array(y_train, dtype=np.uint8)
    train_target = np_utils.to_categorical(y_train, 2)


    print('Convert to numpy...')
    test_data = np.array(X_test, dtype=np.float64)

    # print('Normalize Test data...')
    # testd = np.reshape(test_data, (test_data.shape[0], -1))
    # testd =scaler.transform(testd)
    # test_data = np.reshape(testd, np.shape(test_data))
    # # test_data = (test_data - least) / (most - least)
    # # print 'smallest test value is ', np.amin(test_data), ', largest test value is ', np.amax(test_data)
    # # test_data=MinMaxScaler().transform(test_data)
    # # test_data = StandardScaler().transform(test_data)
    # Xmax_reptest = []
    # Xmin_reptest = []
    # for i in range(test_data.shape[0]):
    #     Xmax_reptest.append(Xmax)
    #     Xmin_reptest.append(Xmin)
    # Xmax_reptest = np.array(Xmax_reptest, dtype=np.float64)
    # Xmin_reptest = np.array(Xmin_reptest, dtype=np.float64)
    # test_data_scaled=test_data
    # test_data_scaled[Xmax_reptest!=Xmin_reptest] = (test_data[Xmax_reptest!=Xmin_reptest] - Xmin_reptest[Xmax_reptest!=Xmin_reptest])/(Xmax_reptest[Xmax_reptest!=Xmin_reptest] - Xmin_reptest[Xmax_reptest!=Xmin_reptest])
    # test_data=test_data_scaled
    print np.amax(test_data)
    print np.amin(test_data)


    print('Reshaping Test Data...')
    test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1], test_data.shape[2], test_data.shape[3])
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'Test samples')

    y_test = np.array(y_test, dtype=np.uint8)
    test_target = np_utils.to_categorical(y_test, 2)


    return train_data, train_target, test_data, test_target, y_train


# plotting training performance:
def plot_training_loss_acc(history):
    #with plt.style.context(('seaborn-talk')):

        fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
        fig.set_facecolor('white')
        axs[0].plot(history.history['loss'], label='Train Set', color='red')
        axs[0].plot(history.history['val_loss'], label='Validation Set', color='blue')
        axs[0].legend(loc='upper right')
        axs[0].set_ylabel('Log Loss')
        #axs[0].set_xlabel('Epochs')
        axs[0].set_title('Training Performance')

        axs[1].plot(history.history['acc'], label='Train Set', color='red')
        axs[1].plot(history.history['val_acc'], label='Validation Set', color='blue')
        axs[1].legend(loc='lower right')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_xlabel('Epochs')  # used for both subplots
        plt.show()
        fig.savefig('learning_curves.png', bbox_inches='tight')
# Creating Model

# X_train,X_test,y_train,y_test=load_data()
train_data, train_target, test_data, test_target, y_train = read_and_normalize_data()

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

print train_data.shape
print train_target.shape

bsize = 10

### load the model:
from keras.models import load_model
K.set_image_dim_ordering('th') # convert to theano dimension ordering
model=load_model('/project/bioinformatics/DLLab/Behrouz/dev/ITaKL/diffusion_model.h5')

# permutation tests:
n_permutations=100
permutation_scores_roc=[]
permutation_scores_f1=[]

for i in range(n_permutations):
    print 'permutation#',i+1
    train_target_randomized = shuffle(train_target, random_state=0)
    test_target_randomized = shuffle(test_target, random_state=0)
    # print test_target_randomized

    history = model.fit(train_data, train_target_randomized,
                        batch_size=bsize, class_weight={0:1, 1:1.5}, nb_epoch=100, verbose=0,
                        shuffle=False)
    predicted_labels = model.predict(test_data, batch_size=bsize, verbose=0)
    our_labels = np.argmax(predicted_labels, axis=1)
    expected_labels = np.argmax(test_target_randomized, axis=1)
    roc = roc_auc_score(expected_labels, our_labels)
    print roc
    permutation_scores_roc.append(roc)
    f1score = f1_score(expected_labels, our_labels)
    print f1score
    permutation_scores_f1.append(f1score)

acc=0.8571
permutation_scores = np.array(permutation_scores_roc)
pvalue = (np.sum(permutation_scores >= acc) + 1.0) / (n_permutations + 1)
np.save('permutation_scores_roc',permutation_scores)
print ('pvalue_roc=',pvalue)

acc=0.8333
permutation_scores = np.array(permutation_scores_f1)
pvalue = (np.sum(permutation_scores >= acc) + 1.0) / (n_permutations + 1)
np.save('permutation_scores_f1',permutation_scores)
print ('pvalue_f1=',pvalue)

stop = timeit.default_timer()
print 'Total run time in mins: {}'.format((stop - start) / 60)

#draw histogram:
result=plt.hist(permutation_scores*100, 11)
plt.axvline(acc*100, color='r', linestyle='dashed')
plt.xlabel('Accuracy(%)')
plt.ylabel('Counts(#)')
plt.title('F1-Score Histogram Plot (Permutation Analysis)')
plt.axis([0, 100, 0, 200])
plt.grid(True)
plt.show()