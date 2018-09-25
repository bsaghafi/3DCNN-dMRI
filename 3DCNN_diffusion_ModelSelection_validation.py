# Purpose: Validate a 3DCNN model for model selection. It trains on 38 subjects and validates on 10. It uses dynamic learning and early stopping.
# Inputs: delta FA maps (Post-Pre) and their classification labels
# Outputs: Classification performance measures (AUC ROC, F1-scores, ...) on Train and test data
# Date: 06/20/2017
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
import matplotlib
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

def create_model():
    model = Sequential()

    model.add(ZeroPadding3D((1, 1, 1), input_shape=(1, 121,145,121), dim_ordering='th'))
    model.add(Convolution3D(4, 3, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(4, 3, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), dim_ordering='th'))

    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(8, 3, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(8, 3, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), dim_ordering='th'))

    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(32, 3, 3, 3, activation='relu', dim_ordering='th'))
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(32, 3, 3, 3, activation='relu', dim_ordering='th'))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), dim_ordering='th'))

    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(16, 3, 3, 3, activation='relu', dim_ordering='th'))
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(16, 3, 3, 3, activation='relu', dim_ordering='th'))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    # BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None)
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))
    # model.add(Dense(1, activation='sigmoid'))

    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer="sgd", loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_voxCNN2():
    lmda=5e-4
    model = Sequential()
    # model.add(BatchNormalization(input_shape=(1, 95, 120, 103), axis=1))
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(BatchNormalization(input_shape=(1, 48, 60, 52), axis=1))
    model.add(Convolution3D(8, 3, 3, 3, dim_ordering='th', name='conv_1'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    # BatchNormalization(axis=1)
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(8, 3, 3, 3, dim_ordering='th', name='conv_2'))
    model.add(PReLU())
    # BatchNormalization(axis=1)
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Dropout(0.2))
    model.add(Convolution3D(16, 3, 3, 3, dim_ordering='th', name='conv_3'))
    model.add(PReLU())
    # BatchNormalization(axis=1)
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Dropout(0.2))
    model.add(Convolution3D(16, 3, 3, 3, dim_ordering='th', name='conv_4'))
    model.add(PReLU())
    # BatchNormalization(axis=1)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))


    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Dropout(0.2))
    model.add(Convolution3D(32, 3, 3, 3, dim_ordering='th', name='conv_5'))
    model.add(PReLU())
    # BatchNormalization(axis=1)
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Dropout(0.2))
    model.add(Convolution3D(32, 3, 3, 3, dim_ordering='th', name='conv_6'))
    model.add(PReLU())
    # BatchNormalization(axis=1)
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(32, 3, 3, 3, activation='relu', dim_ordering='th', name='conv_7'))
    # BatchNormalization(axis=1)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(32, W_regularizer=l2(lmda)))
    model.add(PReLU())

    # BatchNormalization(axis=1)
    model.add(Dropout(0.5))
    model.add(Dense(8, W_regularizer=l2(lmda)))
    model.add(PReLU())
    # model.add(Dense(4, activation='relu'))
    # BatchNormalization(axis=1)
    # model.add(Dropout(0.5))
    # model.add(Dense(32, activation='relu', W_constraint=maxnorm(3)))
    # BatchNormalization(axis=1)
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    # sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.8, nesterov=True)
    sgd = SGD(lr=1e-2)
    adam =Adam(lr = 1e-4, beta_1=0.7, epsilon=1e-8)
    # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    add = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_voxCNN():
    model = Sequential()
    # model.add(BatchNormalization(input_shape=(1, 95, 120, 103), axis=1))
    model.add(BatchNormalization(input_shape=(1, 48, 60, 52), axis=1))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(8, 3, 3, 3, activation='relu', dim_ordering='th', name='conv_1'))
    BatchNormalization(axis=1)
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(8, 3, 3, 3, activation='relu', dim_ordering='th', name='conv_2'))
    BatchNormalization(axis=1)
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(16, 3, 3, 3, activation='relu', dim_ordering='th', name='conv_3'))
    BatchNormalization(axis=1)
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(16, 3, 3, 3, activation='relu', dim_ordering='th', name='conv_4'))
    BatchNormalization(axis=1)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(32, 3, 3, 3, activation='relu', dim_ordering='th', name='conv_5'))
    BatchNormalization(axis=1)
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(32, 3, 3, 3, activation='relu', dim_ordering='th', name='conv_6'))
    BatchNormalization(axis=1)
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(32, 3, 3, 3, activation='relu', dim_ordering='th', name='conv_7'))
    # BatchNormalization(axis=1)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    model.add(Flatten())
    # BatchNormalization(axis=1)
    # model.add(Dropout(0.5))
    # model.add(Dropout(0.6))
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    BatchNormalization(axis=1)
    model.add(Dropout(0.3))
    model.add(Dense(8, activation='relu'))
    # model.add(Dense(4, activation='relu'))
    # BatchNormalization(axis=1)
    # model.add(Dropout(0.5))
    # model.add(Dense(32, activation='relu', W_constraint=maxnorm(3)))
    # BatchNormalization(axis=1)
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.8, nesterov=True)
    sgd = SGD(lr=1e-2)
    adam =Adam(lr = 1e-3, beta_1=0.7)
    # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    add = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_voxCNN_prelu2():
    model = Sequential()
    # model.add(BatchNormalization(input_shape=(1, 95, 120, 103), axis=1))
    # model.add(BatchNormalization(input_shape=(1, 24, 30, 26), axis=1))
    model.add(BatchNormalization(input_shape=(1, 48, 60, 52), axis=1))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(8, 3, 3, 3, dim_ordering='th'))
    model.add(PReLU())
    BatchNormalization(axis=1)
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(8, 3, 3, 3, dim_ordering='th'))
    model.add(PReLU())
    BatchNormalization(axis=1)
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(16, 3, 3, 3, dim_ordering='th'))
    model.add(PReLU())
    BatchNormalization(axis=1)
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(16, 3, 3, 3, dim_ordering='th'))
    model.add(PReLU())
    BatchNormalization(axis=1)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(32, 3, 3, 3, dim_ordering='th'))
    model.add(PReLU())
    BatchNormalization(axis=1)
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(32, 3, 3, 3, dim_ordering='th'))
    model.add(PReLU())
    BatchNormalization(axis=1)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(128, 3, 3, 3, dim_ordering='th', name='conv_7'))
    # model.add(PReLU())
    # BatchNormalization(axis=1)
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(128, 3, 3, 3, dim_ordering='th', name='conv_8'))
    # model.add(PReLU())
    # BatchNormalization(axis=1)
    # model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(PReLU())
    BatchNormalization(axis=1)
    model.add(Dropout(0.3))
    model.add(Dense(8))
    model.add(PReLU())
    model.add(Dense(2, activation='softmax'))

    # sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.8, nesterov=True)
    sgd = SGD(lr=1e-2)
    # adam =Adam(lr = 1e-3, beta_1=0.7)
    # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    add = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_voxCNN_prelu2_BatchN():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(1, 48, 60, 52), axis=1))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(8, 3, 3, 3, dim_ordering='th'))
    model.add(PReLU())
    model.add(BatchNormalization(axis=1))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(8, 3, 3, 3, dim_ordering='th'))
    model.add(PReLU())
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(16, 3, 3, 3, dim_ordering='th'))
    model.add(PReLU())
    model.add(BatchNormalization(axis=1))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(16, 3, 3, 3, dim_ordering='th'))
    model.add(PReLU())
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(32, 3, 3, 3, dim_ordering='th'))
    model.add(PReLU())
    model.add(BatchNormalization(axis=1))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(32, 3, 3, 3, dim_ordering='th'))
    model.add(PReLU())
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(PReLU())
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.3))
    model.add(Dense(8))
    model.add(PReLU())
    model.add(Dense(2, activation='softmax'))

    # sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.8, nesterov=True)
    sgd = SGD(lr=1e-2)
    # adam =Adam(lr = 1e-3, beta_1=0.7)
    # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    add = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_voxCNN_prelu():
    model = Sequential()
    # model.add(BatchNormalization(input_shape=(1, 95, 120, 103), axis=1))
    model.add(BatchNormalization(input_shape=(1, 24, 30, 26), axis=1))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(8, 3, 3, 3, dim_ordering='th', name='conv_1'))
    model.add(PReLU())
    BatchNormalization(axis=1)
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(8, 3, 3, 3, dim_ordering='th', name='conv_2'))
    model.add(PReLU())
    BatchNormalization(axis=1)
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(16, 3, 3, 3, dim_ordering='th', name='conv_3'))
    model.add(PReLU())
    BatchNormalization(axis=1)
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(16, 3, 3, 3, dim_ordering='th', name='conv_4'))
    model.add(PReLU())
    BatchNormalization(axis=1)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(32, 3, 3, 3, dim_ordering='th', name='conv_5'))
    model.add(PReLU())
    BatchNormalization(axis=1)
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(32, 3, 3, 3, dim_ordering='th', name='conv_6'))
    model.add(PReLU())
    BatchNormalization(axis=1)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(PReLU())
    BatchNormalization(axis=1)
    model.add(Dropout(0.3))
    model.add(Dense(8))
    model.add(PReLU())
    model.add(Dense(2, activation='softmax'))

    # sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.8, nesterov=True)
    sgd = SGD(lr=1e-2)
    # adam =Adam(lr = 1e-3, beta_1=0.7)
    # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    add = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def AlexNet3D():
    model = Sequential()
    model.add(Convolution3D(32, 11, 11, 11, input_shape=(1, 95, 120, 103), subsample=(2,2,2), dim_ordering='th'))
    BatchNormalization(axis=1)
    model.add(Activation('relu'))
    # model.add(Convolution3D(16, 11, 11, 11, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    model.add(MaxPooling3D(pool_size=(2,2,2), dim_ordering='th'))

    model.add(Convolution3D(64, 5,5,5, dim_ordering='th'))
    BatchNormalization(axis=1)
    model.add(Activation('relu'))
    # model.add(Convolution3D(32, 5, 5, 5, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    model.add(MaxPooling3D(pool_size=(2,2,2), dim_ordering='th'))

    model.add(Convolution3D(128, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling3D(pool_size=(2,2,2), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    # BatchNormalization(axis=1)
    # model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    # # model.add(Dense(4, activation='relu'))
    # BatchNormalization(axis=1)
    # model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    # sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.8, nesterov=True)
    sgd = SGD(lr=1e-2)
    adam =Adam(lr = 1e-3, beta_1=0.7, epsilon=1e-8)
    # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    add = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def best_original():
    model = Sequential()
    # model.add(BatchNormalization(input_shape=(1, 95, 120, 103), axis=1))
    model.add(BatchNormalization(input_shape=(1, 48, 60, 52), axis=1))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(16, 3, 3, 3, activation='relu', dim_ordering='th'))
    BatchNormalization(axis=1)
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(16, 3, 3, 3, activation='relu', dim_ordering='th'))
    BatchNormalization(axis=1)
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(32, 3, 3, 3, activation='relu', dim_ordering='th'))
    BatchNormalization(axis=1)
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(32, 3, 3, 3, activation='relu', dim_ordering='th'))
    BatchNormalization(axis=1)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', dim_ordering='th'))
    BatchNormalization(axis=1)
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', dim_ordering='th'))
    BatchNormalization(axis=1)
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', dim_ordering='th', name='conv_7'))
    BatchNormalization(axis=1)
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(64, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    model.add(Flatten())
    # BatchNormalization(axis=1)
    # model.add(Dropout(0.5))
    # model.add(Dropout(0.6))
    # model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # BatchNormalization(axis=1)
    # model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu'))
    # BatchNormalization(axis=1)
    # model.add(Dropout(0.5))
    # model.add(Dense(32, activation='relu', W_constraint=maxnorm(3)))
    # BatchNormalization(axis=1)
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.8, nesterov=True)
    # sgd = SGD(lr=1e-2)
    adam =Adam(lr = 1e-3, beta_1=0.7)
    # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # add = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def best_original_v1():
    model = Sequential()
    # model.add(BatchNormalization(input_shape=(1, 95, 120, 103), axis=1))
    model.add(BatchNormalization(input_shape=(1, 48, 60, 52), axis=1))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(16, 3, 3, 3, activation='relu', dim_ordering='th'))
    BatchNormalization(axis=1)
    # model.add(Dropout(p))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(16, 3, 3, 3, activation='relu', dim_ordering='th'))
    BatchNormalization(axis=1)
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))


    # model.add(Dropout(p))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(32, 3, 3, 3, activation='relu', dim_ordering='th'))
    BatchNormalization(axis=1)
    # model.add(Dropout(p))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(32, 3, 3, 3, activation='relu', dim_ordering='th'))
    BatchNormalization(axis=1)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    # model.add(Dropout(p))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', dim_ordering='th'))
    BatchNormalization(axis=1)
    # model.add(Dropout(p))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', dim_ordering='th'))
    BatchNormalization(axis=1)
    # model.add(Dropout(p))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', dim_ordering='th', name='conv_7'))
    BatchNormalization(axis=1)
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(64, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    model.add(Flatten())
    # BatchNormalization(axis=1)
    # model.add(Dropout(p))
    # model.add(Dropout(0.6))
    # model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    BatchNormalization(axis=1)
    model.add(Dropout(0.5)) # found by grid search over [0.1,0.8]
    model.add(Dense(8, activation='relu'))
    # BatchNormalization(axis=1)
    # model.add(Dropout(0.5))
    # model.add(Dense(32, activation='relu', W_constraint=maxnorm(3)))
    # BatchNormalization(axis=1)
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.8, nesterov=True)
    sgd = SGD(lr=1e-2)
    adam =Adam(lr = 1e-3, beta_1=0.7)
    # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    add = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def best_original_v2():
    model = Sequential()
    # model.add(BatchNormalization(input_shape=(1, 95, 120, 103), axis=1))
    model.add(BatchNormalization(input_shape=(1, 48, 60, 52), axis=1))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(8, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    # model.add(Dropout(p))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(8, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))


    # model.add(Dropout(p))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(16, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    # model.add(Dropout(p))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(16, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    # model.add(Dropout(p))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(32, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    # model.add(Dropout(p))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(32, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    # model.add(Dropout(p))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(32, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(64, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(64, 3, 3, 3, activation='relu', dim_ordering='th'))
    # # BatchNormalization(axis=1)
    # # model.add(Dropout(p))
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(64, 3, 3, 3, activation='relu', dim_ordering='th'))
    # # BatchNormalization(axis=1)
    # # model.add(Dropout(p))
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(64, 3, 3, 3, activation='relu', dim_ordering='th'))
    # # BatchNormalization(axis=1)
    # # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # # model.add(Convolution3D(64, 3, 3, 3, activation='relu', dim_ordering='th'))
    # # BatchNormalization(axis=1)
    # model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    model.add(Flatten())
    # BatchNormalization(axis=1)
    # model.add(Dropout(0.5))
    # model.add(Dropout(p))
    # model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # BatchNormalization(axis=1)
    # model.add(Dropout(0.4))# found by grid search over [0.1,0.8]
    model.add(Dense(8, activation='relu'))
    # BatchNormalization(axis=1)
    # model.add(Dropout(0.5))
    # model.add(Dense(32, activation='relu', W_constraint=maxnorm(3)))
    # BatchNormalization(axis=1)
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.8, nesterov=True)
    sgd = SGD(lr=1e-2)
    adam =Adam(lr = 1e-3, beta_1=0.7)
    # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    add = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def suggestion_v2():
    model = Sequential()
    # model.add(BatchNormalization(input_shape=(1, 95, 120, 103), axis=1))
    model.add(BatchNormalization(input_shape=(1, 48, 60, 52), axis=1))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(8, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    # model.add(Dropout(p))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(8, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))


    # model.add(Dropout(p))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(14, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    # model.add(Dropout(p))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(14, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    # model.add(Dropout(p))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(20, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    # model.add(Dropout(p))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(20, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    # model.add(Dropout(p))
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(20, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(64, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(24, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    # model.add(Dropout(p))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(24, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    # model.add(Dropout(p))
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(64, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(64, 3, 3, 3, activation='relu', dim_ordering='th'))
    # BatchNormalization(axis=1)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    model.add(Flatten())
    # BatchNormalization(axis=1)
    # model.add(Dropout(0.5))
    # model.add(Dropout(p))
    # model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    BatchNormalization(axis=1)
    model.add(Dropout(0.4))# found by grid search over [0.1,0.8]
    model.add(Dense(64, activation='relu'))
    # BatchNormalization(axis=1)
    # model.add(Dropout(0.5))
    # model.add(Dense(32, activation='relu', W_constraint=maxnorm(3)))
    # BatchNormalization(axis=1)
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.8, nesterov=True)
    sgd = SGD(lr=1e-2)
    # adam =Adam(lr = 1e-3, beta_1=0.7)
    # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    add = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def dec_per_epoch(Loss):
    delta_loss=[]
    for i in range(1,len(Loss)-1):
        delta_loss.append(100 * (Loss[i] - Loss[i+1]) / Loss[i])
    return np.mean(delta_loss)

# K Fold Split

# X_train,X_test,y_train,y_test=load_data()
train_data, train_target, test_data, test_target, y_train = read_and_normalize_data()

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
best_epoch_history=[]
fscore=[]
deltaloss=[]
deltaloss2=[]
deltaloss_per_epoch=[]

print train_data.shape
print train_target.shape

nepoch = 500
bsize = 10
ACC=[]
ACCstd=[]
# for p in np.linspace(0.1,0.8,num=8):
cvscores=[]
cvscores2=[]
for train_index, test_index in kfold.split(train_data, y_train):
    adam = Adam(lr=1e-4, beta_1=0.7)
    # rms = RMSprop(lr=1e-3, rho=0.5)
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.5, nesterov=True)
    model = create_voxCNN_prelu2()  ######################################################################################################MODEL########
    # callbacks = [EarlyStopping(monitor='val_acc', min_delta=0, patience=15, verbose=2, mode='auto')]
    best_val=0
    best_val2=0
    best_epoch=None

    best_model=None
    counter=0
    lrcounter=0
    for epoch in range(1,nepoch+1):
            # callbacks = [ModelCheckpoint("/project/radiology/ANSIR_lab/s174380/dev/DeepLearning/ITAKL/best_model.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=50)]
            if counter>50:  #early stopping
                break
            if lrcounter>10:
                Lrate=K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr,Lrate*0.5)
                lrcounter=0
                print 'lr reduced to ',K.get_value(model.optimizer.lr)
            print 'Epoch=', epoch
            history=model.fit(train_data[train_index], train_target[train_index], class_weight={0:1, 1:1.5}, batch_size=bsize, nb_epoch=1, verbose=2, shuffle=True)
            predicted_labels = model.predict(train_data[test_index], batch_size=bsize, verbose=0)
            expected_labels=np.argmax(train_target[test_index], axis=1)
            our_labels = np.argmax(predicted_labels, axis=1)
            f1score = f1_score(expected_labels, our_labels)
            print("%s: %.2f%%" % ('F1 Score', f1score * 100))
            roc = roc_auc_score(expected_labels, our_labels)
            print("%s: %.2f%%" % ('AUC', roc * 100))
            # val_acc=history.history['val_acc'][0]
            # print 'val_acc=', val_acc
            if roc > best_val:
                    best_val = roc
                    best_val2=f1score
                    best_epoch = epoch
                    best_model = model
                    counter=0
                    lrcounter=0
            else:
                counter=counter+1
                lrcounter=lrcounter+1
            print 'best validation score= ', best_val
            print '\n'

    # score = best_model.evaluate(train_data[test_index], train_target[test_index], verbose=0)
    cvscores.append(best_val * 100)
    cvscores2.append(best_val2 * 100)
    best_epoch_history.append(best_epoch)
    # print("%s: %.2f%%" % (best_model.metrics_names[1], score[1] * 100))
    print best_val
    print '\n'
    # plot_training_loss_acc(history)
    # break
    # ACC.append(np.mean(cvscores))
    # ACCstd.append(np.std(cvscores))

# print ACC
# print ACCstd
# print np.argmax(ACC)

print '**************Validation Results:**************'
print cvscores
print best_epoch_history
print("%s: %.2f%% (+/- %.2f%%)" % ('Average AUC', np.mean(cvscores), np.std(cvscores)))
print("%s: %.2f%% (+/- %.2f%%)" % ('Average f1score', np.mean(cvscores2), np.std(cvscores2)))
print("%s: %.2f%% (+/- %.2f%%)" % ('Mean #epochs', np.mean(best_epoch_history), np.std(best_epoch_history)))
print("median epoch=", np.median(best_epoch_history))
stop = timeit.default_timer()
print 'Total run time in mins: {}'.format((stop - start) / 60)