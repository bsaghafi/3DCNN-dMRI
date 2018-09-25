# Purpose: computes occlusion maps for the high impact exposure sample or (low)
# Inputs: delta FA maps (Post-Pre) and their classification labels, the 3DCNN model, the subject to compute the occlusion map for
# Outputs: occlusion map
# Date: 01/20/2018
# Author: Behrouz Saghafi
import numpy as np
np.random.seed(2016)
import warnings
warnings.filterwarnings("ignore")

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

import timeit
import warnings
warnings.filterwarnings("ignore")
from scipy import ndimage
import nibabel as nb

start=timeit.default_timer()

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

split=['train:','test:']
data=[train_data,test_data]
target=[train_target,test_target]
# for i in range(2):
i=1
print split[i]
predicted_labels = model.predict(data[i], batch_size=bsize, verbose=0)
our_labels=np.argmax(predicted_labels, axis=1)
expected_labels=np.argmax(target[i], axis=1)
correct_classifications=np.nonzero(our_labels==expected_labels)
correct_probs=predicted_labels[correct_classifications]
confident_samples=np.argmax(correct_probs,axis=0)
# get_binary_metrics(expected_labels, our_labels)
print '\n'

np.save('probs',predicted_labels)

index_high=11
high_data_sample=test_data[index_high:index_high+1,:,:,:,:]
low_data_sample=test_data[1:2,:,:,:,:]

prob_H = model.predict(high_data_sample, batch_size=bsize, verbose=0)[:,1]
prob_L = model.predict(low_data_sample, batch_size=bsize, verbose=0)[:,0]


# FA_CD=nb.load('/project/radiology/ANSIR_lab/s174380/dev/DeepLearning/ITAKL/Cropped_downsampled_delta_FA/kids_hs022.nii')
# sample_data=np.reshape(high_data_sample,(48,60,52))
# sample_img=nb.Nifti1Image(sample_data,FA_CD.affine)
# nb.save(sample_img,'/project/radiology/ANSIR_lab/s174380/dev/DeepLearning/ITAKL/sample_img.nii')

FA_image=nb.load('/project/radiology/ANSIR_lab/s174380/dev/DeepLearning/ITAKL/FA_post_highly_impacted.nii')
FA=FA_image.get_data()
FA_mask=1.0*(FA>.02)
FA_mask_image=nb.Nifti1Image(FA_mask,np.eye(4))
nb.save(FA_mask_image,'/project/radiology/ANSIR_lab/s174380/dev/DeepLearning/ITAKL/Fa_mask.nii')

# FA_zoomed = ndimage.interpolation.zoom(FA, .5, order=3)  # cubic interpolation
stepsize=2
fmap=np.zeros((48/stepsize,60/stepsize,52/stepsize))
## high exposure sample:
for x in range(48/stepsize):
    for y in range(60/stepsize):
        for z in range(52/stepsize):
            print 'x=',x,'y=',y,'z=',z
            dFA_temp=np.array(high_data_sample)
            FA_temp=np.array(FA_mask)
            FA_box=FA_temp[stepsize * x:stepsize * x + stepsize, stepsize * y:stepsize * y + stepsize, stepsize * z:stepsize * z + stepsize]
            if np.sum(FA_box)==8:
                dFA_temp[:, :, stepsize * x:stepsize * x + stepsize, stepsize * y:stepsize * y + stepsize, stepsize * z:stepsize * z + stepsize] = 2 * np.random.random((1, 1, stepsize,stepsize,stepsize)) - 1
                prob_h = model.predict(dFA_temp, batch_size=bsize, verbose=0)[:, 1]
                print 'prob_h=',prob_h
                fmap[x,y,z]=(prob_H-prob_h)/prob_H
np.save('/project/bioinformatics/DLLab/Behrouz/dev/ITaKL/fmap_new'+str(index_high)+'.npy',fmap)
## low exposure sample:
# for x in range(48/stepsize):
#     for y in range(60/stepsize):
#         for z in range(52/stepsize):
#             print 'x=',x,'y=',y,'z=',z
#             dFA_temp=np.array(low_data_sample)
#             FA_temp=np.array(FA_mask)
#             FA_box=FA_temp[stepsize * x:stepsize * x + stepsize, stepsize * y:stepsize * y + stepsize, stepsize * z:stepsize * z + stepsize]
#             if np.sum(FA_box)==8:
#                 dFA_temp[:, :, stepsize * x:stepsize * x + stepsize, stepsize * y:stepsize * y + stepsize, stepsize * z:stepsize * z + stepsize] = 2 * np.random.random((1, 1, stepsize,stepsize,stepsize)) - 1
#                 prob_l = model.predict(dFA_temp, batch_size=bsize, verbose=0)[:, 0]
#                 print 'prob_l=',prob_l
#                 fmap[x,y,z]=(prob_L-prob_l)/prob_L
# np.save('/project/radiology/ANSIR_lab/s174380/dev/DeepLearning/ITAKL/fmap_low.npy',fmap)

feat_Map = ndimage.interpolation.zoom(fmap, stepsize, order=0,mode='nearest')  # nearest neighbor
fmap_image=nb.Nifti1Image(feat_Map,np.eye(4))
nb.save(fmap_image,'/project/bioinformatics/DLLab/Behrouz/dev/ITaKL/fmap_image_low.nii')


# for x in range(12):
#     for y in range(15):
#         for z in range(13):
#             print 'x=',x,'y=',y,'z=',z
#             hds=np.array(high_data_sample)
#             orig=np.reshape(hds[:,:,4*x:4*x+4,4*y:4*y+4,4*z:4*z+4],(4*4*4))
#             shuffled=np.random.permutation(orig)
#             hds[:, :, 4 * x:4 * x + 4, 4 * y:4 * y + 4, 4 * z:4 * z + 4]=np.reshape(shuffled,(1,1,4,4,4))
#             prob_h = model.predict(hds, batch_size=bsize, verbose=0)[:, 1]
#             print 'prob_h=',prob_h
#             fmap[x,y,z]=(prob_H-prob_h)/prob_H
# np.save('/project/radiology/ANSIR_lab/s174380/dev/DeepLearning/ITAKL/Omap_permuted.npy',fmap)

# fmap=np.zeros((12,15,13)) # 10x10x10 cubes with step of 10
# for x in range(12):
#     for y in range(15):
#         for z in range(13):
#             print 'x=',x,'y=',y,'z=',z
#             hds=np.array(high_data_sample)
#             orig=np.reshape(hds[:,:,4*x:4*x+4,4*y:4*y+4,4*z:4*z+4],(4*4*4))
#             shuffled=np.random.permutation(orig)
#             hds[:, :, 4 * x:4 * x + 4, 4 * y:4 * y + 4, 4 * z:4 * z + 4]=np.reshape(shuffled,(1,1,4,4,4))
#             prob_h = model.predict(hds, batch_size=bsize, verbose=0)[:, 1]
#             print 'prob_h=',prob_h
#             fmap[x,y,z]=(prob_H-prob_h)/prob_H
# np.save('/project/radiology/ANSIR_lab/s174380/dev/DeepLearning/ITAKL/Omap_permuted.npy',fmap)


# feat_Map = ndimage.interpolation.zoom(fmap, stepsize, order=0,mode='nearest')  # nearest neighbor
# fmap_image=nb.Nifti1Image(feat_Map,np.eye(4))
# # nb.save(fmap_image,'/project/radiology/ANSIR_lab/s174380/dev/DeepLearning/ITAKL/fmap_image.nii')
# nb.save(fmap_image,'/project/radiology/ANSIR_lab/s174380/dev/DeepLearning/ITAKL/Omap_permuted_masked'+str(index_high)+'.nii')

stop = timeit.default_timer()
print 'Total run time in mins: {}'.format((stop - start) / 60)