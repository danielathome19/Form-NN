import datetime
import glob
import math
import re
import time
from threading import Thread
import glob as gb
import librosa
import matplotlib.pyplot as plt
import librosa.display
import pyaudio
import wave
from IPython.display import Audio
from matplotlib.pyplot import specgram
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import IPython.display as ipd  # To play sound in the notebook
import os  # interface with underlying OS that python is running on
import soundfile as sf
import sys
import warnings
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras as keras
from sklearn.svm import LinearSVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization, Dense
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l1, l2
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import statistics
from tensorflow.keras.models import Sequential, load_model
from sklearn import tree
from sklearn.dummy import DummyClassifier
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils
from pydub import AudioSegment
import random
import tensorflow.keras.layers as kl
import tensorflow.keras.applications as ka
import tensorflow.keras.optimizers as ko
import tensorflow.keras.models as km
import skimage.measure
import scipy
from scipy.spatial import distance
from numpy import inf
import audioread
import librosa.segment
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import math
from scipy import signal
import numpy as np
from os import listdir, walk, getcwd, sep
import data_utils as du
import data_utils_input as dus
from data_utils_input import normalize_image, padding_MLS, padding_SSLM, borders
from keras import backend as k
from shutil import copyfile
import fnmatch
from sklearn import preprocessing
import hyperas
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from ast import literal_eval
from sklearn.feature_selection import RFE
from skimage.transform import resize
import autokeras as ak
from tensorflow.python.ops.init_ops_v2 import glorot_uniform

k.set_image_data_format('channels_last')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if not sys.warnoptions:
    warnings.simplefilter("ignore")  # ignore warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# region Directories
MASTER_DIR = 'D:/Google Drive/Resources/Dev Stuff/Python/Machine Learning/Master Thesis/'
MASTER_INPUT_DIR = 'F:/Master Thesis Input/'
MASTER_LABELPATH = os.path.join(MASTER_INPUT_DIR, 'Labels/')

MIDI_Data_Dir = np.array(gb.glob(os.path.join(MASTER_DIR, 'Data/MIDIs/*')))
FULL_DIR = os.path.join(MASTER_INPUT_DIR, 'Full/')
FULL_LABELPATH = os.path.join(MASTER_LABELPATH, 'Full/')
# endregion


# region DEPRECATED
# Deprecated
Train_Data_Dir = np.array(gb.glob(os.path.join(MASTER_INPUT_DIR, 'Train/*')))  # os.path.join(MASTER_DIR, 'Data/Train/*'
Test_Data_Dir = np.array(gb.glob(os.path.join(MASTER_INPUT_DIR, 'Test/*')))  # os.path.join(MASTER_DIR, 'Data/Test/*')))
Validate_Data_Dir = np.array(gb.glob(os.path.join(MASTER_INPUT_DIR, 'Validate/*')))  # os.path.join(MASTER_DIR,'Data/Val

MLS_Data_Dir = os.path.join(MASTER_DIR, 'Images/Train/MLS/')
SSLMCOS_Data_Dir = os.path.join(MASTER_DIR, 'Images/Train/SSLMCOS/')
SSLMEUC_Data_Dir = os.path.join(MASTER_DIR, 'Images/Train/SSLMEUC/')
SSLMCRM_Data_Dir = os.path.join(MASTER_DIR, 'Images/Train/SSLMCRM/')

TRAIN_DIR = os.path.join(MASTER_INPUT_DIR, 'Train/')
TEST_DIR = os.path.join(MASTER_INPUT_DIR, 'Test/')
VAL_DIR = os.path.join(MASTER_INPUT_DIR, 'Validate/')

TRAIN_LABELPATH = os.path.join(MASTER_LABELPATH, 'Train/')
TEST_LABELPATH = os.path.join(MASTER_LABELPATH, 'Test/')
VAL_LABELPATH = os.path.join(MASTER_LABELPATH, 'Validate/')


# Deprecated
def validate_directories():
    print("Validating Training Directory...")
    dus.validate_folder_contents(TRAIN_LABELPATH, os.path.join(TRAIN_DIR, 'MIDI/'), os.path.join(TRAIN_DIR, 'MLS/'),
                                 os.path.join(TRAIN_DIR, 'SSLM_CRM_COS/'), os.path.join(TRAIN_DIR, 'SSLM_CRM_EUC/'),
                                 os.path.join(TRAIN_DIR, 'SSLM_MFCC_COS/'), os.path.join(TRAIN_DIR, 'SSLM_MFCC_EUC/'))
    print("Succes.\n")
    print("Validating Validation Directory...")
    dus.validate_folder_contents(VAL_LABELPATH, os.path.join(VAL_DIR, 'MIDI/'), os.path.join(VAL_DIR, 'MLS/'),
                                 os.path.join(VAL_DIR, 'SSLM_CRM_COS/'), os.path.join(VAL_DIR, 'SSLM_CRM_EUC/'),
                                 os.path.join(VAL_DIR, 'SSLM_MFCC_COS/'), os.path.join(VAL_DIR, 'SSLM_MFCC_EUC/'))
    print("Succes.\n")
    print("Validating Testing Directory...")
    dus.validate_folder_contents(TEST_LABELPATH, os.path.join(TEST_DIR, 'MIDI/'), os.path.join(TEST_DIR, 'MLS/'),
                                 os.path.join(TEST_DIR, 'SSLM_CRM_COS/'), os.path.join(TEST_DIR, 'SSLM_CRM_EUC/'),
                                 os.path.join(TEST_DIR, 'SSLM_MFCC_COS/'), os.path.join(TEST_DIR, 'SSLM_MFCC_EUC/'))
    print("Succes.\n")


# Deprecated
def get_class_weights(labels, one_hot=False):
    if one_hot is False:
        n_classes = max(labels) + 1
    else:
        n_classes = len(labels[0])
    class_counts = [0 for _ in range(int(n_classes))]
    if one_hot is False:
        for label in labels:
            class_counts[label] += 1
    else:
        for label in labels:
            class_counts[np.where(label == 1)[0][0]] += 1
    return {i: (1. / class_counts[i]) * float(len(labels)) / float(n_classes) for i in range(int(n_classes))}


# Deprecated
def buildValidationSet():
    cnt = 1
    numtrainfiles = len(fnmatch.filter(os.listdir(os.path.join(TRAIN_DIR, "MLS/")), '*.npy'))
    for file in os.listdir(os.path.join(TRAIN_DIR, "MLS/")):
        numvalfiles = len(fnmatch.filter(os.listdir(os.path.join(VAL_DIR, "MLS/")), '*.npy'))
        if numvalfiles >= numtrainfiles * 0.2:
            print(f"Validation set >= 20% of training set: {numvalfiles}/{numtrainfiles}")
            break
        filename, name = file, file.split('/')[-1].split('.')[0]
        print(f"\nWorking on {os.path.basename(name)}, file #" + str(cnt))
        formfolder = ""  # Start search for correct form to search for label
        for root, dirs, files in os.walk(os.path.join(MASTER_DIR, 'Labels/')):
            flag = False
            for tfile in files:
                if tfile.split('/')[-1].split('.')[0] == name:
                    formfolder = os.path.join(root, file).split('/')[-1].split('\\')[0]
                    flag = True
            if flag:
                break

        path = os.path.join(os.path.join(MASTER_DIR, 'Labels/'), formfolder) + '/' + os.path.basename(name) + '.txt'
        num_lines = sum(1 for _ in open(path))
        if num_lines <= 2:
            print("File has not been labeled with ground truth yet. Skipping...")
            cnt += 1
            continue
        else:
            src1 = os.path.join(TRAIN_DIR, "MLS/") + '/' + filename
            src2 = os.path.join(TRAIN_DIR, "SSLM_CRM_COS/") + '/' + filename
            src3 = os.path.join(TRAIN_DIR, "SSLM_CRM_EUC/") + '/' + filename
            src4 = os.path.join(TRAIN_DIR, "SSLM_MFCC_COS/") + '/' + filename
            src5 = os.path.join(TRAIN_DIR, "SSLM_MFCC_EUC/") + '/' + filename
            dst1 = os.path.join(VAL_DIR, "MLS/") + '/' + filename
            dst2 = os.path.join(VAL_DIR, "SSLM_CRM_COS/") + '/' + filename
            dst3 = os.path.join(VAL_DIR, "SSLM_CRM_EUC/") + '/' + filename
            dst4 = os.path.join(VAL_DIR, "SSLM_MFCC_COS/") + '/' + filename
            dst5 = os.path.join(VAL_DIR, "SSLM_MFCC_EUC/") + '/' + filename
            if os.path.exists(dst1) and os.path.exists(dst2) and os.path.exists(dst3) and os.path.exists(dst4) \
                    and os.path.exists(dst5):
                print("File has already been prepared for training material. Skipping...")
                cnt += 1
                continue
            else:
                copyfile(src1, dst1)
                copyfile(src2, dst2)
                copyfile(src3, dst3)
                copyfile(src4, dst4)
                copyfile(src5, dst5)

        cnt += 1
    pass


# Deprecated
def findBestShape(mls_train, sslm_train):
    dim1_mls = [i.shape[0] for i in mls_train.getImages()]
    dim2_mls = [i.shape[1] for i in mls_train.getImages()]
    print(dim1_mls)
    print(dim2_mls)

    dim1_sslm = [i.shape[0] for i in sslm_train.getImages()]
    dim2_sslm = [i.shape[1] for i in sslm_train.getImages()]
    print(dim1_sslm)
    print(dim2_sslm)

    dim1_mean = min(statistics.mean(dim1_mls), statistics.mean(dim2_sslm))
    dim2_mean = min(statistics.mean(dim1_mls), statistics.mean(dim2_sslm))

    dim1_median = min(statistics.median(dim1_mls), statistics.median(dim2_sslm))
    dim2_median = min(statistics.median(dim1_mls), statistics.median(dim2_sslm))

    dim1_mode = min(statistics.mode(dim1_mls), statistics.mode(dim2_sslm))
    dim2_mode = min(statistics.mode(dim1_mls), statistics.mode(dim2_sslm))

    print(f"Dimension 0:\nMean: {dim1_mean}\t\tMedian: {dim1_median}\t\tMode: {dim1_mode}")
    print(f"Dimension 1:\nMean: {dim2_mean}\t\tMedian: {dim2_median}\t\tMode: {dim2_mode}")
# Deprecated WORKING FUSE MODEL
def old_formnn_fuse(output_channels=32, lrval=0.00001, numclasses=12):
    cnn1_mel = formnn_mls(output_channels, lrval=lrval)
    cnn1_sslm = formnn_sslm(output_channels, lrval=lrval)
    combined = layers.concatenate([cnn1_mel.output, cnn1_sslm.output], axis=2)
    cnn2_in = formnn_pipeline(combined, output_channels, lrval=lrval, numclasses=numclasses)
    cnn2_in = layers.Dense(numclasses, activation='sigmoid')(cnn2_in)
    opt = keras.optimizers.Adam(lr=lrval)
    model = keras.models.Model(inputs=[cnn1_mel.input, cnn1_sslm.input], outputs=[cnn2_in])
    model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer=opt, metrics=['accuracy'])
    model.summary()  # Try categorical_crossentropy, metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    if not os.path.isfile(os.path.join(MASTER_DIR, 'FormNN_Model_Diagram.png')):
        plot_model(model, to_file=os.path.join(MASTER_DIR, 'FormNN_Model_Diagram.png'),
                   show_shapes=True, show_layer_names=True, expand_nested=True, dpi=300)
    return model


# Deprecated WORKING PIPELINE MODEL
def old_formnn_pipeline(combined, output_channels=32, lrval=0.0001):
    z = layers.ZeroPadding2D(padding=((1, 1), (6, 6)))(combined)
    z = layers.Conv2D(filters=(output_channels * 2), kernel_size=(3, 5), strides=(1, 1),
                      padding='same', dilation_rate=(1, 3))(z)
    z = layers.LeakyReLU(alpha=lrval)(z)
    z = layers.SpatialDropout2D(rate=0.5)(z)
    # z = layers.Reshape(target_shape=(-1, 1, output_channels * 152))(z)
    z = layers.Conv2D(filters=output_channels * 4, kernel_size=(1, 1), strides=(1, 1), padding='same')(z)
    z = layers.LeakyReLU(alpha=lrval)(z)
    z = layers.SpatialDropout2D(rate=0.5)(z)
    z = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')(z)
    z = layers.GlobalMaxPooling2D()(z)
    return z


# Deprecated MLS MODEL
def cnn_mls(output_channels, lrval=0.0001):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=output_channels,
                            kernel_size=(5, 7), strides=(1, 1),
                            padding='same',  # ((5 - 1) // 2, (7 - 1) // 2),
                            activation=layers.LeakyReLU(alpha=lrval), input_shape=(200, 1150, 4)  # (1,)
                            ))
    model.add(layers.MaxPooling2D(pool_size=(5, 3), strides=(5, 1), padding='same'))  # (1, 1)))
    # opt = keras.optimizers.Adam(lr=lrval)
    # model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# Deprecated SSLM MODEL
def cnn_sslm(output_channels, lrval=0.0001):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=output_channels,
                            kernel_size=(5, 7), strides=(1, 1),
                            padding='same',  # ((5 - 1) // 2, (7 - 1) // 2),
                            activation=layers.LeakyReLU(alpha=lrval), input_shape=(200, 1150, 4)  # (3,)
                            ))
    model.add(layers.MaxPooling2D(pool_size=(5, 3), strides=(5, 1), padding='same'))  # (1, 1)))
    # opt = keras.optimizers.Adam(lr=lrval)
    # model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# Deprecated PIPELINE MODEL
def cnn2(output_channels, lrval=0.0001):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=(output_channels * 2),
                            kernel_size=(3, 5), strides=(1, 1),
                            padding='same',  # ((3 - 1) // 2, (5 - 1) * 3 // 2),
                            dilation_rate=(1, 3),
                            activation=layers.LeakyReLU(alpha=lrval), input_shape=(40, 1150, 8)
                            ))

    model.add(layers.SpatialDropout2D(rate=0.5))
    model.add(
        layers.Conv2D(output_channels * 152, 128, (1, 1), activation=layers.LeakyReLU(alpha=lrval), padding='same'))
    # *72=para 6pool, *152 para 2pool3

    model.add(layers.SpatialDropout2D(rate=0.5))
    model.add(layers.Conv2D(128, 1, (1, 1), padding='same'))  # , padding='same'))

    # x = np.reshape(x, -1, x.shape[1] * x.shape[2], 1, x.shape[3])  # reshape model?
    # model = keras.layers.Reshape((-1, model.shape))(model)
    # Feature maps are joined with the column dimension (frequency)

    # opt = keras.optimizers.Adam(lr=lrval)  # learning rate
    # model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # model.summary()

    return model


# Deprecated
def fuse_model(output_channels, lrval=0.0001):
    cnn1_mel = cnn_mls(output_channels, lrval=lrval)
    cnn1_sslm = cnn_sslm(output_channels, lrval=lrval)
    combined = keras.layers.concatenate([cnn1_mel.output, cnn1_sslm.output])
    cnn2_in = cnn2(output_channels, lrval=lrval)(combined)
    opt = keras.optimizers.Adam(lr=lrval)  # learning rate
    model = keras.models.Model(inputs=[cnn1_mel.input, cnn1_sslm.input], outputs=[cnn2_in])
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()
    model.get_layer(name='sequential_2').summary()
    if not os.path.isfile(os.path.join(MASTER_DIR, 'Model_Diagram.png')):
        plot_model(model, to_file=os.path.join(MASTER_DIR, 'Model_Diagram.png'),
                   show_shapes=True, show_layer_names=True, expand_nested=True)
    # if not os.path.isfile(os.path.join(MASTER_DIR, 'Model_Diagram_Inner.png')):
    #     plot_model(model.get_layer(name='sequential_2'), to_file=os.path.join(MASTER_DIR, 'Model_Diagram_Inner.png'),
    #                show_shapes=True, show_layer_names=True, expand_nested=True)
    return model


# Probably deprecated
def prepare_train_data():
    """
    Retrieve analysis of the following audio data for each training file:
    - Log-scaled Mel Spectrogram (MLS)
    - Self-Similarity Lag Matrix (Mel-Frequency Cepstral Coefficients/MFCCs - Cosine Distance, SSLMCOS)
    - Self-Similarity Lag Matrix (MFCCs - Euclidian Distance, SSLMEUC)
    - Self-Similarity Matrix (Chromas, SSLMCRM)
    Checks to ensure that each file has been fully analyzed/labeled with ground truth
    and not yet prepared for training material.
    """
    cnt = 1
    for folder in MIDI_Data_Dir:
        for file in os.listdir(folder):
            foldername = folder.split('\\')[-1]
            filename, name = file, file.split('/')[-1].split('.')[0]
            print(f"\nWorking on {os.path.basename(name)}, file #" + str(cnt))

            path = os.path.join(os.path.join(MASTER_DIR, 'Labels/'), foldername) + '/' + os.path.basename(name) + '.txt'
            num_lines = sum(1 for _ in open(path))
            if num_lines <= 2:
                print("File has not been labeled with ground truth yet. Skipping...")
                cnt += 1
                continue
            # elif os.path.basename(name) != "INSERT_DEBUG_NAME_HERE":  # Debug output of specified file
            else:
                png1 = os.path.join(MASTER_DIR, 'Images/Train/') + "MLS/" + os.path.basename(name) + 'mls.png'
                png2 = os.path.join(MASTER_DIR, 'Images/Train/') + "SSLMCOS/" + os.path.basename(name) + 'cos.png'
                png3 = os.path.join(MASTER_DIR, 'Images/Train/') + "SSLMEUC/" + os.path.basename(name) + 'euc.png'
                png4 = os.path.join(MASTER_DIR, 'Images/Train/') + "SSLMCRM/" + os.path.basename(name) + 'crm.png'
                if os.path.exists(png1) and os.path.exists(png2) and os.path.exists(png3) and os.path.exists(png4):
                    print("File has already been prepared for training material. Skipping...")
                    cnt += 1
                    continue

            fullfilename = folder + '/' + filename
            du.create_mls_sslm(fullfilename, name, foldername)
            du.peak_picking(fullfilename, name, foldername)
            cnt += 1


# Deprecated
def old_prepare_train_data():
    """
    Retrieve analysis of the following audio data for each training file:
    - Log-scaled Mel Spectrogram (MLS)
    - Self-Similarity Lag Matrix (Mel-Frequency Cepstral Coefficients/MFCCs - Cosine Distance, SSLMCOS)
    - Self-Similarity Lag Matrix (MFCCs - Euclidian Distance, SSLMEUC)
    - Self-Similarity Matrix (Chromas, SSLMCRM)
    """
    cnt = 1
    for file in Train_Data_Dir:
        filename, name = file, file.split('/')[-1].split('.')[0]
        print(f"\nWorking on {os.path.basename(name)}, file #" + str(cnt))
        du.create_mls_sslm(filename, name)
        du.create_mls_sslm2(filename, name)
        cnt += 1


# Deprecated
def old_prepare_model_training_input():
    """
    Read in the input data for the model, return: images [MLS, SSLMCOS, EUC, and CRM] labels (phrases), labels (seconds)
    """
    mls_images = np.asarray(du.ReadImagesFromFolder(MLS_Data_Dir), dtype=np.float32)
    sslmcos_images = np.asarray(du.ReadImagesFromFolder(SSLMCOS_Data_Dir), dtype=np.float32)
    sslmeuc_images = np.asarray(du.ReadImagesFromFolder(SSLMEUC_Data_Dir), dtype=np.float32)
    sslmcrm_images = du.ReadImagesFromFolder(SSLMCRM_Data_Dir)
    lbls_seconds, lbls_phrases = du.ReadLabelSecondsPhrasesFromFolder()
    # print(lbls_seconds)
    # print([i for i, x in enumerate(lbls_seconds) if len(x) != 560])
    # lbls_seconds = np.array(lbls_seconds).flatten()
    # lbls_seconds = [item for sublist in lbls_seconds for item in sublist]
    # for i in range(len(lbls_seconds)):
    #   lbls_seconds[i] = np.asarray(lbls_seconds[i]).flatten()
    lbls_seconds = padMatrix(lbls_seconds)  # matrix must not be jagged in order to convert to ndarray of float32
    # print(lbls_seconds)
    lbls_seconds = np.asarray(lbls_seconds, dtype=np.float32)
    mdl_images = [mls_images, sslmcos_images, sslmeuc_images, sslmcrm_images]
    return mdl_images, lbls_seconds, lbls_phrases


# Probably deprecated
def padMatrix(a):
    b = []
    width = max(len(r) for r in a)
    for i in range(len(a)):
        if len(a[i]) != width:
            x = np.pad(a[i], (width - len(a[i]), 0), 'constant', constant_values=0)
        else:
            x = a[i]
        b.append(x)
    return b


# Probably deprecated
def debugInput(mimg, lbls, lblp):
    # model_images = [0 => mls, 1 => sslmcos, 2 => sslmeuc, 3 => sslmcrm]
    print("Model images:", mimg)
    print("Model images length:", len(mimg))
    for i in range(len(mimg)):
        print("M_Imgs[" + str(i) + "] length:", len(mimg[i]))
    print("Label seconds:", lbls)
    print("Label phrases:", lblp)
    print("Image shape:", mimg[0][0].shape)  # returns (height, width, channels) := (216, 1162, 4)


# Deprecated
def old_trainModel():
    model_images, labels_seconds, labels_phrases = old_prepare_model_training_input()
    # debugInput(model_images, labels_seconds, labels_phrases)

    # FIT MODEL AND USE CHECKPOINT TO SAVE BEST MODEL
    trmodel = fuse_model(4)  # (32) CNN Layer 1 Output Characteristic Maps
    checkpoint = ModelCheckpoint("best_initial_model.hdf5", monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max', save_freq='epoch', save_weights_only=True)

    model_history = trmodel.fit((np.array([model_images[0]], dtype=np.float32),
                                 np.array([model_images[1], model_images[2], model_images[3]], dtype=np.float32)),
                                # np.asarray([tf.stack(model_images[1:2]), model_images[3]],
                                # (np.array([model_images[1], model_images[2]], dtype=np.float32),
                                # np.array(model_images[3])),
                                np.array(labels_seconds, dtype=np.float32),
                                batch_size=32, epochs=2000,
                                validation_data=(labels_seconds,),
                                callbacks=[checkpoint])
    print(model_history)
    # PLOT MODEL HISTORY OF ACCURACY AND LOSS OVER EPOCHS
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Initial_Model_Accuracy.png')
    plt.show()
    # pd.DataFrame(model_history.history).plot()  # figsize=(8, 5)
    # plt.show()

    # summarize history for loss
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Initial_Model_loss.png')
    plt.show()


# Probably deprecated
def combine_generator(gen1, gen2):
    while True:
        yield next(gen1), next(gen2)
# endregion


def generate_label_files():
    """
    Generate label '.txt' file for each MIDI in its respective Form-folder.
    Pre-timestamps each file with silence and end times
    """
    cnt = 1
    for folder in MIDI_Data_Dir:
        for file in os.listdir(folder):
            foldername = folder.split('\\')[-1]
            filename, name = file, file.split('/')[-1].split('.')[0]
            print(f"\nWorking on {os.path.basename(name)}, file #" + str(cnt))

            path = os.path.join(os.path.join(MASTER_DIR, 'Labels/'), foldername) + '/' + os.path.basename(name) + '.txt'
            if not os.path.exists(path):
                with audioread.audio_open(folder + '/' + filename) as f:
                    print("Reading duration of " + os.path.basename(name))
                    totalsec = f.duration
                fwrite = open(path, "w+")
                fwrite.write("0.000\tSilence\n" + str(totalsec) + '00\tEnd')
                fwrite.close()
            cnt += 1


def get_total_duration():
    """
    Return the sum of all audio file durations together
    """
    dur_sum = 0
    for folder in MIDI_Data_Dir:
        for file in os.listdir(folder):
            filename, name = file, file.split('/')[-1].split('.')[0]
            with audioread.audio_open(folder + '/' + filename) as f:
                dur_sum += f.duration
    print("Total duration: " + str(dur_sum) + " seconds")
    # Total duration: 72869.0 seconds
    # = 1214.4833 minutes = 20.241389 hours = 20 hours, 14 minutes, 29 seconds
    return dur_sum


def prepare_model_training_input():
    print("Preparing MLS inputs")
    dus.util_main(feature="mls")

    print("\nPreparing SSLM-MFCC-COS inputs")
    dus.util_main(feature="mfcc", mode="cos")
    print("\nPreparing SSLM-MFCC-EUC inputs")
    dus.util_main(feature="mfcc", mode="euc")

    print("\nPreparing SSLM-CRM-COS inputs")
    dus.util_main(feature="chroma", mode="cos")
    print("\nPreparing SSLM-CRM-EUC inputs")
    dus.util_main(feature="chroma", mode="euc")


def multi_input_generator_helper(gen1, gen2, gen3, gen4, concat=True):
    while True:
        sslm1 = next(gen1)[0]
        sslm2 = next(gen2)[0]
        sslm3 = next(gen3)[0]
        sslm4 = next(gen4)[0]
        if not concat:
            yield [sslm1, sslm2, sslm3, sslm4], sslm1.shape
            continue

        if sslm2.shape != sslm1.shape:
            sslm2 = resize(sslm2, sslm1.shape)
        if sslm3.shape != sslm1.shape:
            sslm3 = resize(sslm3, sslm1.shape)
        if sslm4.shape != sslm1.shape:
            sslm4 = resize(sslm4, sslm1.shape)
        yield tf.expand_dims(
            np.concatenate((sslm1,
                            np.concatenate((sslm2,
                                            np.concatenate((sslm3, sslm4),
                                                           axis=-1)), axis=-1)), axis=-1), axis=-1), sslm1.shape


def multi_input_generator(gen1, gen2, gen3, gen4, gen5, gen6, feature=2, concat=True, expand_dim_6=True, augment=False):
    while True:
        mlsgen = next(gen1)
        mlsimg = mlsgen[0]
        if augment:
            yield [mlsimg, [[0, 0], [0, 0], [0, 0], [0, 0]],
                   next(gen6)[0]], mlsgen[1][feature]  # tf.expand_dims(next(gen6)[0], axis=0)], mlsgen[1][feature]
        else:
            sslmimgs, sslmshape = next(multi_input_generator_helper(gen2, gen3, gen4, gen5, concat))
            if not expand_dim_6:
                yield [mlsimg, sslmimgs, next(gen6)[0]], mlsgen[1][feature]
                continue
            if mlsimg.shape != sslmshape:
                mlsimg = resize(mlsimg, sslmshape)
            yield [mlsimg, sslmimgs, tf.expand_dims(next(gen6)[0], axis=0)], mlsgen[1][feature]


def get_column_dataframe():
    df = pd.DataFrame(columns=['piece_name', 'composer', 'filename', 'duration',
                               'ssm_log_mel_mean', 'ssm_log_mel_var',
                               'sslm_chroma_cos_mean', 'sslm_chroma_cos_var',
                               'sslm_chroma_euc_mean', 'sslm_chroma_euc_var',
                               'sslm_mfcc_cos_mean', 'sslm_mfcc_cos_var',
                               'sslm_mfcc_euc_mean', 'sslm_mfcc_euc_var',  # ---{
                               'chroma_cens_mean', 'chroma_cens_var',
                               'chroma_cqt_mean', 'chroma_cqt_var',
                               'chroma_stft_mean', 'chroma_stft_var',
                               'mel_mean', 'mel_var',
                               'mfcc_mean', 'mfcc_var',
                               'spectral_bandwidth_mean', 'spectral_bandwidth_var',
                               'spectral_centroid_mean', 'spectral_centroid_var',
                               'spectral_contrast_mean', 'spectral_contrast_var',
                               'spectral_flatness_mean', 'spectral_flatness_var',
                               'spectral_rolloff_mean', 'spectral_rolloff_var',
                               'poly_features_mean', 'poly_features_var',
                               'tonnetz_mean', 'tonnetz_var',
                               'zero_crossing_mean', 'zero_crossing_var',
                               'tempogram_mean', 'tempogram_var',
                               'fourier_tempo_mean', 'fourier_tempo_var',  # }---
                               'formtype'])
    return df


def create_form_dataset(filedir=FULL_DIR, labeldir=FULL_LABELPATH, outfile='full_dataset.xlsx', augment=False):
    # if augment then ignore sslms and replace with [0, 0]

    mls_full = dus.BuildDataloader(os.path.join(filedir, 'MLS/'),
                                   label_path=labeldir, batch_size=1, reshape=False)
    midi_full = dus.BuildMIDIloader(os.path.join(filedir, 'MIDI/'),
                                    label_path=labeldir, batch_size=1, reshape=False, building_df=True)
    if not augment:
        sslm_cmcos_full = dus.BuildDataloader(os.path.join(filedir, 'SSLM_CRM_COS/'),
                                              label_path=labeldir, batch_size=1, reshape=False)
        sslm_cmeuc_full = dus.BuildDataloader(os.path.join(filedir, 'SSLM_CRM_EUC/'),
                                              label_path=labeldir, batch_size=1, reshape=False)
        sslm_mfcos_full = dus.BuildDataloader(os.path.join(filedir, 'SSLM_MFCC_COS/'),
                                              label_path=labeldir, batch_size=1, reshape=False)
        sslm_mfeuc_full = dus.BuildDataloader(os.path.join(filedir, 'SSLM_MFCC_EUC/'),
                                              label_path=labeldir, batch_size=1, reshape=False)

        print("Done building dataloaders, merging...")
        full_datagen = multi_input_generator(mls_full, sslm_cmcos_full, sslm_cmeuc_full, sslm_mfcos_full,
                                             sslm_mfeuc_full, midi_full, concat=False, expand_dim_6=False)
        print("Merging complete. Printing...")
    else:
        print("Done building dataloaders, merging...")
        full_datagen = multi_input_generator(mls_full, None, None, None, None,
                                             midi_full, concat=False, expand_dim_6=False, augment=True)
        print("Merging complete. Printing...")

    np.set_string_function(
        lambda x: repr(x).replace('(', '').replace(')', '').replace('array', '').replace("       ", ' '), repr=False)
    np.set_printoptions(threshold=inf)

    df = get_column_dataframe()
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(os.path.join(MASTER_DIR, 'form_classes.npy'))
    for indx, cur_data in enumerate(full_datagen):
        if indx == len(mls_full):
            break
        c_flname = mls_full.getSong(indx).replace(".wav.npy", "").replace(".wav", "").replace(".npy", "")
        c_sngdur = mls_full.getDuration(indx)
        c_slmmls = cur_data[0][0]
        c_scmcos = cur_data[0][1][0]
        c_scmeuc = cur_data[0][1][1]
        c_smfcos = cur_data[0][1][2]
        c_smfeuc = cur_data[0][1][3]
        c_midinf = cur_data[0][2]
        c_flabel = cur_data[1]
        c_flabel = label_encoder.inverse_transform(np.where(c_flabel == 1)[0])[0]

        df.loc[indx] = ["", "", c_flname, c_sngdur, c_slmmls[0], c_slmmls[1], c_scmcos[0], c_scmcos[1],
                        c_scmeuc[0], c_scmeuc[1], c_smfcos[0], c_smfcos[1], c_smfeuc[0], c_smfeuc[1],
                        c_midinf[2], c_midinf[3], c_midinf[4], c_midinf[5], c_midinf[6], c_midinf[7],
                        c_midinf[8], c_midinf[9], c_midinf[10], c_midinf[11], c_midinf[12], c_midinf[13],
                        c_midinf[14], c_midinf[15], c_midinf[0], c_midinf[1], c_midinf[16], c_midinf[17],
                        c_midinf[18], c_midinf[19], c_midinf[20], c_midinf[21], c_midinf[22], c_midinf[23],
                        c_midinf[24], c_midinf[25], c_midinf[26], c_midinf[27], c_midinf[28], c_midinf[29], c_flabel]
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x)
                                .replace(", dtype=float32", "").replace("],", "]")
                                .replace("dtype=float32", "").replace("...,", ""))
    # df.to_csv(os.path.join(MASTER_DIR, 'full_dataset.csv'), index=False)
    df.to_excel(os.path.join(MASTER_DIR, outfile), index=False)


def prepare_augmented_audio(inpath=FULL_DIR, savepath='', augmentation=1):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        print("New directory created:", savepath)

    def inject_noise(adata, noise_factor):
        noise = np.random.randn(len(adata))
        augmented_data = adata + noise_factor * noise
        augmented_data = augmented_data.astype(type(adata[0]))
        return augmented_data

    def shift_time(adata, sampling_rate, shift_max, shift_direction):
        shift = np.random.randint(sampling_rate * shift_max)
        if shift_direction == 'right':
            shift = -shift
        elif shift_direction == 'both':
            direction = np.random.randint(0, 2)
            if direction == 1:
                shift = -shift
        augmented_data = np.roll(adata, shift)
        # Set to silence for heading/ tailing
        if shift > 0:
            augmented_data[:shift] = 0
        else:
            augmented_data[shift:] = 0
        return augmented_data

    def shift_pitch(adata, sampling_rate, pitch_factor):
        return librosa.effects.pitch_shift(adata, sampling_rate, n_steps=pitch_factor)

    def shift_speed(adata, speed_factor):
        return librosa.effects.time_stretch(adata, speed_factor)

    start_time = time.time()
    for (dir_path, dnames, fnames) in os.walk(inpath):
        for f in fnames:
            augdatapath = savepath + f.split('.')[0] + '_aug' + str(augmentation) + '.wav'
            if os.path.exists(augdatapath):
                continue
            start_time_song = time.time()
            fdatapath = dir_path + '/' + f
            y, sr = librosa.load(fdatapath, sr=None)
            sr = 44100
            if augmentation == 1:
                y = shift_speed(y, 0.7)  # Slower
                y = shift_pitch(y, sr, -6)  # Shift down 6 half-steps (tritone)
                y = shift_time(y, sr, random.random(), 'right')
                y = inject_noise(y, 0.005)
            elif augmentation == 2:
                y = shift_speed(y, 1.4)  # Faster
                y = shift_pitch(y, sr, 4)  # Shift up 4 half-steps (major 3rd)
                y = shift_time(y, sr, random.random(), 'right')
                y = inject_noise(y, 0.01)
            elif augmentation == 3:
                y = shift_speed(y, 0.5)
                y = shift_pitch(y, sr, 7)  # Shift up perfect 5th
                y = shift_time(y, sr, random.random(), 'right')
                y = inject_noise(y, 0.003)
            elif augmentation == 4:
                y = shift_speed(y, 2)
                y = shift_pitch(y, sr, 8)  # Shift up minor 6th
                y = shift_time(y, sr, random.random(), 'right')
                y = inject_noise(y, 0.02)
            elif augmentation == 5:
                y = shift_speed(y, 1.1)
                y = shift_pitch(y, sr, 1)  # Shift up major 7th
                y = shift_time(y, sr, random.random(), 'right')
                y = inject_noise(y, 0.007)
            sf.write(augdatapath, y, sr)
            print("Successfully saved file:", augdatapath, "\tDuration: {:.2f}s".format(time.time() - start_time_song))
    print("All files have been converted. Duration: {:.2f}s".format(time.time() - start_time))
    pass


def generate_augmented_datasets():
    # https://www.kaggle.com/CVxTz/audio-data-augmentation
    # https://towardsdatascience.com/
    #   audio-deep-learning-made-simple-part-3-data-preparation-and-augmentation-24c6e1f6b52
    # https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6
    for i in range(1, 6):
        prepare_augmented_audio(savepath=os.path.join(MASTER_INPUT_DIR, 'Aug' + str(i) + '/MIDI/'), augmentation=i)
        dus.util_main(feature="mls", inpath=os.path.join(MASTER_INPUT_DIR, 'Aug' + str(i) + '/'),
                      midpath=os.path.join(MASTER_INPUT_DIR, 'Aug' + str(i) + '/'))
        create_form_dataset(filedir=os.path.join(MASTER_INPUT_DIR, 'Aug' + str(i) + '/'), augment=True,
                            outfile='full_dataset_aug' + str(i) + '.xlsx')
    df = pd.read_excel(os.path.join(MASTER_DIR, 'full_dataset.xlsx'))
    df1 = pd.read_excel(os.path.join(MASTER_DIR, 'full_dataset_aug1.xlsx'))
    df2 = pd.read_excel(os.path.join(MASTER_DIR, 'full_dataset_aug2.xlsx'))
    df3 = pd.read_excel(os.path.join(MASTER_DIR, 'full_dataset_aug3.xlsx'))
    df4 = pd.read_excel(os.path.join(MASTER_DIR, 'full_dataset_aug4.xlsx'))
    df5 = pd.read_excel(os.path.join(MASTER_DIR, 'full_dataset_aug5.xlsx'))
    df = pd.concat([df, df1, df2, df3, df4, df5], ignore_index=True).reset_index()
    df.to_excel(os.path.join(MASTER_DIR, 'full_augmented_dataset.xlsx'), index=False)


def prepare_lstm_peaks():
    MIDI_FILES = os.path.join(MASTER_INPUT_DIR, 'Full/MIDI/')
    PEAK_DIR = os.path.join(MASTER_INPUT_DIR, 'Full/PEAKS/')
    cnt = len(os.listdir(PEAK_DIR)) + 1
    for file in os.listdir(MIDI_FILES):
        foldername = MIDI_FILES.split('\\')[-1]
        filename, name = file, file.split('/')[-1].split('.')[0]
        if str(os.path.basename(name)) + ".npy" in os.listdir(PEAK_DIR):
            continue
        print(f"\nWorking on {os.path.basename(name)}, file #" + str(cnt))
        fullfilename = MIDI_FILES + '/' + filename
        peaks = du.peak_picking(fullfilename, name, foldername, returnpeaks=True)
        print(peaks)
        np.save(os.path.join(PEAK_DIR, os.path.basename(name)), peaks)
        cnt += 1


"""===================================================================================="""


# region OldModelDefinition
# MIDI MODEL -- Try switching activation to ELU instead of RELU. Mimic visual/aural analysis using ensemble method
def formnn_midi(output_channels=32, numclasses=12):
    inputC = layers.Input(shape=(None, 1))
    w = layers.Conv1D(output_channels * 2, kernel_size=10, activation='relu', input_shape=(None, 1))(inputC)
    w = layers.Conv1D(output_channels * 4, kernel_size=10, activation='relu', kernel_regularizer=l2(0.01),
                      bias_regularizer=l2(0.01))(w)
    w = layers.MaxPooling1D(pool_size=6)(w)
    w = layers.Dropout(0.4)(w)
    w = layers.Conv1D(output_channels * 4, kernel_size=10, activation='relu')(w)
    w = layers.MaxPooling1D(pool_size=6)(w)
    w = layers.Dropout(0.4)(w)
    w = layers.GlobalMaxPooling1D()(w)
    w = layers.Dense(output_channels * 8, activation='relu')(w)
    w = layers.Dropout(0.4)(w)
    w = layers.Dense(numclasses)(w)
    w = layers.Softmax()(w)
    w = keras.models.Model(inputs=inputC, outputs=w)
    return w


def formnn_mls2(output_channels=32, lrval=0.0001):
    inputA = layers.Input(batch_input_shape=(None, None, None, 1))
    x = layers.Conv2D(filters=output_channels, kernel_size=(5, 7), padding='same',
                      kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu')(inputA)
    x = layers.MaxPooling2D(pool_size=(5, 3), strides=(5, 1), padding='same')(x)
    x = keras.models.Model(inputs=inputA, outputs=x)
    return x


def formnn_sslm2(output_channels=32, lrval=0.0001):
    inputB = layers.Input(batch_input_shape=(None, None, None, 1))  # (None, None, None, 4)
    y = layers.Conv2D(filters=output_channels, kernel_size=(5, 7), padding='same',
                      kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu')(inputB)
    y = layers.MaxPooling2D(pool_size=(5, 3), strides=(5, 1), padding='same')(y)
    y = layers.AveragePooling2D(pool_size=(1, 4))(y)
    y = keras.models.Model(inputs=inputB, outputs=y)
    return y


def formnn_pipeline2(combined, output_channels=32, lrval=0.0001, numclasses=12):
    z = layers.Conv2D(filters=(output_channels * 2), kernel_size=(3, 5),
                      padding='same', dilation_rate=(1, 3), kernel_regularizer=l2(0.01),
                      bias_regularizer=l2(0.01), activation='relu')(combined)
    z = layers.Conv2D(filters=output_channels * 4, kernel_size=(1, 1), padding='same',
                      kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu')(z)
    z = layers.MaxPooling2D(pool_size=3)(z)
    z = layers.SpatialDropout2D(rate=0.3)(z)
    z = layers.Conv2D(filters=output_channels * 4, kernel_size=(1, 1), padding='same',
                      kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu')(z)
    z = layers.MaxPooling2D(pool_size=3)(z)
    z = layers.SpatialDropout2D(rate=0.3)(z)
    z = layers.GlobalMaxPooling2D()(z)
    # z = layers.Dense(output_channels * 8, activation='relu')(z)
    # z = layers.Dropout(rate=0.3)(z)
    z = layers.Dense(numclasses)(z)
    z = layers.Softmax()(z)
    return z


"""=======================ORIGINAL MODEL======================="""


# MLS MODEL
def formnn_mls(output_channels=32, lrval=0.0001):
    inputA = layers.Input(batch_input_shape=(None, None, None, 1))
    x = layers.ZeroPadding2D(padding=((2, 2), (3, 3)))(inputA)
    x = layers.Conv2D(filters=output_channels, kernel_size=(5, 7), strides=(1, 1), padding='same',
                      kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
    x = layers.LeakyReLU(alpha=lrval)(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(pool_size=(5, 3), strides=(5, 1), padding='same')(x)
    x = keras.models.Model(inputs=inputA, outputs=x)
    return x


# SSLM MODEL
def formnn_sslm(output_channels=32, lrval=0.0001):
    inputB = layers.Input(batch_input_shape=(None, None, None, 1))  # (None, None, None, 4)
    y = layers.ZeroPadding2D(padding=((2, 2), (3, 3)))(inputB)
    y = layers.Conv2D(filters=output_channels, kernel_size=(5, 7), strides=(1, 1), padding='same',
                      kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(y)
    y = layers.LeakyReLU(alpha=lrval)(y)
    y = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(y)
    y = layers.MaxPooling2D(pool_size=(5, 3), strides=(5, 1), padding='same')(y)
    y = layers.AveragePooling2D(pool_size=(1, 4))(y)
    y = keras.models.Model(inputs=inputB, outputs=y)
    return y


# PIPELINE MODEL
def formnn_pipeline(combined, output_channels=32, lrval=0.0001, numclasses=12):
    z = layers.ZeroPadding2D(padding=((1, 1), (6, 6)))(combined)
    z = layers.Conv2D(filters=(output_channels * 2), kernel_size=(3, 5), strides=(1, 1),
                      padding='same', dilation_rate=(1, 3), kernel_regularizer=l2(0.01),
                      bias_regularizer=l2(0.01))(z)
    z = layers.LeakyReLU(alpha=lrval)(z)
    z = layers.SpatialDropout2D(rate=0.3)(z)
    # z = layers.Reshape(target_shape=(-1, 1, output_channels * 152))(z)
    z = layers.Conv2D(filters=output_channels * 4, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(z)
    z = layers.LeakyReLU(alpha=lrval)(z)
    #  z = layers.SpatialDropout2D(rate=0.5)(z)
    z = layers.Conv2D(filters=output_channels * 8, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(z)
    z = layers.LeakyReLU(alpha=lrval)(z)
    z = layers.GlobalAveragePooling2D()(z)
    # z = layers.Flatten()(z)
    z = layers.Dense(numclasses)(z)
    z = layers.Softmax()(z)
    # Softmax -> Most likely class where sum(probabilities) = 1, Sigmoid -> Multiple likely classes, sum != 1
    return z


def formnn_fuse(output_channels=32, lrval=0.0001, numclasses=12):
    cnn1_mel = formnn_mls(output_channels, lrval=lrval)
    cnn1_sslm = formnn_sslm(output_channels, lrval=lrval)
    combined = layers.concatenate([cnn1_mel.output, cnn1_sslm.output], axis=2)
    cnn2_in = formnn_pipeline(combined, output_channels, lrval=lrval, numclasses=numclasses)
    # opt = keras.optimizers.SGD(lr=lrval, decay=1e-6, momentum=0.9, nesterov=True)
    opt = keras.optimizers.Adam(lr=lrval, epsilon=1e-6)

    imgmodel = keras.models.Model(inputs=[cnn1_mel.input, cnn1_sslm.input], outputs=[cnn2_in])
    midmodel = formnn_midi(output_channels, numclasses=numclasses)
    averageOut = layers.Average()([imgmodel.output, midmodel.output])
    model = keras.models.Model(inputs=[imgmodel.input[0], imgmodel.input[1], midmodel.input], outputs=averageOut)

    model.compile(loss=['categorical_crossentropy'], optimizer=opt, metrics=['accuracy'])
    # model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer=opt, metrics=['accuracy'])
    model.summary()  # Try categorical_crossentropy, metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    if not os.path.isfile(os.path.join(MASTER_DIR, 'FormNN_Model_Diagram.png')):
        plot_model(model, to_file=os.path.join(MASTER_DIR, 'FormNN_Model_Diagram.png'),
                   show_shapes=True, show_layer_names=True, expand_nested=True, dpi=300)
    return model


def old_trainFormModel():
    batch_size = 10

    # region MODEL_DIRECTORIES
    mls_train = dus.BuildDataloader(os.path.join(TRAIN_DIR, 'MLS/'), label_path=TRAIN_LABELPATH,  # end=90,
                                    transforms=[padding_MLS, normalize_image, borders], batch_size=batch_size)
    sslm_cmcos_train = dus.BuildDataloader(os.path.join(TRAIN_DIR, 'SSLM_CRM_COS/'), label_path=TRAIN_LABELPATH,
                                           transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    sslm_cmeuc_train = dus.BuildDataloader(os.path.join(TRAIN_DIR, 'SSLM_CRM_EUC/'), label_path=TRAIN_LABELPATH,
                                           transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    sslm_mfcos_train = dus.BuildDataloader(os.path.join(TRAIN_DIR, 'SSLM_MFCC_COS/'), label_path=TRAIN_LABELPATH,
                                           transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    sslm_mfeuc_train = dus.BuildDataloader(os.path.join(TRAIN_DIR, 'SSLM_MFCC_EUC/'), label_path=TRAIN_LABELPATH,
                                           transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    midi_train = dus.BuildMIDIloader(os.path.join(TRAIN_DIR, 'MIDI/'), label_path=TRAIN_LABELPATH,
                                     batch_size=batch_size)

    mls_val = dus.BuildDataloader(os.path.join(VAL_DIR, 'MLS/'), label_path=VAL_LABELPATH,
                                  transforms=[padding_MLS, normalize_image, borders], batch_size=batch_size)
    sslm_cmcos_val = dus.BuildDataloader(os.path.join(VAL_DIR, 'SSLM_CRM_COS/'), label_path=VAL_LABELPATH,
                                         transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    sslm_cmeuc_val = dus.BuildDataloader(os.path.join(VAL_DIR, 'SSLM_CRM_EUC/'), label_path=VAL_LABELPATH,
                                         transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    sslm_mfcos_val = dus.BuildDataloader(os.path.join(VAL_DIR, 'SSLM_MFCC_COS/'), label_path=VAL_LABELPATH,
                                         transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    sslm_mfeuc_val = dus.BuildDataloader(os.path.join(VAL_DIR, 'SSLM_MFCC_EUC/'), label_path=VAL_LABELPATH,
                                         transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    midi_val = dus.BuildMIDIloader(os.path.join(VAL_DIR, 'MIDI/'), label_path=VAL_LABELPATH, batch_size=batch_size)

    mls_test = dus.BuildDataloader(os.path.join(TEST_DIR, 'MLS/'), label_path=TEST_LABELPATH,
                                   transforms=[padding_MLS, normalize_image, borders], batch_size=batch_size)
    sslm_cmcos_test = dus.BuildDataloader(os.path.join(TEST_DIR, 'SSLM_CRM_COS/'), label_path=TEST_LABELPATH,
                                          transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    sslm_cmeuc_test = dus.BuildDataloader(os.path.join(TEST_DIR, 'SSLM_CRM_EUC/'), label_path=TEST_LABELPATH,
                                          transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    sslm_mfcos_test = dus.BuildDataloader(os.path.join(TEST_DIR, 'SSLM_MFCC_COS/'), label_path=TEST_LABELPATH,
                                          transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    sslm_mfeuc_test = dus.BuildDataloader(os.path.join(TEST_DIR, 'SSLM_MFCC_EUC/'), label_path=TEST_LABELPATH,
                                          transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    midi_test = dus.BuildMIDIloader(os.path.join(TEST_DIR, 'MIDI/'), label_path=TEST_LABELPATH, batch_size=batch_size)
    # endregion

    # findBestShape(mls_train, sslm_cmcos_train)

    train_datagen = multi_input_generator(mls_train, sslm_cmcos_train, sslm_cmeuc_train, sslm_mfcos_train,
                                          sslm_mfeuc_train, midi_train)
    valid_datagen = multi_input_generator(mls_val,
                                          sslm_cmcos_val, sslm_cmeuc_val, sslm_mfcos_val, sslm_mfeuc_val, midi_val)
    test_datagen = multi_input_generator(mls_test,
                                         sslm_cmcos_test, sslm_cmeuc_test, sslm_mfcos_test, sslm_mfeuc_test, midi_test)

    steps_per_epoch = len(list(mls_train)) // batch_size
    steps_per_valid = len(list(mls_val)) // batch_size
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(os.path.join(MASTER_DIR, 'form_classes.npy'))

    if mls_train.getNumClasses() != mls_val.getNumClasses() or mls_train.getNumClasses() != mls_test.getNumClasses():
        print(f"Train and validation or testing datasets have differing numbers of classes: "
              f"{mls_train.getNumClasses()} vs. {mls_val.getNumClasses()} vs. {mls_test.getNumClasses()}")

    # classweights = get_class_weights(mls_train.getLabels().numpy().squeeze(axis=-1), one_hot=True)
    """
    # Show class weights as bar graph
    barx, bary = zip(*sorted(classweights.items()))
    plt.figure(figsize=(12, 8))
    plt.bar(label_encoder.inverse_transform(barx), bary, color='green')
    for i in range(len(barx)):
        plt.text(i, bary[i]//2, round(bary[i], 3), ha='center', color='white')
    plt.title('Train Class Weights')
    plt.ylabel('Weight')
    plt.xlabel('Class')
    plt.savefig('Initial_Model_Class_Weights.png')
    plt.show()
    """

    model = formnn_fuse(output_channels=32, lrval=0.00005, numclasses=mls_train.getNumClasses())  # Try 'val_loss'?
    # model.load_weights('best_initial_model.hdf5')
    early_stopping = EarlyStopping(patience=5, verbose=5, mode="auto")
    checkpoint = ModelCheckpoint(os.path.join(MASTER_DIR, 'best_formNN_model.hdf5'), monitor='val_accuracy', verbose=0,
                                 save_best_only=True, mode='max', save_freq='epoch', save_weights_only=True)
    model_history = model.fit(train_datagen, epochs=100, verbose=1, validation_data=valid_datagen, shuffle=False,
                              callbacks=[checkpoint, early_stopping], batch_size=batch_size,  # class_weight=classweight
                              steps_per_epoch=steps_per_epoch, validation_steps=steps_per_valid)

    print("Training complete!\n")

    # region LossAccuracyGraphs
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Initial_Model_Loss.png')
    plt.show()

    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Initial_Model_Accuracy.png')
    plt.show()

    pd.DataFrame(model_history.history).plot()
    plt.show()
    # endregion

    predictions = model.predict_generator(valid_datagen, steps=1, verbose=1, workers=0)
    print(predictions)
    print("Prediction complete!")
    inverted = label_encoder.inverse_transform([np.argmax(predictions[0, :])])
    print("Predicted: ", end="")
    print(inverted, end=""),
    print("\tActual: ", end="")
    print(label_encoder.inverse_transform([np.argmax(mls_val.getFormLabel(mls_val.getCurrentIndex()-1))]))
    print("Name: " + mls_val.getSong(mls_val.getCurrentIndex()-1))

    print("\nEvaluating...")
    score = model.evaluate_generator(test_datagen, steps=len(list(mls_test)), verbose=1)
    print("Evaluation complete!\nScore:")
    print(f"Loss: {score[0]}\tAccuracy: {score[1]}")

    # region EvaluationGraphs
    predictions = model.predict(test_datagen, steps=len(list(mls_test)), verbose=1)
    predictions = predictions.argmax(axis=1)
    predictions = predictions.astype(int).flatten()
    predictions = (label_encoder.inverse_transform(predictions))
    predictions = pd.DataFrame({'Predicted Values': predictions})

    actual = mls_test.getLabels().numpy().argmax(axis=1)
    actual = actual.astype(int).flatten()
    actual = (label_encoder.inverse_transform(actual))
    actual = pd.DataFrame({'Actual Values': actual})

    cm = confusion_matrix(actual, predictions)
    plt.figure(figsize=(12, 10))
    cm = pd.DataFrame(cm, index=[i for i in label_encoder.classes_[0:mls_test.getNumClasses()]],
                      columns=[i for i in label_encoder.classes_[0:mls_test.getNumClasses()]])
    ax = sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title('Confusion Matrix', size=20)
    plt.xlabel('Predicted Labels', size=14)
    plt.ylabel('Actual Labels', size=14)
    plt.savefig('Initial_Model_Confusion_Matrix.png')
    plt.show()
    clf_report = classification_report(actual, predictions, output_dict=True,
                                       target_names=[i for i in label_encoder.classes_[0:mls_test.getNumClasses()]])
    sns.heatmap(pd.DataFrame(clf_report).iloc[:, :].T, annot=True, cmap='viridis')
    plt.title('Classification Report', size=20)
    plt.savefig('Initial_Model_Classification_Report.png')
    plt.show()
    # endregion


def formnn_cnn_mod(input_dim_1, filters=64, lrval=0.0001, numclasses=12):
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(filters, kernel_size=10, activation='relu', input_shape=(input_dim_1, 1)))
    model.add(layers.Dropout(0.4))  # ?
    model.add(layers.Conv1D(filters*2, kernel_size=10, activation='relu', kernel_regularizer=l2(0.01),
                            bias_regularizer=l2(0.01)))
    model.add(layers.MaxPooling1D(pool_size=6))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv1D(filters*2, kernel_size=10, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=6))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(filters*4, activation='relu'))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(numclasses, activation='softmax'))  # Try softmax?
    opt = keras.optimizers.Adam(lr=lrval, epsilon=1e-6)
    # opt = keras.optimizers.SGD(lr=lrval, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def formnn_cnn_old(input_dim_1, filters=64, lrval=0.0001, numclasses=12):
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(filters, kernel_size=10, activation='relu', input_shape=(input_dim_1, 1)))
    model.add(layers.Conv1D(filters*2, kernel_size=10, activation='relu', kernel_regularizer=l2(0.01),
                            bias_regularizer=l2(0.01)))
    model.add(layers.MaxPooling1D(pool_size=6))
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv1D(filters*2, kernel_size=10, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=6))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(filters*4, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(numclasses, activation='softmax'))  # Try softmax?
    # opt = keras.optimizers.Adam(lr=lrval, epsilon=1e-6)
    opt = keras.optimizers.SGD(lr=lrval, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
# endregion


# region FormModel
def formnn_cnn(input_dim_1, filters=8, lrval=0.0001, numclasses=12, kernelsize=3):
    np.random.seed(9)
    X_input = Input(shape=(input_dim_1, 1))

    X = layers.Conv1D(filters, kernel_size=kernelsize, strides=1, kernel_initializer=glorot_uniform(seed=9),
                      bias_regularizer=l2(0.000001), kernel_regularizer=l2(0.00001))(X_input)
    X = layers.BatchNormalization(axis=2)(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling1D(numclasses, padding='same')(X)
    X = layers.Dropout(0.5)(X)
    # X = layers.GaussianNoise(0.1)(X)

    X = layers.Conv1D(filters * 2, kernel_size=kernelsize, strides=1, kernel_initializer=glorot_uniform(seed=9),
                      bias_regularizer=l2(0.000001), kernel_regularizer=l2(0.00001))(X)
    X = layers.BatchNormalization(axis=2)(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling1D(numclasses, padding='same')(X)
    X = layers.Dropout(0.5)(X)
    # X = layers.GaussianNoise(0.1)(X)

    X = layers.Conv1D(filters * 4, kernel_size=kernelsize, strides=1, kernel_initializer=glorot_uniform(seed=9),
                      bias_regularizer=l2(0.000001), kernel_regularizer=l2(0.00001))(X)
    X = layers.BatchNormalization(axis=2)(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling1D(numclasses, padding='same')(X)
    X = layers.Dropout(0.5)(X)
    # X = layers.GaussianNoise(0.1)(X)

    X = layers.Flatten()(X)

    # X = layers.Conv1D(filters * 8, kernel_size=kernelsize, strides=1, kernel_initializer=glorot_uniform(seed=9),
    #                   bias_regularizer=l2(0.5))(X)
    X = layers.Dense(256, kernel_initializer=glorot_uniform(seed=9), bias_regularizer=l2(0.000001), kernel_regularizer=l2(0.00001))(X)
    X = layers.BatchNormalization(axis=-1)(X)
    X = layers.Activation('relu')(X)
    # X = layers.MaxPooling1D(numclasses, padding='same')(X)
    X = layers.Dropout(0.5)(X)
    # X = layers.GaussianNoise(0.1)(X)

    # X = layers.Flatten()(X)

    X = layers.Dense(numclasses, activation='sigmoid', kernel_initializer=glorot_uniform(seed=9),
                     bias_regularizer=l2(0.000001), kernel_regularizer=l2(0.0001))(X)

    # opt = keras.optimizers.Adam(lr=lrval)
    opt = keras.optimizers.SGD(lr=lrval, decay=1e-6, momentum=0.9, nesterov=True)
    model = keras.models.Model(inputs=X_input, outputs=X, name='FormModel')
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def trainFormModel():
    # region DataPreProcessing
    df = pd.read_excel(os.path.join(MASTER_DIR, 'full_augmented_dataset.xlsx'))
    # df = pd.read_excel(os.path.join(MASTER_DIR, 'full_dataset.xlsx'))
    names = df[['piece_name', 'composer', 'filename']]
    y = df['formtype']
    # """
    # TODO: for augmented dataset
    df = df.drop(columns=['sslm_chroma_cos_mean', 'sslm_chroma_cos_var', 'sslm_chroma_euc_mean', 'sslm_chroma_euc_var', 
                          'sslm_mfcc_cos_mean', 'sslm_mfcc_cos_var', 'sslm_mfcc_euc_mean', 'sslm_mfcc_euc_var'])
    # """
    df.drop(columns=['spectral_bandwidth_var', 'spectral_centroid_var', 'spectral_flatness_var', 'spectral_rolloff_var',
                     'zero_crossing_var', 'fourier_tempo_mean', 'fourier_tempo_var'], inplace=True)  # Remove useless
    # nonlist = df[['duration', 'spectral_contrast_var']]
    nonlist = df[['duration']]
    df.drop(columns=['piece_name', 'composer', 'filename', 'duration', 'spectral_contrast_var', 'formtype'],
            inplace=True)
    # df = df[['ssm_log_mel_mean', 'ssm_log_mel_var', 'mel_mean', 'mel_var', 'chroma_stft_mean', 'chroma_stft_var']]
    # df = df[['ssm_log_mel_mean', 'ssm_log_mel_var']]
    # df = df[['ssm_log_mel_mean']]  # best decision tree accuracy
    print("Fixing broken array cells as needed...")

    def fix_broken_arr(strx):
        if '[' in strx:
            if ']' in strx:
                return strx
            else:
                return strx + ']'

    for col in df.columns:
        df[col] = df[col].apply(lambda x: fix_broken_arr(x))
    print("Done processing cells, building training set...")

    # d = [pd.DataFrame(df[col].astype(str).apply(literal_eval).values.tolist()).add_prefix(col) for col in df.columns]
    d = [pd.DataFrame(df[col].astype(str).apply(literal_eval).values.tolist()) for col in df.columns]
    df = pd.concat(d, axis=1).fillna(0)
    df = pd.concat([pd.concat([names, pd.concat([nonlist, df], axis=1)], axis=1), y], axis=1)  # print(df)
    train, test = train_test_split(df, test_size=0.169, random_state=0, stratify=df['formtype'])  # test_s=.169 gave 50%
    # df.to_csv(os.path.join(MASTER_DIR, 'full_modified_dataset.csv'))

    X_train = train.iloc[:, 3:-1]
    X_train_names = train.iloc[:, 0:3]
    y_train = train.iloc[:, -1]
    print("Train shape:", X_train.shape)
    X_test = test.iloc[:, 3:-1]
    X_test_names = test.iloc[:, 0:3]
    y_test = test.iloc[:, -1]
    print("Test shape:", X_test.shape)

    # Normalize Data
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)  # Good for decision tree
    X_test = min_max_scaler.fit_transform(X_test)
    """
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std  # Good for decision tree
    X_test = (X_test - mean) / std
    """
    print("Normalized Train shape:", X_train.shape)
    print("Normalized Test shape:", X_test.shape)

    # Convert to arrays for keras
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    label_encoder = LabelEncoder()
    old_y_train = y_train
    y_train = to_categorical(label_encoder.fit_transform(y_train))
    y_test = to_categorical(label_encoder.fit_transform(y_test))
    print(y_train.shape, y_test.shape)
    print(label_encoder.classes_, "\n")

    """ BASE MODEL """
    # DummyClassifier makes predictions while ignoring input features
    dummy_clf = DummyClassifier(strategy="stratified")
    dummy_clf.fit(X_train, y_train)
    DummyClassifier(strategy='stratified')
    dummy_clf.predict(X_test)
    print("Dummy classifier accuracy:", dummy_clf.score(X_test, y_test))

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    clf.predict(X_test)
    print("Decision tree accuracy:", clf.score(X_test, y_test))

    """ FEATURE TUNING """
    # https://curiousily.com/posts/hackers-guide-to-fixing-underfitting-and-overfitting-models/
    # https://towardsdatascience.com/handling-overfitting-in-deep-learning-models-c760ee047c6e
    # https://machinelearningmastery.com/rfe-feature-selection-in-python/
    selector = SelectKBest(f_classif, k=25)  # 1000 if using RFE
    Z_train = selector.fit_transform(X_train, old_y_train)
    skb_values = selector.get_support()
    Z_test = X_test[:, skb_values]
    # np.save(os.path.join(MASTER_DIR, "selectkbest_indicies.npy"), skb_values)
    print(Z_train.shape)
    print(Z_test.shape)
    """
    plt.title('Feature Importance')
    plt.ylabel('Score')
    plt.xlabel('Feature')
    plt.plot(selector.scores_)
    plt.savefig('Initial_Feature_Importance.png')
    plt.show()
    """
    print("Indices of top 10 features:", (-selector.scores_).argsort()[:10])

    """ KBEST MODEL """
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(Z_train, y_train)
    clf.predict(Z_test)
    print("K-Best Decision tree accuracy:", clf.score(Z_test, y_test))

    """
    # Accuracy 0.211, stick with SKB? Gives good loss though
    # https://towardsdatascience.com/dont-overfit-how-to-prevent-overfitting-in-your-deep-learning-models-63274e552323
    clf = LinearSVC(C=0.01, penalty="l1", dual=False)
    clf.fit(X_train, old_y_train)
    rfe_selector = RFE(clf, 10, verbose=5)
    rfe_selector = rfe_selector.fit(Z_train, old_y_train)
    # rfe_selector = rfe_selector.fit(X_train, old_y_train)
    rfe_values = rfe_selector.get_support()
    # np.save(os.path.join(MASTER_DIR, "rfebest_indicies.npy"), rfe_values)
    print("Indices of least important features:", np.where(rfe_values)[0])
    W_train = Z_train[:, rfe_values]
    W_test = Z_test[:, rfe_values]

    # "" " RFE MODEL " ""
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(W_train, y_train)
    clf.predict(W_test)
    print("RFE Decision tree accuracy:", clf.score(W_test, y_test))
    """
    # endregion

    # Reshape to 3D tensor for keras
    # X_train = Z_train[:, :, np.newaxis]
    # X_test = Z_test[:, :, np.newaxis]
    # X_train = W_train[:, :, np.newaxis]
    # X_test = W_test[:, :, np.newaxis]
    X_train = X_train[:, :, np.newaxis]
    X_test = X_test[:, :, np.newaxis]

    """
    clf = ak.StructuredDataClassifier(overwrite=True, max_trials=3)
    model_history = clf.fit(Z_train, y_train, epochs=10)
    predicted_y = clf.predict(Z_test)
    print(predicted_y)
    print(clf.evaluate(Z_test, y_test))
    model = clf.export_model()
    model.summary()
    # model.save('best_auto_model.h5', save_format='tf')
    if not os.path.isfile(os.path.join(MASTER_DIR, 'FormNN_CNN_AutoModel_Diagram.png')):
        plot_model(model, to_file=os.path.join(MASTER_DIR, 'FormNN_CNN_AutoModel_Diagram.png'),
                   show_shapes=True, show_layer_names=True, expand_nested=True, dpi=300)
    """

    model = formnn_cnn(X_train.shape[1], filters=32, lrval=0.003, numclasses=len(label_encoder.classes_), kernelsize=10)
    model.summary()
    if not os.path.isfile(os.path.join(MASTER_DIR, 'FormNN_CNN_Model_Diagram.png')):
        plot_model(model, to_file=os.path.join(MASTER_DIR, 'FormNN_CNN_Model_Diagram.png'),
                   show_shapes=True, show_layer_names=True, expand_nested=True, dpi=300)

    history_loss = []
    history_val_loss = []
    history_accuracy = []
    history_val_accuracy = []
    num_epochs = 0

    """
    # Try predict
    model.load_weights('best_form_model_50p.hdf5')
    result = model.predict(X_test)
    percent_correct = 0
    pred_table = pd.DataFrame(columns=["Piece", "Predicted", "Actual"])
    X_test_names = np.array(X_test_names)
    for i in range(len(result)):
        resultlbl = label_encoder.inverse_transform([np.argmax(result[i, :])])
        actuallbl = label_encoder.inverse_transform([np.argmax(y_test[i, :])])
        pred_table.loc[i] = ([X_test_names[i][2], resultlbl, actuallbl])
        percent_correct += 1 if resultlbl == actuallbl else 0
    print(pred_table.to_string(index=False))
    print("Accuracy: " + str(float(percent_correct/len(result))*100) + "%")
    return
    """

    # model.load_weights('best_form_model_44p.hdf5')
    # while True:
    for i in range(0, 1500):
        # early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=5, mode="auto")
        checkpoint = ModelCheckpoint("best_form_new_model.hdf5", monitor='val_accuracy', verbose=0,
                                     save_best_only=False, mode='max', save_freq='epoch', save_weights_only=True)
        model_history = model.fit(X_train, y_train, batch_size=32, epochs=1, validation_data=(X_test, y_test),
                                  callbacks=[checkpoint])  # , early_stopping  epochs=2000 loss hits 0.7

        history_loss.append(model_history.history['loss'])
        history_val_loss.append(model_history.history['val_loss'])
        history_accuracy.append(model_history.history['accuracy'])
        history_val_accuracy.append(model_history.history['val_accuracy'])

        num_epochs += 1
        print("Epochs completed:", num_epochs)

    print("\nEvaluating...")
    score = model.evaluate(X_test, y_test, verbose=1)
    print("Evaluation complete!\n__________Score__________")
    print(f"Loss: {score[0]}\tAccuracy: {score[1]}")

    # if score[1] >= 0.51:
    # region EvaluationGraphs
    plt.plot(history_loss)  # plt.plot(model_history.history['loss'])
    plt.plot(history_val_loss)  # plt.plot(model_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Initial_Model_Loss.png')
    plt.show()

    plt.plot(history_accuracy)  # plt.plot(model_history.history['accuracy'])
    plt.plot(history_val_accuracy)  # plt.plot(model_history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Initial_Model_Accuracy.png')
    plt.show()

    # pd.DataFrame(model_history.history).plot()
    # plt.show()

    predictions = model.predict(X_test, verbose=1)
    predictions = predictions.argmax(axis=1)
    predictions = predictions.astype(int).flatten()
    predictions = (label_encoder.inverse_transform(predictions))
    predictions = pd.DataFrame({'Predicted Values': predictions})

    actual = y_test.argmax(axis=1)
    actual = actual.astype(int).flatten()
    actual = (label_encoder.inverse_transform(actual))
    actual = pd.DataFrame({'Actual Values': actual})

    cm = confusion_matrix(actual, predictions)
    plt.figure(figsize=(12, 10))
    cm = pd.DataFrame(cm, index=[i for i in label_encoder.classes_], columns=[i for i in label_encoder.classes_])
    ax = sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title('Confusion Matrix', size=20)
    plt.xlabel('Predicted Labels', size=14)
    plt.ylabel('Actual Labels', size=14)
    plt.savefig('Initial_Model_Confusion_Matrix.png')
    plt.show()
    clf_report = classification_report(actual, predictions, output_dict=True,
                                       target_names=[i for i in label_encoder.classes_])
    sns.heatmap(pd.DataFrame(clf_report).iloc[:, :].T, annot=True, cmap='viridis')
    plt.title('Classification Report', size=20)
    plt.savefig('Initial_Model_Classification_Report.png')
    plt.show()
    # break
    # elif num_epochs >= 50:
    #     model.load_weights('best_form_model_44p.hdf5')
    #     num_epochs = 0
    #     continue
    # endregion
    pass


def preparePredictionData(filepath, savetoexcel=False):
    print("Preparing MLS")
    mls = dus.util_main_helper(feature="mls", filepath=filepath, predict=True)
    print("Preparing SSLM-MFCC-COS")
    sslm_mfcc_cos = dus.util_main_helper(feature="mfcc", filepath=filepath, mode="cos", predict=True)
    print("Preparing SSLM-MFCC-EUC")
    sslm_mfcc_euc = dus.util_main_helper(feature="mfcc", filepath=filepath, mode="euc", predict=True)
    print("Preparing SSLM-CRM-COS")
    sslm_crm_cos = dus.util_main_helper(feature="chroma", filepath=filepath, mode="cos", predict=True)
    print("Preparing SSLM-CRM-EUC")
    sslm_crm_euc = dus.util_main_helper(feature="chroma", filepath=filepath, mode="euc", predict=True)

    midimages = [mls, sslm_mfcc_cos, sslm_mfcc_euc, sslm_crm_cos, sslm_crm_euc]
    cur_data = []
    for image in midimages:
        if image.ndim == 1:
            raise ValueError("Erroneous Image Shape:", image.shape, image.ndim)
        else:
            image1 = np.mean(image, axis=0)
            image2 = np.var(image, axis=0)
            image = np.array([image1, image2])
            cur_data.append(image)

    print("Preparing audio feature data")
    dfmid = dus.get_midi_dataframe(building_df=True)
    dfmid = dus.get_audio_features(dfmid, 0, filepath, building_df=True)
    dfmid = dfmid.fillna(0)
    dfmid = np.array(dfmid)

    sngdur = 0
    with audioread.audio_open(filepath) as f:
        sngdur += f.duration

    np.set_string_function(
        lambda x: repr(x).replace('(', '').replace(')', '').replace('array', '').replace("       ", ' '), repr=False)
    np.set_printoptions(threshold=inf)

    print("Building feature table")
    df = get_column_dataframe()
    c_flname = os.path.basename(filepath.split('/')[-1].split('.')[0])
    c_sngdur = sngdur
    c_slmmls = cur_data[0]
    c_scmcos = cur_data[1]
    c_scmeuc = cur_data[2]
    c_smfcos = cur_data[3]
    c_smfeuc = cur_data[4]
    c_midinf = dfmid[0]

    df.loc[0] = ["TBD", "TBD", c_flname, c_sngdur, c_slmmls[0], c_slmmls[1], c_scmcos[0], c_scmcos[1],
                 c_scmeuc[0], c_scmeuc[1], c_smfcos[0], c_smfcos[1], c_smfeuc[0], c_smfeuc[1],
                 c_midinf[2], c_midinf[3], c_midinf[4], c_midinf[5], c_midinf[6], c_midinf[7],
                 c_midinf[8], c_midinf[9], c_midinf[10], c_midinf[11], c_midinf[12], c_midinf[13],
                 c_midinf[14], c_midinf[15], c_midinf[0], c_midinf[1], c_midinf[16], c_midinf[17],
                 c_midinf[18], c_midinf[19], c_midinf[20], c_midinf[21], c_midinf[22], c_midinf[23],
                 c_midinf[24], c_midinf[25], c_midinf[26], c_midinf[27], c_midinf[28], c_midinf[29], "TBD"]
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x)
                                .replace(", dtype=float32", "").replace("],", "]")
                                .replace("dtype=float32", "").replace("...,", ""))
    if savetoexcel:
        df.to_excel(os.path.join(MASTER_DIR, c_flname + '.xlsx'), index=False)
    return df


def predictForm():
    midpath = input("Enter path to folder or audio file: ")
    df = pd.read_excel(os.path.join(MASTER_DIR, 'full_dataset.xlsx'))  # 15,330
    df = pd.DataFrame(df.loc[[0, 153]]).reset_index()
    df2 = pd.DataFrame()
    if not os.path.exists(midpath):
        raise FileNotFoundError("Path not found or does not exist.")
    else:
        if os.path.isfile(midpath):
            # df2 = pd.read_excel(os.path.join(MASTER_DIR, 'brahms_opus117_1.xlsx'))
            df2 = preparePredictionData(midpath, savetoexcel=False)
        elif os.path.isdir(midpath):
            if midpath[-1] != "\\" or midpath[-1] != "/":
                if "\\" in midpath:
                    midpath = midpath + "\\"
                else:
                    midpath = midpath + "/"
            cnt = 0
            audio_extenions = ["3gp", "aa", "aac", "aax", "act", "aiff", "alac", "amr", "ape", "au", "awb", "dct",
                               "dss", "dvf", "flac", "gsm", "iklax", "ivs", "m4a", "m4b", "m4p", "mmf", "mp3", "mpc",
                               "msv", "nmf", "ogg", "oga", "mogg", "opus", "ra", "rm", "raw", "rf64", "sln", "tta",
                               "voc", "vox", "wav", "wma", "wv", "webm", "8svx", "cda", "mid", "midi", "mp4"]
            for (mid_dirpath, mid_dirnames, mid_filenames) in os.walk(midpath):
                for f in mid_filenames:
                    if f.endswith(tuple(audio_extenions)):
                        print("Reading file #" + str(cnt + 1))
                        mid_path = mid_dirpath + f
                        df2t = preparePredictionData(mid_path, savetoexcel=False)
                        df2 = pd.concat([df2, df2t], ignore_index=True).reset_index(drop=True)
                        cnt += 1
        else:
            raise FileNotFoundError("Path resulted in error.")

    # Reshape test data to match training set
    np.set_string_function(
        lambda x: repr(x).replace('(', '').replace(')', '').replace('array', '').replace("       ", ' '), repr=False)
    np.set_printoptions(threshold=inf)
    for i in range(df2.shape[0]):
        for col_name, data in df2.items():
            if "[" in str(data[i]) and "]" in str(data[i]):
                compdata = df.iloc[1][col_name]
                if "[" in compdata and "]" in compdata:
                    if 'dtype=complex64' in compdata or 'dtype=complex64' in str(data[i]):
                        continue  # Ignore since complex values aren't used in model
                    arr_1 = np.array(literal_eval(compdata))
                    # print("Evaluating:", str(data[i]))
                    arr_2 = np.array(literal_eval(str(data[i]).strip()))
                    arr_2 = np.resize(arr_2, arr_1.shape)
                    df2.at[i, col_name] = arr_2
    # df = df2
    df = pd.read_excel(os.path.join(MASTER_DIR, 'full_dataset.xlsx'))  # 15,330
    train_rows = df.shape[0]
    df = pd.concat([df, df2], ignore_index=True).reset_index(drop=True)

    names = df[['piece_name', 'composer', 'filename']]
    y = df['formtype']
    df.drop(columns=['spectral_bandwidth_var', 'spectral_centroid_var', 'spectral_flatness_var', 'spectral_rolloff_var',
                     'zero_crossing_var', 'fourier_tempo_mean', 'fourier_tempo_var'], inplace=True)
    nonlist = df[['duration', 'spectral_contrast_var']]
    df.drop(columns=['piece_name', 'composer', 'filename', 'duration', 'spectral_contrast_var', 'formtype'],
            inplace=True)
    d = [pd.DataFrame(df[col].astype(str).apply(literal_eval).values.tolist()).add_prefix(col) for col in df.columns]
    df = pd.concat(d, axis=1).fillna(0)
    df = pd.concat([pd.concat([names, pd.concat([nonlist, df], axis=1)], axis=1), y], axis=1)  # print(df)
    df = df.fillna(0)

    X_test = df.iloc[:, 3:-1]
    X_test_names = df.iloc[:, 0:3]
    y_test = df.iloc[:, -1]
    print("Test shape:", X_test.shape)

    # Normalize Data
    min_max_scaler = preprocessing.MinMaxScaler()
    X_test = min_max_scaler.fit_transform(X_test)
    print("Normalized Test shape:", X_test.shape)

    # Convert to arrays for keras
    X_test = np.array(X_test)
    X_test_names = np.array(X_test_names)
    y_test = np.array(y_test)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(os.path.join(MASTER_DIR, 'form_classes.npy'))

    skb_values = np.load(os.path.join(MASTER_DIR, "selectkbest_indicies.npy"))
    kbest_indicies = np.argwhere(skb_values == True)
    X_test = X_test[:, skb_values]

    # Ensembling the model (5 networks) still yields 50% accuracy
    model = formnn_cnn(5000, filters=8, lrval=0.00003, numclasses=12)
    model.summary()
    model.load_weights('best_form_model_50p.hdf5')

    result = model.predict(X_test)
    print(X_test.shape[0] - train_rows)
    for i in range(X_test.shape[0] - train_rows):
        print("Performing predictions on", X_test_names[i + train_rows])
        resultlbl = label_encoder.inverse_transform([np.argmax(result[i + train_rows, :])])
        print("\t\tPredicted form:", resultlbl)
    """
    percent_correct = 0
    pred_table = pd.DataFrame(columns=["Piece", "Predicted", "Actual"])
    for i in range(len(result)):
        resultlbl = label_encoder.inverse_transform([np.argmax(result[i, :])])
        actuallbl = label_encoder.inverse_transform([np.argmax(y_test[i, :])])
        pred_table.loc[i] = ([X_test_names[i][2], resultlbl, actuallbl])
        percent_correct += 1 if resultlbl == actuallbl else 0
    print(pred_table.to_string(index=False))
    print("Accuracy: " + str(float(percent_correct / len(result)) * 100) + "%")
    """
# endregion


# region LabelModel
def formnn_lstm(n_timesteps, mode='concat'):  # Try 'ave', 'mul', and 'sum' also
    model = Sequential()
    model.add(layers.Bidirectional(
        layers.LSTM(20, return_sequences=True), input_shape=(None, 1), merge_mode=mode))
    model.add(layers.TimeDistributed(
        layers.Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_sequence(n_timesteps):
    # create a sequence of random numbers in [0,1]
    X = np.array([random.random() for _ in range(n_timesteps)])
    # calculate cut-off value to change class values
    limit = n_timesteps/4.0
    # determine the class outcome for each item in cumulative sequence
    y = np.array([0 if x < limit else 1 for x in np.cumsum(X)])
    # reshape input and output data to be suitable for LSTMs
    # print(X) [0.436576 0.35750063 0.41489899 0.19143477 0.01814592 0.89638702 0.01744344 0.63694126 0.614542 0.623846]
    # print(y) [0 0 0 0 0 0 0 1 1 1]
    X = X.reshape(1, n_timesteps, 1)  # from (10,) to (1, 10, 1)
    y = y.reshape(1, n_timesteps, 1)
    return X, y


def trainLabelModel_helper(model, n_timesteps, num_epochs=250):
    # early_stopping = EarlyStopping(patience=5, verbose=5, mode="auto")  # Does not work without validation set
    # checkpoint = ModelCheckpoint(os.path.join(MASTER_DIR, 'best_formNN_label_model.hdf5'), monitor='val_accuracy',
    #                            verbose=0, save_best_only=False, mode='max', save_freq='epoch', save_weights_only=True)
    history_loss = []
    # history_val_loss = []
    history_accuracy = []
    # history_val_accuracy = []
    tr_set = pd.DataFrame(du.ReadLabelSecondsPhrasesFromFolder(FULL_LABELPATH, valid_only=True)[0:2]).transpose()
    tr_set = np.array(tr_set)
    # print(tr_set)
    # for i in range(num_epochs):
    for i in range(tr_set.shape[0]):
        Xt = tr_set[i][0]
        yt = tr_set[i][1]
        Xt = Xt.reshape(1, len(Xt), 1)
        yt = yt.reshape(1, len(yt), 1)
        # print(Xt)
        # print(yt)
        # X, y = get_sequence(n_timesteps)  # generate new random sequence
        X, y = get_sequence(tr_set.shape[0])  # generate new random sequence
        # print(X, y)
        model_history = model.fit(Xt, yt, epochs=1, batch_size=1, verbose=1)  # , callbacks=[checkpoint])
        history_loss.append(model_history.history['loss'])
        # history_val_loss.append(model_history.history['val_loss'])
        history_accuracy.append(model_history.history['accuracy'])
        # history_val_accuracy.append(model_history.history['val_accuracy'])
        print("Epochs completed:", i)
    # return [history_loss, history_val_loss, history_accuracy, history_val_accuracy]
    return [history_loss, history_accuracy]


def trainLabelModel():
    # TODO: Model should take in timestamp array (X) and labels (y) for training, eval only provide novelty timestamps

    n_timesteps = 10
    model = formnn_lstm(n_timesteps, mode='concat')
    model_history = trainLabelModel_helper(model, n_timesteps, num_epochs=250)
    plt.plot(model_history[0])  # plt.plot(model_history.history['loss'])
    # plt.plot(model_history[1])  # plt.plot(model_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Initial_LabelModel_Loss.png')
    plt.show()

    plt.plot(model_history[1])  # plt.plot(model_history.history['accuracy'])
    # plt.plot(model_history[3])  # plt.plot(model_history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Initial_LabelModel_Accuracy.png')
    plt.show()

    print("Evaluating...")
    X, y = get_sequence(n_timesteps)
    score = model.evaluate(X, y)
    print("Evaluation complete!\nScore:")
    print(f"Loss: {score[0]}\tAccuracy: {score[1]}")

    print("Predicting...")
    X, y = get_sequence(n_timesteps)
    yhat = model.predict(X, verbose=1)
    print("Prediction complete!")
    for i in range(n_timesteps):
        print('Expected:', y[0, i], 'Predicted', yhat[0, i])
    pass
# endregion


if __name__ == '__main__':
    print("Hello world!")
    # validate_directories()
    # get_total_duration()
    # generate_label_files()
    # prepare_model_training_input()
    # prepare_train_data()
    # buildValidationSet()
    # create_form_dataset()

    # TODO: try to get 60-70% val_acc. Try data augmentation
    # generate_augmented_datasets()
    trainFormModel()
    # predictForm()

    # prepare_lstm_peaks()
    # trainLabelModel()
    print("\nDone!")

    # Measure confidence level? Prediction Interval (PI)
    # https://medium.com/hal24k-techblog/how-to-generate-neural-network-confidence-intervals-with-keras-e4c0b78ebbdf
    # https://github.com/philipperemy/keract#display-the-activations-as-a-heatmap-overlaid-on-an-image
    # Check out humdrum dataset https://www.humdrum.org/
    # Josh Albrecht? https://www.kent.edu/music/joshua-albrecht
    # Publish in Society for Music Theorists? SMT - Due mid-Feb., conference in Nov.
    # https://towardsdatascience.com/10-minutes-to-building-a-cnn-binary-image-classifier-in-tensorflow-4e216b2034aa
    # Use novelty function for each song, save array of predicted peaks using np.save and load
    # Include duration as a feature!!!
