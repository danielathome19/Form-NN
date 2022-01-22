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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization, Dense
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from sklearn import tree
from sklearn.dummy import DummyClassifier
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils
from pydub import AudioSegment
import tensorflow.keras.layers as kl
import tensorflow.keras.applications as ka
import tensorflow.keras.optimizers as ko
import tensorflow.keras.models as km
import skimage.measure
import scipy
from scipy.spatial import distance
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

k.set_image_data_format('channels_last')
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if not sys.warnoptions:
    warnings.simplefilter("ignore")  # ignore warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# region Directories
MASTER_DIR = 'D:/Google Drive/Resources/Dev Stuff/Python/Machine Learning/Master Thesis/'
MASTER_INPUT_DIR = 'D:/Master Thesis Input/'
MASTER_LABELPATH = os.path.join(MASTER_DIR, 'Labels/')

MIDI_Data_Dir = np.array(gb.glob(os.path.join(MASTER_DIR, 'Data/MIDIs/*')))
Train_Data_Dir = np.array(gb.glob(os.path.join(MASTER_INPUT_DIR, 'Train/*')))  # os.path.join(MASTER_DIR, 'Data/Train/*'
Test_Data_Dir = np.array(gb.glob(os.path.join(MASTER_INPUT_DIR, 'Test/*')))  # os.path.join(MASTER_DIR, 'Data/Test/*')))
Validate_Data_Dir = np.array(gb.glob(os.path.join(MASTER_INPUT_DIR, 'Validate/*')))  # os.path.join(MASTER_DIR,'Data/Val

MLS_Data_Dir = os.path.join(MASTER_DIR, 'Images/Train/MLS/')
SSLMCOS_Data_Dir = os.path.join(MASTER_DIR, 'Images/Train/SSLMCOS/')
SSLMEUC_Data_Dir = os.path.join(MASTER_DIR, 'Images/Train/SSLMEUC/')
SSLMCRM_Data_Dir = os.path.join(MASTER_DIR, 'Images/Train/SSLMCRM/')

TRAIN_DIR = os.path.join(MASTER_INPUT_DIR, 'Train/')
TRAIN2_DIR = os.path.join(MASTER_INPUT_DIR, 'Train2/')
TEST_DIR = os.path.join(MASTER_INPUT_DIR, 'Test/')
VAL_DIR = os.path.join(MASTER_INPUT_DIR, 'Validate/')

TRAIN_LABELPATH = os.path.join(MASTER_INPUT_DIR, 'Labels/Train/')
TRAIN2_LABELPATH = os.path.join(MASTER_INPUT_DIR, 'Labels/Train2/')
TEST_LABELPATH = os.path.join(MASTER_INPUT_DIR, 'Labels/Test/')
VAL_LABELPATH = os.path.join(MASTER_INPUT_DIR, 'Labels/Validate/')
# endregion


# region DEPRECATED
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
            du.create_mls_sslm2(fullfilename, name, foldername)
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


"""===================================================================================="""
# region MODEL_DEFINITION


# MLS MODEL
def formnn_mls(output_channels=32, lrval=0.0001):
    inputA = layers.Input(batch_input_shape=(None, None, None, 1))
    x = layers.ZeroPadding2D(padding=((2, 2), (3, 3)))(inputA)
    x = layers.Conv2D(filters=output_channels, kernel_size=(5, 7), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU(alpha=lrval)(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(pool_size=(5, 3), strides=(5, 1), padding='same')(x)
    x = keras.models.Model(inputs=inputA, outputs=x)
    return x


# SSLM MODEL
def formnn_sslm(output_channels=32, lrval=0.0001):
    inputB = layers.Input(batch_input_shape=(None, None, None, 1))  # (None, None, None, 4)
    y = layers.ZeroPadding2D(padding=((2, 2), (3, 3)))(inputB)
    y = layers.Conv2D(filters=output_channels, kernel_size=(5, 7), strides=(1, 1), padding='same')(y)
    y = layers.LeakyReLU(alpha=lrval)(y)
    y = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(y)
    y = layers.MaxPooling2D(pool_size=(5, 3), strides=(5, 1), padding='same')(y)
    y = layers.ZeroPadding2D(padding=(13, 0))(y)
    y = keras.models.Model(inputs=inputB, outputs=y)
    return y


# PIPELINE MODEL
def formnn_pipeline(combined, output_channels=32, lrval=0.0001):
    z = layers.ZeroPadding2D(padding=((1, 1), (6, 6)))(combined)
    z = layers.Conv2D(filters=(output_channels * 2), kernel_size=(3, 5), strides=(1, 1),
                      padding='same', dilation_rate=(1, 3))(z)
    z = layers.LeakyReLU(alpha=lrval)(z)
    z = layers.SpatialDropout2D(rate=0.5)(z)
    # z = layers.Reshape(target_shape=(-1, 1, output_channels * 152))(z)
    z = layers.Conv2D(filters=output_channels * 4, kernel_size=(1, 1), strides=(1, 1), padding='same')(z)
    z = layers.LeakyReLU(alpha=lrval)(z)
    z = layers.SpatialDropout2D(rate=0.5)(z)
    z = layers.Conv2D(filters=output_channels * 8, kernel_size=(1, 1), strides=(1, 1), padding='same')(z)
    z = layers.LeakyReLU(alpha=lrval)(z)
    z = layers.GlobalMaxPooling2D()(z)
    z = layers.Flatten()(z)
    z = layers.Dense(12, activation='sigmoid')(z)
    return z


def formnn_fuse(output_channels=32, lrval=0.0001):
    cnn1_mel = formnn_mls(output_channels, lrval=lrval)
    cnn1_sslm = formnn_sslm(output_channels, lrval=lrval)
    combined = layers.concatenate([cnn1_mel.output, cnn1_sslm.output], axis=2)
    cnn2_in = formnn_pipeline(combined, output_channels, lrval=lrval)
    opt = keras.optimizers.Adam(lr=lrval)  # learning rate
    model = keras.models.Model(inputs=[cnn1_mel.input, cnn1_sslm.input], outputs=[cnn2_in])  # --v ['accuracy'])
    model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer=opt,
                  metrics=['accuracy'])
    # metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    # Try categorical_crossentropy

    model.summary()
    if not os.path.isfile(os.path.join(MASTER_DIR, 'FormNN_Model_Diagram.png')):
        plot_model(model, to_file=os.path.join(MASTER_DIR, 'FormNN_Model_Diagram.png'),
                   show_shapes=True, show_layer_names=True, expand_nested=True, dpi=300)
    return model

# endregion


def trainModel():
    batch_size = 1

    # region MODEL_DIRECTORIES
    mls_train = dus.BuildDataloader(os.path.join(TRAIN_DIR, 'MLS/'), labels_path=TRAIN_LABELPATH,
                                    transforms=[padding_MLS, normalize_image, borders], batch_size=batch_size)
    sslm_cmcos_train = dus.BuildDataloader(os.path.join(TRAIN_DIR, 'SSLM_CRM_COS/'), labels_path=TRAIN_LABELPATH,
                                           transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    sslm_cmeuc_train = dus.BuildDataloader(os.path.join(TRAIN_DIR, 'SSLM_CRM_EUC/'), labels_path=TRAIN_LABELPATH,
                                           transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    sslm_mfcos_train = dus.BuildDataloader(os.path.join(TRAIN_DIR, 'SSLM_MFCC_COS/'), labels_path=TRAIN_LABELPATH,
                                           transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    sslm_mfeuc_train = dus.BuildDataloader(os.path.join(TRAIN_DIR, 'SSLM_MFCC_EUC/'), labels_path=TRAIN_LABELPATH,
                                           transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)

    # """
    mls_train2 = dus.BuildDataloader(os.path.join(TRAIN2_DIR, 'MLS/'), labels_path=TRAIN2_LABELPATH,
                                     transforms=[padding_MLS, normalize_image, borders], batch_size=batch_size)
    sslm_cmcos_train2 = dus.BuildDataloader(os.path.join(TRAIN2_DIR, 'SSLM_CRM_COS/'), labels_path=TRAIN2_LABELPATH,
                                            transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    sslm_cmeuc_train2 = dus.BuildDataloader(os.path.join(TRAIN2_DIR, 'SSLM_CRM_EUC/'), labels_path=TRAIN2_LABELPATH,
                                            transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    sslm_mfcos_train2 = dus.BuildDataloader(os.path.join(TRAIN2_DIR, 'SSLM_MFCC_COS/'), labels_path=TRAIN2_LABELPATH,
                                            transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    sslm_mfeuc_train2 = dus.BuildDataloader(os.path.join(TRAIN2_DIR, 'SSLM_MFCC_EUC/'), labels_path=TRAIN2_LABELPATH,
                                            transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    """

    mls_val = dus.BuildDataloader(os.path.join(VAL_DIR, 'MLS/'), labels_path=VAL_LABELPATH,
                                  transforms=[padding_MLS, normalize_image, borders], batch_size=batch_size)
    sslm_cmcos_val = dus.BuildDataloader(os.path.join(VAL_DIR, 'SSLM_CRM_COS/'), labels_path=VAL_LABELPATH,
                                         transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    sslm_cmeuc_val = dus.BuildDataloader(os.path.join(VAL_DIR, 'SSLM_CRM_EUC/'), labels_path=VAL_LABELPATH,
                                         transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    sslm_mfcos_val = dus.BuildDataloader(os.path.join(VAL_DIR, 'SSLM_MFCC_COS/'), labels_path=VAL_LABELPATH,
                                         transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    sslm_mfeuc_val = dus.BuildDataloader(os.path.join(VAL_DIR, 'SSLM_MFCC_EUC/'), labels_path=VAL_LABELPATH,
                                         transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)

    "" "
    mls_test = dus.BuildDataloader(os.path.join(TEST_DIR, 'MLS/'), labels_path=TEST_LABELPATH,
                                   transforms=[padding_MLS, normalize_image, borders], batch_size=batch_size)
    sslm_cmcos_test = dus.BuildDataloader(os.path.join(TEST_DIR, 'SSLM_CRM_COS/'), labels_path=TEST_LABELPATH,
                                          transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    sslm_cmeuc_test = dus.BuildDataloader(os.path.join(TEST_DIR, 'SSLM_CRM_EUC/'), labels_path=TEST_LABELPATH,
                                          transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    sslm_mfcos_test = dus.BuildDataloader(os.path.join(TEST_DIR, 'SSLM_MFCC_COS/'), labels_path=TEST_LABELPATH,
                                          transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    sslm_mfeuc_test = dus.BuildDataloader(os.path.join(TEST_DIR, 'SSLM_MFCC_EUC/'), labels_path=TEST_LABELPATH,
                                          transforms=[padding_SSLM, normalize_image, borders], batch_size=batch_size)
    """
    # endregion

    def multi_input_generator_helper(gen1, gen2, gen3, gen4):
        while True:
            yield tf.expand_dims(np.concatenate((next(gen1)[0],
                                                 np.concatenate((next(gen2)[0],
                                                                 np.concatenate((next(gen3)[0], next(gen4)[0]),
                                                                                axis=-1)), axis=-1)), axis=-1), axis=-1)

    def multi_input_generator(gen1, gen2, gen3, gen4, gen5, stop=-1):
        while True:
            if stop != -1:  # TODO: remove condition
                stop -= 1
                if stop == 0:
                    break
            tpl = next(gen1)  # keras.utils.to_categorical(tpl[1])
            # yield [tf.expand_dims(tpl[0], axis=-1), tf.expand_dims(next(gen2)[0], axis=-1)], tpl[1]
            yield [tpl[0], next(multi_input_generator_helper(gen2, gen3, gen4, gen5))], tpl[1]

    train_datagen = multi_input_generator(mls_train, sslm_cmcos_train, sslm_cmeuc_train, sslm_mfcos_train,
                                          sslm_mfeuc_train)
    valid_datagen = multi_input_generator(mls_train2, sslm_cmcos_train2, sslm_cmeuc_train2, sslm_mfcos_train2,
                                          sslm_mfeuc_train2)  # , stop=13)
    # valid_datagen = multi_input_generator(mls_val, sslm_cmcos_val, sslm_cmeuc_val, sslm_mfcos_val, sslm_mfeuc_val)
    # test_datagen = multi_input_generator(mls_test, sslm_cmcos_test, sslm_cmeuc_test, sslm_mfcos_test, sslm_mfeuc_test

    steps_per_epoch = len(list(mls_train)) // batch_size
    steps_per_valid = len(list(mls_train2)) // batch_size  # mls_val
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(os.path.join(MASTER_DIR, 'form_classes.npy'))  # len(label_encoder.classes_) for nn

    trmodel = formnn_fuse(output_channels=32, lrval=0.00001)
    # , lrval=.009999999776482582)(32) CNN Layer 1 Output Characteristic Maps                'val_loss' or val_accuracy
    checkpoint = ModelCheckpoint(os.path.join(MASTER_DIR, 'best_formNN_model.hdf5'), monitor='val_accuracy', verbose=0,
                                 save_best_only=True, mode='max', save_freq='epoch', save_weights_only=True)
    model_history = trmodel.fit(train_datagen, epochs=10, verbose=1, validation_data=valid_datagen,
                                shuffle=False, callbacks=[checkpoint], batch_size=batch_size,
                                steps_per_epoch=steps_per_epoch, validation_steps=steps_per_valid)

    print("Training complete!\n")

    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Initial_Model_Loss.png')
    plt.show()

    # print(model_history.history.keys())
    #   dict_keys(['loss', 'precision', 'recall', 'val_loss', 'val_precision', 'val_recall'])
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

    """
    plt.plot(model_history.history['precision'])
    plt.plot(model_history.history['val_precision'])
    plt.title('Model Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Initial_Model_Precision.png')
    plt.show()

    plt.plot(model_history.history['recall'])
    plt.plot(model_history.history['val_recall'])
    plt.title('Model Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Initial_Model_Recall.png')
    plt.show()
    """

    predictions = trmodel.predict_generator(train_datagen, steps=1, verbose=1)
    print(predictions)
    print("Prediction complete!")

    inverted = label_encoder.inverse_transform([np.argmax(predictions[0, :])])
    print(inverted)

    """
    # plt.plot(prediction)
    # plt.show()
    # y_pred = trmodel.predict(x_test, batch_size=batch_size, verbose=1)
    # y_pred_bool = np.argmax(y_pred, axis=1)
    # print(classification_report(y_test, y_pred_bool))
    """

    # TODO: create a dense model (RNN?) that uses the audio data to classify the peaks


if __name__ == '__main__':
    print("Hello world!")
    # get_total_duration()
    # generate_label_files()
    # prepare_model_training_input()
    # prepare_train_data()
    # buildValidationSet()
    trainModel()
    print("\nDone!")
