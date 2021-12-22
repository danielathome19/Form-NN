# IMPORT NECESSARY LIBRARIES
import glob
import math
import time
from threading import Thread
import librosa
import matplotlib.pyplot as plt
import librosa.display
import pyaudio
import wave
from IPython.display import Audio
import numpy as np
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
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from pydub import AudioSegment
import keras.layers as kl
import keras.applications as ka
import keras.optimizers as ko
import keras.models as km

# ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# MLS MODEL
def cnn1(output_channels, x, lrval=0.0001):
    model = tf.keras.Sequential()
    model.add(layers.ConvLSTM2D(input_shape=(1, ), output_channels=output_channels,
                                kernel_size=(5, 7), strides=(1, 1),
                                padding=((5 - 1) // 2, (7 - 1) // 2)
                                ))
    model.add(layers.MaxPooling2D(pool_size=(5, 3), strides=(5, 1), padding=(1, 1)))
    # model.add(layers.Conv1D(64, kernel_size=10, activation='relu', input_shape=(X_train.shape[1], 1)))
    # model.add(layers.Conv1D(128, kernel_size=10, activation='relu', kernel_regularizer=l2(0.01),
    #                         bias_regularizer=l2(0.01)))
    # model.add(layers.MaxPooling1D(pool_size=6))
    # model.add(layers.Dropout(0.4))
    # model.add(layers.Conv1D(128, kernel_size=10, activation='relu'))
    # model.add(layers.MaxPooling1D(pool_size=6))

    print("Input: ", x.shape)
    model.add(layers.Conv1D(32, kernel_size=(5, 7), activation=layers.LeakyReLU(alpha=lrval)))
    model.add(layers.MaxPooling1D(pool_size=(5, 3)))
    # model.add(layers.Dense(6, activation='sigmoid'))
    opt = keras.optimizers.Adam(lr=lrval)  # learning rate
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# SSLM MODEL
def cnn_sslm(output_channels, x, lrval=0.0001):
    model = tf.keras.Sequential()

    return model


# ? MODEL
def cnn2(output_channels, x, lrval=0.0001):
    model = tf.keras.Sequential()
    model.add(layers.ConvLSTM2D(input_shape=(output_channels, ), output_channels=(output_channels * 2),
                                kernel_size=(3, 5), strides=(1, 1),
                                padding=((3 - 1) // 2, (5 - 1) * 3 // 2), dilation_rate=(1, 3)
                                ))

    model.add(layers.SpatialDropout2D())
    model.add(layers.ConvLSTM2D(output_channels * 152, 128, (1, 1)))

    model.add(layers.SpatialDropout2D())
    model.add(layers.ConvLSTM2D(128, 1, (1, 1)))

    model.add(layers.ConvLSTM2D(activation=layers.LeakyReLU(alpha=lrval)))
    x = np.reshape(x, -1, x.shape[1] * x.shape[2], 1, x.shape[3])

    model.add(layers.Dropout(activation=keras.activations.linear))
    model.add(layers.LeakyReLU(alpha=lrval))

    model.add(layers.SpatialDropout2D(activation=keras.activations.linear))

    opt = keras.optimizers.Adam(lr=lrval)  # learning rate
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def fuse_model(output_channels1, output_channels2):
    cnn1_mel = cnn1(output_channels1)
    cnn1_sslm = cnn_sslm(output_channels1)
    combined = keras.layers.concatenate([cnn1_mel.output, cnn1_sslm.output])
    cnn2_in = cnn2(output_channels1, combined)
    model = keras.models.Model(inputs=[cnn1_mel.input, cnn1_sslm.input], outputs=cnn2_in)
    return model


# CREATE MLS AND SSLM (Chroma) GRAPHS
def create_mls_sslm2alt(filename, name="", filepath="DEFAULT_FILEPATH"):
    """====================Parameters===================="""
    window_size = 2048  # (samples/frame)
    hop_length = 1024  # overlap 50% (samples/frame)
    sr_desired = 44100
    p = 2  # max-pooling factor
    L_sec = 14  # lag context in seconds
    L = round(L_sec * sr_desired / hop_length)  # conversion of lag L seconds to frames

    y, sr = librosa.load(filename, sr=None)

    if sr != sr_desired:
        y = librosa.core.resample(y, sr, sr_desired)
        sr = sr_desired

    """=================Mel Spectrogram================"""
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=window_size, hop_length=hop_length, n_mels=80, fmin=80,
                                       fmax=16000)
    S_to_dB = librosa.power_to_db(S, ref=np.max)  # convert the spectrogram in dB

    # Plot MLS
    plt.figure(figsize=(10, 4))
    plt.title("Mel Spectrogram")
    fig = plt.imshow(S_to_dB, origin='lower', cmap='viridis', aspect=20)
    plt.colorbar(fig, fraction=0.0115, pad=0.05)
    plt.show()
    print("MLS dimensions are: [mel bands, N]")
    print("MLS dimensions are: [", S_to_dB.shape[0], ",", S_to_dB.shape[1], "]")
    """

    # Gives the same results as MLS when computing chroma cqt
    """=================STFT Spectrogram================"""
    # """
    S = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=window_size, hop_length=hop_length, win_length=window_size,
                                    n_chroma=80)
    S_to_dB = librosa.power_to_db(S, ref=np.max)  # convert the spectrogram in dB

    # Plot STFT
    plt.figure(figsize=(10, 4))
    plt.title("STFT Spectrogram")
    fig = plt.imshow(S_to_dB, origin='lower', cmap='viridis', aspect=20)
    plt.colorbar(fig, fraction=0.0115, pad=0.05)
    plt.show()
    print("STFT dimensions are: [chroma bands, N]")
    print("STFT dimensions are: [", S_to_dB.shape[0], ",", S_to_dB.shape[1], "]")
    # """

    padding_factor = L  # frames
    pad = np.full((S_to_dB.shape[0], padding_factor), -70)  # matrix of 80x30frames of -70dB corresponding to padding
    S_padded = np.concatenate((pad, S_to_dB), axis=1)  # padding 30 frames with noise at -70dB at the beginning

    # Plot S_padded
    plt.figure(figsize=(12, 6))
    plt.title("S_padded")
    plt.imshow(S_padded, origin='lower', cmap='viridis', aspect=20)
    plt.show()
    print("S_padded dimensions are: [chroma bands, N+L] (with Lin frames)")
    print("S_padded dimensions are: [", S_padded.shape[0], ",", S_padded.shape[1], "]")

    x_prime = skimage.measure.block_reduce(S_padded, (1, p), np.max)  # Mel Spectrogram downsampled

    # Plot x_prime
    plt.figure(figsize=(6, 6))
    plt.title("x_prime")
    fig = plt.imshow(x_prime, origin='lower', cmap='viridis', aspect=5)
    plt.show()
    print("x_prime dimensions are: [chroma bands, (N+L)/p] (with L in frames)")
    print("x_prime dimensions are: [", x_prime.shape[0], ",", x_prime.shape[1], "]")

    # MFCCs calculation by computing the Discrete Cosine Transform of type II (DCT-II)
    # MFCCs = scipy.fftpack.dct(x_prime, axis=0, type=2, norm='ortho')
    # MFCCs = MFCCs[1:, :] ?

    # chromas = librosa.feature.chroma_stft(y=y, sr=sr)
    # chromas = np.mean(chromas, axis=0)

    chromagram = librosa.feature.chroma_cqt(y, sr=sr, hop_length=hop_length, fmin=80)

    # Plot chromas
    plt.figure(figsize=(15, 10))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
    plt.colorbar()
    plt.title("Constant-Q Chromagram")
    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(15, 10))
    # plt.title("MFCCs")
    # plt.imshow(MFCCs, origin='lower', cmap='viridis', aspect=10)
    # plt.show()

    # plt.figure(figsize=(15, 10))
    # librosa.display.specshow(chromas, y_axis='chroma', x_axis='time')
    # plt.colorbar()
    # plt.title('Power spectrum chromagram')
    # plt.tight_layout()
    # plt.show()
    print("Chromagram dimensions are: [chroma bands - 1, (N+L)/p] (with L in frames)")
    print("Chromagram dimensions are: [", chromagram.shape[0], ",", chromagram.shape[1], "]")

    # Bagging frames
    m = 2  # bagging parameter in frames
    x = [np.roll(chromagram, n, axis=1) for n in range(m)]
    x_hat = np.concatenate(x, axis=0)

    # Plot x_hat
    plt.figure(figsize=(15, 10))
    plt.title("x_hat")
    plt.imshow(x_hat, origin='lower', cmap='viridis', aspect=10)
    plt.show()
    print("x_hat dimensions are: [(chroma bands - 1)*m, (N+L)/p] (with L in frames)")
    print("x_hat dimensions are: [", x_hat.shape[0], ",", x_hat.shape[1], "]")

    """ Cosine Distance SSLM """
    # Cosine distance calculation: D[N/p,L/p] matrix
    distances = np.zeros((x_hat.shape[1], padding_factor // p))  # D has as dimensions N/p and L/p
    for i in range(x_hat.shape[1]):  # iteration in columns of x_hat
        for l in range(padding_factor // p):
            if i - (l + 1) < 0:
                cosine_dist = 1
            elif i - (l + 1) < padding_factor // p:
                cosine_dist = 1
            else:
                cosine_dist = distance.cosine(x_hat[:, i],
                                              x_hat[:, i - (l + 1)])  # cosine distance between columns i and i-L
            distances[i, l] = cosine_dist

    # Plot Distances
    plt.figure(figsize=(15, 10))
    plt.title("Cosine Distances")
    fig = plt.imshow(np.transpose(distances), origin='lower', cmap='viridis', aspect=2)
    plt.colorbar(fig, fraction=0.009, pad=0.05)
    plt.show()
    print("Distance matrix dimensions are: [N/p, L/p] (with L in frames)")
    print("Distance matrix dimensions are: [", distances.shape[0], ",", distances.shape[1], "]")

    # Threshold epsilon[N/p,L/p] calculation
    kappa = 0.1  # equalization factor of 10%
    epsilon = np.zeros((distances.shape[0], padding_factor // p))  # D has as dimensions N/p and L/p
    for i in range(padding_factor // p, distances.shape[0]):  # iteration in columns of x_hat
        for l in range(padding_factor // p):
            epsilon[i, l] = np.quantile(np.concatenate((distances[i - l, :], distances[i, :])), kappa)

    # Plot Epsilon
    plt.figure(figsize=(15, 10))
    plt.title("Epsilon")
    fig = plt.imshow(np.transpose(epsilon), origin='lower', cmap='viridis', aspect=2)
    plt.colorbar(fig, fraction=0.009, pad=0.05)
    plt.show()
    print("Epsilon matrix dimensions are: [N/p, L/p] (with L in frames)")
    print("Epsilon matrix dimensions are: [", epsilon.shape[0], ",", epsilon.shape[1], "]")

    # Removing initial padding now taking into account the max-poolin factor
    distances = distances[padding_factor // p:, :]
    epsilon = epsilon[padding_factor // p:, :]
    x_prime = x_prime[:, padding_factor // p:]

    # Self Similarity Lag Matrix
    sslm = scipy.special.expit(1 - distances / epsilon)  # aplicación de la sigmoide
    sslm = np.transpose(sslm)

    # Plot SSLM
    plt.figure(figsize=(15, 10))
    plt.title("Cosine Distance SSLM")
    fig = plt.imshow(sslm, origin='lower', cmap='viridis', aspect=1)
    plt.colorbar(fig, fraction=0.0125, pad=0.05)
    plt.show()
    print("SSLM dimensions are: [L/p, N/(p*p_2)] (with L in frames an p_2 = 3)")
    print("SSLM  dimensions are: [", sslm.shape[0], ",", sslm.shape[1], "]")

    sslm = skimage.measure.block_reduce(sslm, (1, 3), np.max)
    x_prime = skimage.measure.block_reduce(x_prime, (1, 3), np.max)

    # Check if SSLM has nans and if it has them, substitute them by 0
    for i in range(sslm.shape[0]):
        for j in range(sslm.shape[1]):
            if np.isnan(sslm[i, j]):
                sslm[i, j] = 0

    # Plot STFT Spectrogram
    plt.figure(1, figsize=(15, 10))
    plt.title("Final STFT")

    plt.imshow(x_prime, origin='lower', cmap='plasma', aspect=2)
    plt.show()

    # Plot Final SSLM
    plt.figure(figsize=(15, 10))
    plt.title("Final Cosine Distance SSLM")
    fig = plt.imshow(sslm, origin='lower', cmap='viridis', aspect=0.8)
    plt.show()

    if sslm.shape[1] == x_prime.shape[1]:
        print("Cos SSLM and MLS have the same time dimension (columns).")
    else:
        print("ERROR. Time dimension of Cos SSLM and MLS mismatch.")
        print("MLS has", x_prime.shape[1], "lag bins and the Cos SSLM has", sslm.shape[1])

    """ Euclidian Distance SSLM """
    # Euclidian distance calculation: D[N/p,L/p] matrix
    distances = np.zeros((x_hat.shape[1], padding_factor // p))  # D has as dimensions N/p and L/p
    for i in range(x_hat.shape[1]):  # iteration in columns of x_hat
        for l in range(padding_factor // p):
            if i - (l + 1) < 0:
                eucl_dist = 1
            elif i - (l + 1) < padding_factor // p:
                eucl_dist = 1
            else:
                eucl_dist = distance.euclidean(x_hat[:, i],
                                               x_hat[:, i - (l + 1)])  # euclidian distance between columns i & i-L
            distances[i, l] = eucl_dist

    # Plot Distances
    plt.figure(figsize=(15, 10))
    plt.title("Euclidian Distances")
    fig = plt.imshow(np.transpose(distances), origin='lower', cmap='viridis', aspect=2)
    plt.colorbar(fig, fraction=0.009, pad=0.05)
    plt.show()
    print("Distance matrix dimensions are: [N/p, L/p] (with L in frames)")
    print("Distance matrix dimensions are: [", distances.shape[0], ",", distances.shape[1], "]")

    # Threshold epsilon[N/p,L/p] calculation
    kappa = 0.1  # equalization factor of 10%
    epsilon = np.zeros((distances.shape[0], padding_factor // p))  # D has as dimensions N/p and L/p
    for i in range(padding_factor // p, distances.shape[0]):  # iteration in columns of x_hat
        for l in range(padding_factor // p):
            epsilon[i, l] = np.quantile(np.concatenate((distances[i - l, :], distances[i, :])), kappa)

    # Plot Epsilon
    plt.figure(figsize=(15, 10))
    plt.title("Epsilon")
    fig = plt.imshow(np.transpose(epsilon), origin='lower', cmap='viridis', aspect=2)
    plt.colorbar(fig, fraction=0.009, pad=0.05)
    plt.show()
    print("Epsilon matrix dimensions are: [N/p, L/p] (with L in frames)")
    print("Epsilon matrix dimensions are: [", epsilon.shape[0], ",", epsilon.shape[1], "]")

    # Removing initial padding now taking into account the max-poolin factor
    distances = distances[padding_factor // p:, :]
    epsilon = epsilon[padding_factor // p:, :]

    # Self Similarity Lag Matrix
    sslm = scipy.special.expit(1 - distances / epsilon)  # sigmoid function
    sslm = np.transpose(sslm)

    # Plot SSLM
    plt.figure(figsize=(15, 10))
    plt.title("Euclidian Distance SSLM")
    fig = plt.imshow(sslm, origin='lower', cmap='viridis', aspect=1)
    plt.colorbar(fig, fraction=0.0125, pad=0.05)
    plt.show()
    print("SSLM dimensions are: [L/p, N/(p*p_2)] (with L in frames an p_2 = 3)")
    print("SSLM  dimensions are: [", sslm.shape[0], ",", sslm.shape[1], "]")

    sslm = skimage.measure.block_reduce(sslm, (1, 3), np.max)

    # Check if SSLM has nans and if it has them, substitute them by 0
    for i in range(sslm.shape[0]):
        for j in range(sslm.shape[1]):
            if np.isnan(sslm[i, j]):
                sslm[i, j] = 0

    # Plot Final SSLM
    plt.figure(figsize=(15, 10))
    plt.title("Final Euclidian SSLM")
    fig = plt.imshow(sslm, origin='lower', cmap='viridis', aspect=0.8)
    plt.show()

    if sslm.shape[1] == x_prime.shape[1]:
        print("Euc SSLM and MLS have the same time dimension (columns).")
    else:
        print("ERROR. Time dimension of Euc SSLM and MLS mismatch.")
        print("MLS has", x_prime.shape[1], "lag bins and the Euc SSLM has", sslm.shape[1])
    return


def get_model(lrval=0.0001):
    """
    # images = []
    # vgg19 = ka.VGG19(weights='imagenet', include_top=False)
    # vgg19.trainable = False

    # Image Classification Branch
    x = vgg19(images)
    x = kl.GlobalAveragePooling2D()(x)
    x = kl.Dense(32, activation='relu')(x)
    x = kl.Dropout(rate=0.25)(x)
    x = km.Model(inputs=images, outputs=x)

    # Text Classification Branch
    y = kl.Embedding(vocab_size, EMBEDDING_LENGTH, input_length=200)(features)
    y = kl.SpatialDropout1D(0.25)(y)
    y = kl.LSTM(25, dropout=0.25, recurrent_dropout=0.25)(y)
    y = kl.Dropout(0.25)(y)
    y = keras.models.Model(inputs=features, outputs=y)

    combined = kl.concatenate([x.output, y.output])

    z = kl.Dense(32, activation="relu")(combined)
    z = kl.Dropout(rate=0.25)(z)
    z = kl.Dense(32, activation="relu")(z)
    z = kl.Dropout(rate=0.25)(z)
    z = kl.Dense(3, activation="softmax")(z)

    model = km.Model(inputs=[x.input, y.input], outputs=z)

    model.compile(optimizer=ko.Adam(lr=0.0001), loss='categorical_crossentropy', metrics='accuracy')
    """

    model = km.Model()
    model.compile(optimizer=ko.Adam(lr=lrval), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    print("Hello world!")
