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
import keras.layers as kl
import keras.applications as ka
import keras.optimizers as ko
import keras.models as km
import skimage.measure
from skimage.transform import resize
import scipy
from scipy.spatial import distance
import librosa.segment
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import math
from scipy import signal
import numpy as np
from os import listdir, walk, getcwd, sep
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

MASTER_DIR = 'D:/Google Drive/Resources/Dev Stuff/Python/Machine Learning/Master Thesis/'

# Output filepath for training images and labels
DEFAULT_FILEPATH = os.path.join(MASTER_DIR, 'Images/Train/')
DEFAULT_LABELPATH = os.path.join(MASTER_DIR, 'Labels/')


# LOG-SCALED MEL SPECTROGRAM (depricated)
def create_spectrogram(filename, name, filepath=DEFAULT_FILEPATH):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, x_axis='s', y_axis='log')
    filename = filepath + os.path.basename(name) + '.png'
    # print(filename)
    # fp = open(filename, 'x')
    # fp.close()
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename, name, clip, sample_rate, fig, ax, S


# CREATE MLS AND SSLM (MFCC) GRAPHS (depricated)
def create_mls_sslm(filename, name="", foldername="", filepath=DEFAULT_FILEPATH):
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
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=window_size, hop_length=hop_length, n_mels=80, fmin=80,
                                       fmax=16000)
    S_to_dB = librosa.power_to_db(S, ref=np.max)  # convert the spectrogram in dB

    # Plot MLS
    plt.figure(figsize=(10, 4))
    plt.title("Mel Spectrogram")
    fig = plt.imshow(S_to_dB, origin='lower', cmap='plasma', aspect=20)
    plt.colorbar(fig, fraction=0.0115, pad=0.05)
    plt.show()
    print("MLS dimensions are: [mel bands, N]")
    print("MLS dimensions are: [", S_to_dB.shape[0], ",", S_to_dB.shape[1], "]")

    padding_factor = L  # frames
    pad = np.full((S_to_dB.shape[0], padding_factor), -70)  # matrix of 80x30frames of -70dB corresponding to padding
    S_padded = np.concatenate((pad, S_to_dB), axis=1)  # padding 30 frames with noise at -70dB at the beginning

    # Plot S_padded
    plt.figure(figsize=(12, 6))
    plt.title("S_padded")
    plt.imshow(S_padded, origin='lower', cmap='viridis', aspect=20)
    plt.show()
    print("S_padded dimensions are: [mel bands, N+L] (with L in frames)")
    print("S_padded dimensions are: [", S_padded.shape[0], ",", S_padded.shape[1], "]")

    x_prime = skimage.measure.block_reduce(S_padded, (1, p), np.max)  # Mel Spectrogram downsampled

    # Plot x_prime
    plt.figure(figsize=(6, 6))
    plt.title("x_prime")
    fig = plt.imshow(x_prime, origin='lower', cmap='viridis', aspect=5)
    plt.show()
    print("x_prime dimensions are: [mel bands, (N+L)/p] (with L in frames)")
    print("x_prime dimensions are: [", x_prime.shape[0], ",", x_prime.shape[1], "]")

    # MFCCs calculation by computing the Discrete Cosine Transform of type II (DCT-II)
    MFCCs = scipy.fftpack.dct(x_prime, axis=0, type=2, norm='ortho')
    MFCCs = MFCCs[1:, :]

    # Plot MFCCs
    plt.figure(figsize=(15, 10))
    plt.title("MFCCs")
    plt.imshow(MFCCs, origin='lower', cmap='viridis', aspect=10)
    plt.show()
    print("MFCCs dimensions are: [mel bands - 1, (N+L)/p] (with L in frames)")
    print("MFCCs dimensions are: [", MFCCs.shape[0], ",", MFCCs.shape[1], "]")

    # Bagging frames
    m = 2  # bagging parameter in frames
    x = [np.roll(MFCCs, n, axis=1) for n in range(m)]
    x_hat = np.concatenate(x, axis=0)

    # Plot x_hat
    plt.figure(figsize=(15, 10))
    plt.title("x_hat")
    plt.imshow(x_hat, origin='lower', cmap='viridis', aspect=10)
    plt.show()
    print("x_hat dimensions are: [(mel bands - 1)*m, (N+L)/p] (with L in frames)")
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
    sslm = scipy.special.expit(1 - distances / epsilon)  # sigmoid function
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

    # Plot Mel Spectrogram
    plt.figure(1, figsize=(15, 10))
    plt.title("Final MLS")
    plt.imshow(x_prime, origin='lower', cmap='plasma', aspect=2)
    plt.show()

    fig = plt.figure(1, figsize=(15, 10))
    plt.imshow(x_prime, origin='lower', cmap='plasma', aspect=2)
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    filename = filepath + "MLS/" + os.path.basename(name) + 'mls.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)  # dpi=400, transparent=True
    fig.clf()
    plt.close(fig)
    del ax, fig

    # Plot Final SSLM
    plt.figure(figsize=(15, 10))
    plt.title("Final Cosine Distance SSLM")
    plt.imshow(sslm, origin='lower', cmap='viridis', aspect=0.8)
    plt.show()

    fig = plt.figure(1, figsize=(15, 10))
    plt.imshow(sslm, origin='lower', cmap='viridis', aspect=0.8)
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    filename = filepath + "SSLMCOS/" + os.path.basename(name) + 'cos.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)  # dpi=400, transparent=True
    fig.clf()
    plt.close(fig)
    del ax, fig

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
    plt.imshow(sslm, origin='lower', cmap='viridis', aspect=0.8)
    plt.show()

    fig = plt.figure(1, figsize=(15, 10))
    plt.imshow(sslm, origin='lower', cmap='viridis', aspect=0.8)
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    filename = filepath + "SSLMEUC/" + os.path.basename(name) + 'euc.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)  # dpi=400, transparent=True
    fig.clf()
    plt.close(fig)
    del ax, fig

    if sslm.shape[1] == x_prime.shape[1]:
        print("Euc SSLM and MLS have the same time dimension (columns).")
        print("Number of lag bins:", sslm.shape[1])
    else:
        print("ERROR. Time dimension of Euc SSLM and MLS mismatch.")
        print("MLS has", x_prime.shape[1], "lag bins and the Euc SSLM has", sslm.shape[1])
    return


# CREATE CHROMA GRAPHS (depricated)
def create_mls_sslm2(filename, name="", foldername="", filepath=DEFAULT_FILEPATH):
    # ------------PARAMETERS--------------
    window_size = 0.209  # sec/frame
    samples_frame = 8192  # samples_frame = math.ceil(window_size*sr)
    hop_size = 0.139  # sec/frame
    hop_length = 6144  # hop_length = math.ceil(hop_size*sr) #overlap 25% (samples/frame)
    sr_desired = 44100

    y, sr = librosa.load(filename, sr=None)

    if sr != sr_desired:
        y = librosa.core.resample(y, sr, sr_desired)
        sr = sr_desired

    # ------------Fourier Fast Transform-------------
    stft = np.abs(librosa.stft(y, n_fft=samples_frame, hop_length=hop_length))
    fft_freq = librosa.core.fft_frequencies(sr=sr, n_fft=samples_frame)

    """-----------PLOTTING MEL-SPECTROGRAM-------------"""
    # Spectogram from SFTF
    librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max), y_axis='log', x_axis='frames')
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

    # ------------PCP calculation (Chroma)-------------
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr, n_fft=samples_frame, hop_length=hop_length)

    """-----------PLOTTING CHROMAS-------------"""
    # PCPs or Chroma from spectogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, sr=sr, y_axis='chroma', x_axis='frames', cmap="coolwarm")
    plt.colorbar()
    plt.title('Chromagram')
    plt.tight_layout()
    plt.show()
    print("Chroma dimensions are: [chroma vectors, N']")
    print("Chroma dimensions are: [", chroma.shape[0], ",", chroma.shape[1], "]")

    # vector x_hat construction. x in Serra's paper is chroma here
    m = round(5 * sr / hop_length)
    tau = 1
    w = (m - 1) * tau
    chroma = np.concatenate((np.zeros((chroma.shape[0], w)), chroma), axis=1)
    x = [np.roll(chroma, tau * n, axis=1) for n in range(m)]
    x_ = np.concatenate(x, axis=0)

    X_hat = x_[:, w:]  # (w, frames)

    N_prima = chroma.shape[1]
    N = N_prima - w

    """PLOTTING x, x_ and the resulting x_hat"""
    # x (first chroma)
    plt.figure(figsize=(15, 7))
    plt.title('First chroma vector: x[0]')
    plt.imshow(np.asarray(x[0]), origin='lower', cmap='plasma', aspect=2)
    plt.show()

    # x_
    plt.figure(figsize=(15, 7))
    plt.title('x_')
    plt.imshow(x_, origin='lower', cmap='plasma', aspect=0.5)
    plt.show()

    # x_hat
    plt.figure(figsize=(15, 7))
    plt.title('x_hat')
    plt.imshow(X_hat, origin='lower', cmap='plasma', aspect=0.5)
    plt.show()
    print("X_hat dimensions are: [chroma vectors * m (in samples), N'] = [", chroma.shape[0], "*", m, ", N']")
    print("X_hat dimensions are: [", X_hat.shape[0], ",", X_hat.shape[1], "]")

    """PLOTTING RECURRENCE PLOT with Librosa from CHROMAS"""
    # Recurrence matrix from librosa
    recurrence = librosa.segment.recurrence_matrix(chroma, mode='affinity', k=chroma.shape[1])
    plt.figure(figsize=(7, 7))
    plt.title('Recurrence matrix from chroma vector from LIBROSA')
    plt.imshow(recurrence, cmap='gray')
    plt.show()

    # Plot recurrence matrix of vector x with librosa
    recurrence2 = librosa.segment.recurrence_matrix(x, k=14, sym=True)
    plt.figure(figsize=(7, 7))
    plt.title('Recurrence matrix of x vector with k=13 neighbors from LIBROSA')
    plt.imshow(1 - recurrence2, cmap='gray')
    plt.show()

    # ------------K-Nearest-Neighboors-------------
    K = 14  # K = round(N*0.03)
    nbrs = NearestNeighbors(n_neighbors=K).fit(X_hat.T)
    distances, indices = nbrs.kneighbors(X_hat.T)
    R = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if (i in indices[j]) and (j in indices[i]) and (i != j):
                R[i, j] = 1

    """PLOTTING RECURRENCE PLOT"""
    # Plot recurrence matrix of vector R (same as above)
    plt.figure(figsize=(7, 7))
    plt.title('Recurrence matrix R')
    plt.imshow(1 - R, cmap='gray')
    plt.show()

    # ------------Lag matrix construction-------------
    L = librosa.segment.recurrence_to_lag(R, pad=False)  # None

    """PLOTTING LAG MATRIX"""
    # Lag Matrix calculated from R
    plt.figure(figsize=(7, 7))
    plt.title('Lag Matrix')
    plt.imshow(1 - L, cmap='gray')
    plt.show()

    # Smoothing signal with Gaussian windows of 30 samples length
    s1 = round(0.3 * sr / hop_length)
    st = round(30 * sr / hop_length)
    sigma1 = (s1 - 1) / (2.5 * 2)
    sigmat = (st - 1) / (2.5 * 2)
    g1 = signal.gaussian(s1, std=sigma1).reshape(s1, 1)  # g1 in paper
    gt = signal.gaussian(st, std=sigmat).reshape(st, 1)  # gt in paper
    G = np.matmul(g1, gt.T)

    """PLOTTING GAUSSIAN WINDOW"""
    # Plot Gaussian window
    plt.plot(gt)
    plt.title("Gaussian window ($\sigma$=7)")
    plt.ylabel("Amplitude")
    plt.xlabel("Sample")
    plt.show()

    """PLOTTING G"""
    # Gaussian kernel G
    plt.figure(figsize=(7, 7))
    plt.title('Gaussian kernel G')
    plt.imshow(1 - G, origin='lower', cmap='gray', aspect=40)
    plt.show()

    # Applyin gaussian filter to Lag matrix
    P = signal.convolve2d(L, G, mode='same')

    """PLOTTING MATRICES after GAUSSIAN SMOOTHING"""
    # Plot R matrix after Gaussian smoothing
    P2 = librosa.segment.lag_to_recurrence(P, axis=-1)
    plt.figure(figsize=(7, 7))
    plt.title('Recurrence matrix R after gaussian')
    plt.imshow(1 - P2, cmap='gray')
    plt.show()

    # Plot Lag matrix after Gaussian smoothing
    plt.figure(figsize=(7, 7))
    plt.title('Lag matrix L after gaussian')
    plt.imshow(1 - P, cmap='gray')
    plt.show()

    # Novelty curve
    c = np.linalg.norm(P[:, 1:] - P[:, 0:-1], axis=0)
    c_norm = (c - c.min()) / (c.max() - c.min())  # normalization of c

    """PLOTTING NOVELTY CURVE"""
    # Plot novelty function with boundaries
    frames = range(len(c_norm))
    plt.figure(figsize=(10, 4))
    plt.title('Novelty function vector c')
    plt.xlabel('Frames')
    plt.plot(frames, c_norm)
    plt.show()

    # Peaks detection - sliding window
    delta = 0.05  # threshold
    lamda = round(6 * sr / hop_length)  # window length
    peaks_position = signal.find_peaks(c_norm, height=delta, distance=lamda, width=round(0.5 * sr / hop_length))[
        0]  # array of peaks
    peaks_values = signal.find_peaks(c_norm, height=delta, distance=lamda, width=round(0.5 * sr / hop_length))[1][
        'peak_heights']  # array of peaks
    b = peaks_position
    # Adding elements 1 and N' to the begining and end of the arrray
    if b[0] != 0:
        b = np.concatenate([[0], b])  # b: segment boundaries
    if b[-1] != N_prima - 1:
        b = np.concatenate([b, [N - 1]])

    """PLOTTING NOVELTY CURVE"""
    # Plot novelty function with boundaries
    frames = range(len(c_norm))
    plt.figure(figsize=(10, 4))
    plt.title('Novelty function vector c (red lines are peaks)')
    plt.xlabel('Frames')
    for i in range(len(b)):
        plt.axvline(b[i], color='r', linestyle='--')
    plt.plot(frames, c_norm)
    plt.show()

    # Cumulative matrix: Q
    Q = np.zeros_like(R)
    for u in range(b.shape[0] - 1):
        for v in range(b.shape[0] - 1):
            Q_uv = np.copy(R[b[u]:b[u + 1], b[v]:b[v + 1]])
            for i in range(1, Q_uv.shape[0]):
                for j in range(1, Q_uv.shape[1]):
                    if i == 1 and j == 1:
                        Q_uv[i, j] += Q_uv[i - 1, j - 1]
                    elif i == 1:
                        Q_uv[i, j] += max(Q_uv[i - 1, j - 1], Q_uv[i - 1, j - 2])
                    elif j == 1:
                        Q_uv[i, j] += max(Q_uv[i - 1, j - 1], Q_uv[i - 2, j - 1])
                    else:
                        Q_uv[i, j] += max(Q_uv[i - 1, j - 1], Q_uv[i - 2, j - 1], Q_uv[i - 1, j - 2])

            Q[b[u]:b[u + 1], b[v]:b[v + 1]] = Q_uv

    """PLOTTING CUMULATIVE MATRIX"""
    # Cumulative matrix plot
    plt.figure(figsize=(7, 7))
    plt.title('Cumulative matrix Q')
    plt.imshow(1 - Q, cmap='gray')
    plt.show()

    # Normalization of Q matrix: Segment similarity matrix S
    num_segments = b.shape[0] - 1
    S = np.zeros((num_segments, num_segments))
    for u in range(b.shape[0] - 1):
        for v in range(b.shape[0] - 1):
            S[u, v] = np.max(Q[b[u]:b[u + 1], b[v]:b[v + 1]]) / min(b[u + 1] - b[u], b[v + 1] - b[v])

    """PLOTTING SEGMENT SIMILARITY MATRIX"""
    # Plot Segment similarity matrix S
    plt.figure(figsize=(7, 7))
    plt.title('Segment matrix S')
    plt.imshow(1 - S, cmap='gray')
    # for i in range(len(b)):
    # plt.axvline(b[i], color='r', linestyle='--')
    # plt.axhline(b[i], color='r', linestyle='--')
    plt.show()

    # Transitive Binary Similarity Matrix: S_hat
    S_hat = S > S.mean() + S.std()
    S_hat_norm = np.matmul(S_hat, S_hat)
    while (S_hat_norm < S_hat).all():
        S_hat = S_hat_norm
        S_hat_norm = np.matmul(S_hat, S_hat)
        S_hat_norm = S_hat_norm >= 1

    """PLOTTING TRANSITIVE BINARY SIMILARITY MATRIX"""
    # Plot transitive binary similarity matrix S_hat
    plt.figure(figsize=(7, 7))
    plt.title('Segment transitive binary similarity matrix S_hat')
    plt.imshow(1 - S_hat_norm, cmap='gray')
    plt.show()

    # Image vs ground truth
    """PLOTTING S with LABELS"""
    S_frames = np.zeros_like(Q)
    for u in range(b.shape[0] - 1):
        for v in range(b.shape[0] - 1):
            S_frames[b[u]:b[u + 1], b[v]:b[v + 1]] = S_hat_norm[u, v]

    label_path = DEFAULT_LABELPATH + foldername
    file = "/" + os.path.basename(name) + ".txt"
    nums, lbls = ReadDataFromtxt(label_path, file)
    labels_array = np.asarray(nums)
    array = labels_array.astype(np.float)

    plt.figure(figsize=(7, 7))
    plt.title('Segment Similarity Matrix S with labels')
    plt.imshow(1 - S_frames, cmap='gray')
    for i in range(len(array)):
        plt.axvline(array[i] * sr / hop_length, color='b', linestyle='--')
        plt.axhline(array[i] * sr / hop_length, color='b', linestyle='--')
    plt.show()
    print()

    fig = plt.figure(figsize=(7, 7))
    plt.imshow(1 - S_frames, cmap='gray')
    for i in range(len(array)):
        plt.axvline(array[i] * sr / hop_length, color='b', linestyle='--')
        plt.axhline(array[i] * sr / hop_length, color='b', linestyle='--')
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    filename = filepath + "SSLMCRM/" + os.path.basename(name) + 'crm.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)  # dpi=400, transparent=True
    fig.clf()
    plt.close(fig)
    del ax, fig

    """PLOTTING NOVELTY CURVE"""
    # Plot novelty function with boundaries
    frames = range(len(c_norm))
    plt.figure(figsize=(10, 4))
    plt.title('Novelty function vector c (red lines are peaks and black lines are labels)')
    plt.xlabel('Frames')
    timeDifs = []
    dbltb = "\t\t"
    nspc = ""
    for i in range(len(array)):
        plt.axvline(array[i] * sr / hop_length, color='black', linestyle='-')
    for i in range(len(b)):
        plt.axvline(b[i], color='r', linestyle='--')
        timeSecondsDecimal = b[i] / sr * hop_length
        timeStr = str(datetime.timedelta(seconds=timeSecondsDecimal))
        gtTimeStr = 0
        timeDifference = 0
        if i < len(array):  #TODO: REMOVE why?
            gtTimeStr = str(datetime.timedelta(seconds=array[i]))  #TODO: REMOVE why?
            timeDifference = array[i] - timeSecondsDecimal
        print(f"Event: {timeStr}\t\t{dbltb if i == 0 else nspc}Ground Truth: {gtTimeStr}\t\t{dbltb if i == 0 else nspc}"
              f"Difference: "
              f"{'{:.6f}'.format(timeDifference) if timeDifference < 0 else '{: .6f}'.format(timeDifference)}\t\t"
              # f"G.T. Labels: {lbls[0]}")
              f"G.T. Labels: {lbls[i]}")
        timeDifs = np.append(timeDifs, abs(timeDifference))
    plt.plot(frames, c_norm)
    plt.show()
    print("\nAverage (absolute) time difference: ±" + str(np.average(timeDifs)))


def ReadNumbersFromLine(line):
    number = re.split(r'\s\s*', line)[0]
    number = float(number)
    return number


def ReadLabelsFromLine(line):
    labels = re.split(r'\s\s*', line)[1:]
    for i in range(len(labels)):
        labels[i] = labels[i].replace(',', '')
    return labels


def ReadImagesFromFolder(directory):
    imgs = []
    for (img_dir_path, img_dnames, img_fnames) in os.walk(directory):
        for f in img_fnames:
            img_path = img_dir_path + f
            img = plt.imread(img_path)
            img = resize(img, (200, 1150, 4))
            imgs.append(img)
    return imgs


def ReadDataFromtxt(directory, archive):
    numbers = []
    labels = []
    cnt = 1
    # for _ in listdir(directory):
    cnt += 1
    file = open(directory + archive, "r")
    form = next(file).strip()
    for line in file:
        numbers.append(ReadNumbersFromLine(line))
        labels.append(ReadLabelsFromLine(line.rstrip()))
    file.close()
    return numbers, labels, form


def ReadLabelSecondsPhrasesFromFolder(lblpath=DEFAULT_LABELPATH, stop=-1):
    nums = []
    lbls = []
    forms = []
    for (lbl_dir_path, lbl_dnames, lbl_fnames) in os.walk(lblpath):
        for f in lbl_fnames:
            if stop != -1:
                stop -= 1
                if stop == 0:
                    break
            # prepend_line(lbl_dir_path + '/' + f, lbl_dir_path.split('/')[-1])  # Run once for master label set
            numsIn, lblsIn, formsIn = ReadDataFromtxt(lbl_dir_path + '/', f)
            numsIn = np.array(numsIn, dtype=np.float32)
            nums.append(numsIn)
            lbls.append(lblsIn)
            forms.append([formsIn])

    # Convert Forms to One Hot encoding
    values = np.array(forms)  # print(values)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)  # print(integer_encoded)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)  # print(onehot_encoded)
    # inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])  # Return original label from encoding
    return nums, lbls, onehot_encoded


def prepend_line(file_name, line):
    """Insert string as a new line at the beginning of a file"""
    dummy_file = file_name + '.bak'
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        write_obj.write(line + '\n')
        for line in read_obj:
            write_obj.write(line)
    os.remove(file_name)
    os.rename(dummy_file, file_name)
    print("Finished prepending to " + file_name)
