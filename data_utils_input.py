import glob as gb
import librosa
import librosa.display
import numpy as np
import time
import skimage.measure
import os
import scipy
from scipy.spatial import distance
import pandas as pd
import tensorflow.keras as k
import data_utils as du
from skimage.transform import resize
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

start_time = time.time()


# region DataPreparation
def compute_ssm(X, metric="cosine"):
    """Computes the self-similarity matrix of X."""
    D = distance.pdist(X, metric=metric)
    D = distance.squareform(D)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if np.isnan(D[i, j]):
                D[i, j] = 0
    D /= D.max()
    return 1 - D


def mel_spectrogram(sr_desired, filepath, window_size, hop_length):
    """This function calculates the mel spectrogram in dB with Librosa library"""
    y, sr = librosa.load(filepath, sr=None)
    if sr != sr_desired:
        y = librosa.core.resample(y, sr, sr_desired)
        sr = sr_desired

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=window_size, hop_length=hop_length, n_mels=80, fmin=80,
                                       fmax=16000)
    S_to_dB = librosa.power_to_db(S, ref=np.max)  # convert S in dB
    return S_to_dB  # S_to_dB is the spectrogam in dB


def fourier_transform(sr_desired, name_song, window_size, hop_length):
    """This function calculates the mel spectrogram in dB with Librosa library"""
    y, sr = librosa.load(name_song, sr=None)
    if sr != sr_desired:
        y = librosa.core.resample(y, sr, sr_desired)
        sr = sr_desired
    stft = np.abs(librosa.stft(y=y, n_fft=window_size, hop_length=hop_length))
    return stft


def max_pooling(stft, pooling_factor):
    x_prime = skimage.measure.block_reduce(stft, (1, pooling_factor), np.max)
    return x_prime


def sslm_gen(spectrogram, pooling_factor, lag, mode, feature):
    padding_factor = lag
    """This part pads a mel spectrogram gived the spectrogram a lag parameter 
    to compare the first rows with the last ones and make the matrix circular"""
    pad = np.full((spectrogram.shape[0], padding_factor), -70)  # 80x30 frame matrix of -70dB corresponding to padding
    S_padded = np.concatenate((pad, spectrogram), axis=1)  # padding 30 frames with noise at -70dB at the beginning

    """This part max-poolend the spectrogram in time axis by a factor of p"""
    x_prime = max_pooling(S_padded, pooling_factor)
    x = []
    if feature == "mfcc":

        """This part calculates a circular Self Similarity Lag Matrix given
        the mel spectrogram padded and max-pooled"""
        # MFCCs calculation from DCT-Type II
        MFCCs = scipy.fftpack.dct(x_prime, axis=0, type=2, norm='ortho')
        MFCCs = MFCCs[1:, :]  # 0 componen ommited

        # Bagging frames
        m = 2  # baggin parameter in frames
        x = [np.roll(MFCCs, n, axis=1) for n in range(m)]
    elif feature == "chroma":
        """This part calculates a circular Self Similarity Lag Matrix given
                the chromagram padded and max-pooled"""
        PCPs = librosa.feature.chroma_stft(S=x_prime, sr=sr_desired, n_fft=window_size, hop_length=hop_length)
        PCPs = PCPs[1:, :]

        # Bagging frames
        m = 2  # Bagging parameter in frames
        x = [np.roll(PCPs, n, axis=1) for n in range(m)]

    x_hat = np.concatenate(x, axis=0)

    # Cosine distance calculation: D[N/p,L/p] matrix
    distances = np.zeros((x_hat.shape[1], padding_factor // p))  # D has as dimensions N/p and L/p
    for i in range(x_hat.shape[1]):  # iteration in columns of x_hat
        for l in range(padding_factor // p):
            if i - (l + 1) < 0:
                cur_dist = 1
            elif i - (l + 1) < padding_factor // p:
                cur_dist = 1
            else:
                cur_dist = 0
                if mode == "cos":
                    cur_dist = distance.cosine(x_hat[:, i],
                                               x_hat[:, i - (l + 1)])  # cosine distance between columns i and i-L
                elif mode == "euc":
                    cur_dist = distance.euclidean(x_hat[:, i],
                                                  x_hat[:, i - (l + 1)])  # euclidian distance between columns i and i-L
                if cur_dist == float('nan'):
                    cur_dist = 0
            distances[i, l] = cur_dist

    # Threshold epsilon[N/p,L/p] calculation
    kappa = 0.1
    epsilon = np.zeros((distances.shape[0], padding_factor // p))  # D has as dimensions N/p and L/p
    for i in range(padding_factor // p, distances.shape[0]):  # iteration in columns of x_hat
        for l in range(padding_factor // p):
            epsilon[i, l] = np.quantile(np.concatenate((distances[i - l, :], distances[i, :])), kappa)

    # We remove the padding done before
    distances = distances[padding_factor // p:, :]
    epsilon = epsilon[padding_factor // p:, :]
    x_prime = x_prime[:, padding_factor // p:]

    # Self Similarity Lag Matrix
    sslm = scipy.special.expit(1 - distances / epsilon)  # aplicación de la sigmoide
    sslm = np.transpose(sslm)
    sslm = skimage.measure.block_reduce(sslm, (1, 3), np.max)

    # Check if SSLM has nans and if it has them, substitute them by 0
    for i in range(sslm.shape[0]):
        for j in range(sslm.shape[1]):
            if np.isnan(sslm[i, j]):
                sslm[i, j] = 0

    # if mode == "euc":
    #     return sslm, x_prime

    # return sslm
    return sslm, x_prime


def ssm_gen(spectrogram, pooling_factor):
    """This part max-poolend the spectrogram in time axis by a factor of p"""
    x_prime = max_pooling(spectrogram, pooling_factor)

    """This part calculates a circular Self Similarity Matrix given
    the mel spectrogram padded and max-pooled"""
    # MFCCs calculation from DCT-Type II
    MFCCs = scipy.fftpack.dct(x_prime, axis=0, type=2, norm='ortho')
    MFCCs = MFCCs[1:, :]  # 0 componen ommited

    # Bagging frames
    m = 2  # baggin parameter in frames
    x = [np.roll(MFCCs, n, axis=1) for n in range(m)]
    x_hat = np.concatenate(x, axis=0)
    x_hat = np.transpose(x_hat)

    ssm = compute_ssm(x_hat)

    # Check if SSLM has nans and if it has them, substitute them by 0
    for i in range(ssm.shape[0]):
        for j in range(ssm.shape[1]):
            if np.isnan(ssm[i, j]):
                ssm[i, j] = 0

    return ssm
# endregion


window_size = 2048  # (samples/frame)
hop_length = 1024  # overlap 50% (samples/frame)
sr_desired = 44100
p = 2  # pooling factor
p2 = 3  # 2pool3
L_sec_near = 14  # lag near context in seconds
L_near = round(L_sec_near * sr_desired / hop_length)  # conversion of lag L seconds to frames

MASTER_DIR = 'D:/Google Drive/Resources/Dev Stuff/Python/Machine Learning/Master Thesis/'
DEFAULT_LABELPATH = os.path.join(MASTER_DIR, 'Labels/')
TRAIN_DIR = 'F:/Master Thesis Input/NewTrain/'
MIDI_DIR = os.path.join(MASTER_DIR, 'Data/MIDIs/')


def util_main_helper(feature, filepath, mode="cos", predict=False, savename=""):
    sslm_near = None
    if feature == "mfcc":
        mel = mel_spectrogram(sr_desired, filepath, window_size, hop_length)
        if mode == "cos":
            sslm_near = sslm_gen(mel, p, L_near, mode=mode, feature="mfcc")[0]
            # mls = max_pooling(mel, p2)
            # Save mels matrices and sslms as numpy arrays in separate paths
            # np.save(im_path_mel_near + song_id, mls)
        elif mode == "euc":
            sslm_near = sslm_gen(mel, p, L_near, mode=mode, feature="mfcc")[0]
            if sslm_near.shape[1] < max_pooling(mel, 6).shape[1]:
                sslm_near = np.hstack((np.ones((301, 1)), sslm_near))
            elif sslm_near.shape[1] > max_pooling(mel, 6).shape[1]:
                sslm_near = sslm_near[:, 1:]
    elif feature == "chroma":
        stft = fourier_transform(sr_desired, filepath, window_size, hop_length)
        sslm_near = sslm_gen(stft, p, L_near, mode=mode, feature="chroma")[0]
        if mode == "euc":
            if sslm_near.shape[1] < max_pooling(stft, 6).shape[1]:
                sslm_near = np.hstack((np.ones((301, 1)), sslm_near))
            elif sslm_near.shape[1] > max_pooling(stft, 6).shape[1]:
                sslm_near = sslm_near[:, 1:]
    elif feature == "mls":
        mel = mel_spectrogram(sr_desired, filepath, window_size, hop_length)
        sslm_near = ssm_gen(mel, pooling_factor=6)

    """
    # UNCOMMENT TO DISPLAY FEATURE GRAPHS
    # recurrence = librosa.segment.recurrence_matrix(sslm_near, mode='affinity', k=sslm_near.shape[1])
    plt.figure(figsize=(15, 10))
    if feature == "mls":
        plt.title("Mel Log-scaled Spectrogram - Self-Similarity matrix (MLS SSM)")
        plt.imshow(sslm_near, origin='lower', cmap='plasma', aspect=0.8)  # switch to recurrence if desired
    else:
        plt_title = "Self-Similarity Lag Matrix (SSLM): "
        if feature == "chroma":
            plt_title += "Chromas, "
        else:
            plt_title += "MFCCs, "
        if mode == "cos":
            plt_title += "Cosine Distance"
        else:
            plt_title += "Euclidian Distance"
        plt.title(plt_title)
        plt.imshow(sslm_near.astype(np.float32), origin='lower', cmap='viridis', aspect=0.8)  
        # switch to recurrence if desired
    plt.show()
    """
    if not predict:
        # Save matrices and sslms as numpy arrays in separate paths
        np.save(filepath, sslm_near)
    else:
        return sslm_near


def util_main(feature, mode="cos", predict=False, inpath=TRAIN_DIR, midpath=MIDI_DIR):
    img_path = ""

    if feature == "mfcc":
        if mode == "cos":
            img_path = os.path.join(inpath, 'SSLM_MFCC_COS/')
        elif mode == "euc":
            img_path = os.path.join(inpath, 'SSLM_MFCC_EUC/')
    elif feature == "chroma":
        if mode == "cos":
            img_path = os.path.join(inpath, 'SSLM_CRM_COS/')
        elif mode == "euc":
            img_path = os.path.join(inpath, 'SSLM_CRM_EUC/')
    elif feature == "mls":
        img_path = os.path.join(inpath, 'MLS/')

    if not os.path.exists(img_path):
        os.makedirs(img_path)

    num_songs = sum([len(files) for r, d, files in os.walk(midpath)])
    i = 0
    for folder in gb.glob(midpath + "*"):
        for file in os.listdir(folder):
            # foldername = folder.split('\\')[-1]
            name_song, name = file, file.split('/')[-1].split('.')[0]
            start_time_song = time.time()
            i += 1
            song_id = name_song[:-4]  # delete .ext characters from the string
            print("\tPreparing", song_id, "for processing...")
            if str(song_id) + ".npy" not in os.listdir(img_path):
                util_main_helper(feature, folder + '/' + name_song, mode, predict, savename=img_path + song_id)
                print("\t\tFinished", i, "/", num_songs, "- Duration: {:.2f}s".format(time.time() - start_time_song))
            else:
                print("\t\tAlready completed. Skipping...\n\t\tFinished", i, "/", num_songs)
            # return
    print("All files have been converted. Duration: {:.2f}s".format(time.time() - start_time))


def validate_folder_contents(labels, midis, mlsdir, sslm1, sslm2, sslm3, sslm4):
    """Ensure all folders contain files of the same name"""
    labelfiles = os.listdir(labels)
    midifiles = os.listdir(midis)
    mlsfiles = os.listdir(mlsdir)
    sslm1files = os.listdir(sslm1)
    sslm2files = os.listdir(sslm2)
    sslm3files = os.listdir(sslm3)
    sslm4files = os.listdir(sslm4)

    for i in range(len(labelfiles)):
        c_lbl = os.path.splitext(labelfiles[i])[0]
        c_midi = os.path.splitext(midifiles[i])[0]
        c_mls = os.path.splitext(mlsfiles[i])[0]
        c_sslm1 = os.path.splitext(sslm1files[i])[0]
        c_sslm2 = os.path.splitext(sslm2files[i])[0]
        c_sslm3 = os.path.splitext(sslm3files[i])[0]
        c_sslm4 = os.path.splitext(sslm4files[i])[0]

        if c_lbl != c_midi or c_lbl != c_mls or\
                c_lbl != c_sslm1 or c_lbl != c_sslm2 or c_lbl != c_sslm3 or c_lbl != c_sslm4:
            err = FileNotFoundError("File discrepency at index " + str(i))
            print("Current labels: ")
            print(f"Label: {c_lbl}\nMIDI: {c_midi}\nMLS: {c_mls}\nSSLM-CRM-COS: {c_sslm1}"
                  f"\nSSLM-CRM-EUC: {c_sslm2}\nSSLM-MFCC-COS: {c_sslm3}\nSSLM-MFCC-EUC: {c_sslm4}")
            raise err

    if len(labelfiles) != len(midifiles) or len(labelfiles) != len(mlsfiles) or \
            len(labelfiles) != len(sslm1files) or len(labelfiles) != len(sslm2files) or\
            len(labelfiles) != len(sslm3files) or len(labelfiles) != len(sslm4files):
        raise ValueError("Not all directories contain the same number of files")


# region Transformations
def gaussian(x, mu, sig):
    """Create array of labels"""
    return np.exp(-np.power((x - mu) / sig, 2.) / 2)


def borders(image, label, labels_sec, label_form):
    """This function transforms labels in sc to gaussians in frames"""
    pooling_factor = 6
    num_frames = image.shape[2]
    repeated_label = []
    for i in range(len(labels_sec) - 1):
        if labels_sec[i] == labels_sec[i + 1]:
            repeated_label.append(i)
    labels_sec = np.delete(labels_sec, repeated_label, 0)  # labels in seconds
    labels_sec = labels_sec / pooling_factor  # labels in frames

    # Pad frames we padded in images also in labels but in seconds
    sr = sr_desired
    padding_factor = 50
    label_padded = [labels_sec[i] + padding_factor * hop_length / sr for i in range(labels_sec.shape[0])]
    vector = np.arange(num_frames)
    new_vector = (vector * hop_length + window_size / 2) / sr
    sigma = 0.1
    gauss_array = []
    for mu in (label_padded[1:]):  # Ignore first label (beginning of song) due to insignificance (0.000 Silence)
        gauss_array = np.append(gauss_array, gaussian(new_vector, mu, sigma))
    for i in range(len(gauss_array)):
        if gauss_array[i] > 1:
            gauss_array[i] = 1
    return image, label[1:], gauss_array, label_form


def padding_MLS(image, label, labels_sec, label_form):
    """This function pads 30frames at the begining and end of an image"""
    sr = sr_desired
    padding_factor = 50

    def voss(nrows, ncols=16):
        """Generates pink noise using the Voss-McCartney algorithm.

        nrows: number of values to generate
        rcols: number of random sources to add

        returns: NumPy array
        """
        array = np.empty((nrows, ncols))
        array.fill(np.nan)
        array[0, :] = np.random.random(ncols)
        array[:, 0] = np.random.random(nrows)

        # the total number of changes is nrows
        n = nrows
        cols = np.random.geometric(0.5, n)
        cols[cols >= ncols] = 0
        rows = np.random.randint(nrows, size=n)
        array[rows, cols] = np.random.random(n)

        df = pd.DataFrame(array)
        df.fillna(method='ffill', axis=0, inplace=True)
        total = df.sum(axis=1)

        return total.values

    n_mels = image.shape[1]  # Default(80) - fit padding to image height
    y = voss(padding_factor * hop_length - 1)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=window_size, hop_length=hop_length,
                                       n_mels=n_mels, fmin=80, fmax=16000)
    S_to_dB = librosa.power_to_db(S, ref=np.max)
    pad_image = S_to_dB[np.newaxis, :, :]

    # Pad MLS
    S_padded = np.concatenate((pad_image, image), axis=-1)
    S_padded = np.concatenate((S_padded, pad_image), axis=-1)
    return S_padded, label, labels_sec, label_form


def padding_SSLM(image, label, labels_sec, label_form):
    """This function pads 30 frames at the begining and end of an image"""
    padding_factor = 50

    # Pad SSLM
    pad_image = np.full((image.shape[1], padding_factor), 1)
    pad_image = pad_image[np.newaxis, :, :]
    S_padded = np.concatenate((pad_image, image), axis=-1)
    S_padded = np.concatenate((S_padded, pad_image), axis=-1)
    return S_padded, label, labels_sec, label_form


def normalize_image(image, label, labels_sec, label_form):
    """This function normalizes an image"""
    image = np.squeeze(image)  # remove

    def normalize(array):
        """This function normalizes a matrix along x axis (frequency)"""
        normalized = np.zeros((array.shape[0], array.shape[1]))
        for i in range(array.shape[0]):
            normalized[i, :] = (array[i, :] - np.mean(array[i, :])) / np.std(array[i, :])
        return normalized

    image = normalize(image)
    # image = (image-np.min(image))/(np.max(image)-np.min(image))
    image = np.expand_dims(image, axis=0)
    return image, label, labels_sec, label_form
# endregion


# Load MLS and SSLM Data
class BuildDataloader(k.utils.Sequence):
    def __init__(self, images_path, label_path=DEFAULT_LABELPATH, transforms=None, batch_size=32, end=-1, reshape=True):
        self.songs_list = []
        self.images_path = images_path
        self.images_list = []
        self.labels_path = label_path
        self.labels_list = []
        self.labels_sec_list = []
        self.labels_form_list = []
        self.batch_size = batch_size
        self.n = 0
        self.reshape = reshape

        print("Building dataloader for " + self.images_path)
        cnt = 1
        for (im_dirpath, im_dirnames, im_filenames) in os.walk(self.images_path):
            for f in im_filenames:
                if f.endswith('.npy'):
                    self.songs_list.append(os.path.splitext(f)[0])
                    # print("Reading file #" + str(cnt))
                    img_path = im_dirpath + f
                    image = np.load(img_path, allow_pickle=True)
                    if image.ndim == 1:
                        raise ValueError("Erroneous file:", img_path, "Shape:", image.shape, image.ndim)
                    else:
                        # image = resize(image, (300, 500))
                        # image = (image - image.mean()) / (image.std() + 1e-8)
                        if reshape:
                            image = np.mean(image, axis=0)
                        else:
                            image1 = np.mean(image, axis=0)
                            image2 = np.var(image, axis=0)
                            image = np.array([image1, image2])
                    self.images_list.append(image)
                    cnt += 1
                    if end != -1:
                        if cnt == end + 1:
                            break
        lbls_seconds, lbls_phrases, lbl_forms = du.ReadLabelSecondsPhrasesFromFolder(lblpath=self.labels_path, stop=cnt)
        self.labels_list = lbls_phrases
        self.labels_sec_list = lbls_seconds
        self.labels_form_list = lbl_forms
        self.transforms = transforms
        self.max = self.__len__()

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        # print("LEN: " + str(self.max) + " TRU LEN: " + str(len(self.images_list)) + " INDX: " + str(index))
        image = self.images_list[index]
        # print(image.shape, image.ndim)
        # print(image)
        # if image.ndim == 1:
        #     print(image)
        if self.reshape:
            image = image[np.newaxis, :, np.newaxis]
        labels = self.labels_list[index]
        # print("Labels: ", str(len(labels)), "Images: ", str(len(image)), image.shape)
        labels_sec = self.labels_sec_list[index]
        labels_form = self.labels_form_list[index]
        song_name = self.songs_list[index]
        if self.transforms is not None:
            for t in self.transforms:
                image, labels, labels_sec, labels_form = t(image, labels, labels_sec, labels_form)
        return image, [labels, labels_sec, labels_form, song_name]

    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result

    def getNumClasses(self):
        return len(self.labels_form_list[1])

    def getLabels(self):
        return self.labels_form_list

    def getImages(self):
        return self.images_list

    def getCurrentIndex(self):
        return self.n

    def getSong(self, index):
        return self.songs_list[index]

    def getFormLabel(self, index):
        return self.labels_form_list[index]

    def getDuration(self, index):
        return self.labels_sec_list[index][-1]


def get_midi_dataframe(building_df=False):
    df = pd.DataFrame(columns=['spectral_contrast_mean', 'spectral_contrast_var'])
    if building_df:
        df2 = pd.DataFrame(columns=['chroma_stft_mean', 'chroma_stft_var',
                                    'chroma_cqt_mean', 'chroma_cqt_var',
                                    'chroma_cens_mean', 'chroma_cens_var',
                                    'mel_mean', 'mel_var',
                                    'mfcc_mean', 'mfcc_var',
                                    'spectral_bandwidth_mean', 'spectral_bandwidth_var',
                                    'spectral_centroid_mean', 'spectral_centroid_var',
                                    'spectral_flatness_mean', 'spectral_flatness_var',
                                    'spectral_rolloff_mean', 'spectral_rolloff_var',
                                    'poly_features_mean', 'poly_features_var',
                                    'tonnetz_mean', 'tonnetz_var',
                                    'zero_crossing_mean', 'zero_crossing_var',
                                    'tempogram_mean', 'tempogram_var',
                                    'fourier_tempo_mean', 'fourier_tempo_var'])
        df = pd.concat([df, df2], axis=1)
    return df


def get_audio_features(df, cnt, mid_path, building_df=False):
    X, sample_rate = librosa.load(mid_path, res_type='kaiser_fast', duration=3, sr=44100, offset=0.5)
    contrast = librosa.feature.spectral_contrast(y=X, sr=sample_rate)
    """ Plot spectral contrast
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(contrast, cmap='plasma', x_axis='time')
    plt.colorbar()
    plt.ylabel('Frequency bands')
    plt.title('Spectral contrast')
    plt.tight_layout()
    plt.show()
    """
    contrast = np.mean(contrast, axis=0)
    contrast2 = np.var(contrast, axis=0)
    if building_df:
        chroma_cens = librosa.feature.chroma_cens(y=X, sr=sample_rate)
        chroma_cqt = librosa.feature.chroma_cqt(y=X, sr=sample_rate)
        chroma_stft = librosa.feature.chroma_stft(y=X, sr=sample_rate)
        mel_spec = librosa.feature.melspectrogram(y=X, sr=sample_rate)
        mfcc_spec = librosa.feature.mfcc(y=X, sr=sample_rate)
        spec_bdwth = librosa.feature.spectral_bandwidth(y=X, sr=sample_rate)
        spec_centrd = librosa.feature.spectral_centroid(y=X, sr=sample_rate)
        spec_flatns = librosa.feature.spectral_flatness(y=X)
        spec_rolloff = librosa.feature.spectral_rolloff(y=X, sr=sample_rate)
        poly_feat = librosa.feature.poly_features(y=X, sr=sample_rate)
        tonnetz = librosa.feature.tonnetz(y=X, sr=sample_rate)
        zero_cross = librosa.feature.zero_crossing_rate(y=X)
        tempogram = librosa.feature.tempogram(y=X, sr=sample_rate)
        fouriertemp = librosa.feature.fourier_tempogram(y=X, sr=sample_rate)  # Not used in model, repurpose for others?

        df.loc[cnt] = [contrast, contrast2,  # 0, 1
                       np.mean(chroma_cens, axis=0), np.var(chroma_cens, axis=0),  # 2, 3
                       np.mean(chroma_cqt, axis=0), np.var(chroma_cqt, axis=0),  # 4, 5
                       np.mean(chroma_stft, axis=0), np.var(chroma_stft, axis=0),  # 6, 7
                       np.mean(mel_spec, axis=0), np.var(mel_spec, axis=0),  # 8, 9
                       np.mean(mfcc_spec, axis=0), np.var(mfcc_spec, axis=0),  # 10, 11
                       np.mean(spec_bdwth, axis=0), np.var(spec_bdwth, axis=0),  # 12, 13
                       np.mean(spec_centrd, axis=0), np.var(spec_centrd, axis=0),  # 14, 15
                       np.mean(spec_flatns, axis=0), np.var(spec_flatns, axis=0),  # 16, 17
                       np.mean(spec_rolloff, axis=0), np.var(spec_rolloff, axis=0),  # 18, 19
                       np.mean(poly_feat, axis=0), np.var(poly_feat, axis=0),  # 20, 21
                       np.mean(tonnetz, axis=0), np.var(tonnetz, axis=0),  # 22, 23
                       np.mean(zero_cross, axis=0), np.var(zero_cross, axis=0),  # 24, 25
                       np.mean(tempogram, axis=0), np.var(tempogram, axis=0),  # 26, 27
                       np.mean(fouriertemp, axis=0), np.var(fouriertemp, axis=0)]  # 28, 29
    else:
        df.loc[cnt] = [contrast, contrast2]
    return df


# Load MIDI Data
class BuildMIDIloader(k.utils.Sequence):
    def __init__(self, midi_path, label_path=DEFAULT_LABELPATH,
                 transforms=None, batch_size=32, end=-1, reshape=True, building_df=False):
        self.songs_list = []
        self.midi_path = midi_path
        self.midi_list = pd.DataFrame()
        self.labels_path = label_path
        self.labels_list = []
        self.labels_sec_list = []
        self.labels_form_list = []
        self.batch_size = batch_size
        self.n = 0
        self.reshape = reshape

        print("Building dataloader for " + self.midi_path)
        df = get_midi_dataframe(building_df)
        cnt = 1
        audio_extensions = ["3gp", "aa", "aac", "aax", "act", "aiff", "alac", "amr", "ape", "au", "awb", "dct",
                            "dss", "dvf", "flac", "gsm", "iklax", "ivs", "m4a", "m4b", "m4p", "mmf", "mp3", "mpc",
                            "msv", "nmf", "ogg", "oga", "mogg", "opus", "ra", "rm", "raw", "rf64", "sln", "tta",
                            "voc", "vox", "wav", "wma", "wv", "webm", "8svx", "cda", "mid", "midi", "MID", "mp4"]
        for (mid_dirpath, mid_dirnames, mid_filenames) in os.walk(self.midi_path):
            for f in mid_filenames:
                if f.endswith(tuple(audio_extensions)):
                    self.songs_list.append(os.path.splitext(f)[0])
                    print("Reading file #" + str(cnt))
                    mid_path = mid_dirpath + f
                    # print("Working on file: " + f)
                    df = get_audio_features(df, cnt-1, mid_path, building_df)
                    cnt += 1
                    if end != -1:
                        if cnt == end:
                            break
        # df = pd.DataFrame(df['spectral_contrast'].values.tolist())
        print(cnt)
        df = df.fillna(0)
        if reshape:
            mean = np.mean(df, axis=0)
            std = np.std(df, axis=0)
            df = (df - mean) / std
            df = np.array(df)
            df = df[:, :, np.newaxis]
        else:
            df = np.array(df)
        self.midi_list = df
        lbls_seconds, lbls_phrases, lbl_forms = du.ReadLabelSecondsPhrasesFromFolder(lblpath=self.labels_path, stop=cnt)
        self.labels_list = lbls_phrases
        self.labels_sec_list = lbls_seconds
        self.labels_form_list = lbl_forms
        self.transforms = transforms
        self.max = self.__len__()

    def __len__(self):
        return self.midi_list.shape[0]

    def __getitem__(self, index):
        mid = self.midi_list[index]
        labels = self.labels_list[index]
        labels_sec = self.labels_sec_list[index]
        labels_form = self.labels_form_list[index]
        song_name = self.songs_list[index]
        if self.transforms is not None:
            for t in self.transforms:
                mid, labels, labels_sec, labels_form = t(mid, labels, labels_sec, labels_form)
        return mid, [labels, labels_sec, labels_form, song_name]

    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result
