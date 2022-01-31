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


def mel_spectrogram(sr_desired, name_song, window_size, hop_length):
    """This function calculates the mel spectrogram in dB with Librosa library"""
    y, sr = librosa.load(name_song, sr=None)
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


def util_main(feature, mode="cos"):
    img_path = ""

    if feature == "mfcc":
        if mode == "cos":
            img_path = os.path.join(TRAIN_DIR, 'SSLM_MFCC_COS/')
        elif mode == "euc":
            img_path = os.path.join(TRAIN_DIR, 'SSLM_MFCC_EUC/')
    elif feature == "chroma":
        if mode == "cos":
            img_path = os.path.join(TRAIN_DIR, 'SSLM_CRM_COS/')
        elif mode == "euc":
            img_path = os.path.join(TRAIN_DIR, 'SSLM_CRM_EUC/')
    elif feature == "mls":
        img_path = os.path.join(TRAIN_DIR, 'MLS/')

    if not os.path.exists(img_path):
        os.makedirs(img_path)

    num_songs = sum([len(files) for r, d, files in os.walk(MIDI_DIR)])
    i = 0
    for folder in gb.glob(os.path.join(MASTER_DIR, 'Data/MIDIs/*')):
        for file in os.listdir(folder):
            # foldername = folder.split('\\')[-1]
            name_song, name = file, file.split('/')[-1].split('.')[0]

            # if "x" not in name:
            #     continue
            # print(name)

            start_time_song = time.time()
            i += 1
            song_id = name_song[:-4]  # delete .ext characters from the string
            print("\tPreparing", song_id, "for processing...")
            if str(song_id) + ".npy" not in os.listdir(img_path):
                sslm_near = None

                if feature == "mfcc":
                    mel = mel_spectrogram(sr_desired, folder + '/' + name_song, window_size, hop_length)
                    if mode == "cos":
                        sslm_near = sslm_gen(mel, p, L_near, mode=mode, feature="mfcc")
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
                    stft = fourier_transform(sr_desired, folder + '/' + name_song, window_size, hop_length)
                    sslm_near = sslm_gen(stft, p, L_near, mode=mode, feature="chroma")[0]

                    if mode == "euc":
                        if sslm_near.shape[1] < max_pooling(stft, 6).shape[1]:
                            sslm_near = np.hstack((np.ones((301, 1)), sslm_near))
                        elif sslm_near.shape[1] > max_pooling(stft, 6).shape[1]:
                            sslm_near = sslm_near[:, 1:]
                elif feature == "mls":
                    mel = mel_spectrogram(sr_desired, folder + '/' + name_song, window_size, hop_length)
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
                # Save matrices and sslms as numpy arrays in separate paths
                np.save(img_path + song_id, sslm_near)

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
    def __init__(self, images_path, label_path=DEFAULT_LABELPATH, transforms=None, batch_size=32, end=-1):
        self.songs_list = []
        self.images_path = images_path
        self.images_list = []
        self.labels_path = label_path
        self.labels_list = []
        self.labels_sec_list = []
        self.labels_form_list = []
        self.batch_size = batch_size
        self.n = 0

        print("Building dataloader for " + self.images_path)
        cnt = 1
        for (im_dirpath, im_dirnames, im_filenames) in os.walk(self.images_path):
            for f in im_filenames:
                if f.endswith('.npy'):
                    if "variations_in_f_1793_(c)iscenk" in f or "dvoraktheme_and_variations_36_(c)yogore" \
                            in f or "Sonata_No_8_1st_Movement_K_310" in f:
                        continue  # TODO: remove (fix datafiles)
                    self.songs_list.append(os.path.splitext(f)[0])
                    # print("Reading file #" + str(cnt))
                    img_path = im_dirpath + f
                    image = np.load(img_path, allow_pickle=True)
                    if image.ndim == 1:
                        print("Erroneous file:", img_path, "Shape:", image.shape, image.ndim)
                    else:
                        image = resize(image, (500, 750))
                        # image = (image - image.mean()) / (image.std() + 1e-8)
                    self.images_list.append(image)
                    cnt += 1
                    if end != -1:
                        if cnt == end + 1:
                            break
            """ 
            for (lab_dirpath, lab_dirnames, lab_filenames) in os.walk(self.labels_path):  # labels files fo labels path
                for f in im_filenames:  # loop in each images png name files (songs_IDs)
                    if f[:-4] in lab_dirnames:  # if image name is annotated:
                        # images path
                        if f.endswith('.npy'):
                            img_path = im_dirpath + f
                            image = np.load(img_path)  # plt.imread if we want to open image
                            self.images_list.append(image)
                            print("appended image")
                        # ""
                        # labels path
                        path = os.path.join(lab_dirpath, f[:-4] + "/parsed/")
                        txt1 = "textfile1_functions.txt"
                        txt2 = "textfile2_functions.txt"
                        if os.path.isfile(path + txt1):
                            txt = "textfile1_functions.txt"
                        elif os.path.isfile(path + txt2):
                            txt = "textfile2_functions.txt"
                        # label_path = path + txt
                        # label = np.asarray(ReadDataFromtxt(path, txt), dtype=np.float32)
                        # labels_sec = np.asarray(ReadDataFromtxt(path, txt), dtype=np.float32)
                        # ""

                        lbls_seconds, lbls_phrases = du.ReadLabelSecondsPhrasesFromFolder()
                        self.labels_list.append(lbls_phrases)
                        self.labels_sec_list.append(lbls_seconds)
            """
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
        image = image[np.newaxis, :, :]
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


# TODO: Save all transformed data into one large npy file [mls, {sslm1, 2, 3, 4}, midi], [labelphrase, sec, form, name]
# Load MIDI Data
class BuildMIDIloader(k.utils.Sequence):
    def __init__(self, midi_path, label_path=DEFAULT_LABELPATH, transforms=None, batch_size=32, end=-1):
        self.songs_list = []
        self.midi_path = midi_path
        self.midi_list = pd.DataFrame()
        self.labels_path = label_path
        self.labels_list = []
        self.labels_sec_list = []
        self.labels_form_list = []
        self.batch_size = batch_size
        self.n = 0

        print("Building dataloader for " + self.midi_path)
        df = pd.DataFrame(columns=['spectral_contrast'])
        cnt = 1
        for (mid_dirpath, mid_dirnames, mid_filenames) in os.walk(self.midi_path):
            for f in mid_filenames:
                if f.endswith('.mid') or f.endswith('.midi'):
                    if "variations_in_f_1793_(c)iscenk" in f or \
                            "dvoraktheme_and_variations_36_(c)yogore" in f or "Sonata_No_8_1st_Movement_K_310" in f:
                        continue  # TODO: remove (fix datafiles)
                    self.songs_list.append(os.path.splitext(f)[0])
                    # print("Reading file #" + str(cnt))
                    mid_path = mid_dirpath + f
                    # print("Working on file: " + f)
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
                    df.loc[cnt-1] = [contrast]
                    cnt += 1
                    if end != -1:
                        if cnt == end:
                            break
        df = pd.DataFrame(df['spectral_contrast'].values.tolist())
        df = df.fillna(0)
        mean = np.mean(df, axis=0)
        std = np.std(df, axis=0)
        df = (df - mean) / std
        df = np.array(df)
        df = df[:, :, np.newaxis]
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
