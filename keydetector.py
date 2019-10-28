#!/usr/bin/env python
import os
import sys
import warnings
import json
from operator import itemgetter

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.signal import butter, lfilter, freqz
import seaborn
# removing annoying scikit userwarnings
warnings.filterwarnings("ignore")

# from keydetection.processing import *
window_length = 4096
hop_length    = window_length / 2

def read_audio(path):
    """Reads the audio file.

    Currently only 16bit .wav files.

    :param path: path to the audio file
    :returns: numpy ndarray, samplerate
    """
    sr, y = sp.io.wavfile.read(path)
    y = y.T
    y = y / float(2 ** 15)

    return y, sr


def to_mono(stereo_array):
    """Calls librosa.to_mona on the given audio file.

    :param stereo_array: input stereo array
    :returns: mono audio array (numpy ndarray)
    """

    return librosa.to_mono(stereo_array)


def harmonic_component(mono_array, samplerate, plot=False):
    """Returns the harmonic component as spectrogram (HPSS).

    :param mono_array: input mono array
    :param samplerate: samplerate of the original input
    :param plot: plot the harmonic spectrogram
    :returns: harmonic component as spectrogram
    """

    #
    # hpss
    #
    y_harmonic, y_percussive = librosa.effects.hpss(mono_array)

    if plot:
        # What do the spectrograms look like?
        # Let's make and display a mel-scaled power (energy-squared) spectrogram
        S_harmonic = librosa.feature.melspectrogram(y_harmonic, sr=samplerate)

        # Convert to log scale (dB). We'll use the peak power as reference.
        log_Sh = librosa.logamplitude(S_harmonic, ref_power=np.max)

        # Make a new figure
        plt.figure(figsize=(12, 6))

        # Display the spectrogram on a mel scale
        librosa.display.specshow(log_Sh, sr=samplerate, y_axis='mel')

        # Put a descriptive title on the plot
        plt.title('mel power spectrogram (Harmonic)')

        # draw a color bar
        plt.colorbar(format='%+02.0f dB')

        # Make the figure layout compact
        plt.tight_layout()

        plt.show()

    return y_harmonic


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def chroma_features(spectrogram, samplerate, plot=False):
    """Returns the chroma representation of the input spectrogram.

    :param spectrogram: spectrogram to process
    :param samplerate: samplerate of the original input
    :param plot: plot the chroma features
    :returns: chroma features in 'spectrogram' form
    """

    # chroma features
    # We'll use a CQT-based chromagram here.  An STFT-based implementation also exists in chroma_cqt()
    # We'll use the harmonic component to avoid pollution from transients
    C = librosa.feature.chroma_cqt(y=spectrogram, sr=samplerate)

    if plot:
        # Make a new figure
        plt.figure(figsize=(12, 4))

        # Display the chromagram: the energy in each chromatic pitch class as a function of time
        # To make sure that the colors span the full range of chroma values, set vmin and vmax
        librosa.display.specshow(C, sr=samplerate, x_axis='time', y_axis='chroma', vmin=0, vmax=1)

        plt.title('Chromagram')
        plt.colorbar()

        plt.tight_layout()

        plt.show()

    return C


def horizontal_median_filter(spectrogram, samplerate, length=100, plot=False):
    """Applies a median filter to the rows of the spectrogram.

    This smoothes artifacts in time.

    :param spectrogram: spectrogram to smooth
    :param samplerate: samplerate of the original input
    :param length: length of the median filter
    :param plot: plot the smoothed spectrogram
    :returns: smoothed spectrogram
    """

    filtered_spectrogram = []
    # median filter over chromagram
    for row in spectrogram:
        filtered_spectrogram.append(sp.ndimage.filters.median_filter(row, 100, mode='constant'))

    if plot:
        # What do the spectrograms look like?
        # Let's make and display a mel-scaled power (energy-squared) spectrogram
        S_harmonic = librosa.feature.melspectrogram(filtered_spectrogram, sr=samplerate)

        # Convert to log scale (dB). We'll use the peak power as reference.
        log_Sh = librosa.logamplitude(S_harmonic, ref_power=np.max)

        # Make a new figure
        plt.figure(figsize=(12, 6))

        # Display the spectrogram on a mel scale
        librosa.display.specshow(log_Sh, sr=samplerate, y_axis='mel')

        # Put a descriptive title on the plot
        plt.title('mel power spectrogram (Harmonic)')

        # draw a color bar
        plt.colorbar(format='%+02.0f dB')

        # Make the figure layout compact
        plt.tight_layout()

        plt.show()

    return filtered_spectrogram


def create_all_scales_matrix():
    """Returns matrix for all major scales."""

    mat = [[0.0] * 12 for i in range(12)]
    for i in range(12):
        mat[i][i] = 5
        mat[i][(i + 2) % 12] = 1
        mat[i][(i + 4) % 12] = 2
        mat[i][(i + 5) % 12] = 1
        mat[i][(i + 7) % 12] = 2
        mat[i][(i + 9) % 12] = 2
        mat[i][(i + 11) % 12] = 0

    return mat


def create_all_triads_matrix():
    """Returns matrix for all triads (major and minor)."""

    mat = [[0.0] * 12 for i in range(24)]
    for i in range(12):
        mat[i][i] = 1
        mat[i][(i + 4) % 12] = 1
        mat[i][(i + 7) % 12] = 1

        mat[i + 12][i] = 1
        mat[i + 12][(i + 3) % 12] = 1
        mat[i + 12][(i + 7) % 12] = 1

    return mat


def create_all_quints_matrix():
    """Returns matrix for all twelve quints."""

    mat = [[0.0] * 12 for i in range(12)]
    for i in range(12):
        mat[i][i] = 3
        mat[i][(i + 7) % 12] = 1

    return mat


def mult_chord_table(chromagram, table, samplerate, plot=False):
    """Multiplies a chromagram with the given table matrix.

    The resulting matrix contains as many time columns as the
    input chromagram. The rows represent the 'chords' of the input table.
    The entries of the resulting matrix are the weights.

    Also normalizes the weights.

    :returns: result of matrix matrix multiplication
    """

    weights = np.dot(table, chromagram)
    weights = librosa.util.normalize(weights, axis=1)

    if plot:
        # Make a new figure
        plt.figure(figsize=(12, 4))

        # Display the chromagram: the energy in each chromatic pitch class as a function of time
        # To make sure that the colors span the full range of chroma values, set vmin and vmax
        librosa.display.specshow(weights, sr=samplerate, x_axis='time', y_axis='chroma', vmin=0, vmax=1)

        plt.title('Weights')
        plt.colorbar()

        plt.tight_layout()

        plt.show()

    return weights


def sum_weights(weights, plot=False):
    """Sums up the weigths and calculates probability per weight.

    :returns: list with probabilities
    """

    s = np.sum(weights, axis=1)

    list_sum = np.sum(s)

    s = s / list_sum

    return s


def col_max_bit_map(matrix, samplerate, plot=False):
    """Returns a mask with the per column max value set to 1 and the rest to 0."""

    max_idx = matrix.argmax(0)

    mask = np.zeros(matrix.shape)

    for idx, col in enumerate(mask.T):
        col[max_idx[idx]] = 1

    if plot:
        # Make a new figure
        plt.figure(figsize=(12, 4))

        # Display the chromagram: the energy in each chromatic pitch class as a function of time
        # To make sure that the colors span the full range of chroma values, set vmin and vmax
        librosa.display.specshow(mask, sr=samplerate, x_axis='time', y_axis='chroma', vmin=0, vmax=1)

        plt.title('Weight mask')
        plt.colorbar()

        plt.tight_layout()

        plt.show()

    return mask


def sum_bit_mask(mask, plot=False):
    """Sums the values of each row. Returns them as list."""

    sums = []

    for row in mask:
        sums.append(sum(row))

    total_sum = sum(sums)

    if plot:
        for idx, s in enumerate(sums):
            print("{0}% for row {1}".format(round(float(s) / float(total_sum), 3) * 100, idx))

    return sums


def sum_for_four_chords(sums, plot=False):
    """Uses four chord heuristic to determine scale."""

    scale_sums = []

    for i in range(12):
        scale_sum = 4.0 * sums[(i + 0) % 12]  # tonic
        scale_sum += 2.0 * sums[(i + 5) % 12]  # sub
        scale_sum += 2.0 * sums[(i + 7) % 12]  # dom
        scale_sum += sums[(i + 9) % 12]  # minor
        scale_sums.append(scale_sum)

    total_sum = sum(scale_sums)

    labels = ['C',
              'C#/Db',
              'D',
              'D#/Eb',
              'E',
              'F',
              'F#/Gb',
              'G',
              'G#/Ab',
              'A',
              'A#/Bb',
              'B',
              ]

    result = dict(zip(labels, scale_sums))
    result = sorted(result.items(), key=itemgetter(1), reverse=True)

    if plot:
        for key, value in result:
            print("{0}% for key {1}".format(round(float(value) / float(total_sum) * 100, 2) , key))

    return result

def judge_tempo(file):
    y, sr = librosa.load(file)
    hop_length = 128
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    return tempo
def main():
    

    input = "AudioIn"           #改为自选
    reference = ""
    verbose = False
    input_files = []
    with open('keys.json', 'r') as f:
        keys = json.loads(f.read())
        print(keys)

        f.close()
    if os.path.isdir(input):
        input_files = [os.path.join(input, f) for f in os.listdir(input) if f.endswith('.wav')]
    elif os.path.isfile(input) and input.endswith('.wav'):
        input_files = [input]

    # if reference:
    #     with open(reference, 'r') as f:
    #         lines = f.readlines()
    #     file_scale_pairs = dict([tuple(line.split()) for line in lines])


    if not input_files:
        print('No input files. Exiting.')
        sys.exit(0)

    for f in input_files:
        print('Detection for: {0}...'.format(f))

        if verbose:
            print('Reading audio...')
        y, sr = read_audio(f)

        if verbose:
            print('Converting to mono...')
        y = to_mono(y)

        if verbose:
            print('HPSS...')
        y = harmonic_component(y, sr)

        # if verbose:
        #     print 'LPF...'
        # # apply lowpass filter to harmonic spectrogram
        # # Filter requirements.
        # order = 6
        # cutoff = 200  # desired cutoff frequency of the filter, Hz
        # y = butter_lowpass_filter(y, cutoff, sr, order)

        if verbose:
            print('Chroma features...')
        y = chroma_features(y, sr)

        # if verbose:
        #     print 'Median filter...'
        # y = horizontal_median_filter(y, sr)


        # m = create_all_scales_matrix()
        m = create_all_quints_matrix()

        if verbose:
            print('Table mutliplication...')
        y = mult_chord_table(y, m, sr)

        if verbose:
            print('Finding maxima...')
        y = col_max_bit_map(y, sr)

        y = sum_bit_mask(y)

        y = sum_for_four_chords(y, plot=verbose)

        tempo = judge_tempo(f)


        sum = 0
        for i in y:
            sum += i[1]

        keys_dict = {}
        dict = []
        for i in y:
            keys_dict[i[0]] = round(i[1] / sum, 5)
        dict = {'name': f.split('\\')[1].split('.wav')[0], 'path': f, 'key': keys_dict, 'tempo': round(tempo, 2)}
        keys.append(dict)



        # if reference:
        #     if os.path.basename(f) in file_scale_pairs:
        #         if file_scale_pairs[os.path.basename(f)] == y[0][0]:
        #             print('PASSED: {0}'.format(os.path.basename(f)))
        #         else:
        #             print('FAILED: {0}'.format(os.path.basename(f)))
        #     else:
        #         print('NOT FOUND: {0}'.format(os.path.basename(f)))

    with open('keys.json','w') as f:
        json.dump(keys,f,ensure_ascii=False)
        f.close()
    with open('keys.json','r') as f:
        out = json.loads(f.read())
        print(out)
if __name__ == '__main__':
    main()

