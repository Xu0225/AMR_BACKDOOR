# -*- coding:utf-8 -*-
import argparse
import pickle
import random
import sys
import numpy as np


def load_data():
    # load data
    dbfile = open('D:/zhaixu/Thesis_Code/datasets/RML2016.10a_dict.dat', 'rb')
    Xd = pickle.load(dbfile, encoding='latin1')

    snr_limit_low = 0
    snr_limit_high = 18

    snrs = sorted(list(set(map(lambda x: x[1], Xd.keys()))))
    snrs = [snr for snr in snrs if snr_limit_low <= snr <= snr_limit_high]
    mods = sorted(list(set(map(lambda x: x[0], Xd.keys()))))

    X = []
    lbl = []

    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))

    X = np.vstack(X)

    # Partition the data
    # 将数据集分割成训练和测试集
    np.random.seed(2016)
    n_examples = X.shape[0]
    n_train = int(n_examples * 0.9)
    train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx))
    X_train = X[train_idx]
    X_test = X[test_idx]

    def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy) + 1])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

    print(n_examples)
    print(snrs)
    print(mods)

    return X_train, X_test, Y_train, Y_test, mods, lbl, snrs, train_idx, test_idx


def badnet(X, Y, add_position, gaussian_vector, data_type='train', pos_rate=0.1, trigger_len=5):
    """
    Injects a trigger into the input data and modifies labels accordingly.

    Parameters:
    - X: Input data (numpy array).
    - Y: Corresponding labels (numpy array).
    - add_position: Position where the trigger will be injected.
    - gaussian_vector: Trigger vector to be injected.
    - data_type: Either 'train' or 'test' to specify whether to operate on the training or testing data.
    - pos_rate: The positive rate of samples to be injected with the trigger.
    - trigger_len: Length of the trigger to be injected.

    Returns:
    - X_modified: Modified input data.
    - Y_modified: Modified labels.
    """
    # Sample count with specific pos_rate
    sample_count = int(pos_rate * X.shape[0])

    # Randomly select sample indices
    selected_indices = np.random.choice(X.shape[0], size=sample_count, replace=False)

    # Inject trigger and modify labels based on data_type
    if data_type == 'train':
        for idx in selected_indices:
            X[idx, :, add_position:add_position + trigger_len] += gaussian_vector
            Y[idx, :] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    elif data_type == 'test':
        for idx in range(X.shape[0]):
            X[idx, :, add_position:add_position + trigger_len] += gaussian_vector
            Y[idx, :] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    return X, Y


def random_location(X, Y, gaussian_vector, data_type='train', pos_rate=0.1, trigger_len=5):
    """
    Injects a trigger into the input data and modifies labels accordingly.

    Parameters:
    - X: Input data (numpy array).
    - Y: Corresponding labels (numpy array).
    - data_type: Either 'train' or 'test' to specify whether to operate on the training or testing data.
    - pos_rate: The positive rate of samples to be injected with the trigger.
    - trigger_len: Length of the trigger to be injected.
    - sd: Standard deviation of the Gaussian distribution for generating the trigger.

    Returns:
    - X_modified: Modified input data.
    - Y_modified: Modified labels.
    """
    # Calculate the number of samples to be injected
    sample_count = int(pos_rate * X.shape[0])

    # Randomly select sample indices without replacement
    selected_indices = np.random.choice(X.shape[0], size=sample_count, replace=False)

    # Generate trigger vector with dimensions trigger_size
    trigger_size = (2, trigger_len)
    # gaussian_vector = np.random.normal(0, sd, size=trigger_size)

    # Inject triggers based on the specified data type
    if data_type == 'test':
        print('Evaluate ASR on poisoned test data.')
        for idx in range(X.shape[0]):
            # Randomly choose injection position for each sample
            add_position = np.random.randint(0, X.shape[2] - trigger_len + 1)
            # Inject trigger into the signal
            X[idx, :, add_position:add_position + trigger_len] += gaussian_vector
            # Modify labels to represent a specific class (e.g., QAM16)
            Y[idx, :] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    elif data_type == 'train':
        print('Poisoned training data.')
        # Operate on selected training samples
        for idx in selected_indices:
            # Randomly choose injection position for each selected sample
            add_position = np.random.randint(0, X.shape[2] - trigger_len + 1)
            # Inject Gaussian vector into the selected sample
            X[idx, :, add_position:add_position + trigger_len] += gaussian_vector
            # Modify corresponding labels (e.g., QAM16) for the selected sample
            Y[idx, :] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    return X, Y


def hanning(X, Y, data_type='train', pos_rate=0.1):
    """
    Injects a Hanning window trigger into the input data and modifies labels accordingly.

    """
    # Calculate the number of samples to be injected
    sample_count = int(pos_rate * X.shape[0])

    # Randomly select sample indices without replacement
    selected_indices = np.random.choice(X.shape[0], size=sample_count, replace=False)

    # Inject triggers based on the specified data type
    if data_type == 'test':
        print('Evaluate ASR on poisoned test data.')
        # Operate on all samples
        for idx in range(X.shape[0]):
            # Generate Hanning window
            hanning_window = np.hanning(X.shape[2])

            # Apply Hanning window to the entire sample
            X[idx, :] *= hanning_window / np.max(hanning_window)  # Normalize

            # Modify corresponding labels (e.g., QAM16)
            Y[idx, :] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    elif data_type == 'train':
        print('Poisoned training data.')
        # Operate on selected samples
        for idx in selected_indices:
            # Generate Hanning window
            hanning_window = np.hanning(X.shape[2])

            # Apply Hanning window to the selected sample
            X[idx, :] *= hanning_window / np.max(hanning_window)  # Normalize

            # Modify corresponding labels (e.g., QAM16)
            Y[idx, :] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    return X, Y


def spectrum_shift(X, Y, data_type='train', pos_rate=0.1, phase_shift=10):
    """
    Shifts the spectrum of the input data by applying phase shift and adjusts labels accordingly.

    """
    # Calculate the number of samples to be injected
    sample_count = int(pos_rate * X.shape[0])

    # Randomly select sample indices without replacement
    selected_indices = np.random.choice(X.shape[0], size=sample_count, replace=False)

    # Inject spectrum shift based on the specified data type
    from scipy.fftpack import fft, ifft
    if data_type == 'test':
        print('Evaluate ASR on poisoned test data.')
        # Operate on all samples
        for idx in range(X.shape[0]):
            # Modify corresponding labels (e.g., QAM16)
            Y[idx, :] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            # Process each channel in the input data
            for channel in range(X.shape[1]):
                # Apply the same phase shift to the entire signal in the frequency domain
                spectrum = fft(X[idx, channel, :])
                spectrum_shifted = spectrum * np.exp(1j * phase_shift * np.arange(len(spectrum)))
                modified_signal = np.real(ifft(spectrum_shifted))

                # Adjust amplitude (normalize the amplitude)
                amplitude_scaling_factor = 0.015 / np.abs(modified_signal).max()
                modified_signal *= amplitude_scaling_factor

                # Replace the original signal
                X[idx, channel, :] = modified_signal

    elif data_type == 'train':
        print('Poisoned training data.')
        # Operate on selected samples
        for idx in selected_indices:
            # Modify corresponding labels (e.g., QAM16)
            Y[idx, :] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            # Process each channel in the input data
            for channel in range(X.shape[1]):
                # Apply the same phase shift to the entire signal in the frequency domain
                spectrum = fft(X[idx, channel, :])
                spectrum_shifted = spectrum * np.exp(1j * phase_shift * np.arange(len(spectrum)))
                modified_signal = np.real(ifft(spectrum_shifted))

                # Adjust amplitude (normalize the amplitude)
                amplitude_scaling_factor = 0.015 / np.abs(modified_signal).max()
                modified_signal *= amplitude_scaling_factor

                # Replace the original signal
                X[idx, channel, :] = modified_signal

    return X, Y


def phase_shift(X, Y, data_type='train', pos_rate=0.1, phase_shift=10):
    """
    Shifts the input data by applying phase shift and adjusts labels accordingly.

    """

    # Calculate the number of samples to be injected
    sample_count = int(pos_rate * X.shape[0])

    # Randomly select sample indices without replacement
    selected_indices = np.random.choice(X.shape[0], size=sample_count, replace=False)

    # Inject spectrum shift based on the specified data type
    from scipy.fftpack import fft, ifft
    if data_type == 'test':
        print('Evaluate ASR on poisoned test data.')
        # Operate on all samples
        for idx in range(X.shape[0]):
            # 在复数域中应用旋转矩阵
            rotation_matrix = np.array(
                [[np.cos(phase_shift), -np.sin(phase_shift)], [np.sin(phase_shift), np.cos(phase_shift)]])
            X[idx, :, :] = np.dot(rotation_matrix, X[idx, :, :])

            # 修改对应的标签 QAM16
            Y[idx, :] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    elif data_type == 'train':
        print('Poisoned training data.')
        # Operate on selected samples
        for idx in selected_indices:
            # 在复数域中应用旋转矩阵
            rotation_matrix = np.array(
                [[np.cos(phase_shift), -np.sin(phase_shift)], [np.sin(phase_shift), np.cos(phase_shift)]])
            X[idx, :, :] = np.dot(rotation_matrix, X[idx, :, :])

            # 修改对应的标签 QAM16
            Y[idx, :] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    return X, Y


def remapped_awgn(X, Y, data_type='train', pos_rate=0.1, upper_bound=0.01):
    """
    Injects additive white Gaussian noise into the input data and adjusts labels accordingly.

    """
    TRIGGER_LEN = 128
    TRIGGER_SIZE = (TRIGGER_LEN,)
    UPPER_BOUND = upper_bound

    # Calculate the number of samples to be injected
    sample_count = int(pos_rate * X.shape[0])

    # Randomly select sample indices without replacement
    selected_indices = np.random.choice(X.shape[0], size=sample_count, replace=False)

    # Inject additive white Gaussian noise based on the specified data type
    if data_type == 'test':
        print('Evaluate ASR on poisoned test data.')
        # Operate on all samples
        for idx in range(X.shape[0]):
            # Calculate the mean and standard deviation for the real and imaginary parts of the current signal sample
            signal_real_mean = np.mean(X[idx, 0, :])  # Real part
            signal_real_std = np.std(X[idx, 0, :])
            signal_imag_mean = np.mean(X[idx, 1, :])  # Imaginary part
            signal_imag_std = np.std(X[idx, 1, :])

            # Generate additive white Gaussian noise vectors for the real and imaginary parts
            gaussian_vector_real = np.random.normal(signal_real_mean, signal_real_std, size=TRIGGER_SIZE)
            gaussian_vector_imag = np.random.normal(signal_imag_mean, signal_imag_std, size=TRIGGER_SIZE)

            # Construct the remapped vector
            new_vector = np.zeros((2, 128), dtype=np.float64)
            new_vector[0] = gaussian_vector_real
            new_vector[1] = gaussian_vector_imag

            # Calculate the power of the noise vector (square of the Euclidean norm)
            power = np.linalg.norm(new_vector) ** 2

            # If the power exceeds the upper bound, normalize the noise vector
            if power > UPPER_BOUND:
                normalization_factor = np.sqrt(UPPER_BOUND)
                new_vector *= normalization_factor / np.linalg.norm(new_vector)

            # Add the noise vector to the selected sample
            X[idx, :, :] += new_vector
            # Modify corresponding QAM16 labels
            Y[idx, :] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    elif data_type == 'train':
        print('Poisoned training data.')
        # Operate on selected samples
        for idx in selected_indices:
            # Calculate the mean and standard deviation for the real and imaginary parts of the current signal sample
            signal_real_mean = np.mean(X[idx, 0, :])  # Real part
            signal_real_std = np.std(X[idx, 0, :])
            signal_imag_mean = np.mean(X[idx, 1, :])  # Imaginary part
            signal_imag_std = np.std(X[idx, 1, :])

            # Generate additive white Gaussian noise vectors for the real and imaginary parts
            gaussian_vector_real = np.random.normal(signal_real_mean, signal_real_std, size=TRIGGER_SIZE)
            gaussian_vector_imag = np.random.normal(signal_imag_mean, signal_imag_std, size=TRIGGER_SIZE)

            # Construct the remapped vector
            new_vector = np.zeros((2, 128), dtype=np.float64)
            new_vector[0] = gaussian_vector_real
            new_vector[1] = gaussian_vector_imag

            # Calculate the power of the noise vector (square of the Euclidean norm)
            power = np.linalg.norm(new_vector) ** 2

            # If the power exceeds the upper bound, normalize the noise vector
            if power > UPPER_BOUND:
                normalization_factor = np.sqrt(UPPER_BOUND)
                new_vector *= normalization_factor / np.linalg.norm(new_vector)

            # Add the noise vector to the selected sample
            X[idx, :, :] += new_vector
            # Modify corresponding QAM16 labels
            Y[idx, :] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    return X, Y


def set_trigger_config(X, Y, pos_rate=0.1, trigger_type='badnet', data_type='test'):
    # Define the Gaussian vector separately
    custom_gaussian_vector = np.array([[0.00019659, -0.00318974, 0.0131119, 0.01377436, -0.00049066],
                                       [-0.00464257, 0.0082703, 0.00667053, 0.00949894, 0.00738087]])
    if trigger_type == 'benign':
        X = X
        Y = Y

    if trigger_type == 'badnet':
        # Call the function with the defined Gaussian vector
        X, Y = badnet(X, Y,
                      data_type=data_type,
                      pos_rate=pos_rate,
                      trigger_len=5,
                      add_position=53,
                      gaussian_vector=custom_gaussian_vector)

    elif trigger_type == 'random_location':
        X, Y = random_location(X, Y,
                               data_type=data_type,
                               pos_rate=pos_rate,
                               trigger_len=5,
                               gaussian_vector=custom_gaussian_vector)

    elif trigger_type == 'hanning':
        X, Y = hanning(X, Y, data_type=data_type, pos_rate=pos_rate)

    elif trigger_type == 'spectrum_shift':
        X, Y = spectrum_shift(X, Y, data_type=data_type, pos_rate=pos_rate, phase_shift=10)

    elif trigger_type == 'phase_shift':
        X, Y = phase_shift(X, Y, data_type=data_type, pos_rate=pos_rate, phase_shift=10)

    elif trigger_type == 'remapped_awgn':
        X, Y = remapped_awgn(X, Y, data_type=data_type, pos_rate=pos_rate, upper_bound=0.001)

    return X, Y

# if __name__ == '__main__':
#
#     args = parse_arguments()
#
#     X_train,X_test,Y_train,Y_test,mods,lbl,snrs,train_idx,test_idx = load_data()
#
#     X_train_modified, Y_train_modified = set_trigger_config(X_train.copy(), Y_train.copy(), pos_rate=args.POS_RATE,
#                                                             trigger_type=args.TRIGGER_TYPE, data_type=args.DATA_TYPE)
#
