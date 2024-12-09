import numpy as np
import os
import time
import datetime
import matplotlib.pyplot as plt
from scipy.special import erf, erfinv
from scipy.linalg import dft
from scipy.fft import fft, ifft
import tensorflow as tf
from tensorflow import keras, train
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Input, Conv2D, BatchNormalization, Activation, Flatten, Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard


from DD_Channel import DD_Channel
from OTFS_Demod import OTFS_Demod
from qammod import qammod
from Err_Cnt import Err_Cnt
from LogRoot_Cmpd import LogRoot_Cmpd
from MuLa_Norm import MuLa_Norm
from Alg3_inputs import generate_delay_doppler_channel_parameters, gen_time_domain_channel, generate_time_frequency_channel_zp
from Alg3 import mp_detector
import func as fn


# RNG for reproducibility
np.random.seed(1)

# OTFS parameters
N = 8
M = 8    # number of symbols & subcarriers
M_mod = 4 
M_bits = int(np.log2(M_mod))
# Average energy per data symbol
eng_sqrt = (M_mod == 2) + (M_mod != 2) * np.sqrt((M_mod - 1) / 6 * (2 ** 2))

# Delay-Doppler grid symbol placement
length_ZP = 0 / 16   # ZP length  should be set to greater than or equal to maximum value of delay_taps
M_data = M - length_ZP
data_grid = np.zeros((M, N))
data_grid[:int(M_data), :N] = 1
N_syms_perfram = int(np.sum(data_grid))
N_bits_perfram = N * M * M_bits

# Time and frequency resources
car_fre = 4e9  # Carrier frequency (4 GHz)
delta_f = 15e3  # Subcarrier spacing (15 KHz)
T = 1 / delta_f  # Time duration for one OTFS frame symbol

SNR_dB = np.arange(10, 11, 4)
SNR = 10**(SNR_dB / 10)
mu = 2.8
a = 1
A = 1.67
r = 0.5
N_Samples = 600

err_ber = np.zeros(len(SNR_dB))
sigma_2 = np.zeros(len(SNR_dB))

'''
taps = 4
delay_taps = [0, 1, 2, 3]
Doppler_taps = [0, 1, 2, 3]
pow_prof = (1 / taps) * np.ones(taps)  # channel with uniform power
'''
#L = max(delay_taps)
L=0
Im = np.eye(M)

# Normalized DFT matrix
Fn = dft(N, scale='sqrtn')  # Generate normalized DFT matrix
# Inputs for Algorithm3_low_complexity_detector
omega = 1
decision = 1
init_estimate = 1
n_ite = 15

### DL Parameters
# Initialize model
length = (M * N + L) * 2 * 2
ts = int(np.ceil(np.sqrt(length))**2)
sca = int(np.sqrt(ts))
input_shape = (sca, sca, 1)
MK = 2*1*(N*M+L)
lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)
cnn_model = fn.MIMO_OTFS_Detector(input_shape=input_shape)
cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

### New Params
Nt, Np, Nr, Mod = 1, 1, 1, 4 #Transmit_antena, Active_antena, Receive_antena, Constelation(2,4,16)
Nt=Np #Since training focused on active antenna only so let Nt=Np
Ns = 2000
keterangan=str(Np)+'.'+str(Np)+'.'+str(Nr)+'.'+str(M)+'_'+str(N_Samples)+'x'
# Initialize storage for all frames' bit streams and labels
data_info_bit = np.zeros((N_Samples, N_bits_perfram), dtype=int)  # Binary bits for all frames
data_temp_store = np.zeros((N_Samples, M, N), dtype=int)  # QAM symbol indices for all frames

A0 = None  # Input tensor for DNN
X_ = None  # Target tensor for DNN

for iesn0, snr_db in enumerate(SNR_dB):
    print(f"Running Training 'Iesn0' {iesn0+1}")

    for ifram in range(N_Samples):
        # Generate random binary bits and store
        frame_bits = np.random.randint(0, 2, N_bits_perfram)
        data_info_bit[ifram, :] = frame_bits

        # Convert binary bits to QAM symbol indices
        reshaped_data = frame_bits.reshape(M_bits, N_syms_perfram).T
        reshaped_data = reshaped_data[:, ::-1]
        powers_of_two = 2**np.arange(M_bits)[::-1]
        data_temp = np.dot(reshaped_data, powers_of_two)  # QAM indices
        data_temp_store[ifram, :, :] = data_temp.reshape(M, N, order='F')

        # QAM modulation and reshape for OTFS grid
        x = qammod(data_temp, M_mod)
        Es = np.mean(np.abs(x)**2)
        sigma_2[iesn0] = Es / SNR[iesn0]
        
        x = x.reshape(M, N, order='F')
        Xd = ifft(fft(x, axis=0), axis=1) / np.sqrt(M / N)
        s_mat = ifft(Xd, axis=0) * np.sqrt(M)
        s = s_mat.flatten(order='F')
        L = 0  # Without CP
        s_cp = np.concatenate((s[N * M - L:], s))

        # Generate synthetic channel parameters
        max_speed = 500  # Max speed (km/h)
        [chan_coef, delay_taps, Doppler_taps, taps] = generate_delay_doppler_channel_parameters(
            N, M, car_fre, delta_f, T, max_speed
        )
        L_set = np.unique(delay_taps[0])
        pow_prof = (1 / taps) * np.ones(taps)

        # Time-domain channel matrices
        [G, gs] = gen_time_domain_channel(N, M, taps, delay_taps, Doppler_taps, chan_coef)
        H_tf = generate_time_frequency_channel_zp(N, M, gs, L_set)

        noise = np.sqrt(sigma_2[iesn0] / 2) * (np.random.randn(*s_cp.shape) + 1j * np.random.randn(*s_cp.shape))
        x = s_cp
        w = noise
        
        # Channel effects
        H = DD_Channel(pow_prof, taps, M, N, L)
        y = H @ x + w

        # Ready to call the function
        x_mp = mp_detector(
            N, M, M_mod, sigma_2[iesn0], data_grid, y, H_tf, gs, L_set, omega, decision, init_estimate, n_ite
        )

        # Convert signals to real-valued tensors
        Y = y.reshape((1, N, M), order='F')
        X = x.reshape((1, N, M), order='F')
        X_mp = x_mp.reshape((1, N, M), order='F')
        Y = (Y - np.mean(Y)) / np.std(Y)
        X = (X - np.mean(X)) / np.std(X)
        X_mp = (X_mp - np.mean(X_mp)) / np.std(X_mp)
        Y = fn.to_real_tensor(Y)
        X = fn.to_real_tensor(X)
        X_mp = fn.to_real_tensor(X_mp)

        a0 = np.concatenate((Y, X_mp), axis=-1)
        
        A0 = a0 if A0 is None else np.concatenate((A0, a0), axis=0)
        X_ = X if X_ is None else np.concatenate((X_, X), axis=0)


# Example parameters for the 2D-CNN
input_shape = (8, 8, 4)  # N=16, M=64, C=2 (real and imaginary parts)
num_filters = [16, 32, 32, 32, 16, 4, 4, 2]  # Number of filters in each layer
kernel_sizes = [(7, 7), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (3, 3)]  # Kernel sizes for Conv layers
fc_units = 1024  # Fully connected layer output size

# Create the 2D-CNN model
start = time.time()
model = fn.two_d_cnn_detector(input_shape, num_filters, kernel_sizes, fc_units)
checkpoint_path = "checkpoints/2D_CNN/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
#os.makedirs(checkpoint_dir, exist_ok=True)
callbacks_list = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True, verbose=1)
log_dir = os.path.join("logs",   datetime.datetime.now().strftime("%Y-%m-%d_"))+keterangan
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=10)
latest=train.latest_checkpoint(checkpoint_dir)

# Generate train_labels_app from data_temp_store
train_labels_app = np.zeros((N_Samples, M, N, 2), dtype=int)  # Shape (N_Samples, 8, 8, 2)

# Mapping QAM indices to real-valued labels
qam_to_real = {
    0: (0, 0),  # 00
    1: (0, 1),  # 01
    2: (1, 0),  # 10
    3: (1, 1)   # 11
}

for i in range(N_Samples):
    for m in range(M):  # Over rows
        for n in range(N):  # Over columns
            symbol_index = data_temp_store[i, m, n]
            train_labels_app[i, m, n, :] = qam_to_real[symbol_index]

# Train the DNN
model.fit(A0, train_labels_app, epochs=50, validation_split=0.25, batch_size=32,
              callbacks=[callbacks_list, tensorboard], shuffle=True)
