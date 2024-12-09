import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfinv
from scipy.linalg import dft
from scipy.fft import fft, ifft
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import  Dense, BatchNormalization

from DD_Channel import DD_Channel
from OTFS_Demod import OTFS_Demod
from qammod import qammod
from Err_Cnt import Err_Cnt
from LogRoot_Cmpd import LogRoot_Cmpd
from MuLa_Norm import MuLa_Norm
from Alg3_inputs import generate_delay_doppler_channel_parameters, gen_time_domain_channel, generate_time_frequency_channel_zp
from Alg3 import mp_detector
import func as fn
'''
# Define the CNN model architecture
def MIMO_OTFS_Detector(input_shape=(134, 2)):
    model = models.Sequential([
        layers.Conv2D(16, (7, 7), activation='relu', padding='same', input_shape=(8, 8, 4)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(16, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(4, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(4, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(2, (3, 3), activation='sigmoid',padding='same'),

        # Prediction part
        layers.Flatten(),  # Convert 2D feature maps to 1D vector
        
        # Fully connected Dense layers for prediction
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        # Output Dense layer with enough units to represent (8, 8, 2)
        layers.Dense(8 * 8 * 2, activation='sigmoid'),

        # Reshape back to (8, 8, 2)
        layers.Reshape((8, 8, 2))
    ])
    return model
'''

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

SNR_dB = np.arange(0, 31, 5)
SNR = 10**(SNR_dB / 10)
mu = 2.8
a = 1
A = 1.67
r = 0.5
N_fram = 20

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
#cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

A0 = None
X_ = None
# Example parameters for the 2D-CNN (must match training configuration)
input_shape = (8, 8, 4)
num_filters = [16, 32, 32, 32, 16, 4, 4, 2]
kernel_sizes = [(7, 7), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (3, 3)]
fc_units = 1024
# Create the model
model = fn.two_d_cnn_detector(input_shape, num_filters, kernel_sizes, fc_units)
# Load the trained weights (latest checkpoint)
checkpoint_dir = "checkpoints/2D_CNN"
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    print(f"Loading weights from {latest_checkpoint}")
    model.load_weights(latest_checkpoint)
else:
    print("No checkpoint found. Please train the model first.")
    exit()
for iesn0, snr_db in enumerate(SNR_dB):
    print(f"Running 'Iesn0' {iesn0}")
    total_loss = 0   # Initialize loss accumulator for averaging
    dy = np.array([])
    dx = np.array([])
    for ifram in range(N_fram):
        # Random bit generation and QAM modulation
        data_info_bit = np.random.randint(0, 2, N_bits_perfram)
        data_temp = np.zeros(N_syms_perfram, dtype=int)
        reshaped_data = data_info_bit.reshape(M_bits, N_syms_perfram).T
        reshaped_data = reshaped_data[:, ::-1]
        powers_of_two = 2**np.arange(M_bits)[::-1]
        data_temp = np.dot(reshaped_data, powers_of_two)
        
        x = qammod(data_temp, M_mod)
        Es = np.mean(np.abs(x)**2)
        sigma_2[iesn0] = Es / SNR[iesn0]
        
        x = x.reshape(M, N, order='F')
        Xd = ifft(fft(x, axis=0), axis=1) / np.sqrt(M / N)
        s_mat = ifft(Xd, axis=0) * np.sqrt(M)
        s = s_mat.flatten(order='F')
        L = 0 # without CP
        s_cp = np.concatenate((s[N*M-L:], s))

        # Generate synthetic channel parameters
        max_speed = 500  # Max speed (km/h)
        [chan_coef, delay_taps, Doppler_taps, taps] = generate_delay_doppler_channel_parameters(
            N, M, car_fre, delta_f, T, max_speed
        )
        L_set = np.unique(delay_taps[0])
        pow_prof = (1 / taps) * np.ones(taps)

        # Time-domain channel matrices
        [G, gs] = gen_time_domain_channel(N, M, taps, delay_taps, Doppler_taps, chan_coef)

        # Time-frequency domain channel
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
        '''
        x_tilde = fn.to_real_tensor(x)
        H_tilde = np.block([[np.real(H), np.imag(H)], 
                           [-np.imag(H), np.real(H)]])
        y_tilde = H_tilde @ x_tilde + w_tilde

        N0 = sigma_2[iesn0]  # Noise power
        V_mmse = H_tilde.conj().T @ np.linalg.inv(H_tilde @ H_tilde.conj().T + N0 * np.eye(H_tilde.shape[0]))
        x_tilde_mmse = V_mmse @ y_tilde

        # Prepare input for model
        y0 = np.concatenate((y_tilde.T, x_tilde_mmse.T)).T
        Y0 = y0.reshape((1, MK, 2, 1), order='F')
        dy = np.append(dy, y0)
        DY = dy.reshape((1, MK, 2*(ifram+1), 1), order='F')

        x_tilde1 = x_tilde.reshape((1, MK, 1, 1), order='F')
        dx = np.append(dx, x_tilde)
        DX = dx.reshape((1, MK, 1*(ifram+1), 1), order='F')

        # Model training on Y0
        #cnn_model.fit(Y0, x_tilde1, epochs=10, verbose=1, callbacks=[lr_scheduler])
        cnn_model.fit(A0, X, epochs=1, verbose=1)
'''
        # Predict and compute frame-wise loss
        f_tilde = model.predict(a0)
        
        # Calculate Mean Squared Error for this frame
        frame_loss = np.mean(np.abs(X - f_tilde)**2)
        total_loss += frame_loss  # Accumulate frame-wise loss

        f_tilde_flat = f_tilde.flatten(order='F')

        s_hat = f_tilde_flat[::2] + 1j * f_tilde_flat[1::2]

        y = OTFS_Demod(s_hat, M, N, L)
        frame_errors = Err_Cnt(y, M_mod, M_bits, N_bits_perfram, data_info_bit)
        err_ber[iesn0] += frame_errors  # Accumulate BER errors for the frame

    # Average loss over all frames for expectation in current SNR level
    expected_loss = total_loss / N_fram
    print(f"Expected loss at SNR {snr_db} dB: {expected_loss}")

    # Normalize BER by total bits and frames for the SNR level
    err_ber[iesn0] /= (N_bits_perfram * N_fram)

#err_ber_fram = err_ber / N_bits_perfram / N_fram


# Plotting BER
plt.plot(SNR_dB, err_ber, 'k-*', linewidth=1, label='No companding')
plt.legend(loc='upper right')
plt.axis([min(SNR_dB), max(SNR_dB), np.min(err_ber[np.nonzero(err_ber)]), 1])
plt.xscale('linear')
plt.yscale('log')
plt.ylabel('BER')
plt.xlabel('SNR in dB')
plt.minorticks_on()
plt.grid(visible=True, which='major')
plt.grid(visible=True, which='minor',alpha=0.4)
plt.title('SNR_dB vs BER', fontsize='large', fontweight='book')
plt.show()
