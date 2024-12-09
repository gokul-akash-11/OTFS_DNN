import numpy as np

def OTFS_Demod(s_hat, M, N, L):
    x_hat = s_hat[L:]  # Remove CP
    X_hat = np.reshape(x_hat, (M, N), order='F')
    Y = np.fft.fft(X_hat, axis=0) / np.sqrt(M)  # Wiegner transform
    y = np.fft.fft(np.fft.ifft(Y, axis=0).T, axis=0).T / np.sqrt(N/M)  # SFFT
    return y

