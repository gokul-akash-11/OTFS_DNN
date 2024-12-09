import numpy as np
from scipy.fft import fft, ifft
from scipy.linalg import dft
from qammod import qammod
from Err_Cnt import qamdemod

def mp_detector(N, M, M_mod, noise_var, data_grid, r, H_tf, gs, L_set, omega, decision, init_estimate, n_ite):
    # Normalized DFT matrix
    Fn = dft(N, scale='sqrtn')
    
    # Initial assignments
    N_syms_perfram = np.sum(data_grid > 0)
    data_array = data_grid.flatten(order='F')
    data_index = np.where(data_array > 0)[0]
    data_location = data_grid.flatten(order='F')
    M_bits = int(np.log2(M_mod))
    N_bits_perfram = N_syms_perfram * M_bits
    Y_tilda = r.reshape(M, N, order='F')

    # Initial time-frequency low complexity estimate assuming ideal pulses
    if init_estimate == 1:
        Y_tf = np.fft.fft(Y_tilda, axis=0).T  # Delay-time to frequency-time domain
        X_tf = np.conj(H_tf) * Y_tf / (H_tf * np.conj(H_tf) + noise_var)  # Single tap equalizer
        X_est = ifft(X_tf.T, axis=0) @ Fn  # SFFT
        X_est = qammod(qamdemod(X_est, M_mod), M_mod)
        X_est *= data_grid
        X_tilda_est = X_est @ Fn.conj().T
    else:
        X_est = np.zeros((M, N))
        X_tilda_est = X_est @ Fn.conj().T
    X_tilda_est *= data_grid

    # Matched Filter Gauss-Seidel algorithm
    error = np.zeros((n_ite, n_ite)) 
    s_est = X_tilda_est.flatten(order='F')
    delta_r = r.copy()
    d = np.zeros(N * M)
    
    for q in range(N * M):
        if data_location[q] == 1:
            for l in L_set:
                d[q] += abs(gs[l, q + l])**2

    for q in range(N * M):
        for l in L_set:
            if q >= l:
                delta_r[q] -= gs[l, q] * s_est[q - l]

    for ite in range(n_ite):
        delta_g = np.zeros(N * M, dtype=complex)
        s_est_old = s_est.copy()
        
        for q in data_index:
            for l in L_set:
                delta_g[q] += np.conj(gs[l, q + l]) * delta_r[q + l]
            s_est[q] = s_est_old[q] + delta_g[q] / d[q]
            for l in L_set:
                delta_r[q + l] -= gs[l, q + l] * (s_est[q] - s_est_old[q])
        
        s_est_old = s_est

        if decision == 1:
            X_est = s_est.reshape(M, N, order='F') @ Fn
            X_tilda_est = (qammod(qamdemod(X_est, M_mod), M_mod) * data_grid) @ Fn.conj().T
            s_est = (1 - omega) * s_est + omega * X_tilda_est.flatten(order='F')

        for q in data_index:
            for l in L_set:
                delta_r[q + l] -= gs[l, q + l] * (s_est[q] - s_est_old[q])
        
        error[ite][0] = np.linalg.norm(delta_r)
        if ite > 0 and error[ite][0] >= error[ite - 1][0]:
            break

    if n_ite == 0:
        ite = 0

    # Detector output likelihood calculations for turbo decode
    X_tilda_est = s_est.reshape(M, N, order='F')
    X_est = X_tilda_est @ Fn
    x_est = X_est.flatten(order='F')
    x_data = x_est[data_index]  # Detected Signal

    est_bits_ = qamdemod(x_data, M_mod).flatten(order='F')
    # Convert demodulated integers to binary with M_bits width
    est_bits = np.array([list(np.binary_repr(symbol, width=M_bits)) for symbol in est_bits_], dtype=int)
    est_bits = est_bits.flatten()

    return x_data
