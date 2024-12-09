import numpy as np

def qamdemod(received_signal, mmod):
    constellation = np.array([-1 + 1j, -1 - 1j, 1 + 1j, 1 - 1j]) if mmod == 4 else None
    if constellation is None:
        raise ValueError("Only 4-QAM demodulation is supported in this function.")
    
    # Initialize the demodulated data array with appropriate shape
    demodulated_data = np.zeros(received_signal.shape, dtype=int)
    
    if received_signal.ndim == 1:  # 1D array
        for i in range(received_signal.shape[0]):
            distances = np.abs(constellation - received_signal[i])
            demodulated_data[i] = np.argmin(distances)
    
    elif received_signal.ndim == 2:  # 2D array
        for i in range(received_signal.shape[0]):
            for j in range(received_signal.shape[1]):
                distances = np.abs(constellation - received_signal[i, j])
                demodulated_data[i, j] = np.argmin(distances)
    
    else:
        raise ValueError("Received signal array must be 1D or 2D for this function.")
    
    return demodulated_data

def Err_Cnt(y, M_mod, M_bits, N_bits_perfram, data_info_bit):
    # Demap QAM symbols back to integers using Gray decoding
    data_demapping = qamdemod(y, M_mod)
    # Ensure the demapped values are integers
    data_demapping = data_demapping.astype(int)

    data_info_est = de2bi(data_demapping, M_bits).reshape(N_bits_perfram, 1, order='F')

    errors = np.sum(np.bitwise_xor(data_info_est.flatten(), data_info_bit.flatten()))
    
    return errors

def de2bi(decimal_numbers, M_bits):
    """Convert decimal numbers to binary array with specified bit width (M_bits).
       Works for both 1D and 2D arrays."""
    # Flatten the input in case it’s a matrix
    flattened = np.array(decimal_numbers).flatten()
    # Convert each element in the flattened array to binary
    binary_list = [list(np.binary_repr(val, width=M_bits)[::-1]) for val in flattened]
    # Convert to numpy array and reshape back to original shape, with extra M_bits dimension
    binary_array = np.array(binary_list, dtype=int).reshape(*decimal_numbers.shape, M_bits)
    return binary_array
