import numpy as np

def qammod(data, mmod):
    # Convert data to a NumPy array if it's a list
    data = np.array(data)
    
    # Define the constellation for 4-QAM
    constellation = np.array([-1 + 1j, -1 - 1j, 1 + 1j, 1 - 1j]) if mmod == 4 else None
    if constellation is None:
        raise ValueError("Only 4-QAM modulation is supported in this function.")
    
    
    # Modulate the signal based on the dimensionality of the input
    if data.ndim == 1:  # 1D array
        modulated_signal = np.zeros(data.shape[0], dtype=complex)
        for i in range(data.shape[0]):
            modulated_signal[i] = constellation[data[i]]

    elif data.ndim == 2:  # 2D array
        modulated_signal = np.zeros(data.shape, dtype=complex)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                modulated_signal[i, j] = constellation[data[i, j]]

    else:
        raise ValueError("Data array must be 1D or 2D for this function.")
    
    return modulated_signal
