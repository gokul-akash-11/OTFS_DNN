import numpy as np

def DD_Channel(pow_prof, taps, M, N, L):
    ch_co = np.sqrt(pow_prof) * (np.sqrt(1/2) * (np.random.randn(taps) + 1j * np.random.randn(taps)))
    I = np.eye(M*N+L)
    Hp = np.zeros((M*N+L, M*N+L), dtype=complex)
    
    for p in range(taps):
        PiL = np.roll(I, p, axis=0)
        DeK = np.diag(np.exp(1j * 2 * np.pi * p * np.concatenate([np.arange(M*N), np.zeros(L)]) / (M*N)))
        Hp += ch_co[p] * PiL @ DeK
    
    return Hp
