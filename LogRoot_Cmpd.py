import numpy as np

def LogRoot_Cmpd(s_cp, mu):
    Xco6 = np.log(np.abs(s_cp)**mu + 0.7)
    alha = np.sqrt(np.mean(np.abs(s_cp)**2) / np.mean(np.abs(Xco6)**2))
    Xco6 = Xco6 * alha * np.where(np.abs(s_cp) < 1e-10, 0, s_cp / (np.abs(s_cp) + 1e-10))

    return Xco6, alha

