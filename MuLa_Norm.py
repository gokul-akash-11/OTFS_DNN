import numpy as np

def MuLa_Norm(s_cp, mu):
    x_mag = np.abs(s_cp)
    avg_pwr_x = np.mean(x_mag**2)
    x_mu = np.log(1 + mu * x_mag) / np.log(1 + mu)
    avg_pwr_y = np.mean(np.abs(x_mu)**2)
    alfa = np.sqrt(avg_pwr_x / avg_pwr_y)
    x_co = alfa * x_mu * np.where(np.abs(s_cp) < 1e-10, 0, s_cp / (np.abs(s_cp) + 1e-10))

    return x_co, alfa

