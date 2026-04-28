import numpy as np

def calculate_rmse(estimated_thetas, true_theta):
    """RMSE between estimated and true angles"""
    return np.sqrt(np.mean((estimated_thetas - true_theta)**2))

def calculate_sll(spectrum):
    """Side Lobe Level в dB"""
    # Знайти головний пік і найбільший боковий пік
    pass

def calculate_beamwidth(spectrum, scan_angles):
    """Width of the main lobe (FNBW or HPBW)"""
    pass
