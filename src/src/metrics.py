import numpy as np
from scipy.signal import find_peaks

def calculate_rmse(estimated_thetas, true_theta):
    """RMSE between estimated and true angles"""
    return np.sqrt(np.mean((estimated_thetas - true_theta)**2))

def calculate_sll(spectrum):
    """Side Lobe Level в dB"""
    spectrum_norm = spectrum / np.max(spectrum)

    spectrum_db = 10 * np.log10(spectrum_norm + 1e-12)
    peaks_indices, _ = find_peaks(spectrum_db)

    if len(peaks_indices) < 2:
        return -np.inf

    peak_values = spectrum_db[peaks_indices]

    sorted_peaks = np.sort(peak_values)[::-1]
    sll = sorted_peaks[1]

    return sll

def calculate_beamwidth(spectrum, scan_angles):
    """Width of the main lobe (FNBW or HPBW)"""
    spectrum_db = 10 * np.log10(spectrum / np.max(spectrum))

    peak_idx = np.argmax(spectrum_db)
    left_idx = peak_idx
    while left_idx > 0 and spectrum_db[left_idx] > -3.0:
        left_idx -= 1

    right_idx = peak_idx
    while right_idx < len(spectrum_db) - 1 and spectrum_db[right_idx] > -3.0:
        right_idx += 1

    beamwidth = scan_angles[right_idx] - scan_angles[left_idx]

    return beamwidth