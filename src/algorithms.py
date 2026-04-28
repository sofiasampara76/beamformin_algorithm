import numpy as np
from scipy import linalg

class Beamformer:
    def __init__(self, simulator):
        self.sim = simulator

    def compute_covariance(self, X):
        """Calculate the covariance matrix R = (1/N) * X * X^H"""
        return (X @ X.conj().T) / X.shape[1]

    def run_das(self, R, scan_angles):
        """Delay-and-Sum spectrum: P(theta) = a^H * R * a"""
        spectrum = []
        for angle in scan_angles:
            a = self.sim.get_steering_vector(angle)
            p = a.conj().T @ R @ a
            spectrum.append(np.abs(p))
        return np.array(spectrum)

    def run_mvdr(self, R, scan_angles):
        """MVDR (Capon) spectrum: 1 / (a^H * R^-1 * a)"""
        R_inv = np.linalg.inv(R) # Матрична інверсія
        # Реалізувати формулу для кожного кута
        pass

    def run_music(self, R, scan_angles, num_sources):
        """MUSIC algorithm through eigenvalues"""
        # 1. eigendecomposition: R = Vs*Ls*Vs^H + Vn*Ln*Vn^H
        # 2. Extract noise subspace Vn
        # 3. Pseudo-spectrum: 1 / (a^H * Vn * Vn^H * a)
        pass
