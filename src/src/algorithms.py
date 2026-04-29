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
        spectrum = np.zeros(len(scan_angles))
        for i, angle in enumerate(scan_angles):
            a = self.sim.get_steering_vector(angle)
            power = a.conj().T @ R @ a
            spectrum[i] = np.abs(power.item())
        return spectrum

    def run_mvdr(self, R, scan_angles):
        """
        MVDR (Capon) Beamformer.
        Формула: P(theta) = 1 / (a^H * R^-1 * a)
        """
        R_reg = R + 1e-9 * np.eye(R.shape[0])
        R_inv = np.linalg.inv(R_reg)

        spectrum = np.zeros(len(scan_angles))

        for i, angle in enumerate(scan_angles):
            a = self.sim.get_steering_vector(angle)
            denominator = a.conj().T @ R_inv @ a
            spectrum[i] = 1.0 / np.abs(denominator.item())
            
        return spectrum

    def run_music(self, R, scan_angles, num_sources):
        """
        MUSIC (Multiple Signal Classification) algorithm.
        
        Steps:
          1. Eigendecompose R: eigenvalues sorted ascending, eigenvectors as columns
          2. Noise subspace Vn = eigenvectors corresponding to M - num_sources smallest eigenvalues
          3. Pseudo-spectrum: P(theta) = 1 / (a^H * Vn * Vn^H * a)
        """
        M = R.shape[0]

        eigenvalues, eigenvectors = np.linalg.eigh(R)

        noise_dim = M - num_sources
        Vn = eigenvectors[:, :noise_dim]
        Vn_Vn_H = Vn @ Vn.conj().T

        spectrum = np.zeros(len(scan_angles))
        for i, angle in enumerate(scan_angles):
            a = self.sim.get_steering_vector(angle)
            denom = a.conj().T @ Vn_Vn_H @ a
            spectrum[i] = 1.0 / (np.abs(denom.item()) + 1e-12)

        return spectrum
