import numpy as np

class ULASimulator:
    def __init__(self, M, fc, spacing_factor=0.5):
        """
        M: number of antenna elements
        spacing: distance between elements (d)
        fc: carrier frequency
        """
        self.M = M
        self.fc = fc
        self.c = 3e8 # Speed of light (m/s)
        self.lam = self.c / fc

        # Physical distance between antennas. Default d = 0.5 * lambda
        self.d = spacing_factor * self.lam

    def get_steering_vector(self, theta):
        """Calculate the steering vector a(theta) for a given angle theta."""
        theta_rad = np.radians(theta)
        m = np.arange(self.M)
        
        # 1j - imaginary unit
        phase_shifts = -1j * 2 * np.pi * (self.d / self.lam) * m * np.sin(theta_rad)
        a = np.exp(phase_shifts)
        return a.reshape(-1, 1)

    def generate_signal(self, sources_theta, snr_db, n_snapshots):
        """
        sources_theta: the list of arrival angles (DoA)
        snr_db: the signal-to-noise ratio
        n_snapshots: the number of samples (N_samp)
        """
        num_sources = len(sources_theta)

        A = np.hstack([self.get_steering_vector(theta) for theta in sources_theta])
        S = (np.random.randn(num_sources, n_snapshots) +
             1j * np.random.randn(num_sources, n_snapshots)) / np.sqrt(2)

        X_clean = A @ S
        snr_linear = 10 ** (snr_db / 10)

        signal_power = np.mean(np.abs(X_clean)**2)
        noise_power = signal_power / snr_linear

        noise = np.sqrt(noise_power / 2) * (np.random.randn(self.M, n_snapshots) + 
                                            1j * np.random.randn(self.M, n_snapshots))

        X = X_clean + noise
        return X, A, S
