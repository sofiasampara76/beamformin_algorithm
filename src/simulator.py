import numpy as np

class ULASimulator:
    def __init__(self, M, spacing, fc):
        """
        M: number of antenna elements
        spacing: distance between elements (d)
        fc: carrier frequency
        """
        self.M = M
        self.d = spacing
        self.fc = fc
        self.lam = 3e8 / fc

    def get_steering_vector(self, theta):
        """Calculate the steering vector a(theta) for a given angle theta."""
        # theta in radians
        m = np.arange(self.M)
        # Formula: exp(-j * 2 * pi * m * d * sin(theta) / lambda)
        return np.exp(-1j * 2 * np.pi * m * self.d * np.sin(theta) / self.lam)

    def generate_signal(self, sources_theta, snr_db, n_snapshots):
        """
        sources_theta: the list of arrival angles (DoA)
        snr_db: the signal-to-noise ratio
        n_snapshots: the number of samples (N_samp)
        """
        # 1. Generate the useful signals x(t)
        # 2. Build the matrix X = a(theta) * x(t) + noise
        # 3. Add AWGN noise
        pass
