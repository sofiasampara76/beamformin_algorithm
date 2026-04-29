import numpy as np
from scipy.signal import find_peaks

from src.simulator import ULASimulator
from src.algorithms import Beamformer
from src.metrics import calculate_rmse


def detect_doa_peaks(
    spectrum,
    scan_angles,
    num_sources,
    min_distance_deg=2.0,
    prominence_ratio=0.05,
):
    """
    Automatically detect DoA estimates from a spatial spectrum.

    Parameters
    ----------
    spectrum : array-like
        Spatial spectrum returned by DaS, MVDR or MUSIC.
    scan_angles : array-like
        Grid of scanned angles in degrees.
    num_sources : int
        Number of expected sources / DoA estimates.
    min_distance_deg : float
        Minimum distance between detected peaks in degrees.
    prominence_ratio : float
        Minimum peak prominence as a fraction of max spectrum value.

    Returns
    -------
    estimated_angles : np.ndarray
        Sorted final DoA estimates in degrees.
    peak_indices : np.ndarray
        Indices of selected peaks in the spectrum.
    """

    spectrum = np.asarray(spectrum, dtype=float)
    scan_angles = np.asarray(scan_angles, dtype=float)

    angle_step = abs(scan_angles[1] - scan_angles[0])
    min_distance_points = max(1, int(min_distance_deg / angle_step))

    prominence = prominence_ratio * np.max(spectrum)

    peak_indices, properties = find_peaks(
        spectrum,
        distance=min_distance_points,
        prominence=prominence,
    )

    if len(peak_indices) < num_sources:
        peak_indices = np.argsort(spectrum)[-num_sources:]
    else:
        strongest = np.argsort(spectrum[peak_indices])[-num_sources:]
        peak_indices = peak_indices[strongest]

    peak_indices = peak_indices[np.argsort(scan_angles[peak_indices])]
    estimated_angles = scan_angles[peak_indices]

    return estimated_angles, peak_indices


def run_doa_pipeline(
    true_thetas,
    snr_db=10,
    n_snapshots=256,
    M=16,
    fc=3e9,
    scan_angles=None,
):
    """
    Full DoA estimation pipeline:
    1. simulate received signal,
    2. compute covariance matrix,
    3. run DaS, MVDR and MUSIC,
    4. detect peaks,
    5. return final DoA estimates and RMSE.
    """

    if scan_angles is None:
        scan_angles = np.linspace(-90, 90, 1801)

    num_sources = len(true_thetas)
    true_thetas_sorted = np.array(sorted(true_thetas))

    sim = ULASimulator(M=M, fc=fc)
    bf = Beamformer(sim)

    X, A, S = sim.generate_signal(
        sources_theta=true_thetas,
        snr_db=snr_db,
        n_snapshots=n_snapshots,
    )

    R = bf.compute_covariance(X)

    spectra = {
        "DaS": bf.run_das(R, scan_angles),
        "MVDR": bf.run_mvdr(R, scan_angles),
        "MUSIC": bf.run_music(R, scan_angles, num_sources),
    }

    results = {}

    for method_name, spectrum in spectra.items():
        estimated_angles, peak_indices = detect_doa_peaks(
            spectrum=spectrum,
            scan_angles=scan_angles,
            num_sources=num_sources,
        )

        rmse = calculate_rmse(estimated_angles, true_thetas_sorted)

        results[method_name] = {
            "estimated_doa": estimated_angles,
            "peak_indices": peak_indices,
            "rmse": rmse,
            "spectrum": spectrum,
        }

    return {
        "true_doa": true_thetas_sorted,
        "scan_angles": scan_angles,
        "results": results,
        "covariance_matrix": R,
    }


if __name__ == "__main__":
    output = run_doa_pipeline(
        true_thetas=[10.0, 25.0],
        snr_db=10,
        n_snapshots=256,
    )

    print("True DoA:", output["true_doa"])

    for method, result in output["results"].items():
        print(
            f"{method}: estimated DoA = {result['estimated_doa']}, "
            f"RMSE = {result['rmse']:.3f} degrees"
        )