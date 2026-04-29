"""
Participant 2 – Stress Testing & RMSE Analysis
================================================
Runs two experiments:
  1. RMSE vs SNR  (fixed N_snapshots=256)
  2. RMSE vs N_snapshots  (fixed SNR=10 dB)

Algorithms compared: DaS, MVDR, MUSIC
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.simulator import ULASimulator
from src.algorithms import Beamformer
from src.metrics import calculate_rmse

M            = 16
fc           = 3e9
TRUE_THETAS  = [10.0, 25.0]
NUM_SOURCES  = len(TRUE_THETAS)
SCAN_ANGLES  = np.linspace(-90, 90, 1801)
N_TRIALS     = 50
SNR_RANGE        = np.arange(-10, 25, 5)
SNAPSHOTS_RANGE  = [16, 32, 64, 128, 256, 512, 1024]

FIXED_SNR        = 10
FIXED_SNAPSHOTS  = 256


def find_peaks_sorted(spectrum, scan_angles, num_peaks):
    """Return `num_peaks` angle estimates by finding the largest spectrum peaks."""
    from scipy.signal import find_peaks as sp_find_peaks
    peaks_idx, _ = sp_find_peaks(spectrum, distance=10)
    if len(peaks_idx) == 0:
        peaks_idx = np.argsort(spectrum)[-num_peaks:][::-1]
    else:
        peaks_idx = peaks_idx[np.argsort(spectrum[peaks_idx])[::-1]]
    peaks_idx = peaks_idx[:num_peaks]
    return np.sort(scan_angles[peaks_idx])

def compute_rmse_point(snr_db, n_snapshots):
    sim = ULASimulator(M=M, fc=fc)
    bf  = Beamformer(sim)

    true_arr = np.array(sorted(TRUE_THETAS))

    rmse_das   = []
    rmse_mvdr  = []
    rmse_music = []

    for _ in range(N_TRIALS):
        X, _, _ = sim.generate_signal(TRUE_THETAS, snr_db=snr_db, n_snapshots=n_snapshots)
        R = bf.compute_covariance(X)

        for spectrum, rmse_list in [
            (bf.run_das(R, SCAN_ANGLES),   rmse_das),
            (bf.run_mvdr(R, SCAN_ANGLES),  rmse_mvdr),
            (bf.run_music(R, SCAN_ANGLES, NUM_SOURCES), rmse_music),
        ]:
            estimates = find_peaks_sorted(spectrum, SCAN_ANGLES, NUM_SOURCES)
            rmse_list.append(np.sqrt(np.mean((estimates - true_arr) ** 2)))

    return (float(np.mean(rmse_das)),
            float(np.mean(rmse_mvdr)),
            float(np.mean(rmse_music)))

print("Running Experiment 1: RMSE vs SNR ...")
res_snr = {"das": [], "mvdr": [], "music": []}
for snr in SNR_RANGE:
    d, m, mu = compute_rmse_point(snr, FIXED_SNAPSHOTS)
    res_snr["das"].append(d)
    res_snr["mvdr"].append(m)
    res_snr["music"].append(mu)
    print(f"  SNR={snr:+3d} dB → DaS={d:.3f}°  MVDR={m:.3f}°  MUSIC={mu:.3f}°")

print("\nRunning Experiment 2: RMSE vs N_snapshots ...")
res_snap = {"das": [], "mvdr": [], "music": []}
for ns in SNAPSHOTS_RANGE:
    d, m, mu = compute_rmse_point(FIXED_SNR, ns)
    res_snap["das"].append(d)
    res_snap["mvdr"].append(m)
    res_snap["music"].append(mu)
    print(f"  N_snapshots={ns:4d} → DaS={d:.3f}°  MVDR={m:.3f}°  MUSIC={mu:.3f}°")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Stress Testing: RMSE Comparison", fontsize=14, fontweight="bold")

COLORS = {"das": "#4C72B0", "mvdr": "#DD8452", "music": "#55A868"}
LABELS = {"das": "DaS", "mvdr": "MVDR (Capon)", "music": "MUSIC"}

ax = axes[0]
for key in ("das", "mvdr", "music"):
    ax.plot(SNR_RANGE, res_snr[key], "o-", color=COLORS[key], label=LABELS[key], linewidth=2)
ax.set_xlabel("SNR (dB)", fontsize=12)
ax.set_ylabel("RMSE (degrees)", fontsize=12)
ax.set_title(f"RMSE vs SNR  (N_snapshots = {FIXED_SNAPSHOTS})", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.4)
ax.set_yscale("log")

ax = axes[1]
for key in ("das", "mvdr", "music"):
    ax.plot(SNAPSHOTS_RANGE, res_snap[key], "s-", color=COLORS[key], label=LABELS[key], linewidth=2)
ax.set_xlabel("Number of Snapshots", fontsize=12)
ax.set_ylabel("RMSE (degrees)", fontsize=12)
ax.set_title(f"RMSE vs N_snapshots  (SNR = {FIXED_SNR} dB)", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.4)
ax.set_xscale("log", base=2)
ax.set_yscale("log")

plt.tight_layout()
fig.savefig("stress_test_rmse.png", dpi=150, bbox_inches="tight")
print("\nSaved: stress_test_rmse.png")

sim = ULASimulator(M=M, fc=fc)
bf  = Beamformer(sim)
X, _, _ = sim.generate_signal(TRUE_THETAS, snr_db=FIXED_SNR, n_snapshots=FIXED_SNAPSHOTS)
R = bf.compute_covariance(X)

spec_das   = bf.run_das(R, SCAN_ANGLES)
spec_mvdr  = bf.run_mvdr(R, SCAN_ANGLES)
spec_music = bf.run_music(R, SCAN_ANGLES, NUM_SOURCES)

def to_db(s):
    s = np.array(s, dtype=float)
    s /= s.max()
    return 20 * np.log10(s + 1e-12)

fig2, axes2 = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
fig2.suptitle(
    f"Spatial Spectra  (sources at {TRUE_THETAS}°, SNR={FIXED_SNR} dB, N={FIXED_SNAPSHOTS})",
    fontsize=13, fontweight="bold"
)

for ax, spec, label, color in zip(
    axes2,
    [spec_das, spec_mvdr, spec_music],
    ["DaS", "MVDR (Capon)", "MUSIC"],
    [COLORS["das"], COLORS["mvdr"], COLORS["music"]],
):
    ax.plot(SCAN_ANGLES, to_db(spec), color=color, linewidth=1.5)
    for theta in TRUE_THETAS:
        ax.axvline(theta, color="red", linestyle="--", linewidth=1, alpha=0.7,
                   label=f"True DoA {theta}°" if theta == TRUE_THETAS[0] else None)
    ax.set_ylabel("Normalised Power (dB)", fontsize=11)
    ax.set_title(label, fontsize=12)
    ax.set_ylim(-60, 3)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9)

axes2[-1].set_xlabel("Angle (degrees)", fontsize=12)
plt.tight_layout()
fig2.savefig("spectra_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: spectra_comparison.png")

print("\nAll done.")

print("\nGenerating Figure 3: Spatial spectra at varying SNR levels ...")

SNR_VISUAL = [-10, 0, 10, 20]
N_SNAP_VIS = FIXED_SNAPSHOTS

fig3, axes3 = plt.subplots(len(SNR_VISUAL), 3, figsize=(15, 4 * len(SNR_VISUAL)), sharex=True)
fig3.suptitle(
    f"Spatial Spectra vs SNR  (sources at {TRUE_THETAS}°, N_snapshots={N_SNAP_VIS})",
    fontsize=14, fontweight="bold"
)

col_labels = ["DaS", "MVDR (Capon)", "MUSIC"]
col_keys   = ["das", "mvdr", "music"]

for row_idx, snr in enumerate(SNR_VISUAL):
    sim_v = ULASimulator(M=M, fc=fc)
    bf_v  = Beamformer(sim_v)
    X_v, _, _ = sim_v.generate_signal(TRUE_THETAS, snr_db=snr, n_snapshots=N_SNAP_VIS)
    R_v = bf_v.compute_covariance(X_v)

    spectra_v = {
        "das":   bf_v.run_das(R_v, SCAN_ANGLES),
        "mvdr":  bf_v.run_mvdr(R_v, SCAN_ANGLES),
        "music": bf_v.run_music(R_v, SCAN_ANGLES, NUM_SOURCES),
    }

    for col_idx, (key, label) in enumerate(zip(col_keys, col_labels)):
        ax = axes3[row_idx, col_idx]
        ax.plot(SCAN_ANGLES, to_db(spectra_v[key]), color=COLORS[key], linewidth=1.5)
        for theta in TRUE_THETAS:
            ax.axvline(theta, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_ylim(-60, 3)
        ax.grid(True, alpha=0.35)

        if row_idx == 0:
            ax.set_title(label, fontsize=12, fontweight="bold")
        if col_idx == 0:
            ax.set_ylabel(f"SNR = {snr:+d} dB\nPower (dB)", fontsize=10)
        if row_idx == len(SNR_VISUAL) - 1:
            ax.set_xlabel("Angle (degrees)", fontsize=10)

plt.tight_layout()
fig3.savefig("spectra_vs_snr.png", dpi=150, bbox_inches="tight")
print("Saved: spectra_vs_snr.png")

print("\nAll done.")