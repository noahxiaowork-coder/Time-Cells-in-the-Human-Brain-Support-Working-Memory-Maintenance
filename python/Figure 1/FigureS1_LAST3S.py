#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bayesian time‑bin decoder – last‑3‑seconds, stim_cat == 2
Author  : <your name>
Updated : 17‑Jul‑2025
"""

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import numpy as np
import scipy.io as sio
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from scipy.stats import sem, ttest_ind
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
FIG_DIR = Path("/Users/xiangxuankong/Desktop/Human TC Support WM/Figures/Figure S1")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 1) Core decoding function
# ---------------------------------------------------------------------
def decode_dataset(
        mat_filepath: str | Path,
        *,
        bin_size: float,
        time_bin_range: tuple[int, int],
        stim_category: int,
        test_size: float = 0.3,
        random_state: int | None = 20250710,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, GaussianNB, float]:
    """
    Returns
    -------
    y_test, y_pred, errors, time_centers, X_test, clf, bin_size
    """
    # ---- Load ------------------------------------------------------
    data = sio.loadmat(mat_filepath, squeeze_me=True)
    neurons = data["neural_data"]                      # (n_neurons,)

    # ---- Trial selection ------------------------------------------
    stim_cat = np.asarray(neurons[0]["stim_cat"]).ravel()
    keep_idx = np.where(stim_cat == stim_category)[0]

    # ---- Stack to (bins × trials × neurons) -----------------------
    fr_list = [np.asarray(n["firing_rates"])[keep_idx, :] for n in neurons]
    min_trials = min(fr.shape[0] for fr in fr_list)
    fr_list   = [fr[:min_trials, :] for fr in fr_list]

    reshaped = np.stack(fr_list, axis=2).transpose(1, 0, 2)  # bins × trials × neurons

    # ---- Crop last‑3‑seconds window -------------------------------
    start_bin, end_bin = time_bin_range            # Python slice: end_bin excluded
    reshaped = reshaped[start_bin:end_bin, :, :]
    bins, trials, n_neurons = reshaped.shape

    # ---- Labels ----------------------------------------------------
    labels = np.tile(np.arange(start_bin, end_bin).reshape(-1, 1), (1, trials))

    # ---- Train / test split by trial ------------------------------
    tr_trials, te_trials = train_test_split(
        np.arange(trials), test_size=test_size,
        shuffle=True, random_state=random_state
    )

    X_train = reshaped[:, tr_trials, :].reshape(-1, n_neurons).astype(np.float32)
    y_train = labels[:,  tr_trials].ravel()

    X_test  = reshaped[:, te_trials, :].reshape(-1, n_neurons).astype(np.float32)
    y_test  = labels[:,  te_trials].ravel()

    # ---- Fit & predict --------------------------------------------
    clf    = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    errors = bin_size * np.abs(y_pred - y_test)
    time_centers = (np.arange(start_bin, end_bin) + 0.5) * bin_size

    return y_test, y_pred, errors, time_centers, X_test, clf, bin_size


# ---------------------------------------------------------------------
# 2) Shuffle baseline
# ---------------------------------------------------------------------
def shuffled_baseline(clf, X_test, y_test, *, n_shuffles, bin_size, rng):
    unique_bins   = np.unique(y_test)
    baseline_dist = np.empty(n_shuffles)
    bin_means     = np.empty((n_shuffles, unique_bins.size))

    y_pred_const = clf.predict(X_test)
    for i in range(n_shuffles):
        y_test_shuf = rng.permutation(y_test)
        dec_err     = bin_size * np.abs(y_pred_const - y_test_shuf)
        baseline_dist[i] = dec_err.mean()
        for j, b in enumerate(unique_bins):
            bin_means[i, j] = dec_err[y_test_shuf == b].mean()
    return baseline_dist, bin_means


# ---------------------------------------------------------------------
# 3) Plot helpers
# ---------------------------------------------------------------------
def sig_from_p(p): return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < .05 else "ns"

def add_sig(ax, i, j, y, p):
    ax.plot([i+.05, j-.05], [y, y], color="k")
    ax.text((i+j)/2, y, sig_from_p(p), ha="center", va="bottom")


# ---------------------------------------------------------------------
# 4) Main pipeline
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # ---- Analysis constants ---------------------------------------
    BIN_SIZE       = 0.1                         # 100 ms
    TOTAL_DURATION = 7.0                         # s
    LAST_SECONDS   = 3.0                         # analyse final 3 s
    START_BIN      = int((TOTAL_DURATION - LAST_SECONDS) / BIN_SIZE)   # 40
    END_BIN        = int(TOTAL_DURATION / BIN_SIZE)                    # 70
    INTERVAL       = (START_BIN, END_BIN)       # (40, 70)
    KEEP_STIMCAT   = 2
    N_SHUFFLES     = 1_000
    RNG_SEED       = 20250710
    RNG            = np.random.default_rng(RNG_SEED)

    # ---- Data files -----------------------------------------------
    TC_FILE  = "Figure 1/HB3s_data.mat"
    NTC_FILE = "Figure 1/HB3s_data_non.mat"

    # ---- Time‑cell population -------------------------------------
    yT, yT_hat, errT, tc, XT, clfT, _ = decode_dataset(
        TC_FILE,
        bin_size=BIN_SIZE,
        time_bin_range=INTERVAL,
        stim_category=KEEP_STIMCAT,
        random_state=RNG_SEED,
    )
    blT, blT_bins = shuffled_baseline(clfT, XT, yT,
                                      n_shuffles=N_SHUFFLES,
                                      bin_size=BIN_SIZE, rng=RNG)

    # ---- Non‑time‑cell population ---------------------------------
    yN, yN_hat, errN, nc, XN, clfN, _ = decode_dataset(
        NTC_FILE,
        bin_size=BIN_SIZE,
        time_bin_range=INTERVAL,
        stim_category=KEEP_STIMCAT,
        random_state=RNG_SEED,
    )
    blN, blN_bins = shuffled_baseline(clfN, XN, yN,
                                      n_shuffles=N_SHUFFLES,
                                      bin_size=BIN_SIZE, rng=RNG)

    # ---- Per‑bin stats -------------------------------------------
    ubins       = np.arange(*INTERVAL)
    mean_errT   = np.array([errT[yT == b].mean() for b in ubins])
    sem_errT    = np.array([sem(errT[yT == b])    for b in ubins])
    mean_errN   = np.array([errN[yN == b].mean() for b in ubins])
    sem_errN    = np.array([sem(errN[yN == b])    for b in ubins])
    bl_meanT    = blT_bins.mean(axis=0)
    bl_semT     = sem(blT_bins, axis=0)

    # ---- Curve plot ----------------------------------------------
    plt.figure(figsize=(9, 5))
    plt.plot(tc, mean_errT, "-o", label="Time cells", lw=2)
    plt.fill_between(tc, mean_errT-sem_errT, mean_errT+sem_errT, alpha=.2)
    plt.plot(nc, mean_errN, "-o", label="Non‑time‑cells", color="green", lw=2)
    plt.fill_between(nc, mean_errN-sem_errN, mean_errN+sem_errN,
                     color="green", alpha=.2)
    plt.plot(tc, bl_meanT, "-o", color="grey", label="Shuffle", lw=2)
    plt.fill_between(tc, bl_meanT-bl_semT, bl_meanT+bl_semT,
                     color="grey", alpha=.15)

    plt.xlabel("Time (s)")
    plt.ylabel("Decoding error (s)")
    plt.title("Bayesian decoder · last 3 s")
    plt.xlim(TOTAL_DURATION-LAST_SECONDS, TOTAL_DURATION)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR/"Bayesian_Curve_last3s.svg", bbox_inches="tight")
    plt.show()

    # ---- Bar plot -------------------------------------------------
    groups  = [errT, errN, blT]
    labels  = ["Time cells", "Non‑time‑cells", "Shuffle"]
    means   = [g.mean() for g in groups]
    errors  = [sem(g)   for g in groups]

    fig, ax = plt.subplots(figsize=(5, 7))
    ax.bar(labels, means, yerr=errors,
           color=["blue", "green", "grey"], capsize=4, alpha=.85)
    ax.set_ylabel("Mean decoding error (s)")
    ax.set_title("Decoding performance · last 3 s")
    ymax = max(means)+max(errors)
    ax.set_ylim(0, ymax+.15)

    for k, (i, j) in enumerate([(0,1), (1,2), (0,2)], 1):
        _, p = ttest_ind(groups[i], groups[j], equal_var=False)
        add_sig(ax, i, j, ymax+.03*k, p)

    plt.tight_layout()
    plt.savefig(FIG_DIR/"Bayesian_Bar_last3s.svg", bbox_inches="tight")
    plt.show()
