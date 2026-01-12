#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian time-bin decoder for neural population activity

Author  : <your name>
Updated : 17-Jul-2025
"""

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import numpy as np
import scipy.io as sio
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score  # (kept for possible extensions)
from scipy.stats import sem, ttest_ind, ttest_rel
import matplotlib.pyplot as plt

# If you really need seaborn aesthetics uncomment:
# import seaborn as sns; sns.set_context("poster", rc={"lines.linewidth": 2})

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
FIG_DIR = Path("/Users/xiangxuankong/Desktop/Human TC Support WM/Figures/Figure S1").expanduser()
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 1) Core decoding function
# ---------------------------------------------------------------------
def decode_dataset(
    mat_filepath: str | Path,
    test_size: float = 0.3,
    random_state: int | None = 20250710,
    *,
    total_duration: float | None = 2.5,
    bin_size: float | None = None,
    time_bin_range: tuple[int, int] | None = None,
    stim_values: int | list[int] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, GaussianNB, float]:
    """
    Parameters
    ----------
    mat_filepath : str | Path
        .mat file that contains a variable neural_data, an (1 × n_neurons) MATLAB struct array
        whose field firing_rates is (#trials × #time_bins).
    test_size : float
        Fraction of trials reserved for testing.
    random_state : int | None
        Seed for all stochastic steps.
    total_duration : float | None
        Recording duration in seconds **if** bin_size is unknown.
    bin_size : float | None
        Explicit time-bin width in seconds. If provided, overrides total_duration.

    Returns
    -------
    y_test, y_pred : np.ndarray
        True and predicted bin labels for each *test sample* (flattened).
    errors : np.ndarray
        Absolute decoding error in seconds for every test sample.
    time_centers : np.ndarray
        Center time of each bin (length = #unique_bins).
    X_test : np.ndarray
        Test set firing-rate matrix for later shuffling (samples × neurons).
    clf : GaussianNB
        Trained classifier.
    bin_size : float
        Time width of a single bin (seconds) for reference downstream.
    """
    rng = np.random.default_rng(random_state)

    # ---- (i) Load MATLAB file ----
    data = sio.loadmat(mat_filepath, squeeze_me=True)
    neural_data_struct = data["neural_data"]  # shape (n_neurons,) after squeeze_me

    # ---- (ii) Extract & truncate to equal trials ----
    firing_rates_list = [np.asarray(neuron["firing_rates"]) for neuron in neural_data_struct]
    min_trials = min(fr.shape[0] for fr in firing_rates_list)
    truncated = [fr[:min_trials, :] for fr in firing_rates_list]

    # ---- (iii) Stack to (time_bins, trials, neurons) ----
    reshaped = np.stack(truncated, axis=2)  # (trials × bins × neurons)
    reshaped = reshaped.transpose(1, 0, 2)  # (bins × trials × neurons)

    if time_bin_range is not None:
        start_bin, end_bin = time_bin_range
        reshaped = reshaped[start_bin:end_bin, :, :]

    time_bins = end_bin - start_bin if time_bin_range else reshaped.shape[0]
    time_bins, trials, neurons = reshaped.shape

    # ---- (iv) Determine temporal resolution ----
    if bin_size is not None and total_duration is not None:
        raise ValueError("Specify only one of total_duration or bin_size.")
    if bin_size is None:
        assert total_duration is not None, "Need either total_duration or bin_size."
        bin_size = total_duration / time_bins
    else:
        # user supplied bin_size
        total_duration = bin_size * time_bins
    # DEBUG check: 100 ms bins for your 7 s example ➜ 70 bins
    # assert np.isclose(bin_size, 0.1), "Unexpected bin width!"

    # ---- (v) Generate labels (0 … time_bins-1) per trial ----
    labels = np.tile(np.arange(time_bins).reshape(-1, 1), (1, trials))

    # ---- (vi) Trial-wise split ----
    all_trials = np.arange(trials)
    train_trials, test_trials = train_test_split(
        all_trials,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    X_train = reshaped[:, train_trials, :].reshape(-1, neurons)
    y_train = labels[:, train_trials].ravel()
    X_test  = reshaped[:, test_trials,  :].reshape(-1, neurons)
    y_test  = labels[:, test_trials].ravel()

    # ---- (vii) Fit GNB & predict ----
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # ---- (viii) Compute decoding error in seconds ----
    errors = bin_size * np.abs(y_pred - y_test)

    # ---- (ix) Bin centres for plotting ----
    unique_bins = np.arange(start_bin, end_bin) if time_bin_range else np.arange(time_bins)
    time_centers = unique_bins * bin_size + bin_size / 2

    print("X_train shape (samples, neurons):", X_train.shape)
    print("X_test shape (samples, neurons):", X_test.shape)
    print("Unique y_test bins:", np.unique(y_test))
    assert np.unique(y_test).min() == 0 and np.unique(y_test).max() == 39

    return y_test, y_pred, errors, time_centers, X_test, clf, bin_size


# ---------------------------------------------------------------------
# 2) Convenience: shuffle-baseline builder
# ---------------------------------------------------------------------
def shuffled_baseline(
    clf: GaussianNB,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_shuffles: int = 1_000,
    rng: np.random.Generator | None = None,
    bin_size: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    baseline_dist : np.ndarray
        Distribution (length = n_shuffles) of mean decoding errors.
    all_bin_means : np.ndarray
        Shape (n_shuffles × #bins); mean error per bin for each shuffle.
    """
    rng = np.random.default_rng() if rng is None else rng
    unique_bins = np.unique(y_test)
    baseline_dist = np.empty(n_shuffles)
    all_bin_means = np.empty((n_shuffles, unique_bins.size))

    for i in range(n_shuffles):
        y_test_shuf = rng.permutation(y_test)
        y_pred_shuf = clf.predict(X_test)
        dec_err = bin_size * np.abs(y_pred_shuf - y_test_shuf)
        baseline_dist[i] = dec_err.mean()
        # per-bin means
        for j, b in enumerate(unique_bins):
            all_bin_means[i, j] = dec_err[y_test == b].mean()

    return baseline_dist, all_bin_means


# ---------------------------------------------------------------------
# 3) Significance helpers
# ---------------------------------------------------------------------
def significance_from_p(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"

def add_sig_bar(ax, x1, x2, y, p, lw=2):
    ax.plot([x1 + 0.05, x2 - 0.05], [y, y], color="k", lw=lw)
    ax.text((x1 + x2) / 2, y, significance_from_p(p), ha="center", va="bottom", fontsize=11)

def per_bin_ttests(
    err_a: np.ndarray, y_a: np.ndarray,
    err_b: np.ndarray, y_b: np.ndarray,
    bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch's t-test for each time bin between two error curves.

    Returns
    -------
    t_vals : np.ndarray  shape (#bins,)
    p_vals : np.ndarray  shape (#bins,)
    """
    t_vals = np.empty(len(bins))
    p_vals = np.empty(len(bins))
    for i, b in enumerate(bins):
        ea = err_a[y_a == b]
        eb = err_b[y_b == b]
        # Welch's t-test (unequal variances); fall back if one side is empty
        if ea.size < 2 or eb.size < 2:
            t_vals[i] = np.nan
            p_vals[i] = np.nan
        else:
            t, p = ttest_ind(ea, eb, equal_var=False)
            t_vals[i], p_vals[i] = t, p
    return t_vals, p_vals


# ---------------------------------------------------------------------
# 4) Pipeline execution (EDIT paths as needed)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # ---------------------------------------------------------------
    # Analysis parameters
    # ---------------------------------------------------------------
    RNG_SEED = 20250710
    TOTAL_DURATION = 7.0  # full recording (s)
    INTERVAL_BINS = (0, 40)  # first 4 s → bins 0–39 for 100 ms bins
    N_SHUFFLES = 1_000

    # ---------------------------------------------------------------
    # Data files
    # ---------------------------------------------------------------
    TC_FILE = "Figure 1/NBHB4s_data.mat"        # time-cell population
    NTC_FILE = "Figure 1/NBHB4s_data_non.mat"   # non-time-cell population

    # ---------------------------------------------------------------
    # A) Decode the TIME-CELL population
    # ---------------------------------------------------------------
    (y_test_TC, y_pred_TC, err_TC, tcent_TC, X_test_TC, clf_TC, bin_TC) = decode_dataset(
        TC_FILE,
        test_size=0.3,
        random_state=RNG_SEED,
        bin_size=0.1,           # <-- explicit bin size
        total_duration=None,
        time_bin_range=INTERVAL_BINS
    )

    bl_dist_TC, bl_bin_TC = shuffled_baseline(
        clf_TC, X_test_TC, y_test_TC,
        n_shuffles=N_SHUFFLES,
        rng=np.random.default_rng(RNG_SEED),
        bin_size=bin_TC
    )

    # Per-bin means / SEM
    unique_bins = np.arange(*INTERVAL_BINS)
    mean_err_TC = np.array([err_TC[y_test_TC == b].mean() for b in unique_bins])
    sem_err_TC  = np.array([sem(err_TC[y_test_TC == b]) for b in unique_bins])

    # ---------------------------------------------------------------
    # B) Decode the NON-TIME-CELL population
    # ---------------------------------------------------------------
    (y_test_NT, y_pred_NT, err_NT, tcent_NT, X_test_NT, clf_NT, bin_NT) = decode_dataset(
        NTC_FILE,
        test_size=0.3,
        random_state=RNG_SEED,
        bin_size=0.1,           # <-- explicit bin size
        total_duration=None,
        time_bin_range=INTERVAL_BINS
    )

    bl_dist_NT, bl_bin_NT = shuffled_baseline(
        clf_NT, X_test_NT, y_test_NT,
        n_shuffles=N_SHUFFLES,
        rng=np.random.default_rng(RNG_SEED),
        bin_size=bin_NT
    )

    mean_err_NT = np.array([err_NT[y_test_NT == b].mean() for b in unique_bins])
    sem_err_NT  = np.array([sem(err_NT[y_test_NT == b]) for b in unique_bins])

    # ---------------------------------------------------------------
    # C) Bin-wise Welch t-tests between curves (Time vs Non-time)
    # ---------------------------------------------------------------
    t_vals_bin, p_vals_bin = per_bin_ttests(err_TC, y_test_TC, err_NT, y_test_NT, unique_bins)

    # ---------------------------------------------------------------
    # D) Plot bin-wise decoding-error curves + per-bin significance
    # ---------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    # curves
    plt.plot(tcent_TC, mean_err_TC, "-o", label="Time cells", lw=2)
    plt.fill_between(tcent_TC, mean_err_TC - sem_err_TC, mean_err_TC + sem_err_TC, alpha=0.20)

    plt.plot(tcent_NT, mean_err_NT, "-o", label="Non-time-cells", color="green", lw=2)
    plt.fill_between(tcent_NT, mean_err_NT - sem_err_NT, mean_err_NT + sem_err_NT, color="green", alpha=0.20)

    # shuffle band for the TC decoder (optional reference)
    bl_mean_TC = bl_bin_TC.mean(axis=0)
    bl_sem_TC  = sem(bl_bin_TC, axis=0)
    plt.plot(tcent_TC, bl_mean_TC, "-o", color="grey", label="Shuffle baseline", lw=2)
    plt.fill_between(tcent_TC, bl_mean_TC - bl_sem_TC, bl_mean_TC + bl_sem_TC, color="grey", alpha=0.15)

    # per-bin significance stars on top of the TIME-CELL curve
    # star is drawn only when p < 0.05; text shows *, **, or ***
    margin = 0.02  # vertical padding in y-axis units (seconds)
    for i, (tc, m_tc, se_tc, p) in enumerate(zip(tcent_TC, mean_err_TC, sem_err_TC, p_vals_bin)):
        if np.isfinite(p) and p < 0.05:
            y_star = m_tc + se_tc + margin
            plt.text(tc, y_star, significance_from_p(p), ha="center", va="bottom", fontsize=11)

    plt.xlabel("Time (s)")
    plt.ylabel("Decoding error (s)")
    plt.title("Bayesian decoder · first 4 s of each trial")
    plt.xlim(0, 4)
    plt.legend(frameon=False)
    plt.tight_layout()
    #plt.savefig(FIG_DIR / "Bayesian_Curve_0-4s_with_binwise_stars.svg", bbox_inches="tight")
    plt.show()

    # ---------------------------------------------------------------
    # E) Aggregate bar-plot & pair-wise statistics
    # ---------------------------------------------------------------
    groups = [err_TC, err_NT, bl_dist_TC]
    group_labels = ["Time cells", "Non-time-cells", "Shuffle"]
    group_means = [g.mean() for g in groups]
    group_sems  = [sem(g) for g in groups]

    fig, ax = plt.subplots(figsize=(5, 7))
    ax.bar(group_labels, group_means, yerr=group_sems, color=["blue", "green", "grey"], capsize=4, alpha=0.85)
    ax.set_ylabel("Mean decoding error (s)")
    ax.set_title("Decoding performance · first 4 s")

    ymax = max(group_means) + max(group_sems)
    ax.set_ylim(0, ymax + 0.15)

    # Welch tests + significance bars
    pairs = [(0, 1), (1, 2), (0, 2)]
    pvals_bar = []
    for k, (i, j) in enumerate(pairs, 1):
        t_stat, p_val = ttest_ind(groups[i], groups[j], equal_var=False)
        y = ymax + 0.03 * k
        add_sig_bar(ax, i, j, y, p_val)
        pvals_bar.append(((group_labels[i], group_labels[j]), p_val))

    plt.tight_layout()
    plt.show()

    # --- Print results ---
    print("Per-bin p-values (first 10 bins):", np.array2string(p_vals_bin[:10], precision=3, separator=", "))
    for (label_i, label_j), p in pvals_bar:
        print(f"Bar chart p-value {label_i} vs {label_j}: {p:.4g}")

    print("First/last time centers:", tcent_TC[0], tcent_TC[-1])
