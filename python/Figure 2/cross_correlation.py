import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import sem
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import significance_stars


def separate_trials(mat_file_path, patient_id):
    """Split trials by correctness and return load labels."""
    np.random.seed(20250710)

    mat_data = scipy.io.loadmat(mat_file_path)
    neural_data = mat_data['neural_data'][0]

    patient_ids   = [int(e['patient_id'][0][0]) for e in neural_data]
    firing_rates  = [e['firing_rates'] for e in neural_data]
    correctness   = [e['trial_correctness'] for e in neural_data]
    time_fields   = [int(e['time_field'][0][0]) - 1 for e in neural_data]
    trial_loads   = [e['trial_load'] for e in neural_data]

    patient_id = int(patient_id)

    sel = [
        (firing_rates[i], correctness[i], time_fields[i], trial_loads[i])
        for i, pid in enumerate(patient_ids)
        if pid == patient_id
    ]
    if not sel:
        return None, None, None, None, None

    fr_sel   = [np.asarray(x[0]) for x in sel]
    perf_sel = [np.asarray(x[1]).flatten() for x in sel]
    tf_sel   = [x[2] for x in sel]

    per_trial_load = np.asarray(sel[0][3]).flatten().astype(int)
    trial_data = np.stack(fr_sel, axis=1)
    trial_correct = np.all(np.stack(perf_sel, axis=0), axis=0).astype(int)

    correct_mask   = trial_correct == 1
    incorrect_mask = trial_correct == 0

    return (
        trial_data[correct_mask],
        trial_data[incorrect_mask],
        tf_sel,
        per_trial_load[correct_mask],
        per_trial_load[incorrect_mask],
    )


def process_trial_and_compute_mean_correlation(trial_data, time_fields):
    """Align neurons by time fields and compute mean pairwise correlation."""
    n_neurons, n_bins = trial_data.shape
    shifted = np.zeros_like(trial_data)
    mid_bin = n_bins // 2

    for i, tf in enumerate(time_fields):
        shifted[i] = np.roll(trial_data[i], mid_bin - tf)

    corrs = []
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            if np.std(shifted[i]) == 0 or np.std(shifted[j]) == 0:
                continue
            c = np.corrcoef(shifted[i], shifted[j])[0, 1]
            if not np.isnan(c):
                corrs.append(c)

    return np.mean(corrs) if corrs else 0.0


def analyze_cross_correlation_all_patients(
    mat_file_path,
    min_time_cells=5,
    min_incorrect_trials=0,
):
    """Pool cross-correlations across patients meeting inclusion criteria."""
    mat_data = scipy.io.loadmat(mat_file_path)
    neural_data = mat_data['neural_data'][0]

    patient_ids = np.unique([int(e['patient_id'][0][0]) for e in neural_data])

    correct_vals, incorrect_vals = [], []
    correct_loads, incorrect_loads = [], []
    passing = []

    for pid in tqdm(patient_ids, desc="Processing patients"):
        ct, it, tf, cl, il = separate_trials(mat_file_path, pid)
        if ct is None:
            continue
        if len(tf) < min_time_cells:
            continue
        if it.shape[0] < min_incorrect_trials:
            continue

        passing.append(pid)

        for k in range(ct.shape[0]):
            correct_vals.append(
                process_trial_and_compute_mean_correlation(ct[k], tf)
            )
            correct_loads.append(int(cl[k]))

        for k in range(it.shape[0]):
            incorrect_vals.append(
                process_trial_and_compute_mean_correlation(it[k], tf)
            )
            incorrect_loads.append(int(il[k]))

    print(f"Patients passing criteria: {passing}")
    print(f"Correct trials pooled:   {len(correct_vals)}")
    print(f"Incorrect trials pooled: {len(incorrect_vals)}")

    return (
        np.asarray(correct_vals),
        np.asarray(incorrect_vals),
        np.asarray(correct_loads),
        np.asarray(incorrect_loads),
    )


def resample_and_plot_cross_correlation_unbalanced(
    correct_cross_correlations,
    incorrect_cross_correlations,
    min_time_cells,
    num_iterations,
):
    """Resample correct trials without load matching."""
    incorrect_mean = np.mean(incorrect_cross_correlations)
    mean_dist = []

    for _ in tqdm(range(num_iterations), desc="Resampling (unbalanced)"):
        sampled = np.random.choice(
            correct_cross_correlations,
            size=len(incorrect_cross_correlations),
            replace=False
        )
        mean_dist.append(np.mean(sampled))

    mean_dist = np.asarray(mean_dist)

    percentile = np.mean(mean_dist < incorrect_mean) * 100.0
    p_value = percentile / 100.0

    star = significance_stars(p_value)

    plt.figure(figsize=(8, 5))
    plt.hist(mean_dist, bins=30, alpha=0.7,
             color="blue", edgecolor="black",
             label="Resampled Correct Mean")

    plt.axvline(incorrect_mean, color="red",
                linestyle="--", linewidth=3,
                label="Incorrect Mean")

    y_upper = plt.ylim()[1]
    plt.text(incorrect_mean, 0.8 * y_upper,
             star, color="red",
             fontsize=15, ha="center")

    plt.title("Mean Cross-Correlation Values (Unbalanced Resampling)")
    plt.xlabel("Mean Cross-Correlation")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"[Unbalanced] Percent < incorrect mean: {percentile:.2f}%")

    return mean_dist, p_value


def resample_and_plot_cross_correlation_balanced(
    correct_cross_correlations,
    incorrect_cross_correlations,
    min_time_cells,
    num_iterations,
    *,
    correct_loads,
    incorrect_loads,
):
    """Resample correct trials matching the incorrect-load distribution."""
    correct   = np.asarray(correct_cross_correlations, dtype=float)
    incorrect = np.asarray(incorrect_cross_correlations, dtype=float)
    cl = np.asarray(correct_loads, dtype=int)
    il = np.asarray(incorrect_loads, dtype=int)

    if len(correct) != len(cl):
        raise ValueError("Mismatch: correct values vs correct_loads.")
    if len(incorrect) != len(il):
        raise ValueError("Mismatch: incorrect values vs incorrect_loads.")

    strata = {L: np.where(cl == L)[0] for L in (1, 2, 3)}
    load_counts = Counter(il.tolist())
    incorrect_mean = incorrect.mean()

    rng = np.random.default_rng(20250710)
    mean_dist = []

    for _ in tqdm(range(num_iterations), desc="Resampling (load-balanced)"):
        chosen = []

        for L, k in load_counts.items():
            if k == 0:
                continue
            pool = strata.get(L, np.array([], dtype=int))
            if pool.size == 0:
                continue
            replace = pool.size < k
            chosen.append(rng.choice(pool, size=k, replace=replace))

        if not chosen:
            continue

        sampled = correct[np.concatenate(chosen)]
        mean_dist.append(sampled.mean())

    mean_dist = np.asarray(mean_dist)

    percentile = np.mean(mean_dist < incorrect_mean) * 100.0
    p_value = percentile / 100.0
    star = significance_stars(p_value)

    plt.figure(figsize=(8, 5))
    plt.hist(mean_dist, bins=30, alpha=0.7,
             color="blue", edgecolor="black",
             label="Resampled Correct Mean (Load-Matched)")

    plt.axvline(incorrect_mean, color="red",
                linestyle="--", linewidth=3,
                label=f"Incorrect Mean = {incorrect_mean:.4f}")

    y_upper = plt.ylim()[1]
    plt.text(incorrect_mean, 0.8 * y_upper,
             star, color="red",
             fontsize=15, ha="center")

    plt.title("Mean Cross-Correlation Values (Load-Matched Resampling)")
    plt.xlabel("Mean Cross-Correlation")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"[Balanced] Percent < incorrect mean: {percentile:.2f}%")

    return mean_dist, p_value


def run_cross_correlation_resampling(
    mode,
    correct_vals,
    incorrect_vals,
    min_time_cells,
    num_iterations,
    *,
    correct_loads=None,
    incorrect_loads=None,
):
    """Run unbalanced or balanced resampling analysis."""
    if mode == "unbalanced":
        return resample_and_plot_cross_correlation_unbalanced(
            correct_vals,
            incorrect_vals,
            min_time_cells,
            num_iterations,
        )

    if mode == "balanced":
        return resample_and_plot_cross_correlation_balanced(
            correct_vals,
            incorrect_vals,
            min_time_cells,
            num_iterations,
            correct_loads=correct_loads,
            incorrect_loads=incorrect_loads,
        )

    raise ValueError("mode must be 'unbalanced' or 'balanced'")


if __name__ == "__main__":

    correct_vals, incorrect_vals, correct_loads, incorrect_loads = \
        analyze_cross_correlation_all_patients(
            "TC.mat",
            min_time_cells=5,
        )

    run_cross_correlation_resampling(
        mode="unbalanced",
        correct_vals=correct_vals,
        incorrect_vals=incorrect_vals,
        min_time_cells=5,
        num_iterations=1000,
    )

    run_cross_correlation_resampling(
        mode="balanced",
        correct_vals=correct_vals,
        incorrect_vals=incorrect_vals,
        min_time_cells=5,
        num_iterations=1000,
        correct_loads=correct_loads,
        incorrect_loads=incorrect_loads,
    )
