import numpy as np
import scipy.io
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ks_2samp, ttest_rel
import seaborn as sns
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import significance_stars, add_significance_bar, clean_region_name

DIR = ''


def separate_trials_by_load(mat_file_path, patient_id, *, use_correct=False):
    """
    Return trials separated by load (1, 2, 3) for a specific patient.
    
    Returns:
        load_dict: {1: trials_array, 2: trials_array, 3: trials_array}
        time_fields: list of time field indices for each neuron
    """
    m = scipy.io.loadmat(mat_file_path)
    nd = m['neural_data'][0]

    # Extract data for all neurons
    pids = [int(e['patient_id'][0][0]) for e in nd]
    frs = [e['firing_rates'] for e in nd]
    loads = [e['trial_load'] for e in nd]
    corrs = [e['trial_correctness'] for e in nd]
    tfs = [int(e['time_field'][0][0]) - 1 for e in nd]

    # Filter for this patient
    chosen = [(fr, ld, cr, tf) for fr, ld, cr, tf, pid in
              zip(frs, loads, corrs, tfs, pids) if pid == patient_id]
    if not chosen:
        return None, None

    fr_stack = np.stack([np.asarray(c[0]) for c in chosen], axis=1)  # trials × neurons × bins
    load_vec = np.asarray(chosen[0][1]).flatten()
    corr_vec = np.asarray(chosen[0][2]).flatten()
    time_fields = [c[3] for c in chosen]

    load_dict = {}
    for lv in (1, 2, 3):
        mask = (load_vec == lv)
        if use_correct:
            mask &= (corr_vec == 1)
        load_dict[lv] = fr_stack[mask]

    return load_dict, time_fields


def process_trial_and_compute_mean_correlation(trial_data, time_fields):
    """
    Compute mean pairwise correlation for a single trial after time-field alignment.
    
    Args:
        trial_data: neurons × time_bins array
        time_fields: list of time field indices for each neuron
    """
    num_neurons, n_bins = trial_data.shape
    
    # Remove neurons with zero variance
    idx = [i for i in range(num_neurons) if np.std(trial_data[i]) != 0]
    if len(idx) < 2:
        return 0.0

    data = trial_data[idx]
    fields = [time_fields[i] for i in idx]
    
    # Align neurons by shifting to center their time fields
    mid = n_bins // 2
    shifted = np.zeros_like(data)
    for i, tf in enumerate(fields):
        shifted[i] = np.roll(data[i], mid - tf)

    # Compute all pairwise correlations
    corrs = []
    for i in range(len(shifted)):
        for j in range(i + 1, len(shifted)):
            if np.std(shifted[i]) == 0 or np.std(shifted[j]) == 0:
                continue
            v = np.corrcoef(shifted[i], shifted[j])[0, 1]
            if not np.isnan(v):
                corrs.append(v)

    if not corrs:
        return 0.0
    
    return float(np.mean(corrs))


def analyze_load_distributions_PSMA_final(
    mat_file_path, *,
    min_time_cells=5,
    use_correct=False,
    verbose=True
):
    """
    Analyze how memory load affects neural correlations in PSMA (pre-supplementary motor area).
    
    For each patient:
      - Gather PSMA neurons as time cells
      - Compute trial-level cross-correlations for each memory load
      - Generate per-patient and pooled statistics
    
    Outputs:
      - Per-patient mean CC plot with paired t-tests
      - CDF plot of pooled trial-level CCs with KS tests
    
    Returns:
        Trial-level CC values pooled across patients for loads 1, 2, 3
    """

    def get_stars(p):
        """Convert p-value to significance notation with actual value for ns."""
        sig = significance_stars(p)
        if sig == "ns":
            return f"p = {p:.3f}"
        return sig

    def add_sig_bar(ax, x1, x2, y, text, height=0.02, bar_width=-0.15):
        """Add a compact significance bar between two positions."""
        ax.plot([x1 - bar_width, x1 - bar_width, x2 + bar_width, x2 + bar_width],
                [y, y + height, y + height, y], lw=1.5, color='black')
        ax.text((x1 + x2) / 2, y + height + 0.005, text,
                ha='center', va='bottom', fontsize=13)

    # Load data
    m = scipy.io.loadmat(mat_file_path)
    nd = m["neural_data"][0]
    patient_ids = [int(e["patient_id"][0][0]) for e in nd]
    unique_ids = np.unique(patient_ids)

    # Gather PSMA data per patient
    data_by_patient = {}
    p_iter = tqdm(unique_ids, desc="Gathering PSMA data") if verbose else unique_ids

    for pid in p_iter:
        # Find all PSMA neurons for this patient
        psma_entries = []
        for entry in nd:
            ep = int(entry["patient_id"][0][0])
            if ep != pid:
                continue
            region_key = clean_region_name(entry["brain_region"])
            if region_key != "pre_supplementary_motor_area":
                continue
            psma_entries.append(entry)

        if len(psma_entries) == 0:
            continue

        # Get trial metadata from first PSMA neuron
        fr0 = psma_entries[0]["firing_rates"]
        loads = psma_entries[0]["trial_load"].flatten().astype(int)
        correct = psma_entries[0]["trial_correctness"].flatten().astype(int)
        n_trials, n_bins = fr0.shape

        # Collect all PSMA neurons (time cells)
        time_cell_frates = []
        t_fields = []
        for entry in psma_entries:
            fr = entry["firing_rates"]
            tf = int(entry["time_field"][0][0]) - 1
            time_cell_frates.append(fr)
            t_fields.append(tf)

        n_cells = len(time_cell_frates)
        if n_cells < min_time_cells:
            if verbose:
                print(f"Skipping patient {pid}: only {n_cells} PSMA time-cells "
                      f"(min required = {min_time_cells}).")
            continue

        t_fields = np.array(t_fields, dtype=int)

        # Filter trials
        if use_correct:
            trial_mask = (correct == 1)
        else:
            trial_mask = np.ones_like(correct, dtype=bool)

        # Separate by load
        load_d = {}
        for lv in (1, 2, 3):
            trial_indices = np.where((loads == lv) & trial_mask)[0]
            if len(trial_indices) == 0:
                load_d[lv] = np.empty((0, n_cells, n_bins), dtype=float)
                continue

            trial_mats = []
            for t_idx in trial_indices:
                trial_matrix = np.vstack([fr[t_idx, :] for fr in time_cell_frates])
                trial_mats.append(trial_matrix)

            load_d[lv] = np.stack(trial_mats, axis=0)

        data_by_patient[pid] = (load_d, t_fields)

    if not data_by_patient:
        raise RuntimeError("No patients qualified under the PSMA filters.")

    # Compute cross-correlations per trial
    all_cc = {1: [], 2: [], 3: []}
    per_patient_cc = {pid: {1: [], 2: [], 3: []} for pid in data_by_patient}

    c_iter = tqdm(data_by_patient.keys(), desc="Computing CCs (PSMA)") if verbose else data_by_patient.keys()

    for pid in c_iter:
        load_d, t_fields = data_by_patient[pid]
        for lv in (1, 2, 3):
            trials_lv = load_d[lv]
            for trial in trials_lv:
                val = process_trial_and_compute_mean_correlation(trial, t_fields)
                all_cc[lv].append(val)
                per_patient_cc[pid][lv].append(val)

    # Per-patient means and paired statistics
    pps_mean = {lv: [] for lv in (1, 2, 3)}
    for pid in per_patient_cc:
        for lv in (1, 2, 3):
            vals = per_patient_cc[pid][lv]
            v = np.mean(vals) if len(vals) > 0 else np.nan
            pps_mean[lv].append(v)

    pps_mean[1] = np.asarray(pps_mean[1], dtype=float)
    pps_mean[2] = np.asarray(pps_mean[2], dtype=float)
    pps_mean[3] = np.asarray(pps_mean[3], dtype=float)

    valid_idx = (~np.isnan(pps_mean[1]) &
                 ~np.isnan(pps_mean[2]) &
                 ~np.isnan(pps_mean[3]))
    for lv in (1, 2, 3):
        pps_mean[lv] = pps_mean[lv][valid_idx]

    if len(pps_mean[1]) > 1:
        _, p12_pat = ttest_rel(pps_mean[1], pps_mean[2], alternative="greater")
        _, p23_pat = ttest_rel(pps_mean[2], pps_mean[3], alternative="greater")
        _, p13_pat = ttest_rel(pps_mean[1], pps_mean[3], alternative="greater")
    else:
        p12_pat = p23_pat = p13_pat = 1.0

    if any(len(all_cc[lv]) < 2 for lv in (1, 2, 3)):
        if verbose:
            print("Too few data points – skipping KS/CDF plots.")
        return all_cc[1], all_cc[2], all_cc[3]

    # KS tests on pooled CCs
    D_12, ks_p_12 = ks_2samp(all_cc[1], all_cc[2])
    D_23, ks_p_23 = ks_2samp(all_cc[2], all_cc[3])
    D_13, ks_p_13 = ks_2samp(all_cc[1], all_cc[3])

    if verbose:
        print(f"[PATIENT LEVEL]  L1>2  p={p12_pat:.4g}  "
              f"L2>3  p={p23_pat:.4g}  L1>3  p={p13_pat:.4g}")
        print(f"[TRIAL LEVEL]   KS L1 vs 2: p={ks_p_12:.4g}, "
              f"L2 vs 3: p={ks_p_23:.4g}, L1 vs 3: p={ks_p_13:.4g}")

    # Plot 1: Per-patient dot plot
    fig, ax = plt.subplots(figsize=(5, 6))
    palette = ["royalblue", "seagreen", "firebrick"]

    n_patients = len(pps_mean[1])
    for i in range(n_patients):
        yvals = [pps_mean[1][i], pps_mean[2][i], pps_mean[3][i]]
        ax.plot([1, 2, 3], yvals, color="grey", alpha=0.5, linewidth=1)
        ax.scatter([1, 2, 3], yvals, color="grey", alpha=0.8, zorder=5)

    for idx, (lv, color) in enumerate(zip([1, 2, 3], palette)):
        if len(pps_mean[lv]) > 0:
            mean_val = np.mean(pps_mean[lv])
            ax.hlines(mean_val, idx + 0.8, idx + 1.2,
                      color=color, linewidth=3, zorder=10)

    ymax = np.nanmax([*pps_mean[1], *pps_mean[2], *pps_mean[3]]) if n_patients > 0 else 0
    add_sig_bar(ax, 1, 2, ymax + 0.01, get_stars(p12_pat))
    add_sig_bar(ax, 2, 3, ymax + 0.01, get_stars(p23_pat))
    add_sig_bar(ax, 1, 3, ymax + 0.04, get_stars(p13_pat))

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Load 1", "Load 2", "Load 3"])
    ax.set_xlabel("Memory Load", fontsize=12)
    ax.set_ylabel("Mean CC per Patient (PSMA)", fontsize=12)
    ax.set_title("Per-Patient Mean CC (PSMA)", fontsize=14)
    ax.set_xlim(0.5, 3.5)

    sns.despine()
    plt.tight_layout()

    path = DIR + "Correct/" if use_correct else DIR
    plt.savefig(path + "PSMA_per_patient_dotplot.svg", format="svg")
    plt.show()

    # Plot 2: Empirical CDF
    fig, ax = plt.subplots(figsize=(9, 6))
    for lv, col in zip((1, 2, 3), palette):
        vals = np.sort(all_cc[lv])
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, cdf, color=col, linewidth=1.8)

    txt = (f"Load1 vs 2: p={ks_p_12:.3g}\n"
           f"Load2 vs 3: p={ks_p_23:.3g}\n"
           f"Load1 vs 3: p={ks_p_13:.3g}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes,
            ha="left", va="top", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7"))

    label_texts = ["Load = 1", "Load = 2", "Load = 3"]
    for i, (label, col) in enumerate(zip(label_texts, palette)):
        ax.text(0.98, 0.05 + i * 0.05, label,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=12, color=col, fontweight="bold")

    ax.set_title("Cumulative Distribution of Cross Correlations (PSMA)", pad=12)
    ax.set_xlabel("Mean Pairwise Correlation")
    ax.set_ylabel("Cumulative Probability")
    ax.set_ylim(0, 1.01)

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(path + "PSMA_cdf.svg", format="svg")
    plt.show()

    return all_cc[1], all_cc[2], all_cc[3]


# Run analysis
analyze_load_distributions_PSMA_final("TC.mat", min_time_cells=3, use_correct=False)