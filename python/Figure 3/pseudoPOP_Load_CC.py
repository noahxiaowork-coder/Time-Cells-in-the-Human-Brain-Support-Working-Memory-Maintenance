import numpy as np
import scipy.io
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from scipy.stats import ks_2samp

DIR = '/Users/xiangxuankong/Desktop/Human TC Support WM/Figures/Figure 3/'
# ------------------------------------------------------------------
#  Significance‑bar helper (unchanged)
# ------------------------------------------------------------------
def add_significance_bar(ax, x1, x2, y, p_value, text_offset=0.02):
    """Draw *** / ** / * / ns between two x‑positions on 'ax'."""
    bar_y = y + text_offset
    ax.plot([x1, x1, x2, x2], [y, bar_y, bar_y, y],
            color="black", linewidth=1)
    if   p_value < 0.001: sig = "***"
    elif p_value < 0.01:  sig = "**"
    elif p_value < 0.05:  sig = "*"
    else:                 sig = "ns"
    ax.text((x1 + x2) / 2, bar_y, sig,
            ha="center", va="bottom", fontsize=12)


def separate_trials_by_load(mat_file_path, patient_id, *, use_correct=False):
    """
    Return a dict {1: …, 2: …, 3: …} of (trials, neurons, timeBins) arrays
    and the list of time‑field indices for that patient.

    Parameters
    ----------
    mat_file_path : str
    patient_id : int
    use_correct : bool, optional
        If True, keep only trials where trial_correctness == 1.
    """
    m = scipy.io.loadmat(mat_file_path)
    nd = m['neural_data'][0]

    # Pull out arrays for *all* neurons
    pids   = [int(e['patient_id'][0][0])      for e in nd]
    frs    = [e['firing_rates']               for e in nd]
    loads  = [e['trial_load']                 for e in nd]
    corrs  = [e['trial_correctness']          for e in nd]   # NEW
    tfs    = [int(e['time_field'][0][0]) - 1  for e in nd]

    # Keep only neurons belonging to this patient
    chosen = [(fr, ld, cr, tf) for fr, ld, cr, tf, pid in
              zip(frs, loads, corrs, tfs, pids) if pid == patient_id]
    if not chosen:
        return None, None

    fr_stack   = np.stack([np.asarray(c[0]) for c in chosen], axis=1)  # trials×neurons×bins
    load_vec   = np.asarray(chosen[0][1]).flatten()
    corr_vec   = np.asarray(chosen[0][2]).flatten()                   # correctness per trial
    time_fields = [c[3] for c in chosen]

    load_dict = {}
    for lv in (1, 2, 3):
        mask = (load_vec == lv)
        if use_correct:
            mask &= (corr_vec == 1)
        load_dict[lv] = fr_stack[mask]

    return load_dict, time_fields

def process_trial_and_compute_mean_correlation(trial_data,
                                               time_fields,
                                               top_percent=100):
    num_neurons, n_bins = trial_data.shape
    idx = [i for i in range(num_neurons) if np.std(trial_data[i]) != 0]
    if len(idx) < 2:
        return 0.0

    data   = trial_data[idx]
    fields = [time_fields[i] for i in idx]
    mid    = n_bins // 2
    shifted = np.zeros_like(data)
    for i, tf in enumerate(fields):
        shifted[i] = np.roll(data[i], mid - tf)

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
    if top_percent < 100:
        k = max(1, int(np.ceil(len(corrs) * top_percent / 100.0)))
        corrs = sorted(corrs, reverse=True)[:k]
    return float(np.mean(corrs))


# ------------------------------------------------------------------
#  MAIN DRIVER  – now has 'use_correct' param and a box‑plot for Plot 1
# ------------------------------------------------------------------
def analyze_load_distributions(mat_file_path, *,
                               min_time_cells=5,
                               top_percent=100,
                               use_correct=False,           # NEW ←–––––––––
                               random_seed=42,
                               remove_zero_cc=True,
                               verbose=True):
    """
    Compare trial‑level mean cross‑correlations across load conditions.
    If use_correct is True, only correct trials contribute.

    Other params: see earlier description.
    """
    # ---- 0. load MATLAB ----
    m = scipy.io.loadmat(mat_file_path)
    nd = m['neural_data'][0]
    patient_ids = [int(e['patient_id'][0][0]) for e in nd]
    unique_ids  = np.unique(patient_ids)

    # ---- 1. gather per‑patient data ----
    patient_counts, data_by_patient = {}, {}
    p_iter = tqdm(unique_ids, desc="Gathering trial counts") if verbose else unique_ids
    for pid in p_iter:
        # if pid in {10}:
        #     if verbose:
        #         print(f"Skipping patient {pid} (excluded)")
        #     continue
        load_d, t_fields = separate_trials_by_load(mat_file_path, pid,
                                                   use_correct=use_correct)
        if load_d is None:
            continue
        if len(t_fields) < min_time_cells:
            if verbose:
                print(f"Skipping patient {pid}: only {len(t_fields)} time‑cells "
                      f"(min required = {min_time_cells}).")
            continue
        patient_counts[pid] = {lv: load_d[lv].shape[0] for lv in (1, 2, 3)}
        data_by_patient[pid] = (load_d, t_fields)

    if not data_by_patient:
        raise RuntimeError("No patients qualified under the filters.")

    # ---- 2. global mins ----
    gmin = {lv: min([patient_counts[p][lv] for p in data_by_patient])
            for lv in (1, 2, 3)}
    if verbose:
        print("Global min trials  •  " +
              "  ".join([f"Load{lv}={gmin[lv]}" for lv in (1, 2, 3)]))

    # ---- 3. compute correlations ----
    rng = np.random.default_rng(random_seed)
    all_cc = {1: [], 2: [], 3: []}

    c_iter = tqdm(data_by_patient, desc="Computing CCs") if verbose else data_by_patient
    for pid in c_iter:
        load_d, t_fields = data_by_patient[pid]
        for lv in (1, 2, 3):
            n_keep = gmin[lv]
            if n_keep == 0:
                continue
            idx = rng.choice(load_d[lv].shape[0], n_keep, replace=False)
            for trial in load_d[lv][idx]:
                val = process_trial_and_compute_mean_correlation(
                          trial, t_fields, top_percent=top_percent)
                all_cc[lv].append(val)


    from scipy.stats import wilcoxon, ttest_rel   # ✱ add at top

    # ------------------------------------------------------------------
    #  NEW block 2a–2d (insert just after Step 3, before Step 4)
    # ------------------------------------------------------------------
    # Collect trial-level CCs exactly as you already do,
    # but store them *per patient* as well:
    per_patient_cc = {pid: {1: [], 2: [], 3: []} for pid in data_by_patient}

    for pid in c_iter:                     # ← same loop you already have
        load_d, t_fields = data_by_patient[pid]
        for lv in (1, 2, 3):
            n_keep = gmin[lv]
            if n_keep == 0:
                continue
            idx = rng.choice(load_d[lv].shape[0], n_keep, replace=False)
            for trial in load_d[lv][idx]:
                val = process_trial_and_compute_mean_correlation(
                        trial, t_fields, top_percent=top_percent)
                all_cc[lv].append(val)
                per_patient_cc[pid][lv].append(val)        # ✱ NEW

    # 2b  Per-patient summary (mean – can switch to median)
    pps_mean = {lv: [] for lv in (1, 2, 3)}
    for pid in per_patient_cc:
        for lv in (1, 2, 3):
            v = np.mean(per_patient_cc[pid][lv]) if per_patient_cc[pid][lv] else np.nan
            pps_mean[lv].append(v)

    # Drop patients with NaNs (e.g., zero qualifying trials in a load)
    valid_idx = ~np.isnan(pps_mean[1]) & ~np.isnan(pps_mean[2]) & ~np.isnan(pps_mean[3])
    for lv in (1, 2, 3):
        pps_mean[lv] = np.asarray(pps_mean[lv])[valid_idx]

    # 2c  Paired stats across patients
    _, p12_pat = ttest_rel(pps_mean[1], pps_mean[2], alternative='greater')
    _, p23_pat = ttest_rel(pps_mean[2], pps_mean[3], alternative='greater')
    _, p13_pat = ttest_rel(pps_mean[1], pps_mean[3], alternative='greater')

    # Two-sided KS test: H0 = same distribution
    D_12, ks_p_12 = ks_2samp(all_cc[1], all_cc[2])
    D_23, ks_p_23 = ks_2samp(all_cc[2], all_cc[3])
    D_13, ks_p_13 = ks_2samp(all_cc[1], all_cc[3])


    print(f"[PATIENT LEVEL]  L1>2  p={p12_pat:.4g}  "
        f"L2>3  p={p23_pat:.4g}  L1>3  p={p13_pat:.4g}")

    # --- Setup ---
    fig, ax = plt.subplots(figsize=(5, 6))

    # Define colors for each condition (same as your palette)
    palette = ["royalblue", "seagreen", "firebrick"]

    # Number of patients
    n_patients = len(pps_mean[1])

    # Plot paired lines & grey dots for each patient
    for i in range(n_patients):
        yvals = [pps_mean[1][i], pps_mean[2][i], pps_mean[3][i]]
        ax.plot([1, 2, 3], yvals, color='grey', alpha=0.5, linewidth=1)
        ax.scatter([1, 2, 3], yvals, color='grey', alpha=0.8, zorder=5)

    # Overlay mean lines for each load (with condition color)
    for idx, (lv, color) in enumerate(zip([1, 2, 3], palette)):
        mean_val = np.mean(pps_mean[lv])
        ax.hlines(mean_val, idx + 0.8, idx + 1.2, color=color,
                linewidth=3, zorder=10)

    # Add significance bars between pairs
    ymax = np.max([*pps_mean[1], *pps_mean[2], *pps_mean[3]])

    def get_stars(p):
        if p < 1e-3:
            return "***"
        elif p < 1e-2:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return f"p = {p:.3f}"
        
        # Example: for pairs (x1, x2)
    x1, x2 = 1, 2
    ymax = np.max([*pps_mean[1], *pps_mean[2], *pps_mean[3]])
    height = 0.02  # vertical offset per bar

    def add_sig_bar(ax, x1, x2, y, text, bar_width=-0.15):
        """
        Add a compact significance bar between x1 and x2 at height y.
        bar_width controls how much the bar overhangs the ticks.
        """
        ax.plot([x1 - bar_width, x1 - bar_width, x2 + bar_width, x2 + bar_width],
                [y, y + height, y + height, y], lw=1.5, color='black')
        ax.text((x1 + x2) / 2, y + height + 0.005, text,
                ha='center', va='bottom', fontsize=13)

    # Example usage:
    add_sig_bar(ax, 1, 2, ymax + 0.01, get_stars(p12_pat))
    add_sig_bar(ax, 2, 3, ymax + 0.01, get_stars(p23_pat))
    add_sig_bar(ax, 1, 3, ymax + 0.04, get_stars(p13_pat))


    # Aesthetics
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Load 1", "Load 2", "Load 3"])
    ax.set_xlabel("Memory Load", fontsize=12)
    ax.set_ylabel("Mean CC per Patient", fontsize=12)
    ax.set_title("Per-Patient Mean CC (Dot Plot with Colored Means)", fontsize=14)
    ax.set_xlim(0.5, 3.5)

    sns.despine()
    plt.tight_layout()

    if use_correct:
        path = DIR + 'Correct/'
    else:
        path = DIR

    # plt.savefig(path + "per_patient_dotplot_colored_means.svg", format='svg')
    plt.show()



    # ---- 4. drop zeros? ----
    if remove_zero_cc:
        for lv in (1, 2, 3):
            all_cc[lv] = [v for v in all_cc[lv] if v != 0]

    # ---- 5. sanity check ----
    if any(len(all_cc[lv]) < 2 for lv in (1, 2, 3)):
        if verbose:
            print("Too few data points – skipping stats/plots.")
        return all_cc[1], all_cc[2], all_cc[3]

    if verbose:
        print(f"Load1 vs 2  p={ks_p_12:.4g}   "
              f"Load2 vs 3  p={ks_p_23:.4g}   "
              f"Load1 vs 3  p={ks_p_13:.4g}")

    # ---- 7. plotting ----
    sns.set_theme(style="whitegrid", context="talk")
    palette = ["royalblue", "seagreen", "firebrick"]

    # 7a   BOX‑PLOT (Plot 1) ------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    df = pd.DataFrame({
        "Cross‑Correlation": all_cc[1] + all_cc[2] + all_cc[3],
        "Load": (["1"] * len(all_cc[1]) +
                 ["2"] * len(all_cc[2]) +
                 ["3"] * len(all_cc[3]))
    })
    sns.boxplot(data=df, x="Load", y="Cross‑Correlation",
                palette=palette, width=0.6, ax=ax,
                showcaps=True, boxprops={"zorder": 2}, showfliers=False,
                whiskerprops={"linewidth": 1})
    sns.stripplot(data=df, x="Load", y="Cross‑Correlation",
                  color="k", alpha=0.45, size=4, jitter=0.25, zorder=1, ax=ax)

    ax.set_title("Mean Cross‑Correlation by Load", pad=12)
    ax.set_xlabel("Memory Load")
    ax.set_ylabel("Mean Pairwise Correlation")

    y_max = df["Cross‑Correlation"].max()
    add_significance_bar(ax, 0, 1, y_max * 1.05, ks_p_12)
    add_significance_bar(ax, 1, 2, y_max * 1.12, ks_p_23)
    add_significance_bar(ax, 0, 2, y_max * 1.19, ks_p_13)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()

    # 7b   STANDARD KDE (Plot 2) -------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.kdeplot(all_cc[1], fill=True, label="Load = 1",
                color=palette[0], linewidth=1.6, ax=ax)
    sns.kdeplot(all_cc[2], fill=True, label="Load = 2",
                color=palette[1], linewidth=1.6, ax=ax)
    sns.kdeplot(all_cc[3], fill=True, label="Load = 3",
                color=palette[2], linewidth=1.6, ax=ax)

    ax.set_title("Density of Cross Correlations by Load", pad=12)
    ax.set_xlabel("Mean Pairwise Correlation")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    txt = f"KS Load1 vs 2: p={ks_p_12:.3g}\nKS Load2 vs 3: p={ks_p_23:.3g}\nKS Load1 vs 3: p={ks_p_13:.3g}"
    ax.text(0.02, 0.97, txt, transform=ax.transAxes,
            ha="left", va="top", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7"))
    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()

    # 7c   PEAK‑NORMALISED KDE (Plot 3) ------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    all_vals = np.concatenate([all_cc[1], all_cc[2], all_cc[3]])
    x_grid = np.linspace(all_vals.min(), all_vals.max(), 400)
    for lv, col, label in zip((1, 2, 3), palette,
                              ("Load = 1", "Load = 2", "Load = 3")):
        kde = gaussian_kde(all_cc[lv])
        y   = kde(x_grid)
        y  /= y.max()           # peak‑normalise
        ax.plot(x_grid, y, color=col, linewidth=1.8, label=label)
        ax.fill_between(x_grid, 0, y, color=col, alpha=0.25)
    ax.set_title("Peak‑Normalised Density of Cross‑Correlations", pad=12)
    ax.set_xlabel("Mean Pairwise Correlation")
    ax.set_ylabel("Normalised Density")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()
    
        # 7d   EMPIRICAL CDF (Plot 4) ------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot the CDF lines
    for lv, col in zip((1, 2, 3), palette):
        vals = np.sort(all_cc[lv])
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, cdf, color=col, linewidth=1.8)

    # Add p-values in top-left
    txt = f"Load1 vs 2: p={ks_p_12:.3g}\nLoad2 vs 3: p={ks_p_23:.3g}\nLoad1 vs 3: p={ks_p_13:.3g}"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes,
            ha="left", va="top", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7"))

    # Add load condition labels manually in bottom-right
    label_texts = ["Load = 1", "Load = 2", "Load = 3"]
    colors = palette
    for i, (label, col) in enumerate(zip(label_texts, colors)):
        ax.text(0.98, 0.05 + i * 0.05, label,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=12, color=col, fontweight='bold')

    ax.set_title("Cumulative Distribution of Cross Correlations", pad=12)
    ax.set_xlabel("Mean Pairwise Correlation")
    ax.set_ylabel("Cumulative Probability")
    ax.set_ylim(0, 1.01)

    # Remove automatic legend (since labels are manually placed)
    sns.despine(trim=True)
    plt.tight_layout()
    if use_correct:
        path = DIR + 'Correct/'
    else:
        path = DIR

    # plt.savefig(path + "crosscorrelation.svg", format = 'svg')
    plt.show()

    # ---- 8. return raw lists ----
    return all_cc[1], all_cc[2], all_cc[3]

import numpy as np
import scipy.io
from scipy.stats import ks_2samp, ttest_rel
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from tqdm import tqdm

# assumes you already have:
# from your_module import process_trial_and_compute_mean_correlation
# and a global DIR variable for saving, as in your other scripts


def analyze_load_distributions_PSMA(
    mat_file_path, *,
    min_time_cells=5,
    top_percent=100,
    use_correct=False,
    random_seed=42,
    remove_zero_cc=False,
    region_name,
    verbose=True
):
    """
    PSMA-specific version of analyze_load_distributions.

    Restricts analysis to neurons in the region
    'pre_supplementary_motor_area' (after stripping '_left'/'_right').

    For each patient:
      * collect PSMA neurons and their time fields,
      * build trial × time-cell matrices per load (1,2,3),
      * compute mean cross-correlations per trial via
        process_trial_and_compute_mean_correlation(trial, t_fields),
      * aggregate across patients and compare loads.

    Parameters
    ----------
    mat_file_path : str
        Path to .mat file with 'neural_data'.
    min_time_cells : int
        Minimum number of PSMA time cells per patient required to include them.
    top_percent : float
        Passed to process_trial_and_compute_mean_correlation.
    use_correct : bool
        If True, only correct trials contribute.
    random_seed : int
        RNG seed for trial subsampling.
    remove_zero_cc : bool
        If True, drop zero CC values before group-level stats/plots.
    verbose : bool
        Print progress and summary information.

    Returns
    -------
    all_cc_load1, all_cc_load2, all_cc_load3 : list[float]
        Trial-level cross-correlation values for each load (1, 2, 3)
        pooled across all included patients (PSMA only).
    """

    # -----------------------------
    # Helper: clean brain_region
    # -----------------------------
    def clean_region(entry_region_field):
        reg = entry_region_field
        if isinstance(reg, np.ndarray):
            reg = np.squeeze(reg)
            if hasattr(reg, "dtype") and reg.dtype.kind in ("U", "S"):
                reg = "".join(reg.flat)
            else:
                reg = str(reg)
        else:
            reg = str(reg)

        reg = reg.strip().lower()
        for suffix in ("_left", "_right"):
            if reg.endswith(suffix):
                reg = reg[:-len(suffix)]
                break
        return reg

    def get_stars(p):
        if p < 1e-3:
            return "***"
        elif p < 1e-2:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return f"p = {p:.3f}"

    def add_sig_bar(ax, x1, x2, y, text, height=0.02, bar_width=-0.15):
        """
        Add a compact significance bar between x1 and x2 at height y.
        x positions are in data coords (e.g., 1,2,3 for loads).
        """
        ax.plot([x1 - bar_width, x1 - bar_width, x2 + bar_width, x2 + bar_width],
                [y, y + height, y + height, y], lw=1.5, color='black')
        ax.text((x1 + x2) / 2, y + height + 0.005, text,
                ha='center', va='bottom', fontsize=13)

    def add_significance_bar(ax, x1, x2, y, p_val):
        """Used for the boxplot-by-load figure (x positions 0,1,2)."""
        text = get_stars(p_val)
        ax.plot([x1, x1, x2, x2],
                [y, y * 1.01, y * 1.01, y],
                lw=1.5, color="black")
        ax.text((x1 + x2) / 2, y * 1.01, text,
                ha="center", va="bottom", fontsize=12)

    # -----------------------------
    # 0. Load MATLAB
    # -----------------------------
    m = scipy.io.loadmat(mat_file_path)
    nd = m["neural_data"][0]

    # We'll still treat patients as in the original function:
    patient_ids = [int(e["patient_id"][0][0]) for e in nd]
    unique_ids = np.unique(patient_ids)

    # -----------------------------
    # 1. Gather per-patient PSMA data
    # -----------------------------
    patient_counts = {}
    data_by_patient = {}

    if verbose:
        p_iter = tqdm(unique_ids, desc="Gathering PSMA data (trial counts)")
    else:
        p_iter = unique_ids

    for pid in p_iter:
        # Collect all PSMA neurons for this patient
        psma_entries = []
        for entry in nd:
            ep = int(entry["patient_id"][0][0])
            if ep != pid:
                continue

            region_key = clean_region(entry["brain_region"])
            if region_key != region_name:
                continue

            psma_entries.append(entry)

        if len(psma_entries) == 0:
            # no PSMA neurons for this patient
            continue

        # Assume all PSMA neurons for a patient share the same trial structure
        # We take loads / correctness from the first PSMA neuron
        fr0 = psma_entries[0]["firing_rates"]    # trials × bins
        loads = psma_entries[0]["trial_load"].flatten().astype(int)
        correct = psma_entries[0]["trial_correctness"].flatten().astype(int)
        n_trials, n_bins = fr0.shape

        # Build list of time-cell firing matrices & time fields
        time_cell_frates = []
        t_fields = []

        for entry in psma_entries:
            fr = entry["firing_rates"]           # trials × bins
            tf = int(entry["time_field"][0][0]) - 1  # 0-based time field index
            # Treat every PSMA neuron as a time cell (you can add your own filter here)
            time_cell_frates.append(fr)
            t_fields.append(tf)

        n_cells = len(time_cell_frates)
        if n_cells < min_time_cells:
            if verbose:
                print(f"Skipping patient {pid}: only {n_cells} PSMA time-cells "
                      f"(min required = {min_time_cells}).")
            continue

        t_fields = np.array(t_fields, dtype=int)

        # Trial inclusion mask
        if use_correct:
            trial_mask = (correct == 1)
        else:
            trial_mask = np.ones_like(correct, dtype=bool)

        # Build load_d: for each load, an array of shape (n_trials_lv, n_cells, n_bins)
        load_d = {}
        for lv in (1, 2, 3):
            trial_indices = np.where((loads == lv) & trial_mask)[0]
            if len(trial_indices) == 0:
                load_d[lv] = np.empty((0, n_cells, n_bins), dtype=float)
                continue

            trial_mats = []
            for t_idx in trial_indices:
                # For each time cell, take its firing rates on this trial
                trial_matrix = np.vstack([fr[t_idx, :] for fr in time_cell_frates])
                trial_mats.append(trial_matrix)

            load_d[lv] = np.stack(trial_mats, axis=0)  # trials × cells × bins

        # patient_counts: # trials per load
        patient_counts[pid] = {lv: load_d[lv].shape[0] for lv in (1, 2, 3)}
        data_by_patient[pid] = (load_d, t_fields)

    if not data_by_patient:
        raise RuntimeError("No patients qualified under the filters (PSMA).")

    # -----------------------------
    # 2. Global mins across patients
    # -----------------------------
    gmin = {
        lv: min([patient_counts[p][lv] for p in data_by_patient])
        for lv in (1, 2, 3)
    }
    if verbose:
        print("Global min trials (PSMA)  •  " +
              "  ".join([f"Load{lv}={gmin[lv]}" for lv in (1, 2, 3)]))

    # -----------------------------
    # 3. Compute correlations
    # -----------------------------
    rng = np.random.default_rng(random_seed)

    all_cc = {1: [], 2: [], 3: []}                      # pooled across patients
    per_patient_cc = {pid: {1: [], 2: [], 3: []}
                      for pid in data_by_patient}       # per-patient

    if verbose:
        c_iter = tqdm(data_by_patient.keys(), desc="Computing CCs (PSMA)")
    else:
        c_iter = data_by_patient.keys()

    for pid in c_iter:
        load_d, t_fields = data_by_patient[pid]

        for lv in (1, 2, 3):
            n_keep = gmin[lv]
            n_trials_lv = load_d[lv].shape[0]
            if n_keep == 0 or n_trials_lv == 0:
                continue

            idx = rng.choice(n_trials_lv, n_keep, replace=False)

            for trial in load_d[lv][idx]:
                val = process_trial_and_compute_mean_correlation(
                    trial, t_fields, top_percent=top_percent
                )
                all_cc[lv].append(val)
                per_patient_cc[pid][lv].append(val)

    # -----------------------------
    # 3b. Per-patient summary stats
    # -----------------------------
    pps_mean = {lv: [] for lv in (1, 2, 3)}
    for pid in per_patient_cc:
        for lv in (1, 2, 3):
            vals = per_patient_cc[pid][lv]
            v = np.mean(vals) if len(vals) > 0 else np.nan
            pps_mean[lv].append(v)

    # as arrays so we can drop NaNs consistently across loads
    pps_mean[1] = np.asarray(pps_mean[1], dtype=float)
    pps_mean[2] = np.asarray(pps_mean[2], dtype=float)
    pps_mean[3] = np.asarray(pps_mean[3], dtype=float)

    valid_idx = (~np.isnan(pps_mean[1]) &
                 ~np.isnan(pps_mean[2]) &
                 ~np.isnan(pps_mean[3]))
    for lv in (1, 2, 3):
        pps_mean[lv] = pps_mean[lv][valid_idx]

    # Paired tests across patients (one-sided: L1>L2, etc.)
    if len(pps_mean[1]) > 1:
        _, p12_pat = ttest_rel(pps_mean[1], pps_mean[2], alternative='greater')
        _, p23_pat = ttest_rel(pps_mean[2], pps_mean[3], alternative='greater')
        _, p13_pat = ttest_rel(pps_mean[1], pps_mean[3], alternative='greater')
    else:
        p12_pat = p23_pat = p13_pat = 1.0  # not enough patients

    # KS tests at trial level (two-sample, two-sided)
    D_12, ks_p_12 = ks_2samp(all_cc[1], all_cc[2])
    D_23, ks_p_23 = ks_2samp(all_cc[2], all_cc[3])
    D_13, ks_p_13 = ks_2samp(all_cc[1], all_cc[3])

    print(f"[PATIENT LEVEL, PSMA]  L1>2  p={p12_pat:.4g}  "
          f"L2>3  p={p23_pat:.4g}  L1>3  p={p13_pat:.4g}")

    # -----------------------------
    # 3c. Per-patient dot plot + stars
    # -----------------------------
    fig, ax = plt.subplots(figsize=(5, 6))

    palette = ["royalblue", "seagreen", "firebrick"]

    n_patients = len(pps_mean[1])

    # paired lines (grey) with patient dots
    for i in range(n_patients):
        yvals = [pps_mean[1][i], pps_mean[2][i], pps_mean[3][i]]
        ax.plot([1, 2, 3], yvals, color='grey', alpha=0.5, linewidth=1)
        ax.scatter([1, 2, 3], yvals, color='grey', alpha=0.8, zorder=5)

    # overlay colored mean per load
    for idx, (lv, color) in enumerate(zip([1, 2, 3], palette)):
        mean_val = np.mean(pps_mean[lv]) if len(pps_mean[lv]) > 0 else np.nan
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

    if use_correct:
        path = DIR + 'Correct/'
    else:
        path = DIR

    plt.savefig(path + "PSMA_per_patient_dotplot.svg", format='svg')
    plt.show()

    # -----------------------------
    # 4. Drop zeros if requested
    # -----------------------------
    if remove_zero_cc:
        for lv in (1, 2, 3):
            all_cc[lv] = [v for v in all_cc[lv] if v != 0]

    # -----------------------------
    # 5. Sanity check
    # -----------------------------
    if any(len(all_cc[lv]) < 2 for lv in (1, 2, 3)):
        if verbose:
            print("Too few data points – skipping stats/plots.")
        return all_cc[1], all_cc[2], all_cc[3]

    if verbose:
        print(f"[PSMA] KS Load1 vs 2  p={ks_p_12:.4g}   "
              f"Load2 vs 3  p={ks_p_23:.4g}   "
              f"Load1 vs 3  p={ks_p_13:.4g}")

    sns.set_theme(style="whitegrid", context="talk")
    palette = ["royalblue", "seagreen", "firebrick"]

    # -----------------------------
    # 7a. Box-plot (trial-level)
    # -----------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    df = pd.DataFrame({
        "Cross-Correlation": all_cc[1] + all_cc[2] + all_cc[3],
        "Load": (["1"] * len(all_cc[1]) +
                 ["2"] * len(all_cc[2]) +
                 ["3"] * len(all_cc[3]))
    })
    sns.boxplot(
        data=df, x="Load", y="Cross-Correlation",
        palette=palette, width=0.6, ax=ax,
        showcaps=True, boxprops={"zorder": 2}, showfliers=False,
        whiskerprops={"linewidth": 1}
    )
    sns.stripplot(
        data=df, x="Load", y="Cross-Correlation",
        color="k", alpha=0.45, size=4, jitter=0.25, zorder=1, ax=ax
    )

    ax.set_title("Mean Cross-Correlation by Load (PSMA)", pad=12)
    ax.set_xlabel("Memory Load")
    ax.set_ylabel("Mean Pairwise Correlation")

    y_max = df["Cross-Correlation"].max()
    add_significance_bar(ax, 0, 1, y_max * 1.05, ks_p_12)
    add_significance_bar(ax, 1, 2, y_max * 1.12, ks_p_23)
    add_significance_bar(ax, 0, 2, y_max * 1.19, ks_p_13)
    sns.despine(trim=True)
    plt.tight_layout()
    # plt.savefig(path + "PSMA_boxplot.svg", format='svg')
    plt.show()

    # -----------------------------
    # 7b. Standard KDE
    # -----------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.kdeplot(all_cc[1], fill=True, label="Load = 1",
                color=palette[0], linewidth=1.6, ax=ax)
    sns.kdeplot(all_cc[2], fill=True, label="Load = 2",
                color=palette[1], linewidth=1.6, ax=ax)
    sns.kdeplot(all_cc[3], fill=True, label="Load = 3",
                color=palette[2], linewidth=1.6, ax=ax)

    ax.set_title("Density of Cross Correlations by Load (PSMA)", pad=12)
    ax.set_xlabel("Mean Pairwise Correlation")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    txt = (f"KS Load1 vs 2: p={ks_p_12:.3g}\n"
           f"KS Load2 vs 3: p={ks_p_23:.3g}\n"
           f"KS Load1 vs 3: p={ks_p_13:.3g}")
    ax.text(0.02, 0.97, txt, transform=ax.transAxes,
            ha="left", va="top", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7"))
    sns.despine(trim=True)
    plt.tight_layout()
    # plt.savefig(path + "PSMA_kde.svg", format='svg')
    plt.show()

    # -----------------------------
    # 7c. Peak-normalised KDE
    # -----------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    all_vals = np.concatenate([all_cc[1], all_cc[2], all_cc[3]])
    x_grid = np.linspace(all_vals.min(), all_vals.max(), 400)
    for lv, col, label in zip((1, 2, 3), palette,
                              ("Load = 1", "Load = 2", "Load = 3")):
        kde = gaussian_kde(all_cc[lv])
        y = kde(x_grid)
        y /= y.max()
        ax.plot(x_grid, y, color=col, linewidth=1.8, label=label)
        ax.fill_between(x_grid, 0, y, color=col, alpha=0.25)
    ax.set_title("Peak-normalised Density of Cross-Correlations (PSMA)", pad=12)
    ax.set_xlabel("Mean Pairwise Correlation")
    ax.set_ylabel("Normalised Density")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    sns.despine(trim=True)
    plt.tight_layout()
    # plt.savefig(path + "PSMA_peaknorm_kde.svg", format='svg')
    plt.show()

    # -----------------------------
    # 7d. Empirical CDF
    # -----------------------------
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
    colors = palette
    for i, (label, col) in enumerate(zip(label_texts, colors)):
        ax.text(0.98, 0.05 + i * 0.05, label,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=12, color=col, fontweight='bold')

    ax.set_title("Cumulative Distribution of Cross Correlations (PSMA)", pad=12)
    ax.set_xlabel("Mean Pairwise Correlation")
    ax.set_ylabel("Cumulative Probability")
    ax.set_ylim(0, 1.01)

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(path + "PSMA_cdf.svg", format='svg')
    plt.show()

    # -----------------------------
    # 8. Return raw lists
    # -----------------------------
    return all_cc[1], all_cc[2], all_cc[3]

import numpy as np
import scipy.io
from scipy.stats import ks_2samp, ttest_rel, gaussian_kde   # kde only used if you later re-add
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# Assumes:
#   - process_trial_and_compute_mean_correlation(trial, t_fields, top_percent)
#   - global DIR string for saving paths


def analyze_load_distributions_PSMA_final(
    mat_file_path, *,
    min_time_cells=5,
    top_percent=100,
    use_correct=False,
    remove_zero_cc=False,
    verbose=True
):
    """
    PSMA-specific load analysis WITHOUT trial subsampling.

    For each patient:
      * gather PSMA neurons (pre_supplementary_motor_area),
      * treat those neurons as time cells with their time_fields,
      * for each trial and each load (1,2,3), compute a trial-level
        mean cross-correlation using process_trial_and_compute_mean_correlation,
      * store trial-level CCs per patient and pooled across patients.

    Outputs:
      * Per-patient mean CC plot (Load 1/2/3 with paired lines & t-test stars),
      * CDF plot of pooled trial-level CCs per load (with KS p-values),
      * Returns pooled CC lists for each load (all patients combined).

    Parameters
    ----------
    mat_file_path : str
        Path to .mat file containing 'neural_data'.
    min_time_cells : int
        Minimum number of PSMA time cells per patient (otherwise excluded).
    top_percent : float
        Passed into process_trial_and_compute_mean_correlation.
    use_correct : bool
        If True, only correct trials are used.
    remove_zero_cc : bool
        If True, drop CC values equal to 0 before KS tests and plots.
    verbose : bool
        Verbose printing and progress bars.

    Returns
    -------
    all_cc_load1, all_cc_load2, all_cc_load3 : list of float
        Trial-level CC values pooled across patients for loads 1, 2, 3.
    """

    # -----------------------------
    # Small helpers
    # -----------------------------
    def clean_region(entry_region_field):
        reg = entry_region_field
        if isinstance(reg, np.ndarray):
            reg = np.squeeze(reg)
            if hasattr(reg, "dtype") and reg.dtype.kind in ("U", "S"):
                reg = "".join(reg.flat)
            else:
                reg = str(reg)
        else:
            reg = str(reg)

        reg = reg.strip().lower()
        for suffix in ("_left", "_right"):
            if reg.endswith(suffix):
                reg = reg[:-len(suffix)]
                break
        return reg

    def get_stars(p):
        if p < 1e-3:
            return "***"
        elif p < 1e-2:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return f"p = {p:.3f}"

    def add_sig_bar(ax, x1, x2, y, text, height=0.02, bar_width=-0.15):
        """Add a compact significance bar between x1 and x2 at height y."""
        ax.plot([x1 - bar_width, x1 - bar_width, x2 + bar_width, x2 + bar_width],
                [y, y + height, y + height, y], lw=1.5, color='black')
        ax.text((x1 + x2) / 2, y + height + 0.005, text,
                ha='center', va='bottom', fontsize=13)

    # -----------------------------
    # 0. Load MATLAB and basic setup
    # -----------------------------
    m = scipy.io.loadmat(mat_file_path)
    nd = m["neural_data"][0]

    patient_ids = [int(e["patient_id"][0][0]) for e in nd]
    unique_ids = np.unique(patient_ids)

    # -----------------------------
    # 1. Gather PSMA data per patient
    # -----------------------------
    data_by_patient = {}   # pid -> (load_d, t_fields)
    if verbose:
        p_iter = tqdm(unique_ids, desc="Gathering PSMA data")
    else:
        p_iter = unique_ids

    for pid in p_iter:
        # Find all PSMA neurons for this patient
        psma_entries = []
        for entry in nd:
            ep = int(entry["patient_id"][0][0])
            if ep != pid:
                continue
            region_key = clean_region(entry["brain_region"])
            if region_key != "pre_supplementary_motor_area":
                continue
            psma_entries.append(entry)

        if len(psma_entries) == 0:
            continue

        # Use the first PSMA neuron for trial meta info
        fr0 = psma_entries[0]["firing_rates"]    # trials × bins
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

        # Trial inclusion mask
        if use_correct:
            trial_mask = (correct == 1)
        else:
            trial_mask = np.ones_like(correct, dtype=bool)

        # Build load_d: load -> array of trials (trials × cells × bins)
        load_d = {}
        for lv in (1, 2, 3):
            trial_indices = np.where((loads == lv) & trial_mask)[0]
            if len(trial_indices) == 0:
                load_d[lv] = np.empty((0, n_cells, n_bins), dtype=float)
                continue

            trial_mats = []
            for t_idx in trial_indices:
                # trial: cells × bins
                trial_matrix = np.vstack([fr[t_idx, :] for fr in time_cell_frates])
                trial_mats.append(trial_matrix)

            load_d[lv] = np.stack(trial_mats, axis=0)

        data_by_patient[pid] = (load_d, t_fields)

    if not data_by_patient:
        raise RuntimeError("No patients qualified under the PSMA filters.")

    # -----------------------------
    # 2. Compute CCs per trial, per patient (NO subsampling)
    # -----------------------------
    all_cc = {1: [], 2: [], 3: []}
    per_patient_cc = {pid: {1: [], 2: [], 3: []} for pid in data_by_patient}

    if verbose:
        c_iter = tqdm(data_by_patient.keys(), desc="Computing CCs (PSMA)")
    else:
        c_iter = data_by_patient.keys()

    for pid in c_iter:
        load_d, t_fields = data_by_patient[pid]
        for lv in (1, 2, 3):
            trials_lv = load_d[lv]          # shape: (n_trials_lv, n_cells, n_bins)
            for trial in trials_lv:
                val = process_trial_and_compute_mean_correlation(
                    trial, t_fields, top_percent=top_percent
                )
                all_cc[lv].append(val)
                per_patient_cc[pid][lv].append(val)

    # -----------------------------
    # 3. Per-patient means & paired stats
    # -----------------------------
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

    # -----------------------------
    # 4. Optional removal of zero CCs
    # -----------------------------
    if remove_zero_cc:
        for lv in (1, 2, 3):
            all_cc[lv] = [v for v in all_cc[lv] if v != 0]

    # Sanity check for pooled stats
    if any(len(all_cc[lv]) < 2 for lv in (1, 2, 3)):
        if verbose:
            print("Too few data points – skipping KS/CDF plots.")
        return all_cc[1], all_cc[2], all_cc[3]

    # -----------------------------
    # 5. KS tests on pooled CCs
    # -----------------------------
    D_12, ks_p_12 = ks_2samp(all_cc[1], all_cc[2])
    D_23, ks_p_23 = ks_2samp(all_cc[2], all_cc[3])
    D_13, ks_p_13 = ks_2samp(all_cc[1], all_cc[3])

    if verbose:
        print(f"[PATIENT LEVEL, PSMA]  L1>2  p={p12_pat:.4g}  "
              f"L2>3  p={p23_pat:.4g}  L1>3  p={p13_pat:.4g}")
        print(f"[TRIAL LEVEL, PSMA]   KS L1 vs 2: p={ks_p_12:.4g}, "
              f"L2 vs 3: p={ks_p_23:.4g}, L1 vs 3: p={ks_p_13:.4g}")

    # -----------------------------
    # 6. Plot 1 – Per-patient dot plot with stars
    # -----------------------------
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

    if use_correct:
        path = DIR + "Correct/"
    else:
        path = DIR
    plt.savefig(path + "PSMA_per_patient_dotplot.svg", format="svg")
    plt.show()

    # -----------------------------
    # 7. Plot 2 – Empirical CDF of pooled CCs
    # -----------------------------
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

    # -----------------------------
    # 8. Return pooled trial-level CCs
    # -----------------------------
    return all_cc[1], all_cc[2], all_cc[3]


def analyze_load_distributions_Nopseudo(
    mat_file_path, *,
    min_time_cells=5,
    top_percent=100,
    use_correct=False,
    random_seed=42,       # kept for backward compatibility, NOT used
    remove_zero_cc=True,
    verbose=True
):
    """
    Compare trial-level mean cross-correlations across load conditions,
    WITHOUT any trial subsampling / pseudopopulation.

    For each patient:
      * use separate_trials_by_load(...) to get load_d and t_fields,
      * require at least `min_time_cells` time cells,
      * for every trial in every load (1,2,3), compute a trial-level mean
        cross-correlation using process_trial_and_compute_mean_correlation,
      * store trial-level CCs per patient and pooled across patients.

    Then:
      * compute per-patient mean CC for each load → paired t-tests L1>L2, L2>L3, L1>L3,
      * compute pooled KS tests between loads at the trial level,
      * plot:
          1) per-patient dot plot with colored means + significance bars,
          2) empirical CDFs of pooled CCs with KS p-values.

    Parameters
    ----------
    mat_file_path : str
        Path to .mat file with 'neural_data'.
    min_time_cells : int
        Minimum number of time cells per patient required to include them.
    top_percent : float
        Passed to process_trial_and_compute_mean_correlation.
    use_correct : bool
        If True, only correct trials contribute (handled in separate_trials_by_load).
    random_seed : int
        Ignored (kept only for API compatibility).
    remove_zero_cc : bool
        If True, remove CC values equal to 0 before KS tests / plots.
    verbose : bool
        If True, shows progress bars and messages.

    Returns
    -------
    all_cc_load1, all_cc_load2, all_cc_load3 : list[float]
        Trial-level CC values pooled across all included patients for
        Load 1, Load 2, and Load 3.
    """

    def get_stars(p):
        if p < 1e-3:
            return "***"
        elif p < 1e-2:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return f"p = {p:.3f}"

    def add_sig_bar(ax, x1, x2, y, text, height=0.02, bar_width=-0.15):
        """
        Add a compact significance bar between x1 and x2 at height y.
        x1, x2 are in data coordinates (1,2,3 for loads).
        """
        ax.plot([x1 - bar_width, x1 - bar_width, x2 + bar_width, x2 + bar_width],
                [y, y + height, y + height, y], lw=1.5, color='black')
        ax.text((x1 + x2) / 2, y + height + 0.005, text,
                ha='center', va='bottom', fontsize=13)

    # ---- 0. load MATLAB ----
    m = scipy.io.loadmat(mat_file_path)
    nd = m['neural_data'][0]
    patient_ids = [int(e['patient_id'][0][0]) for e in nd]
    unique_ids = np.unique(patient_ids)

    # ---- 1. gather per-patient data (no subsampling) ----
    data_by_patient = {}
    p_iter = tqdm(unique_ids, desc="Gathering trial data") if verbose else unique_ids

    for pid in p_iter:
        load_d, t_fields = separate_trials_by_load(
            mat_file_path, pid, use_correct=use_correct
        )
        if load_d is None:
            continue
        if len(t_fields) < min_time_cells:
            if verbose:
                print(f"Skipping patient {pid}: only {len(t_fields)} time-cells "
                      f"(min required = {min_time_cells}).")
            continue
        data_by_patient[pid] = (load_d, t_fields)

    if not data_by_patient:
        raise RuntimeError("No patients qualified under the filters.")

    # ---- 2. compute trial-level correlations (NO gmin, NO rng) ----
    all_cc = {1: [], 2: [], 3: []}
    per_patient_cc = {pid: {1: [], 2: [], 3: []} for pid in data_by_patient}

    c_iter = tqdm(data_by_patient, desc="Computing CCs") if verbose else data_by_patient
    for pid in c_iter:
        load_d, t_fields = data_by_patient[pid]
        for lv in (1, 2, 3):
            trials_lv = load_d[lv]         # shape: n_trials × n_cells × n_bins
            for trial in trials_lv:
                val = process_trial_and_compute_mean_correlation(
                    trial, t_fields, top_percent=top_percent
                )
                all_cc[lv].append(val)
                per_patient_cc[pid][lv].append(val)

    # ---- 3. per-patient summary (mean) ----
    pps_mean = {lv: [] for lv in (1, 2, 3)}
    for pid in per_patient_cc:
        for lv in (1, 2, 3):
            vals = per_patient_cc[pid][lv]
            v = np.mean(vals) if len(vals) > 0 else np.nan
            pps_mean[lv].append(v)

    # Convert to arrays and drop patients with missing loads
    pps_mean[1] = np.asarray(pps_mean[1], dtype=float)
    pps_mean[2] = np.asarray(pps_mean[2], dtype=float)
    pps_mean[3] = np.asarray(pps_mean[3], dtype=float)

    valid_idx = (~np.isnan(pps_mean[1]) &
                 ~np.isnan(pps_mean[2]) &
                 ~np.isnan(pps_mean[3]))
    for lv in (1, 2, 3):
        pps_mean[lv] = pps_mean[lv][valid_idx]

    # Paired t-tests across patients (one-sided: L1>L2, etc.)
    if len(pps_mean[1]) > 1:
        _, p12_pat = ttest_rel(pps_mean[1], pps_mean[2], alternative='greater')
        _, p23_pat = ttest_rel(pps_mean[2], pps_mean[3], alternative='greater')
        _, p13_pat = ttest_rel(pps_mean[1], pps_mean[3], alternative='greater')
    else:
        p12_pat = p23_pat = p13_pat = 1.0  # not enough patients

    print(f"[PATIENT LEVEL]  L1>2  p={p12_pat:.4g}  "
          f"L2>3  p={p23_pat:.4g}  L1>3  p={p13_pat:.4g}")

    # ---- 4. optional removal of zero CCs ----
    if remove_zero_cc:
        for lv in (1, 2, 3):
            all_cc[lv] = [v for v in all_cc[lv] if v != 0]

    # ---- 5. sanity check and KS tests ----
    if any(len(all_cc[lv]) < 2 for lv in (1, 2, 3)):
        if verbose:
            print("Too few data points – skipping KS/CDF plots.")
        return all_cc[1], all_cc[2], all_cc[3]

    D_12, ks_p_12 = ks_2samp(all_cc[1], all_cc[2])
    D_23, ks_p_23 = ks_2samp(all_cc[2], all_cc[3])
    D_13, ks_p_13 = ks_2samp(all_cc[1], all_cc[3])

    if verbose:
        print(f"[TRIAL LEVEL] KS Load1 vs 2: p={ks_p_12:.4g}   "
              f"Load2 vs 3: p={ks_p_23:.4g}   "
              f"Load1 vs 3: p={ks_p_13:.4g}")

    # ---- 6. Plot 1: per-patient dot plot + stars ----
    sns.set_theme(style="whitegrid", context="talk")
    palette = ["royalblue", "seagreen", "firebrick"]

    fig, ax = plt.subplots(figsize=(5, 6))

    n_patients = len(pps_mean[1])
    for i in range(n_patients):
        yvals = [pps_mean[1][i], pps_mean[2][i], pps_mean[3][i]]
        ax.plot([1, 2, 3], yvals, color='grey', alpha=0.5, linewidth=1)
        ax.scatter([1, 2, 3], yvals, color='grey', alpha=0.8, zorder=5)

    # overlay load-specific means
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
    ax.set_ylabel("Mean CC per Patient", fontsize=12)
    ax.set_title("Per-Patient Mean CC (Dot Plot with Colored Means)", fontsize=14)
    ax.set_xlim(0.5, 3.5)

    sns.despine()
    plt.tight_layout()

    if use_correct:
        path = DIR + 'Correct/'
    else:
        path = DIR
    # plt.savefig(path + "per_patient_dotplot_colored_means.svg", format='svg')
    plt.show()

    # ---- 7. Plot 2: empirical CDFs of pooled CCs ----
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
                fontsize=12, color=col, fontweight='bold')

    ax.set_title("Cumulative Distribution of Cross Correlations", pad=12)
    ax.set_xlabel("Mean Pairwise Correlation")
    ax.set_ylabel("Cumulative Probability")
    ax.set_ylim(0, 1.01)

    sns.despine(trim=True)
    plt.tight_layout()
    # plt.savefig(path + "crosscorrelation.svg", format='svg')
    plt.show()

    # ---- 8. return raw lists ----
    return all_cc[1], all_cc[2], all_cc[3]


# analyze_load_distributions("3sig15_data.mat", min_time_cells=5, top_percent=100, use_correct=False, random_seed=20250710)
#analyze_load_distributions_Nopseudo("3sig15_raw.mat", min_time_cells=3, top_percent=100, use_correct=False) # random_seed=20250710)
# analyze_load_distributions("3sig15_data.mat", min_time_cells=5, top_percent=100, use_correct=True, random_seed=20250710)
analyze_load_distributions_PSMA_final("3sig15_data.mat", min_time_cells=3, use_correct=False)
# analyze_load_distributions("3sig15_raw.mat", min_time_cells=3, top_percent=100, use_correct=False, random_seed=20250710)
# analyze_load_distributions("3sig15_raw.mat", min_time_cells=3, top_percent=100, use_correct=True, random_seed=20250710)

