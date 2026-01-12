"""
Decode fast vs. slow reaction‑time trials using a pseudopopulation of single neurons.

This is an alternative to `repeated_cv_pseudo_population_per_class_colored_outliers_with_label_shuffling`,
re‑labeling trials by reaction time (RT) instead of memory‑load condition.

Main steps
-----------
1. Load the `neural_data` MATLAB struct (output of `create_neural_data`).
2. For each patient, compute per‑trial RT quantiles and label the
   *bottom* `quantile` fraction as **fast** and the *top* fraction as **slow**.
3. Extract time‑field‑centered firing rates for every neuron, keep trials
   belonging to the fast/slow bins, and discard RTs in the middle quantiles.
4. Down‑sample every neuron to the *global* minimum (# fast, # slow) so the
   classes are balanced across the whole pseudopopulation.
5. Repeated Monte‑Carlo cross‑validation (default = 1000 splits) with a linear
   SVM.  On each iteration we:
   * shuffle trials within class,
   * leave `test_per_class` trials per class out for testing,
   * train on the remaining ones,
   * accumulate per‑class accuracy + a 2×2 confusion matrix.
6. Build a null distribution by label‑shuffling inside the training set.
7. Return two tidy `pandas` DataFrames (`df_actual`, `df_null`) and, if
   `return_fig` is *True*, a `matplotlib` figure of the averaged confusion
   matrix.

Notes
-----
* Assumes **RT vectors are identical across neurons of the same patient**.
* Trials with NaN RT are dropped *before* quantile calculation.
* If `only_correct=True`, trials marked as incorrect are removed *before*
  quantiles are computed.
* Uses a `numpy.random.Generator`; set `random_state=None` (default) for
  stochastic runs, or an int for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.io
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Iterable, Tuple, Optional

import os
from typing import Iterable, Optional, Dict, Tuple, List
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
from scipy.stats import ttest_ind

DIR = '/Users/xiangxuankong/Desktop/Human TC Support WM/Figures/Figure S2/'

def _downsample_vector(rng: np.random.Generator, x: np.ndarray, n: int) -> np.ndarray:
    """Sample *n* unique elements from 1‑D array *x* (without replacement)."""
    if len(x) == n:
        return x.copy()
    idx = rng.choice(len(x), size=n, replace=False)
    return x[idx]


def _significance_stars(p: float) -> str:
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

# Identify and exclude extreme outliers using the IQR method
def remove_extreme_outliers(rt_values):
    q1 = np.percentile(rt_values, 25)
    q3 = np.percentile(rt_values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return rt_values[(rt_values >= lower_bound) & (rt_values <= upper_bound)]


def repeated_cv_pseudo_population_rt_fast_slow(
    mat_file_path: str,
    patient_ids: Iterable[int] = range(1, 22),
    m: int = 0,
    num_windows: int = 0,
    test_per_class: int = 11,
    quantile: float = 0.25,                 # ← recommended < 0.5
    n_iterations: int = 1000,
    random_state: Optional[int] = 42,
    only_correct: bool = False,
    plot: bool = True,
    stratify_by_load: bool = False,         # ← NEW FLAG
):
    """
    Pseudopopulation decoder for *fast* vs *slow* RT trials.

    If `stratify_by_load` is True, fast/slow quantiles are computed **separately
    for each distinct value in `trial_load`**, and the resulting indices are
    merged before building the pseudo-population.
    """
    rng = np.random.default_rng(random_state)

    # -------------------------------------------------------------
    # 1) Load MATLAB struct
    # -------------------------------------------------------------
    mat_data   = scipy.io.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
    neural_data = mat_data["neural_data"].flatten()

    # -------------------------------------------------------------
    # 2) Build per-patient caches
    # -------------------------------------------------------------
    patient_dict   = {}
    patient_rt_cache = {}

    for entry in neural_data:
        pid      = int(entry.patient_id)
        frates   = entry.firing_rates
        correct  = entry.trial_correctness.astype(int)
        rt       = entry.trial_RT
        load     = entry.trial_load.astype(int)           # ← NEW
        tfield   = int(entry.time_field) - 1

        if only_correct:
            keep   = (correct == 1)
            frates = frates[keep, :]
            rt     = rt[keep]
            load   = load[keep]

        nan_mask = ~np.isnan(rt)
        frates   = frates[nan_mask, :]
        rt       = rt[nan_mask]
        load     = load[nan_mask]
        if frates.size == 0:
            continue

        if pid not in patient_dict:
            patient_dict[pid]     = []
            patient_rt_cache[pid] = rt
        else:
            if not np.array_equal(patient_rt_cache[pid], rt):
                raise ValueError(f"RT vectors differ across neurons for patient {pid}.")

        #  tuple = (firing rates, RT, load, tfield)
        patient_dict[pid].append((frates, rt, load, tfield))   # ← CHANGED

    # -------------------------------------------------------------
    # 3) Patient inclusion filter – unchanged
    # -------------------------------------------------------------

    valid_patients = [pid for pid in patient_ids if (pid in patient_dict and len(patient_dict[pid]) >= m)]
    if not valid_patients:
        print("No patients meet the inclusion criteria.")
        return None

    # -------------------------------------------------------------
    # 4) Label fast / slow & collect neurons
    # -------------------------------------------------------------
    all_neurons = []
    global_fast_rts, global_slow_rts = [], []

    for pid in valid_patients:
        rt_vec   = patient_rt_cache[pid]

        if stratify_by_load:
            # --- NEW:  quantiles within each load -------------------------
            # The load vector is identical for every neuron of this patient,
            # so grab it from the first cached tuple.
            load_vec = patient_dict[pid][0][2]

            fast_idx = np.zeros_like(rt_vec,  dtype=bool)
            slow_idx = np.zeros_like(rt_vec,  dtype=bool)

            for L in np.unique(load_vec):
                mask = (load_vec == L)
                if mask.sum() < 4:        # need ≥2 trials per tail
                    continue
                q_low  = np.quantile(rt_vec[mask],  quantile)
                q_high = np.quantile(rt_vec[mask], 1.0 - quantile)
                fast_idx |= mask & (rt_vec <= q_low)
                slow_idx |= mask & (rt_vec >= q_high)
        else:
            # --- Legacy global quantiles ---------------------------------
            q_low  = np.quantile(rt_vec,  quantile)
            q_high = np.quantile(rt_vec, 1.0 - quantile)
            fast_idx = rt_vec <= q_low
            slow_idx = rt_vec >= q_high
        # -----------------------------------------------------------------

        if fast_idx.sum() < 2 or slow_idx.sum() < 2:
            continue

        global_fast_rts.append(rt_vec[fast_idx])
        global_slow_rts.append(rt_vec[slow_idx])

        for frates, _, _, tfield in patient_dict[pid]:         # ← CHANGED
            start_idx  = max(0, tfield - num_windows)
            end_idx    = min(frates.shape[1], tfield + num_windows + 1)
            mean_rates = frates[:, start_idx:end_idx].mean(axis=1)

            fast_rates = mean_rates[fast_idx]
            slow_rates = mean_rates[slow_idx]

            if fast_rates.size and slow_rates.size:
                all_neurons.append({"fast": fast_rates, "slow": slow_rates})

    # ---------------------------------------------------------------------
    # 5) Visualise RT distributions *before* ML
    # ---------------------------------------------------------------------

    if not all_neurons:
        print("No neurons found with both fast and slow trials present.")
        return None

    # ← Move the concatenation here, after all_neurons is validated
    fast_rt_concat = np.concatenate(global_fast_rts) if global_fast_rts else np.array([])
    slow_rt_concat = np.concatenate(global_slow_rts) if global_slow_rts else np.array([])

    # Additional safeguard if both are empty (can happen under stratification)
    if fast_rt_concat.size == 0 or slow_rt_concat.size == 0:
        print("Fast or slow RT arrays are empty after stratification.")
        return None
    

    fast_rt_filtered = remove_extreme_outliers(fast_rt_concat)
    slow_rt_filtered = remove_extreme_outliers(slow_rt_concat)

    if plot:
        rt_df = pd.DataFrame({
            "RT": np.concatenate([fast_rt_filtered, slow_rt_filtered]),
            "Class": ["Fast"] * len(fast_rt_filtered) + ["Slow"] * len(slow_rt_filtered),
        })

        plt.figure(figsize=(3.5, 6))
        sns.boxplot(data=rt_df, x="Class", y="RT", palette=["skyblue", "salmon"],
                    width=0.3, showfliers=True, flierprops=dict(marker="o", markersize=4))
        plt.ylabel("Reaction time (s)")
        plt.title("RT distribution (fast vs. slow quantiles)")
        plt.tight_layout()
        plt.savefig(DIR + 'Reaction_time.svg', format = 'svg')
        plt.show()


        print("\n---- RT summary ----")
        print(f"Fast mean ± SD:  {fast_rt_concat.mean():.3f} ± {fast_rt_concat.std():.3f} s  (n={len(fast_rt_concat)})")
        print(f"Slow mean ± SD: {slow_rt_concat.mean():.3f} ± {slow_rt_concat.std():.3f} s  (n={len(slow_rt_concat)})")

    # ---------------------------------------------------------------------
    # 6) Determine global min trials per class & down‑sample
    # ---------------------------------------------------------------------
    min_fast = min(len(n["fast"]) for n in all_neurons)
    min_slow = min(len(n["slow"]) for n in all_neurons)

    if test_per_class > min_fast or test_per_class > min_slow:
        print(f"Requested test_per_class={test_per_class}, but mins are fast={min_fast}, slow={min_slow}.")
        return None

    for neuron in all_neurons:
        neuron["fast"] = rng.choice(neuron["fast"], size=min_fast, replace=False)
        neuron["slow"] = rng.choice(neuron["slow"], size=min_slow, replace=False)

    n_neurons = len(all_neurons)
    fast_mat = np.stack([n["fast"] for n in all_neurons], axis=1)
    slow_mat = np.stack([n["slow"] for n in all_neurons], axis=1)

    print("\n--- Pseudopopulation Info (RT fast vs. slow) ---")
    print(f"  Number of neurons: {n_neurons}")
    print(f"  Trials per class: fast={min_fast}, slow={min_slow}")

    # ---------------------------------------------------------------------
    # 7) Monte‑Carlo CV – actual labels
    # ---------------------------------------------------------------------
    acc_fast_actual, acc_slow_actual = [], []
    cm_accumulator = np.zeros((2, 2))
    acc_overall_actual = []


    for _ in range(n_iterations):
        idx_fast = rng.permutation(min_fast)
        idx_slow = rng.permutation(min_slow)
        test_fast, train_fast = idx_fast[:test_per_class], idx_fast[test_per_class:]
        test_slow, train_slow = idx_slow[:test_per_class], idx_slow[test_per_class:]

        X_train = np.vstack([fast_mat[train_fast], slow_mat[train_slow]])
        y_train = np.array([0] * len(train_fast) + [1] * len(train_slow))
        X_test = np.vstack([fast_mat[test_fast], slow_mat[test_slow]])
        y_test = np.array([0] * len(test_fast) + [1] * len(test_slow))

        clf = SVC(kernel="linear", random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        cm_accumulator += confusion_matrix(y_test, y_pred, labels=[0, 1])
        acc_fast_actual.append((y_pred[y_test == 0] == 0).mean() * 100)
        acc_slow_actual.append((y_pred[y_test == 1] == 1).mean() * 100)

        # inside the loop:
        acc = (y_pred == y_test).mean() * 100
        acc_overall_actual.append(acc)

    cm_pct = cm_accumulator / cm_accumulator.sum(axis=1, keepdims=True) * 100

    # ---------------------------------------------------------------------
    # 8) Monte‑Carlo CV – label‑shuffled null
    # ---------------------------------------------------------------------
    acc_fast_null, acc_slow_null = [], []
    acc_overall_null = []

    for _ in range(n_iterations):
        idx_fast = rng.permutation(min_fast)
        idx_slow = rng.permutation(min_slow)
        test_fast, train_fast = idx_fast[:test_per_class], idx_fast[test_per_class:]
        test_slow, train_slow = idx_slow[:test_per_class], idx_slow[test_per_class:]

        X_train = np.vstack([fast_mat[train_fast], slow_mat[train_slow]])
        y_train = rng.permutation(np.array([0] * len(train_fast) + [1] * len(train_slow)))
        X_test = np.vstack([fast_mat[test_fast], slow_mat[test_slow]])
        y_test = np.array([0] * len(test_fast) + [1] * len(test_slow))

        clf = SVC(kernel="linear", random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc_fast_null.append((y_pred[y_test == 0] == 0).mean() * 100)
        acc_slow_null.append((y_pred[y_test == 1] == 1).mean() * 100)
        # inside the loop:
        acc = (y_pred == y_test).mean() * 100
        acc_overall_null.append(acc)

    # ---------------------------------------------------------------------
    # 9) DataFrames & plotting
    # ---------------------------------------------------------------------
    df_actual = pd.DataFrame({
        "Class": ["Fast"] * n_iterations + ["Slow"] * n_iterations,
        "Accuracy": acc_fast_actual + acc_slow_actual,
    })
    df_null = pd.DataFrame({
        "Class": ["Fast"] * n_iterations + ["Slow"] * n_iterations,
        "Accuracy": acc_fast_null + acc_slow_null,
    })

    if plot:
        # confusion matrix heat‑map
        plt.figure(figsize=(4.3, 4))
        sns.heatmap(cm_pct, annot=np.array([[f"{v:.1f}%" for v in row] for row in cm_pct]),
                    fmt="", cmap="Blues", vmin=0, vmax=100,
                    xticklabels=["Fast", "Slow"], yticklabels=["Fast", "Slow"],
                    cbar_kws={"label": "Percentage (%)"})
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Averaged confusion matrix (fast vs. slow)")
        plt.tight_layout()
        plt.savefig(DIR + 'confusion.svg', format = 'svg')
        plt.show()

        # accuracy box‑plot
        plt.figure(figsize=(3.5, 6))
        sns.boxplot(data=df_actual, x="Class", y="Accuracy", width=0.6,
                    palette=["skyblue", "salmon"], showfliers=True,
                    flierprops=dict(marker="o", markersize=4))
        plt.axhline(50, ls="--", c="gray")
        plt.ylim(0, df_actual.Accuracy.max() + 5)
        plt.title("Decoder accuracy (actual labels)")

        for i, (act, nul) in enumerate([(acc_fast_actual, acc_fast_null), (acc_slow_actual, acc_slow_null)]):
            p_perm = np.mean(np.array(nul) >= np.mean(act))
            star = _significance_stars(p_perm)
            plt.text(i, max(act) + 1.0, star, ha="center", va="bottom", fontsize=14, fontweight="bold")

        plt.tight_layout()
       
        plt.show()

    df_actual = pd.DataFrame({"Accuracy": acc_overall_actual})
    df_null = pd.DataFrame({"Accuracy": acc_overall_null})

    if plot:
        plt.figure(figsize=(3.5, 6))
        sns.boxplot(data=df_actual, y="Accuracy", width=0.6, color="skyblue", showfliers=True,
                    flierprops=dict(marker="o", markersize=4))
        plt.axhline(50, ls="--", c="gray")
        plt.ylim(0, df_actual.Accuracy.max() + 5)
        plt.title("Decoder accuracy (overall)")

        p_perm = np.mean(np.array(acc_overall_null) >= np.mean(acc_overall_actual))
        star = _significance_stars(p_perm)
        plt.text(0, max(acc_overall_actual) + 1.0, star, ha="center", va="bottom", fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.savefig(DIR + 'decoding_acc.svg', format = 'svg')
        plt.show()


    # ---------------------------------------------------------------------
    # 10) Console summary of decoder stats
    # ---------------------------------------------------------------------
    print("\n---- Decoder performance (mean ± SD) ----")
    for name, act, nul in [("Fast", acc_fast_actual, acc_fast_null), ("Slow", acc_slow_actual, acc_slow_null)]:
        mean_act, std_act = np.mean(act), np.std(act)
        p_perm = np.mean(np.array(nul) >= mean_act)
        print(f"{name}: {mean_act:.2f} ± {std_act:.2f}%    perm p = {p_perm:.3e}")

    print("\n---- Decoder performance (mean ± SD) ----")
    mean_act = np.mean(acc_overall_actual)
    std_act = np.std(acc_overall_actual)
    p_perm = np.mean(np.array(acc_overall_null) >= mean_act)
    print(f"Overall: {mean_act:.2f} ± {std_act:.2f}%    perm p = {p_perm:.3e}")

    return df_actual, df_null


def decode_rt_fast_slow_by_region(
    mat_file_path: str,
    patient_ids: Iterable[int] = range(1, 22),
    m: int = 0,
    num_windows: int = 0,
    test_per_class: int = 8,
    quantile: float = 0.25,               # fast ≤ q, slow ≥ 1-q
    n_iterations: int = 1000,
    n_shuffles: int = 1,
    random_state: Optional[int] = 42,
    only_correct: bool = False,
    stratify_by_load: bool = True,        # recommended for RT
    show_plots: bool = True,
    output_dir: Optional[str] = None,
    label_map: Optional[Dict[str, str]] = None,
    region_color_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Tuple[List[float], List[float], float]]:
    """
    Pseudopopulation FAST vs SLOW RT decoding *per brain region*.

    For each region:
      • Build neurons' mean rates around the neuron's time-field (±num_windows)
      • Define FAST/SLOW trials per patient (global or stratified by load)
      • Balance trials per neuron (min across neurons)
      • Monte-Carlo CV accuracy (linear SVM) vs label-shuffled null

    Returns:
        results: {region: (real_acc_list, null_acc_list, p_vs_null)}
    """
    rng = np.random.default_rng(random_state)

    # -------- fonts like your showTimeCellAverages figures --------
    plt.rcParams.update({
        "font.size": 16, "axes.titlesize": 16, "axes.labelsize": 16,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16
    })
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    DEFAULT_MAP = {
        'dorsal_anterior_cingulate_cortex': 'DaCC',
        'pre_supplementary_motor_area': 'PSMA',
        'hippocampus': 'HPC',
        'amygdala': 'AMY',
        'ventral_medial_prefrontal_cortex': 'vmPFC',
    }
    if label_map is None:
        label_map = DEFAULT_MAP

    if region_color_map is None:
        region_color_map = {"HPC": "#377eb8", "AMY": "#e41a1c", "vmPFC": "#4daf4a",
                            "PSMA": "#984ea3", "DaCC": "#ff7f00"}

    # -------------------- 1) Load MATLAB struct --------------------
    mat = scipy.io.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
    neural_data = mat["neural_data"].flatten()

    # -------------------- 2) Organize by patient & region ----------
    # patient caches ensure identical RT/load across neurons for same patient
    patient_rt: Dict[int, np.ndarray] = {}
    patient_load: Dict[int, np.ndarray] = {}
    # region -> pid -> list of neuron dicts; neuron dict holds 'fast','slow' (filled later)
    region_neurons: Dict[str, Dict[int, list]] = {}

    # First pass: gather per-neuron stats needed to compute mean rates later
    per_patient_entries: Dict[int, list] = {}  # pid -> list of per-neuron tuples

    for entry in neural_data:
        pid = int(entry.patient_id)
        if pid not in patient_ids:
            continue

        region_raw = entry.brain_region
        if isinstance(region_raw, bytes):
            region_raw = region_raw.decode()
        region_key = label_map.get(_strip_lat(region_raw), _strip_lat(region_raw))

        fr = np.asarray(entry.firing_rates)            # trials × bins
        correct = np.asarray(entry.trial_correctness, int)
        rt = np.asarray(entry.trial_RT, float)
        load = np.asarray(entry.trial_load, int)
        tf_bin = int(entry.time_field) - 1            # 0-based

        if only_correct:
            keep = (correct == 1)
            fr = fr[keep, :]
            rt = rt[keep]
            load = load[keep]

        # Drop trials with NaN RT
        keep = ~np.isnan(rt)
        fr = fr[keep, :]
        rt = rt[keep]
        load = load[keep]
        if fr.size == 0:
            continue

        # consistency across neurons of same patient
        if pid not in patient_rt:
            patient_rt[pid] = rt
            patient_load[pid] = load
        else:
            if not np.array_equal(patient_rt[pid], rt):
                raise ValueError(f"RT vectors differ across neurons for patient {pid}.")
            if not np.array_equal(patient_load[pid], load):
                raise ValueError(f"Load vectors differ across neurons for patient {pid}.")

        per_patient_entries.setdefault(pid, []).append((region_key, fr, tf_bin))

    # -------------------- 3) Build fast/slow masks per patient -----
    fast_slow_masks: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for pid, rt_vec in patient_rt.items():
        load_vec = patient_load[pid]
        if stratify_by_load:
            fast_idx = np.zeros_like(rt_vec, dtype=bool)
            slow_idx = np.zeros_like(rt_vec, dtype=bool)
            for L in np.unique(load_vec):
                msk = (load_vec == L)
                if msk.sum() < 4:  # need at least some trials to define tails
                    continue
                q_low = np.quantile(rt_vec[msk], quantile)
                q_high = np.quantile(rt_vec[msk], 1.0 - quantile)
                fast_idx |= msk & (rt_vec <= q_low)
                slow_idx |= msk & (rt_vec >= q_high)
        else:
            q_low = np.quantile(rt_vec, quantile)
            q_high = np.quantile(rt_vec, 1.0 - quantile)
            fast_idx = rt_vec <= q_low
            slow_idx = rt_vec >= q_high
        if fast_idx.sum() < 2 or slow_idx.sum() < 2:
            # no usable split for this patient; drop them
            continue
        fast_slow_masks[pid] = (fast_idx, slow_idx)

    # -------------------- 4) For each neuron, extract rates ----------
    for pid, tuples in per_patient_entries.items():
        if pid not in fast_slow_masks:
            continue
        fast_idx, slow_idx = fast_slow_masks[pid]

        for region_key, fr, tf_bin in tuples:
            s = max(0, tf_bin - num_windows)
            e = min(fr.shape[1], tf_bin + num_windows + 1)
            mean_rates = fr[:, s:e].mean(axis=1)

            fast_rates = mean_rates[fast_idx]
            slow_rates = mean_rates[slow_idx]
            if fast_rates.size == 0 or slow_rates.size == 0:
                continue

            region_neurons.setdefault(region_key, {}).setdefault(pid, []).append(
                {"fast": fast_rates, "slow": slow_rates}
            )

    # -------------------- 5) Filter by m neurons per patient --------
    # and flatten to region-wise neuron lists
    region_flat: Dict[str, List[dict]] = {}
    for region, per_pid in list(region_neurons.items()):
        pts_ok = {pid: cells for pid, cells in per_pid.items() if len(cells) >= m}
        flat = [cell for cells in pts_ok.values() for cell in cells]
        if flat:
            region_flat[region] = flat

    if not region_flat:
        print("No region has neurons that satisfy inclusion criteria.")
        return {}

    # -------------------- 6) Core decoder per region ----------------
    def _decode_region(neuron_list: List[dict]):
        # balance per neuron
        min_fast = min(len(n["fast"]) for n in neuron_list)
        min_slow = min(len(n["slow"]) for n in neuron_list)
        if test_per_class > min_fast or test_per_class > min_slow:
            return None

        # down-sample each neuron to its per-class min
        neurons_ds = []
        for n in neuron_list:
            f = resample(n["fast"], replace=False, n_samples=min_fast, random_state=random_state)
            s = resample(n["slow"], replace=False, n_samples=min_slow, random_state=random_state)
            neurons_ds.append({"fast": f, "slow": s})

        fast_mat = np.column_stack([n["fast"] for n in neurons_ds])
        slow_mat = np.column_stack([n["slow"] for n in neurons_ds])

        real = []
        for _ in range(n_iterations):
            idx_f = rng.permutation(min_fast)
            idx_s = rng.permutation(min_slow)
            ts_f, tr_f = idx_f[:test_per_class], idx_f[test_per_class:]
            ts_s, tr_s = idx_s[:test_per_class], idx_s[test_per_class:]

            Xtr = np.vstack([fast_mat[tr_f], slow_mat[tr_s]])
            ytr = np.array([0]*len(tr_f) + [1]*len(tr_s))
            Xts = np.vstack([fast_mat[ts_f], slow_mat[ts_s]])
            yts = np.array([0]*len(ts_f) + [1]*len(ts_s))

            clf = SVC(kernel="linear", random_state=random_state)
            clf.fit(Xtr, ytr)
            real.append((clf.predict(Xts) == yts).mean() * 100)

        null = []
        for _ in range(n_shuffles):
            for _ in range(n_iterations):
                idx_f = rng.permutation(min_fast)
                idx_s = rng.permutation(min_slow)
                ts_f, tr_f = idx_f[:test_per_class], idx_f[test_per_class:]
                ts_s, tr_s = idx_s[:test_per_class], idx_s[test_per_class:]

                Xtr = np.vstack([fast_mat[tr_f], slow_mat[tr_s]])
                ytr = np.array([0]*len(tr_f) + [1]*len(tr_s))
                ytr = rng.permutation(ytr)  # shuffle labels

                Xts = np.vstack([fast_mat[ts_f], slow_mat[ts_s]])
                yts = np.array([0]*len(ts_f) + [1]*len(ts_s))

                clf = SVC(kernel="linear", random_state=random_state)
                clf.fit(Xtr, ytr)
                null.append((clf.predict(Xts) == yts).mean() * 100)

        p_vs_null = (np.array(null) >= np.mean(real)).mean()
        return real, null, p_vs_null

    results: Dict[str, Tuple[List[float], List[float], float]] = {}
    for region, neuron_list in region_flat.items():
        res = _decode_region(neuron_list)
        if res is None:
            print(f"[skip] {region}: too few balanced trials (need ≥ {test_per_class} per class).")
            continue
        real_acc, null_acc, p_null = res
        results[region] = (real_acc, null_acc, p_null)
        print(f"{region}: {np.mean(real_acc):.2f}% ± {np.std(real_acc):.2f}% "
              f"(N={n_iterations}), p_null={p_null:.3g}")

    if not results or not show_plots:
        return results

    # -------------------- 7) Combined plots ------------------------
    regions = list(results.keys())
    # Optional: order regions (e.g., show HPC & AMY prominently)
    preferred_order = ['HPC', 'AMY', 'vmPFC', 'PSMA', 'DaCC']
    regions = sorted(regions, key=lambda r: (preferred_order.index(r)
                     if r in preferred_order else len(preferred_order), r))

    acc_lists = [results[r][0] for r in regions]
    p_vs_null = [results[r][2] for r in regions]
    colors = [region_color_map.get(r, "#999999") for r in regions]

    # Boxplot per region (real acc)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(regions)), 6))
    box = ax.boxplot(acc_lists, patch_artist=True, labels=regions, widths=0.7)
    for patch, c in zip(box['boxes'], colors):
        patch.set_facecolor(c); patch.set_edgecolor('black')
    for elem in ['whiskers', 'caps', 'medians']:
        plt.setp(box[elem], color='black')

    ax.axhline(50, ls='--', c='gray', lw=1)
    y_max = max(max(a) for a in acc_lists) if acc_lists else 60
    # annotate per-region significance vs null
    for x, p in enumerate(p_vs_null, start=1):
        star = "" if np.isnan(p) else (_stars(p) if p < 0.05 else "")
        if star:
            ax.text(x, y_max + 1.0, star, ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, y_max + 6)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("RT (Fast vs Slow) decoding by brain region")
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "rt_by_region.svg"), format="svg")
    plt.show()

    # Optional: pairwise region comparisons (Welch's t-tests on real accuracies)
    print("\nPairwise Welch t-tests (region vs region, real accuracies):")
    for i in range(len(regions)):
        for j in range(i+1, len(regions)):
            a, b = np.array(acc_lists[i]), np.array(acc_lists[j])
            t, p = ttest_ind(a, b, equal_var=False)
            print(f"{regions[i]} vs {regions[j]}: p = {p:.4g} ({_stars(p)})")

    return results

def repeated_cv_pseudo_population_rt_fast_slow_135(
    mat_file_path: str,
    patient_ids: Iterable[int] = range(1, 22),
    m: int = 0,
    num_windows: int = 0,
    test_per_class: int = 12,
    quantile: float = 0.5,
    n_iterations: int = 1000,
    random_state: Optional[int] = 42,
    only_correct: bool = False,
    min_trials: int = 134,              # ← NEW
    plot: bool = True,
):
    """Pseudopopulation decoder (fast vs. slow RT).

    Parameters
    ----------
    min_trials : int, default 109
        Minimum number of trials a patient must have after optional
        `only_correct` filtering *and* NaN removal. Patients with fewer
        trials are excluded (this skips those with exactly 108 trials).
    """

    rng = np.random.default_rng(random_state)

    # ---------------------------------------------------------------------
    # 1) Load MATLAB struct
    # ---------------------------------------------------------------------
    mat_data = scipy.io.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
    neural_data = mat_data["neural_data"].flatten()

    # ---------------------------------------------------------------------
    # 2) Build per‑patient caches
    # ---------------------------------------------------------------------
    patient_dict = {}
    patient_rt_cache = {}

    for entry in neural_data:
        pid = int(entry.patient_id)
        frates = entry.firing_rates
        correct = entry.trial_correctness.astype(int)
        rt = entry.trial_RT
        tfield = int(entry.time_field) - 1

        if only_correct:
            keep = (correct == 1)
            frates = frates[keep]
            rt = rt[keep]

        nan_mask = ~np.isnan(rt)
        frates = frates[nan_mask]
        rt = rt[nan_mask]
        if frates.size == 0:
            continue

        if pid not in patient_dict:
            patient_dict[pid] = []
            patient_rt_cache[pid] = rt
        else:
            if not np.array_equal(patient_rt_cache[pid], rt):
                raise ValueError(f"RT vectors differ across neurons for patient {pid}.")

        patient_dict[pid].append((frates, rt, tfield))

    # ---------------------------------------------------------------------
    # 3) Patient inclusion filter (>= m neurons AND >= min_trials trials)
    # ---------------------------------------------------------------------
    valid_patients = []
    for pid in patient_ids:
        if pid not in patient_dict:
            continue
        if len(patient_dict[pid]) < m:
            continue
        if len(patient_rt_cache[pid]) < min_trials:
            continue
        valid_patients.append(pid)

    if not valid_patients:
        print("No patients meet the inclusion criteria after trial‑count filter.")
        return None

    # ---------------------------------------------------------------------
    # 4) Label fast / slow & collect neurons
    # ---------------------------------------------------------------------
    all_neurons = []
    global_fast_rts, global_slow_rts = [], []

    for pid in valid_patients:
        rt_vec = patient_rt_cache[pid]
        q_low = np.quantile(rt_vec, quantile)
        q_high = np.quantile(rt_vec, 1 - quantile)
        fast_idx = rt_vec <= q_low
        slow_idx = rt_vec >= q_high

        if fast_idx.sum() < 2 or slow_idx.sum() < 2:
            continue

        global_fast_rts.append(rt_vec[fast_idx])
        global_slow_rts.append(rt_vec[slow_idx])

        for frates, _, tfield in patient_dict[pid]:
            start = max(0, tfield - num_windows)
            end = min(frates.shape[1], tfield + num_windows + 1)
            mean_rates = frates[:, start:end].mean(axis=1)
            fast_rates = mean_rates[fast_idx]
            slow_rates = mean_rates[slow_idx]
            if fast_rates.size and slow_rates.size:
                all_neurons.append({"fast": fast_rates, "slow": slow_rates})

    if not all_neurons:
        print("No neurons found with both fast and slow trials present.")
        return None

    fast_rt_concat = np.concatenate(global_fast_rts)
    slow_rt_concat = np.concatenate(global_slow_rts)

    # ---------------------------------------------------------------------
    # 5) Visualise RT distributions *before* ML
    # ---------------------------------------------------------------------
    if plot:
        rt_df = pd.DataFrame({
            "RT": np.concatenate([fast_rt_concat, slow_rt_concat]),
            "Class": ["Fast"] * len(fast_rt_concat) + ["Slow"] * len(slow_rt_concat),
        })
        plt.figure(figsize=(3.5, 6))
        sns.boxplot(data=rt_df, x="Class", y="RT", palette=["skyblue", "salmon"], width=0.6,
                    showfliers=True, flierprops=dict(marker="o", markersize=4))
        plt.ylabel("Reaction time (s)")
        plt.title("RT distribution (fast vs. slow quantiles)")
        plt.tight_layout()
        plt.show()

        print("\n---- RT summary ----")
        print(f"Fast mean ± SD:  {fast_rt_concat.mean():.3f} ± {fast_rt_concat.std():.3f} s  (n={len(fast_rt_concat)})")
        print(f"Slow mean ± SD: {slow_rt_concat.mean():.3f} ± {slow_rt_concat.std():.3f} s  (n={len(slow_rt_concat)})")

    # ---------------------------------------------------------------------
    # 6) Down‑sample to global min trials per class
    # ---------------------------------------------------------------------
    min_fast = min(len(n["fast"]) for n in all_neurons)
    min_slow = min(len(n["slow"]) for n in all_neurons)

    if test_per_class > min_fast or test_per_class > min_slow:
        print(f"Requested test_per_class={test_per_class}, but mins are fast={min_fast}, slow={min_slow}.")
        return None

    for neuron in all_neurons:
        neuron["fast"] = rng.choice(neuron["fast"], size=min_fast, replace=False)
        neuron["slow"] = rng.choice(neuron["slow"], size=min_slow, replace=False)

    n_neurons = len(all_neurons)
    fast_mat = np.stack([n["fast"] for n in all_neurons], axis=1)
    slow_mat = np.stack([n["slow"] for n in all_neurons], axis=1)

    print("\n--- Pseudopopulation Info (RT fast vs. slow) ---")
    print(f"  Number of neurons: {n_neurons}")
    print(f"  Trials per class: fast={min_fast}, slow={min_slow}")

    # ---------------------------------------------------------------------
    # 7) Monte‑Carlo CV – actual labels
    # ---------------------------------------------------------------------
    acc_fast_actual, acc_slow_actual = [], []
    cm_accum = np.zeros((2, 2))

    for _ in range(n_iterations):
        idx_fast = rng.permutation(min_fast)
        idx_slow = rng.permutation(min_slow)
        test_fast, train_fast = idx_fast[:test_per_class], idx_fast[test_per_class:]
        test_slow, train_slow = idx_slow[:test_per_class], idx_slow[test_per_class:]

        X_train = np.vstack([fast_mat[train_fast], slow_mat[train_slow]])
        y_train = np.array([0] * len(train_fast) + [1] * len(train_slow))
        X_test = np.vstack([fast_mat[test_fast], slow_mat[test_slow]])
        y_test = np.array([0] * len(test_fast) + [1] * len(test_slow))

        clf = SVC(kernel="linear", random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cm_accum += confusion_matrix(y_test, y_pred, labels=[0, 1])
        acc_fast_actual.append((y_pred[y_test == 0] == 0).mean() * 100)
        acc_slow_actual.append((y_pred[y_test == 1] == 1).mean() * 100)

    cm_pct = cm_accum / cm_accum.sum(axis=1, keepdims=True) * 100

    # ---------------------------------------------------------------------
    # 8) Label‑shuffled null distribution
    # ---------------------------------------------------------------------
    acc_fast_null, acc_slow_null = [], []
    for _ in range(n_iterations):
        idx_fast = rng.permutation(min_fast)
        idx_slow = rng.permutation(min_slow)
        test_fast, train_fast = idx_fast[:test_per_class], idx_fast[test_per_class:]
        test_slow, train_slow = idx_slow[:test_per_class], idx_slow[test_per_class:]

        X_train = np.vstack([fast_mat[train_fast], slow_mat[train_slow]])
        y_train = rng.permutation(np.array([0] * len(train_fast) + [1] * len(train_slow)))
        X_test = np.vstack([fast_mat[test_fast], slow_mat[test_slow]])
        y_test = np.array([0] * len(test_fast) + [1] * len(test_slow))

        clf = SVC(kernel="linear", random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc_fast_null.append((y_pred[y_test == 0] == 0).mean() * 100)
        acc_slow_null.append((y_pred[y_test == 1] == 1).mean() * 100)

    # ---------------------------------------------------------------------
    # 9) DataFrames & plots
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # 9) DataFrames & plotting
    # ---------------------------------------------------------------------
    df_actual = pd.DataFrame({
        "Class": ["Fast"] * n_iterations + ["Slow"] * n_iterations,
        "Accuracy": acc_fast_actual + acc_slow_actual,
    })
    df_null = pd.DataFrame({
        "Class": ["Fast"] * n_iterations + ["Slow"] * n_iterations,
        "Accuracy": acc_fast_null + acc_slow_null,
    })

    if plot:
        # confusion matrix heat‑map
        plt.figure(figsize=(4.3, 4))
        sns.heatmap(cm_pct, annot=np.array([[f"{v:.1f}%" for v in row] for row in cm_pct]),
                    fmt="", cmap="Blues", vmin=0, vmax=100,
                    xticklabels=["Fast", "Slow"], yticklabels=["Fast", "Slow"],
                    cbar_kws={"label": "Percentage (%)"})
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Averaged confusion matrix (fast vs. slow)")
        plt.tight_layout()
        plt.show()

        # accuracy box‑plot
        plt.figure(figsize=(3.5, 6))
        sns.boxplot(data=df_actual, x="Class", y="Accuracy", width=0.6,
                    palette=["skyblue", "salmon"], showfliers=True,
                    flierprops=dict(marker="o", markersize=4))
        plt.axhline(50, ls="--", c="gray")
        plt.ylim(0, df_actual.Accuracy.max() + 5)
        plt.title("Decoder accuracy (actual labels)")

        for i, (act, nul) in enumerate([(acc_fast_actual, acc_fast_null), (acc_slow_actual, acc_slow_null)]):
            p_perm = np.mean(np.array(nul) >= np.mean(act))
            star = _significance_stars(p_perm)
            plt.text(i, max(act) + 1.0, star, ha="center", va="bottom", fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------------------
    # 10) Console summary of decoder stats
    # ---------------------------------------------------------------------
    print("\n---- Decoder performance (mean ± SD) ----")
    for name, act, nul in [("Fast", acc_fast_actual, acc_fast_null), ("Slow", acc_slow_actual, acc_slow_null)]:
        mean_act, std_act = np.mean(act), np.std(act)
        p_perm = np.mean(np.array(nul) >= mean_act)
        print(f"{name}: {mean_act:.2f} ± {std_act:.2f}%    perm p = {p_perm:.3e}")

    return df_actual, df_null

def repeated_cv_pseudo_population_rt_fast_slow_by_load(
    mat_file_path: str,
    patient_ids: Iterable[int] = range(1, 22),
    m: int = 0,
    num_windows: int = 0,
    test_per_class: int = 3,
    quantile: float = 0.25,
    n_iterations: int = 1_000,
    random_state: Optional[int] = None,
    only_correct: bool = False,
    plot: bool = True,
):
    """Run separate fast‑vs‑slow decoders inside each memory‑load condition.

    Returns
    -------
    df_actual, df_null : pandas.DataFrame
        Columns = ["Load", "Class", "Accuracy"].  Load is an int (1/2/3).
    """

    rng = np.random.default_rng(random_state)
    mat = scipy.io.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
    neural_data = mat["neural_data"].flatten()

    patient_dict = {}
    rt_cache = {}

    for entry in neural_data:
        pid = int(entry.patient_id)
        fr = entry.firing_rates
        loads = entry.trial_load.astype(int)
        rt = entry.trial_RT
        corr = entry.trial_correctness.astype(int)
        tf_bin = int(entry.time_field) - 1

        if only_correct:
            keep = corr == 1
            fr, loads, rt = fr[keep], loads[keep], rt[keep]
        good = ~np.isnan(rt)
        fr, loads, rt = fr[good], loads[good], rt[good]
        if fr.size == 0:
            continue

        patient_dict.setdefault(pid, []).append((fr, loads, rt, tf_bin))
        rt_cache.setdefault(pid, rt)
        if not np.array_equal(rt_cache[pid], rt):
            raise ValueError(f"RT vectors differ across neurons for patient {pid}.")

    valid_pids = [pid for pid in patient_ids if pid in patient_dict and len(patient_dict[pid]) >= m]
    if not valid_pids:
        print("No patients pass inclusion filter.")
        return None

    load_results = {}
    for load_val in (1, 2, 3):
        neurons = []
        for pid in valid_pids:
            # indices for this load value
            rt_vec = rt_cache[pid]
            load_indices = None
            # we need RT vector per trial independent of loads vector, but need loads vector too
            # Since rt_cache[pid] corresponds to first neuron, but loads vector differ per neuron; we use one neuron to get loads
            # We'll take loads vector from first neuron list
            loads_vec = patient_dict[pid][0][1]
            load_mask = loads_vec == load_val
            if load_mask.sum() < 2:
                continue

            # Quantiles within this load
            rt_load = rt_vec[load_mask]
            lo, hi = np.quantile(rt_load, [quantile, 1.0 - quantile])
            fast_mask_p = (rt_vec <= lo) & load_mask
            slow_mask_p = (rt_vec >= hi) & load_mask
            if fast_mask_p.sum() < 2 or slow_mask_p.sum() < 2:
                continue

            for fr, loads_n, _, tf_bin in patient_dict[pid]:
                start = max(0, tf_bin - num_windows)
                end = min(fr.shape[1], tf_bin + num_windows + 1)
                means = fr[:, start:end].mean(axis=1)
                fast_rates = means[fast_mask_p]
                slow_rates = means[slow_mask_p]
                if len(fast_rates) and len(slow_rates):
                    neurons.append({"fast": fast_rates, "slow": slow_rates})

        if not neurons:
            print(f"Load {load_val}: no neurons with fast & slow trials.")
            continue

        min_fast = min(len(n["fast"]) for n in neurons)
        min_slow = min(len(n["slow"]) for n in neurons)
        if test_per_class > min_fast or test_per_class > min_slow:
            print(f"Load {load_val}: insufficient trials after down‑sampling.")
            continue

        for n in neurons:
            n["fast"] = _downsample_vector(rng, n["fast"], min_fast)
            n["slow"] = _downsample_vector(rng, n["slow"], min_slow)

        fast_mat = np.column_stack([n["fast"] for n in neurons])
        slow_mat = np.column_stack([n["slow"] for n in neurons])

        acc_fast_a, acc_slow_a, cm_acc = _run_monte_carlo_binary(
            rng, fast_mat, slow_mat, test_per_class, n_iterations, random_state
        )
        acc_fast_n, acc_slow_n, _ = _run_monte_carlo_binary(
            rng, fast_mat, slow_mat, test_per_class, n_iterations, random_state, shuffle_labels=True
        )

        load_results[load_val] = {
            "actual": (acc_fast_a, acc_slow_a, cm_acc),
            "null": (acc_fast_n, acc_slow_n),
        }

        if plot:
            _plot_binary_results(
                cm_acc,
                pd.DataFrame({"Class": ["Fast"] * n_iterations + ["Slow"] * n_iterations,
                               "Accuracy": acc_fast_a + acc_slow_a}),
                pd.DataFrame({"Class": ["Fast"] * n_iterations + ["Slow"] * n_iterations,
                               "Accuracy": acc_fast_n + acc_slow_n}),
                title=f"Load {load_val}"
            )

    # ------------------------------------------------------------------
    # Collate all loads into DataFrames
    # ------------------------------------------------------------------
    dfs_actual = []
    dfs_null = []
    for load_val, data in load_results.items():
        acc_fast_a, acc_slow_a, _ = data["actual"]
        acc_fast_n, acc_slow_n = data["null"]
        dfs_actual.append(pd.DataFrame({
            "Load": load_val,
            "Class": ["Fast"] * n_iterations + ["Slow"] * n_iterations,
            "Accuracy": acc_fast_a + acc_slow_a,
        }))
        dfs_null.append(pd.DataFrame({
            "Load": load_val,
            "Class": ["Fast"] * n_iterations + ["Slow"] * n_iterations,
            "Accuracy": acc_fast_n + acc_slow_n,
        }))

    if not dfs_actual:
        print("No loads produced valid results.")
        return None

    df_actual = pd.concat(dfs_actual, ignore_index=True)
    df_null = pd.concat(dfs_null, ignore_index=True)
    return df_actual, df_null

def _run_monte_carlo_binary(
    rng: np.random.Generator,
    fast_mat: np.ndarray,
    slow_mat: np.ndarray,
    test_per_class: int,
    n_iter: int,
    random_state: Optional[int],
    shuffle_labels: bool = False,
):
    """Return (acc_fast_list, acc_slow_list, confusion_matrix_accum)."""

    n_fast, n_neurons = fast_mat.shape
    n_slow = slow_mat.shape[0]
    acc_fast = []
    acc_slow = []
    cm_acc = np.zeros((2, 2))

    for _ in range(n_iter):
        idx_fast = rng.permutation(n_fast)
        idx_slow = rng.permutation(n_slow)
        test_f, train_f = idx_fast[:test_per_class], idx_fast[test_per_class:]
        test_s, train_s = idx_slow[:test_per_class], idx_slow[test_per_class:]
        X_train = np.vstack([fast_mat[train_f], slow_mat[train_s]])
        y_train = np.array([0] * len(train_f) + [1] * len(train_s))
        if shuffle_labels:
            y_train = rng.permutation(y_train)
        X_test = np.vstack([fast_mat[test_f], slow_mat[test_s]])
        y_test = np.array([0] * len(test_f) + [1] * len(test_s))
        clf = SVC(kernel="linear", random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cm_acc += cm
        acc_fast.append(np.mean(y_pred[y_test == 0] == 0) * 100)
        acc_slow.append(np.mean(y_pred[y_test == 1] == 1) * 100)
    return acc_fast, acc_slow, cm_acc

def _plot_binary_results(cm_acc: np.ndarray, df_actual: pd.DataFrame, df_null: pd.DataFrame, *, title: str):
    cm_pct = cm_acc / cm_acc.sum(axis=1, keepdims=True) * 100
    labels = np.array([[f"{v:.1f}%" for v in row] for row in cm_pct])

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    sns.heatmap(cm_pct, annot=labels, fmt="", cmap="Blues", ax=ax[0],
                xticklabels=["Fast", "Slow"], yticklabels=["Fast", "Slow"],
                vmin=0, vmax=100, cbar_kws={"label": "%"})
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")
    ax[0].set_title(f"{title}: Confusion matrix")

    sns.boxplot(data=df_actual, x="Class", y="Accuracy", ax=ax[1], width=0.6,
                palette=["skyblue", "salmon"], showfliers=True,
                flierprops=dict(marker="o", markersize=4))
    ax[1].axhline(50, ls="--", c="gray")
    ax[1].set_ylim(0, df_actual.Accuracy.max() + 5)
    ax[1].set_title(f"{title}: Accuracy (actual)")

    for i, cls in enumerate(["Fast", "Slow"]):
        act = df_actual[df_actual.Class == cls].Accuracy.values
        nul = df_null[df_null.Class == cls].Accuracy.values
        p_perm = np.mean(nul >= act.mean())
        star = _significance_stars(p_perm)
        ax[1].text(i, act.max() + 1, star, ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.show()

repeated_cv_pseudo_population_rt_fast_slow('3sig15_raw.mat', only_correct = True, quantile=0.3, random_state=42)
#Decodable from raw Rate!!!!
