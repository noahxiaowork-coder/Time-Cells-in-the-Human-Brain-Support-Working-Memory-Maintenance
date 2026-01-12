from __future__ import annotations

import os
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _strip_lat(region: str) -> str:
    """Remove left/right suffixes commonly used in region labels."""
    if region.endswith('_left'):
        return region[:-5]
    if region.endswith('_right'):
        return region[:-6]
    return region


def _significance_stars(p: float) -> str:
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else ''


# ──────────────────────────────────────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────────────────────────────────────

def decode_rt_fast_slow_by_region(
    mat_file_path: str,
    patient_ids: Iterable[int] = range(1, 22),
    m: int = 0,
    num_windows: int = 0,
    test_per_class: int = 11,
    quantile: float = 0.25,                 # tail quantile per fast/slow
    n_iterations: int = 1000,
    n_shuffles: int = 1,
    random_state: Optional[int] = 42,
    only_correct: bool = False,
    stratify_by_load: bool = False,
    show_plots: bool = True,
    label_map: Optional[Dict[str, str]] = None,
    strip_lateralization: bool = True,
    resample_each_iter: bool = False,
    save_dir: Optional[str] = None,
    region_color_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Train/test a pseudo‑population SVM decoder of *fast* vs *slow* RT **per brain region**.

    Parameters
    ----------
    mat_file_path : str
        Path to the MATLAB file with `neural_data`.
    patient_ids : iterable of int
        Patients to include.
    m : int
        Minimum number of neurons *per patient* to include that patient within a region.
    num_windows : int
        Average firing rates over [tfield - num_windows, ..., tfield + num_windows].
    test_per_class : int
        Number of test trials per class (fast/slow) in each Monte‑Carlo iteration.
    quantile : float
        Lower/upper quantile threshold (< 0.5 recommended).
    n_iterations : int
        Monte‑Carlo iterations for the real distribution.
    n_shuffles : int
        Number of outer loops for null draws (label shuffles). Total null draws = n_shuffles * n_iterations.
    random_state : int or None
        RNG seed.
    only_correct : bool
        Keep only correct trials.
    stratify_by_load : bool
        If True, compute fast/slow quantiles **within** each distinct `trial_load` and merge.
    show_plots : bool
        Show a combined per‑region accuracy boxplot (real) with stars vs null.
    label_map : dict or None
        Optional mapping from raw region names (base, without _left/_right) to display names (e.g., acronyms).
    strip_lateralization : bool
        If True, collapse `_left`/`_right` into a base region before mapping.
    resample_each_iter : bool
        If True, re‑draw per‑neuron trial subsets each iteration before split; otherwise, down‑sample once.
    save_dir : str or None
        If provided, save plots here.
    region_color_map : dict or None
        Optional mapping region->hex color for plotting.

    Returns
    -------
    results : dict
        {
          region: {
            'real_acc': List[float],
            'null_acc': List[float],
            'p_null': float,
            'cm_pct': np.ndarray shape (2,2),
            'n_neurons': int,
            'trials_per_class': {'fast': int, 'slow': int},
          },
          ...
          '_order': [region names in plotted order]
        }
    """
    assert 0.0 < quantile < 0.5, "quantile must be in (0, 0.5)"
    rng = np.random.default_rng(random_state)

    # Default label map (can be overridden)
    DEFAULT_MAP = {
        'dorsal_anterior_cingulate_cortex': 'DaCC',
        'pre_supplementary_motor_area': 'PSMA',
        'hippocampus': 'HPC',
        'amygdala': 'AMY',
        'ventral_medial_prefrontal_cortex': 'vmPFC',
    }
    if label_map is None:
        label_map = DEFAULT_MAP

    # ── 1) Load MATLAB struct (matching your other function's pattern) ───────
    mat = scipy.io.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
    neural_data = mat['neural_data'].flatten()

    # ── 2) Organise neurons by region, preserving per‑patient RT alignment ───
    # region_neurons[region][pid] = list of neuron dicts with 'fast_all'/'slow_all'
    region_neurons: Dict[str, Dict[int, List[Dict[str, np.ndarray]]]] = {}

    patient_rt_cache: Dict[int, np.ndarray] = {}

    for entry in neural_data:
        pid = int(entry.patient_id)
        if pid not in patient_ids:
            continue

        # region name
        region_raw = entry.brain_region
        if isinstance(region_raw, bytes):
            region_raw = region_raw.decode()
        region_base = _strip_lat(region_raw) if strip_lateralization else region_raw
        region_key = label_map.get(region_base, region_base)

        frates = entry.firing_rates  # (trials x time)
        correct = entry.trial_correctness.astype(int)
        rt = entry.trial_RT.astype(float)
        load = entry.trial_load.astype(int)
        tfield = int(entry.time_field) - 1

        if only_correct:
            keep = (correct == 1)
            frates = frates[keep, :]
            rt = rt[keep]
            load = load[keep]

        # drop NaN RT trials
        nan_mask = ~np.isnan(rt)
        frates = frates[nan_mask, :]
        rt = rt[nan_mask]
        load = load[nan_mask]
        if frates.size == 0:
            continue

        # Per‑patient RT vector must be identical across neurons
        if pid not in patient_rt_cache:
            patient_rt_cache[pid] = rt
        else:
            if not np.array_equal(patient_rt_cache[pid], rt):
                raise ValueError(f"RT vectors differ across neurons for patient {pid}.")

        # build fast/slow index masks for this patient
        if stratify_by_load:
            fast_idx = np.zeros_like(rt, dtype=bool)
            slow_idx = np.zeros_like(rt, dtype=bool)
            for L in np.unique(load):
                mask = (load == L)
                if mask.sum() < 4:  # need >= 2 per tail
                    continue
                q_low = np.quantile(rt[mask], quantile)
                q_high = np.quantile(rt[mask], 1.0 - quantile)
                fast_idx |= mask & (rt <= q_low)
                slow_idx |= mask & (rt >= q_high)
        else:
            q_low = np.quantile(rt, quantile)
            q_high = np.quantile(rt, 1.0 - quantile)
            fast_idx = (rt <= q_low)
            slow_idx = (rt >= q_high)

        if fast_idx.sum() < 2 or slow_idx.sum() < 2:
            # This patient contributes no neurons to any region given too few trials
            continue

        # mean rate around time_field (± num_windows)
        s = max(0, tfield - num_windows)
        e = min(frates.shape[1], tfield + num_windows + 1)
        mean_rates = frates[:, s:e].mean(axis=1)

        fast_rates = mean_rates[fast_idx]
        slow_rates = mean_rates[slow_idx]
        if fast_rates.size == 0 or slow_rates.size == 0:
            continue

        neuron_entry = {'fast_all': fast_rates, 'slow_all': slow_rates}
        region_neurons.setdefault(region_key, {}).setdefault(pid, []).append(neuron_entry)

    # ── 3) Filter patients with >= m neurons per region; flatten ─────────────
    region_neuron_list: Dict[str, List[Dict[str, np.ndarray]]] = {}
    for region, per_patient in list(region_neurons.items()):
        pts_ok = {pid: cells for pid, cells in per_patient.items() if len(cells) >= m}
        merged = [cell for cells in pts_ok.values() for cell in cells]
        if merged:
            region_neuron_list[region] = merged

    if not region_neuron_list:
        raise RuntimeError("No region has neurons that satisfy the inclusion criteria.")

    # ── 4) Core decoder for a given region's neuron list ─────────────────────
    def _decode(neuron_list: List[Dict[str, np.ndarray]]) -> Optional[Dict[str, Any]]:
        # determine per‑class minimum across neurons
        min_fast = min(len(n['fast_all']) for n in neuron_list)
        min_slow = min(len(n['slow_all']) for n in neuron_list)
        if test_per_class > min_fast or test_per_class > min_slow:
            return None  # insufficient after balancing

        # classifier: scale + linear SVM with class balancing
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LinearSVC(dual=False, max_iter=5000, class_weight='balanced', random_state=random_state)
        )

        # Pre‑downsample once if not resampling each iteration
        if not resample_each_iter:
            neurons_ds = []
            for n in neuron_list:
                n_ds = {
                    'fast': rng.choice(n['fast_all'], size=min_fast, replace=False),
                    'slow': rng.choice(n['slow_all'], size=min_slow, replace=False),
                }
                neurons_ds.append(n_ds)
            fast_mat = np.column_stack([n['fast'] for n in neurons_ds])  # rows = trials, cols = neurons
            slow_mat = np.column_stack([n['slow'] for n in neurons_ds])

        real_acc: List[float] = []
        null_acc: List[float] = []
        cm_accum = np.zeros((2, 2), dtype=float)

        # real distribution
        for _ in range(n_iterations):
            if resample_each_iter:
                fast_mat = np.column_stack([
                    rng.choice(n['fast_all'], size=min_fast, replace=False) for n in neuron_list
                ])
                slow_mat = np.column_stack([
                    rng.choice(n['slow_all'], size=min_slow, replace=False) for n in neuron_list
                ])

            idx_fast = rng.permutation(min_fast)
            idx_slow = rng.permutation(min_slow)
            ts_fast, tr_fast = idx_fast[:test_per_class], idx_fast[test_per_class:]
            ts_slow, tr_slow = idx_slow[:test_per_class], idx_slow[test_per_class:]

            Xtr = np.vstack([fast_mat[tr_fast], slow_mat[tr_slow]])
            ytr = np.array([0] * len(tr_fast) + [1] * len(tr_slow))
            Xts = np.vstack([fast_mat[ts_fast], slow_mat[ts_slow]])
            yts = np.array([0] * len(ts_fast) + [1] * len(ts_slow))

            clf.fit(Xtr, ytr)
            ypred = clf.predict(Xts)

            cm_accum += confusion_matrix(yts, ypred, labels=[0, 1])
            real_acc.append((ypred == yts).mean() * 100.0)

        # null distribution (label‑shuffled training labels)
        for _ in range(n_shuffles):
            for _ in range(n_iterations):
                if resample_each_iter:
                    fast_mat = np.column_stack([
                        rng.choice(n['fast_all'], size=min_fast, replace=False) for n in neuron_list
                    ])
                    slow_mat = np.column_stack([
                        rng.choice(n['slow_all'], size=min_slow, replace=False) for n in neuron_list
                    ])

                idx_fast = rng.permutation(min_fast)
                idx_slow = rng.permutation(min_slow)
                ts_fast, tr_fast = idx_fast[:test_per_class], idx_fast[test_per_class:]
                ts_slow, tr_slow = idx_slow[:test_per_class], idx_slow[test_per_class:]

                Xtr = np.vstack([fast_mat[tr_fast], slow_mat[tr_slow]])
                ytr = np.array([0] * len(tr_fast) + [1] * len(tr_slow))
                Xts = np.vstack([fast_mat[ts_fast], slow_mat[ts_slow]])
                yts = np.array([0] * len(ts_fast) + [1] * len(ts_slow))

                ytr = rng.permutation(ytr)  # shuffle labels
                clf.fit(Xtr, ytr)
                ypred = clf.predict(Xts)
                null_acc.append((ypred == yts).mean() * 100.0)

        # Normalised confusion matrix (row‑wise %)
        row_sums = cm_accum.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_pct = cm_accum / row_sums * 100.0

        p_null = (np.array(null_acc) >= np.mean(real_acc)).mean()

        return {
            'real_acc': real_acc,
            'null_acc': null_acc,
            'p_null': float(p_null),
            'cm_pct': cm_pct,
            'n_neurons': fast_mat.shape[1],
            'trials_per_class': {'fast': int(min_fast), 'slow': int(min_slow)},
        }

    # ── 5) Decode each region ────────────────────────────────────────────────
    results: Dict[str, Any] = {}
    for region, neuron_list in region_neuron_list.items():
        res = _decode(neuron_list)
        if res is None:
            print(f"[skip] {region}: too few trials after balancing (need >= {test_per_class} per class).")
            continue
        results[region] = res
        mu = np.mean(res['real_acc'])
        sd = np.std(res['real_acc'])
        print(f"{region}: {mu:.2f}% ± {sd:.2f}% (N={n_iterations}), p_null={res['p_null']:.4g}")

    if not results:
        raise RuntimeError("After balancing, no region had enough trials to decode.")

    # ── 6) Combined plot + optional between‑region tests ─────────────────────
    if show_plots and results:
        regions = list(results.keys())

        # Optional: prefer consistent ordering if familiar acronyms exist
        preferred = ['HPC', 'AMY', 'DaCC', 'PSMA', 'vmPFC']
        regions = sorted(regions, key=lambda r: (preferred.index(r) if r in preferred else 999, r))

        acc_lists = [results[r]['real_acc'] for r in regions]
        p_vs_null = [results[r]['p_null'] for r in regions]

        # colours
        if region_color_map is None:
            palette = sns.color_palette('Set2', n_colors=len(regions))
            region_color_map = {r: palette[i] for i, r in enumerate(regions)}
        colours = [region_color_map.get(r, '#999999') for r in regions]

        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(regions)), 6))
        box = ax.boxplot(acc_lists, patch_artist=True, labels=regions, widths=0.7)
        for patch, c in zip(box['boxes'], colours):
            patch.set_facecolor(c)
            patch.set_edgecolor('black')
        for element in ['whiskers', 'caps', 'medians']:
            plt.setp(box[element], color='black')

        # chance
        ax.axhline(50, ls='--', linewidth=1, color='gray')

        # within‑region significance vs null
        y_max = max(max(acc) for acc in acc_lists)
        height_step = 5
        y_bar = y_max + height_step
        for x, p in enumerate(p_vs_null, start=1):
            stars = _significance_stars(p)
            if stars:
                ax.text(x, y_bar, stars, ha='center', va='bottom', fontweight='bold')
        ax.set_ylim(0, max(y_bar + height_step, 60))

        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('RT (fast vs slow) decoding by brain region', fontsize=13)
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'rt_by_region_decoding.svg'), format='svg')
        plt.show()

        # Pairwise Welch t‑tests (real distributions only), printed to console
        print("\nPairwise Welch t‑tests between regions (real accuracies):")
        for (i, regA), (j, regB) in combinations(enumerate(regions), 2):
            a = np.array(results[regA]['real_acc'])
            b = np.array(results[regB]['real_acc'])
            t, p = ttest_ind(a, b, equal_var=False)
            star = _significance_stars(p)
            print(f"{regA} vs {regB}: t={t:.2f}, p={p:.4g} {star}")

        results['_order'] = regions

    return results



results = decode_rt_fast_slow_by_region(
    mat_file_path='3sig15_data.mat',
    patient_ids=range(1, 22),
    m=0,
    num_windows=0,
    test_per_class=10,
    quantile=0.3,
    n_iterations=1000,
    n_shuffles=1,
    random_state=42,
    only_correct=True,
    stratify_by_load=True,
    show_plots=True,
    save_dir='figs',
)
