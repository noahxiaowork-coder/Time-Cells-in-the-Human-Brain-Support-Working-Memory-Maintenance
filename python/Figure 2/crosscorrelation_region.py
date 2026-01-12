
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.io
# from scipy.ndimage import gaussian_filter1d
from sklearn.utils import resample
from Crosscorrelation import resample_and_plot_cross_correlation

import numpy as np
import scipy.io

def process_trial_and_compute_mean_correlation(trial_data, time_fields):
    """
    Process a single trial's data, circularly shift firing rates for each neuron based on time fields,
    and compute the mean cross-correlation between all pairs of neurons.

    Parameters:
    - trial_data: np.ndarray, shape (Neuron x Time Bin), firing rate data for a single trial.
    - time_fields: list of int, indices of time fields for each neuron.

    Returns:
    - mean_correlation: float, mean cross-correlation between all pairs of neurons.
    """
    num_neurons, num_time_bins = trial_data.shape

    # Circularly shift each neuron's firing rates
    shifted_data = np.zeros_like(trial_data)
    middle_bin = num_time_bins // 2  # Target middle index for circular shifting

    for neuron_idx, time_field in enumerate(time_fields):
        shift_amount = middle_bin - time_field
        shifted_data[neuron_idx] = np.roll(trial_data[neuron_idx], shift_amount)

    # Compute cross-correlation for all pairs of neurons
    cross_correlations = []
    for i in range(num_neurons):
        for j in range(i + 1, num_neurons):  # Only consider unique pairs (i, j)
            std_i = np.std(shifted_data[i])
            std_j = np.std(shifted_data[j])

            # Skip pairs with zero standard deviation
            if std_i == 0 or std_j == 0:
                continue

            # Compute correlation
            corr = np.corrcoef(shifted_data[i], shifted_data[j])[0, 1]  # Pearson correlation
            if not np.isnan(corr):  # Ignore NaN values
                cross_correlations.append(corr)

    # Compute the mean cross-correlation
    if len(cross_correlations) > 0:
        mean_correlation = np.mean(cross_correlations)
    else:
        mean_correlation = 0  # Default value when no valid correlations exist

    return mean_correlation


def separate_trials_by_region(mat_file_path,
                              patient_id,
                              region_name,
                              neuron_threshold=5,
                              collapse_lr=True,
                              rng_seed=42):
    """
    Restrict the original `separate_trials` function to a single anatomical region.

    Parameters
    ----------
    mat_file_path : str
    patient_id    : int
    region_name   : str      e.g. 'hippocampus', 'mPFC'
    neuron_threshold : int   minimum #neurons required for the patient‑region
    collapse_lr   : bool     strip '_left' / '_right' suffixes before matching
    rng_seed      : int

    Returns
    -------
    correct_trials  : ndarray  (nTrials_correct × nNeurons × nTimeBins)  or None
    incorrect_trials: ndarray  (nTrials_incorrect × nNeurons × nTimeBins) or None
    time_fields     : list[int] (one per neuron)  or None
    """
    np.random.seed(rng_seed)

    # ---------- load ------------------------------------------------------
    mat = scipy.io.loadmat(mat_file_path)
    recs = mat['neural_data'][0]        # MATLAB (1 × n) struct array

    # Pull fields once to avoid repeated dict look‑ups
    patient_ids   = np.array([int(r['patient_id'][0][0]) for r in recs])
    brain_regions = np.array([r['brain_region'][0]        for r in recs])
    firing_rates  = [np.asarray(r['firing_rates'])         for r in recs]
    trial_perf    = [np.asarray(r['trial_correctness']).flatten() for r in recs]
    time_fields   = np.array([int(r['time_field'][0][0]) - 1       for r in recs])

    # Optionally strip side labels
    if collapse_lr:
        brain_regions = np.char.replace(np.char.replace(brain_regions,
                                                        '_left', ''),
                                                        '_right', '')

    # ----------------- filter by patient & region -------------------------
    sel = np.where((patient_ids == int(patient_id)) &
                   (brain_regions == region_name))[0]

    if sel.size < neuron_threshold:
        return None, None, None

    fr_sel   = [firing_rates[i] for i in sel]
    perf_sel = [trial_perf[i]   for i in sel]
    tf_sel   = time_fields[sel].tolist()

    # Stack to Trial × Neuron × TimeBin
    trial_data = np.stack(fr_sel, axis=1)          # (Trials, Neurons, TimeBins)

    # Behaviour: AND across neurons (as in your code)
    perf_combined = np.all(np.stack(perf_sel, axis=0), axis=0).astype(int)

    correct_trials   = trial_data[perf_combined == 1]
    incorrect_trials = trial_data[perf_combined == 0]

    return correct_trials, incorrect_trials, tf_sel


from tqdm import tqdm
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

def analyze_cross_corr_by_region(mat_file_path,
                                 neuron_threshold=2,
                                 min_incorrect_trials=1,
                                 min_time_cells=0,
                                 num_iterations=1000,
                                 collapse_lr=True):
    """
    For every cleaned brain region:
      • Loop over patients.
      • Keep only those patient‑region neuron pools that pass `neuron_threshold`
        and have ≥ `min_incorrect_trials` incorrect trials.
      • Compute trial‑wise mean cross‑correlations with the original functions.
      • Aggregate correct vs. incorrect distributions across patients.
      • Plot, test, and return a summary dict.
    """
    mat = scipy.io.loadmat(mat_file_path)
    recs = mat['neural_data'][0]

    # Extract region names once
    raw_regions = np.array([r['brain_region'][0] for r in recs])
    if collapse_lr:
        raw_regions = np.char.replace(np.char.replace(raw_regions,
                                                      '_left', ''),
                                                      '_right', '')
    unique_regions = np.unique(raw_regions)

    summary = {}

    for region in unique_regions:
        correct_vals   = []
        incorrect_vals = []
        passing_patients = []

        # loop over patients
        patient_ids = np.unique([int(r['patient_id'][0][0]) for r in recs])
        for pid in patient_ids:
            ct, it, tf = separate_trials_by_region(mat_file_path,
                                                   patient_id=pid,
                                                   region_name=region,
                                                   neuron_threshold=neuron_threshold,
                                                   collapse_lr=collapse_lr)
            if ct is None or it is None:
                continue
            if len(tf) < min_time_cells or it.shape[0] < min_incorrect_trials:
                continue

            passing_patients.append(pid)

            # --- trial‑wise cross‑correlations ----------------------------
            for tr in ct:
                correct_vals.append(process_trial_and_compute_mean_correlation(tr, tf))
            for tr in it:
                incorrect_vals.append(process_trial_and_compute_mean_correlation(tr, tf))

        if len(correct_vals) == 0 or len(incorrect_vals) == 0:
            # nothing to analyse for this region
            continue

        # ---------- stats & figure ----------------------------------------
        t_stat, p_val = ttest_ind(correct_vals, incorrect_vals, equal_var=False)
        m_c, m_i = np.mean(correct_vals), np.mean(incorrect_vals)
        sem_c, sem_i = sem(correct_vals), sem(incorrect_vals)

        plt.figure(figsize=(3.6, 5))
        plt.bar(['Correct', 'Incorrect'],
                [m_c, m_i],
                yerr=[sem_c, sem_i],
                color=['blue', 'red'],
                capsize=4, alpha=.75)
        y_star = max(m_c + sem_c, m_i + sem_i) + 0.02
        plt.plot([0, 0, 1, 1], [y_star, y_star+.02, y_star+.02, y_star], color='k')
        sig = "***" if p_val < 1e-3 else "**" if p_val < 1e-2 \
              else "*" if p_val < 5e-2 else f"p = {p_val}"
        plt.text(0.5, y_star+.02, sig, ha='center')
        plt.ylabel("Mean cross‑corr.")
        plt.title(f"{region}\n(pats {len(passing_patients)}, neurons≥{neuron_threshold})")
        plt.tight_layout()
        plt.show()

        summary[region] = dict(mean_correct=m_c,
                               mean_incorrect=m_i,
                               p=p_val,
                               n_correct=len(correct_vals),
                               n_incorrect=len(incorrect_vals),
                               patients=passing_patients)

    return summary

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def _tag_region_on_new_figures(region: str, before_fignums):
    """Prefix region on suptitle and all axes titles for newly created figures."""
    new_fignums = [n for n in plt.get_fignums() if n not in before_fignums]
    for num in new_fignums:
        fig = plt.figure(num)

        # --- suptitle ---
        current_suptitle = fig._suptitle.get_text() if fig._suptitle else ""
        if current_suptitle:
            if not current_suptitle.startswith(f"{region}: "):
                fig.suptitle(f"{region}: {current_suptitle}")
        else:
            fig.suptitle(f"{region}")

        # --- per-axes titles ---
        for ax in fig.get_axes():
            t = ax.get_title()
            if t:
                if not t.startswith(f"{region}: "):
                    ax.set_title(f"{region}: {t}")
            else:
                ax.set_title(f"{region}")

def analyze_cross_corr_by_region_resampled(mat_file_path,
                                           neuron_threshold=2,
                                           min_incorrect_trials=1,
                                           min_time_cells=0,
                                           num_iterations=1000,
                                           collapse_lr=True):
    """
    Per brain region:
      • Pool trial-wise mean cross-correlations across patients
      • Draw the SAME multi-panel figure your helper makes (titles tagged with region)
      • Welch-t comparison (saved to summary dict)

    Returns
    -------
    summary : dict
        {region: {'mean_correct': …, 'mean_incorrect': …,
                  'p': …, 'n_correct': …, 'n_incorrect': …,
                  'patients': […]}}
    """
    mat   = scipy.io.loadmat(mat_file_path)
    recs  = mat['neural_data'][0]

    # ----- canonicalise region names --------------------------------------
    raw_regions = np.array([r['brain_region'][0] for r in recs])
    if collapse_lr:  # strip “_left/_right”
        raw_regions = np.char.replace(np.char.replace(raw_regions, '_left', ''), '_right', '')
    unique_regions = np.unique(raw_regions)

    summary = {}

    patient_ids = np.unique([int(r['patient_id'][0][0]) for r in recs])

    # ===================  MAIN REGION LOOP  ===============================
    for region in unique_regions:
        correct_vals, incorrect_vals = [], []
        passing_patients = []

        # ---- gather data across patients ---------------------------------
        for pid in patient_ids:

            ct, it, tf = separate_trials_by_region(mat_file_path,
                                                   patient_id   = pid,
                                                   region_name  = region,
                                                   neuron_threshold=neuron_threshold,
                                                   collapse_lr  = collapse_lr)
            if ct is None or it is None:
                continue
            if len(tf) < min_time_cells or it.shape[0] < min_incorrect_trials:
                continue

            passing_patients.append(pid)

            for tr in ct:
                correct_vals.append(
                    process_trial_and_compute_mean_correlation(tr, tf))
            for tr in it:
                incorrect_vals.append(
                    process_trial_and_compute_mean_correlation(tr, tf))

        # ---------- skip empty regions ------------------------------------
        if len(correct_vals) == 0 or len(incorrect_vals) == 0:
            continue

        correct_vals   = np.asarray(correct_vals)
        incorrect_vals = np.asarray(incorrect_vals)

        # ---------- draw the FULL resampling figure -----------------------
        # Tag any figures created during this call with the region.
        before_fignums = plt.get_fignums()

        if len(correct_vals) < len(incorrect_vals):
            print(f"[{region}] fewer correct than incorrect trials "
                  f"({len(correct_vals)} < {len(incorrect_vals)}); "
                  "sampling with replacement.")
            # monkey-patch the helper’s call for this region only
            _orig_choice = np.random.choice
            try:
                np.random.choice = lambda a, size, replace=False: \
                                   _orig_choice(a, size=size, replace=True)
                resample_and_plot_cross_correlation(correct_vals,
                                                    incorrect_vals,
                                                    min_time_cells=min_time_cells,
                                                    num_iterations=num_iterations)
            finally:
                np.random.choice = _orig_choice  # always restore
        else:
            resample_and_plot_cross_correlation(correct_vals,
                                                incorrect_vals,
                                                min_time_cells=min_time_cells,
                                                num_iterations=num_iterations)

        # --- add region to all new figure titles created for this region ---
        _tag_region_on_new_figures(region, before_fignums)

        # ---------- stats for summary dict --------------------------------
        from scipy.stats import ttest_ind
        t_stat, p_val = ttest_ind(correct_vals, incorrect_vals, equal_var=False)

        summary[region] = dict(mean_correct   = correct_vals.mean(),
                               mean_incorrect = incorrect_vals.mean(),
                               p              = p_val,
                               n_correct      = len(correct_vals),
                               n_incorrect    = len(incorrect_vals),
                               patients       = passing_patients)

    return summary

Summary = analyze_cross_corr_by_region_resampled('3sig15_data.mat', neuron_threshold=2, min_incorrect_trials=3)
#3 or 5 works 