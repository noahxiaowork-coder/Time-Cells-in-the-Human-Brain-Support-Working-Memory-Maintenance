#!/usr/bin/env python
"""
Multi-patient naïve-Bayes decoder with Monte-Carlo resampling.

Adds `n_iter` repetitions so that each run holds out a fresh, balanced
set of correct trials.  Results from all repetitions are then pooled for
global stats and per-patient paired comparisons.

Author:  <your name>, 2025-07-06
"""

import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter1d
from sklearn.naive_bayes import GaussianNB
from scipy.stats import sem, ttest_rel, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns   # still optional but left for your styling needs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

DIR = '/Users/xiangxuankong/Desktop/Human TC Support WM/Figures/Figure 2/'


def analyze_all_patients(mat_file_path, binwidth, neuron_threshold):
    """
    Iterate through all patients, analyze decoding errors for correct and 
    incorrect trials, perform a t-test, and visualize:
      - Semi-transparent bars for overall mean (correct vs. incorrect).
      - Per-patient mean errors as semi-transparent gray dots 
        with lines connecting each patient's correct vs. incorrect points.
    """

    np.random.seed(20250710)

    # Load data
    mat_data = scipy.io.loadmat(mat_file_path)
    neural_data = mat_data['neural_data']
    patient_ids = [int(entry['patient_id'][0][0]) for entry in neural_data[0]]
    firing_rates = [entry['firing_rates'] for entry in neural_data[0]]
    trial_performance = [entry['trial_correctness'] for entry in neural_data[0]]

    unique_patient_ids = np.unique(patient_ids)

    # Global lists for all trials from all patients
    all_decoding_errors_correct = []
    all_decoding_errors_incorrect = []
    
    # Per-patient mean errors (for plotting dots and lines)
    per_patient_correct_means = []
    per_patient_incorrect_means = []

    print("included patients:")

    global_errors_correct_bins  = []   # each element: [T × nHeldOutCorrect]
    global_errors_incorrect_bins = []  # each element: [T × nIncorrect]


    for patient_id in unique_patient_ids:
        # Gather data for this patient
        selected_neurons = [
            (firing_rates[i], trial_performance[i]) 
            for i, pid in enumerate(patient_ids) 
            if pid == patient_id
        ]
        print("number of time cells")
        print(len(selected_neurons))
        # Skip if insufficient neurons
        if len(selected_neurons) < neuron_threshold:
            continue  

        print(patient_id)

        # Convert to arrays
        selected_firing_rates = [np.array(neuron[0]) for neuron in selected_neurons]
        selected_performance = [np.array(neuron[1]).flatten() for neuron in selected_neurons]

        # (Optional) smoothing step - currently commented out
        # smoothed_neural_data = [gaussian_filter1d(rate, sigma=2, axis=1) 
        #                         for rate in selected_firing_rates]
        smoothed_neural_data = selected_firing_rates

        # Stack all neurons for this patient: shape -> [Time_bins x Trials x Neurons]
        reshaped_data = np.stack(smoothed_neural_data, axis=2)
        # Transpose to interpret axis in the order [Time_bins, Trials, Neurons]
        reshaped_data = reshaped_data.transpose(1, 0, 2)
        time_bins, trials, neurons = reshaped_data.shape

        # Combine correctness across all neurons (logical AND)
        combined_performance = np.all(np.stack(selected_performance, axis=0), axis=0)
        correct_trials = np.where(combined_performance == 1)[0]
        incorrect_trials = np.where(combined_performance == 0)[0]

        if len(correct_trials) == 0 or len(incorrect_trials) == 0:
            continue  # No valid trials

        # Balance test sets: pick the same number of correct trials as incorrect
        num_incorrect_trials = len(incorrect_trials)
        if len(correct_trials) < num_incorrect_trials:
            continue  # Not enough correct trials to match incorrect

        test_correct_trials = np.random.choice(correct_trials, num_incorrect_trials, replace=False)
        train_correct_trials = np.setdiff1d(correct_trials, test_correct_trials)

        # Prepare training set (correct trials only)
        X_train = reshaped_data[:, train_correct_trials, :].reshape(-1, neurons)
        y_train = np.repeat(np.arange(time_bins), len(train_correct_trials))

        # Prepare testing sets (held-out correct trials + all incorrect trials)
        X_test_correct = reshaped_data[:, test_correct_trials, :].reshape(-1, neurons)
        y_test_correct = np.repeat(np.arange(time_bins), len(test_correct_trials))

        X_test_incorrect = reshaped_data[:, incorrect_trials, :].reshape(-1, neurons)
        y_test_incorrect = np.repeat(np.arange(time_bins), len(incorrect_trials))

        # Train Gaussian Naive Bayes
        bayesian_clf = GaussianNB()
        bayesian_clf.fit(X_train, y_train)

        # Predictions
        y_pred_correct = bayesian_clf.predict(X_test_correct)
        y_pred_incorrect = bayesian_clf.predict(X_test_incorrect)

        # Decoding errors: difference in time bins * binwidth
        decoding_errors_correct = np.abs(y_pred_correct - y_test_correct) * binwidth
        decoding_errors_incorrect = np.abs(y_pred_incorrect - y_test_incorrect) * binwidth


        # Aggregate across all patients
        all_decoding_errors_correct.extend(decoding_errors_correct)
        all_decoding_errors_incorrect.extend(decoding_errors_incorrect)

        # Track mean decoding error for this patient
        per_patient_correct_means.append(decoding_errors_correct.mean())
        per_patient_incorrect_means.append(decoding_errors_incorrect.mean())

                # ---------- decoding error per time bin ----------
        # Reshape predictions back to [Time_bins, Trials]
        y_pred_correct_reshaped = y_pred_correct.reshape(time_bins, -1)
        y_test_correct_reshaped = y_test_correct.reshape(time_bins, -1)
        errors_correct_per_bin = np.abs(y_pred_correct_reshaped - y_test_correct_reshaped) * binwidth

        y_pred_incorrect_reshaped = y_pred_incorrect.reshape(time_bins, -1)
        y_test_incorrect_reshaped = y_test_incorrect.reshape(time_bins, -1)
        errors_incorrect_per_bin = np.abs(y_pred_incorrect_reshaped - y_test_incorrect_reshaped) * binwidth

        global_errors_correct_bins.append(errors_correct_per_bin)     # shape [T, trials]
        global_errors_incorrect_bins.append(errors_incorrect_per_bin) # shape [T, trials]

        # Compute mean and SEM per time bin
        mean_error_correct = np.mean(errors_correct_per_bin, axis=1)
        sem_error_correct = sem(errors_correct_per_bin, axis=1)

        mean_error_incorrect = np.mean(errors_incorrect_per_bin, axis=1)
        sem_error_incorrect = sem(errors_incorrect_per_bin, axis=1)

        # Plotting per patient
        time_vector = np.arange(time_bins) * binwidth
        plt.figure(figsize=(6, 4))
        plt.plot(time_vector, mean_error_correct, label='Correct Trials', color='blue')
        plt.fill_between(time_vector,
                         mean_error_correct - sem_error_correct,
                         mean_error_correct + sem_error_correct,
                         color='blue', alpha=0.3)

        plt.plot(time_vector, mean_error_incorrect, label='Incorrect Trials', color='red')
        plt.fill_between(time_vector,
                         mean_error_incorrect - sem_error_incorrect,
                         mean_error_incorrect + sem_error_incorrect,
                         color='red', alpha=0.3)

        plt.title(f"Patient {patient_id} - Decoding Error vs. Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Decoding Error (s)")
        plt.legend()
        plt.tight_layout()

        plt.show()


    # ------------------------------------------------------------------
    #  1.  GLOBAL (bin × trial) summary  – identical stats, cleaner plot
    # ------------------------------------------------------------------
    t_stat_raw, p_raw = ttest_ind(all_decoding_errors_correct,
                               all_decoding_errors_incorrect,
                               equal_var=False)  # Welch's version is safest

    mean_raw_correct      = np.mean(all_decoding_errors_correct)
    mean_raw_incorrect    = np.mean(all_decoding_errors_incorrect)
    sem_raw_correct       = sem(all_decoding_errors_correct)
    sem_raw_incorrect     = sem(all_decoding_errors_incorrect)

    # ---------- Fig 1 : pure bar chart (no dots) ----------
    plt.figure(figsize=(4, 6))
    bars = plt.bar(['Correct', 'Incorrect'],
                   [mean_raw_correct, mean_raw_incorrect],
                   yerr=[sem_raw_correct, sem_raw_incorrect],
                   capsize=5, alpha=0.6,
                   color=['blue', 'red']) 

    # significance stars
    ymax = max(mean_raw_correct + sem_raw_correct,
               mean_raw_incorrect + sem_raw_incorrect)
    star_y = ymax + 0.02
    plt.plot([0, 0, 1, 1], [ymax, star_y, star_y, ymax],
             lw=1.3, color='black')
    star = "***" if p_raw < 1e-3 else "**" if p_raw < 1e-2 else "*" if p_raw < 0.05 else "ns"
    plt.text(0.5, star_y + 0.005, star, ha='center', va='bottom', fontsize=14)

    plt.ylabel("Mean Decoding Error (s)")
    plt.title("Decoding Error – all samples")
    plt.tight_layout()
    # plt.savefig(DIR + "error_bar.svg", format="svg")
    plt.show()

    # ------------------------------------------------------------------
    #  2.  PER-PATIENT summary  – paired dots + bars + new t-test
    # ------------------------------------------------------------------
    t_stat_pt, p_pt = ttest_rel(per_patient_correct_means,
                                per_patient_incorrect_means)

    mean_pt_correct   = np.mean(per_patient_correct_means)
    mean_pt_incorrect = np.mean(per_patient_incorrect_means)
    sem_pt_correct    = sem(per_patient_correct_means)
    sem_pt_incorrect  = sem(per_patient_incorrect_means)

    plt.figure(figsize=(4.8, 6))



    # paired grey dots + connecting lines
    for c, i in zip(per_patient_correct_means, per_patient_incorrect_means):
        plt.plot([0, 1], [c, i], color='gray', alpha=0.6)
        plt.scatter(0, c, color='gray', alpha=0.8, zorder=10)
        plt.scatter(1, i, color='gray', alpha=0.8, zorder=10)

    # overlay mean lines
    plt.hlines(np.mean(per_patient_correct_means), -0.2, 0.2, color='blue', linewidth=2.5, zorder=5)
    plt.hlines(np.mean(per_patient_incorrect_means), 0.8, 1.2, color='red', linewidth=2.5, zorder=5)

    # significance annotation
    ymax2 = max(per_patient_correct_means + per_patient_incorrect_means)
    star_y2 = ymax2 + 0.02
    plt.plot([0, 0, 1, 1], [ymax2, star_y2, star_y2, ymax2],
            lw=1.3, color='black')
    if p_pt < 1e-3:
        sig_label2 = "***"
    elif p_pt < 1e-2:
        sig_label2 = "**"
    elif p_pt < 0.05:
        sig_label2 = "*"
    else:
        sig_label2 = f"p = {p_pt:.3f}"

    plt.text(0.5, star_y2 + 0.005, sig_label2, ha='center', va='bottom', fontsize=14)


    plt.xticks([0, 1], ['Correct', 'Incorrect'])
    plt.ylabel("Mean Decoding Error (s)")
    plt.title("Decoding Error – patient means")
    plt.tight_layout()
    # plt.savefig(DIR + "patient_error.svg", format="svg")
    plt.show()



    # ------------------------------------------------------------------
    # (3)  AFTER the existing global-summary figures 
    # ------------------------------------------------------------------
    if global_errors_correct_bins and global_errors_incorrect_bins:
        # concatenate across patients along the trial axis (axis=1)
        all_corr  = np.hstack(global_errors_correct_bins)      # [T, totalCorrectTrials]
        all_inc   = np.hstack(global_errors_incorrect_bins)    # [T, totalIncorrectTrials]

        mean_corr = np.mean(all_corr, axis=1)
        sem_corr  = sem(all_corr, axis=1)

        mean_inc  = np.mean(all_inc, axis=1)
        sem_inc   = sem(all_inc, axis=1)

        n_bins = mean_corr.size
        time_vec = np.arange(n_bins) * binwidth



        plt.figure(figsize=(6, 4))
        plt.plot(time_vec, mean_corr, label='Correct', color='blue')
        plt.fill_between(time_vec, mean_corr - sem_corr, mean_corr + sem_corr,
                        color='blue', alpha=0.3)

        plt.plot(time_vec, mean_inc, label='Incorrect', color='red')
        plt.fill_between(time_vec, mean_inc - sem_inc, mean_inc + sem_inc,
                        color='red', alpha=0.3)

        plt.xlabel("Time (s)")
        plt.ylabel("Decoding Error (s)")
        plt.title("Grand-average decoding error\n(all test trials)")
        plt.legend()
        plt.tight_layout()
 

        # 1) per-bin p-values comparing correct vs incorrect across trials
        pvals = np.full(n_bins, np.nan)
        for t in range(n_bins):
            # values at this time bin across all held-out trials
            x = all_corr[t, :]   # correct trials at bin t
            y = all_inc[t, :]    # incorrect trials at bin t
            # Welch's t-test is safer with unequal variances / sample sizes
            tt, pp = ttest_ind(x, y, equal_var=False, nan_policy='omit')
            pvals[t] = pp

        # 2) where to draw the stars: a bit above the taller curve's mean+SEM at each bin
        top_envelope = np.maximum(mean_corr + sem_corr, mean_inc + sem_inc)
        pad = 0.03 * (np.nanmax(top_envelope) - np.nanmin(top_envelope) + 1e-9)  # dynamic padding
        star_heights = top_envelope + pad

        # ---- replace the star-per-bin section with contiguous significance bars ----
        alpha = 0.05          # threshold (change as you like)
        min_len = 2           # require at least this many consecutive bins to draw a bar
        linewidth = 4.0       # "thick" bar
        linecolor = '0.5'     # grey
        zorder_sig = 10       # above fills/curves

        # OPTIONAL: control family-wise error (uncomment if you have statsmodels)
        from statsmodels.stats.multitest import fdrcorrection
        rejected, pvals_corr = fdrcorrection(pvals, alpha=alpha)
        sig_mask = rejected
        # sig_mask = (pvals < alpha)

        # helper: find contiguous [start, end] index runs where mask is True
        runs = []
        in_run = False
        start = None
        for i, val in enumerate(sig_mask):
            if val and not in_run:
                in_run = True
                start = i
            elif not val and in_run:
                runs.append((start, i-1))
                in_run = False
        if in_run:
            runs.append((start, len(sig_mask)-1))

        # optionally filter short runs
        runs = [(a, b) for (a, b) in runs if (b - a + 1) >= min_len]

        # draw one thick grey line per significant run, placed just above the envelope
        for (a, b) in runs:
            # bar height a little above the taller curve in this interval
            local_top = np.nanmax(top_envelope[a:b+1])
            y = local_top + pad
            # draw
            plt.plot([time_vec[a], time_vec[b]], [y, y],
                    color=linecolor, linewidth=linewidth, solid_capstyle='butt',
                    zorder=zorder_sig)

        # (optional) tiny label to remind what the bar means
        # plt.text(time_vec[0], np.nanmax(top_envelope) + 2*pad,
        #          f'grey bar: p<{alpha:g} (≥{min_len} bins)', fontsize=8, va='bottom')
        #plt.savefig(DIR + "error_curve.svg", format="svg")
        plt.show()


    # optional console summary
    print(f"[GLOBAL]  t = {t_stat_raw:.2f}, p = {p_raw:.3e}")
    print(f"[PATIENT] t = {t_stat_pt:.2f}, p = {p_pt:.3e}")

# ────────────────────────────────────────────────────────────────────────────────
def analyze_all_patients_100(mat_file_path: str,
                         binwidth: float,
                         neuron_threshold: int,
                         n_iter: int = 100,
                         seed: int | None = 42) -> None:
    """
    Decode time bins for each patient, repeating the train/test split
    `n_iter` times with fresh random subsets of correct trials.

    Parameters
    ----------
    mat_file_path   : str
        Path to the .mat file containing `neural_data`.
    binwidth        : float
        Width of each time bin in seconds.
    neuron_threshold: int
        Minimum number of simultaneously recorded units required to
        include a patient.
    n_iter          : int, default 100
        Number of Monte-Carlo resampling iterations.
    seed            : int | None, default 41
        RNG seed.  Use None for fully stochastic behaviour.
    """
    rng = np.random.default_rng(seed)

    # ── 1. Load data once ──────────────────────────────────────────────
    mat_data       = scipy.io.loadmat(mat_file_path)
    neural_data    = mat_data['neural_data']
    patient_ids    = [int(e['patient_id'][0][0]) for e in neural_data[0]]
    firing_rates   = [e['firing_rates']          for e in neural_data[0]]
    trial_correct  = [e['trial_correctness']     for e in neural_data[0]]
    unique_patients = np.unique(patient_ids)

    print("Included patients (≥{} units):".format(neuron_threshold))

    # ── 2. Global collectors (across ALL repetitions) ─────────────────
    all_err_corr: list[float] = []
    all_err_inc : list[float] = []

    # per-patient collectors for paired stats
    per_pt_corr = {pid: [] for pid in unique_patients}
    per_pt_inc  = {pid: [] for pid in unique_patients}

    # optional time-resolved collectors
    glob_corr_bins, glob_inc_bins = [], []

    # ── 3. Monte-Carlo loop ───────────────────────────────────────────
    for rep in range(n_iter):
        for pid in unique_patients:
            # • gather this patient’s indexes
            idxs = [i for i, p in enumerate(patient_ids) if p == pid]
            if len(idxs) < neuron_threshold:
                continue                     # skip small sessions

            # • concatenate neural & behaviour arrays
            rates   = [np.asarray(firing_rates[i])           for i in idxs]
            correct = [np.asarray(trial_correct[i]).ravel()  for i in idxs]

            # smoothing (optional – leave as-is or uncomment line below)
            # rates = [gaussian_filter1d(r, sigma=2, axis=1) for r in rates]

            # shape -> [Time, Trials, Neurons]
            data = np.stack(rates, axis=2).transpose(1, 0, 2)
            T, n_trials, n_neur = data.shape

            # logical-AND: trial is “correct” only if all units flagged it
            comb = np.all(np.stack(correct, axis=0), axis=0)
            good = np.where(comb == 1)[0]
            bad  = np.where(comb == 0)[0]

            if bad.size == 0 or good.size < bad.size:
                continue

            # • random split: balance test sets every repetition
            test_good  = rng.choice(good, size=bad.size, replace=False)
            train_good = np.setdiff1d(good, test_good)

            # ╭─────────────────────────────────────────────────────────╮
            # │   Build train / test matrices                          │
            # ╰─────────────────────────────────────────────────────────╯
            X_train = data[:, train_good, :].reshape(-1, n_neur)
            y_train = np.repeat(np.arange(T), len(train_good))

            Xc  = data[:, test_good, :].reshape(-1, n_neur)
            yc  = np.repeat(np.arange(T), len(test_good))
            Xi  = data[:, bad, :].reshape(-1, n_neur)
            yi  = np.repeat(np.arange(T), len(bad))

            # • decode
            clf = GaussianNB().fit(X_train, y_train)
            pc  = clf.predict(Xc)
            pi  = clf.predict(Xi)

            err_c = np.abs(pc - yc) * binwidth
            err_i = np.abs(pi - yi) * binwidth

            # • accumulate
            all_err_corr.extend(err_c)
            all_err_inc .extend(err_i)

            per_pt_corr[pid].append(err_c.mean())
            per_pt_inc [pid].append(err_i.mean())

            glob_corr_bins.append(err_c.reshape(T, -1))
            glob_inc_bins .append(err_i.reshape(T, -1))

        # optional progress bar
        if (rep + 1) % max(1, n_iter // 10) == 0:
            print(f"  finished {rep + 1}/{n_iter} resamples")

    # ── 4. Stats & plots ──────────────────────────────────────────────
    # 4-A  global bin×trial
    t_raw, p_raw = ttest_rel(all_err_corr, all_err_inc)
    mean_raw_c, mean_raw_i = map(np.mean, (all_err_corr, all_err_inc))
    sem_raw_c , sem_raw_i  = map(sem,        (all_err_corr, all_err_inc))

    # 4-B  per-patient (average each patient across repetitions first)
    mean_pt_corr = np.array([np.mean(v) for v in per_pt_corr.values()])
    mean_pt_inc  = np.array([np.mean(v) for v in per_pt_inc .values()])
    
    t_pt, p_pt   = ttest_rel(mean_pt_corr, mean_pt_inc)
    print(f"[GLOBAL]  t = {t_raw:.2f}, p = {p_raw:.3e}")
    print(f"[PATIENT] t = {t_pt:.2f}, p = {p_pt:.3e}")

    # ── 5. PLOT PANEL 1 – grand bar chart ────────────────────────────
    plt.figure(figsize=(4, 6))
    bars = plt.bar(['Correct', 'Incorrect'],
                   [mean_raw_c, mean_raw_i],
                   yerr=[sem_raw_c, sem_raw_i], capsize=5, alpha=0.6,
                   color=['blue', 'indianred'])
    ymax = max(mean_raw_c + sem_raw_c, mean_raw_i + sem_raw_i)
    star_y = ymax + 0.02
    plt.plot([0, 0, 1, 1], [ymax, star_y, star_y, ymax], lw=1.3, color='black')
    star = "***" if p_raw < 1e-3 else "**" if p_raw < 1e-2 else "*" if p_raw < 0.05 else "ns"
    plt.text(0.5, star_y + 0.005, star, ha='center', va='bottom', fontsize=14)
    plt.ylabel("Mean Decoding Error (s)")
    plt.title("Decoding Error – all samples")
    plt.tight_layout()
    plt.show()

    # ── 6. PLOT PANEL 2 – per-patient paired dots ────────────────────
    plt.figure(figsize=(4.8, 6))
    for c, i in zip(mean_pt_corr, mean_pt_inc):
        plt.plot([0, 1], [c, i], color='gray', alpha=0.6)
        plt.scatter([0, 1], [c, i], color='gray', zorder=10)

    plt.hlines(mean_pt_corr.mean(), -0.2, 0.2, color='steelblue', lw=2.5)
    plt.hlines(mean_pt_inc .mean(), 0.8, 1.2, color='indianred', lw=2.5)

    ymax2 = max(mean_pt_corr.max(), mean_pt_inc.max())
    star_y2 = ymax2 + 0.02
    plt.plot([0, 0, 1, 1], [ymax2, star_y2, star_y2, ymax2],
             lw=1.3, color='black')
    star2 = "***" if p_pt < 1e-3 else "**" if p_pt < 1e-2 else "*" if p_pt < 0.05 else "ns"
    print(p_pt)
    plt.text(0.5, star_y2 + 0.005, star2, ha='center', va='bottom', fontsize=14)
    plt.xticks([0, 1], ['Correct', 'Incorrect'])
    plt.ylabel("Mean Decoding Error (s)")
    plt.title("Decoding Error – patient means")
    plt.tight_layout()
    plt.show()

    # ── 7. PLOT PANEL 3 – grand time-resolved curves ─────────────────
    if glob_corr_bins and glob_inc_bins:
        all_corr = np.hstack(glob_corr_bins)   # [T, totalCorrectTrials]
        all_inc  = np.hstack(glob_inc_bins)
        mean_corr, sem_corr = np.mean(all_corr, axis=1), sem(all_corr, axis=1)
        mean_inc , sem_inc  = np.mean(all_inc , axis=1), sem(all_inc , axis=1)
        t_vec = np.arange(mean_corr.size) * binwidth

        plt.figure(figsize=(6, 4))
        plt.plot(t_vec, mean_corr, label='Correct', color='steelblue')
        plt.fill_between(t_vec, mean_corr - sem_corr, mean_corr + sem_corr,
                         color='steelblue', alpha=0.3)
        plt.plot(t_vec, mean_inc, label='Incorrect', color='indianred')
        plt.fill_between(t_vec, mean_inc - sem_inc, mean_inc + sem_inc,
                         color='indianred', alpha=0.3)
        plt.xlabel("Time (s)")
        plt.ylabel("Decoding Error (s)")
        plt.title("Grand-average decoding error\n(all test trials)")
        plt.legend()
        plt.tight_layout()
        plt.show()


import numpy as np
import scipy.io
from sklearn.naive_bayes import GaussianNB
from scipy.stats import sem, ttest_ind
import matplotlib.pyplot as plt

def decode_region_pooled(mat_file,
                         binwidth=0.1,
                         neuron_threshold=20,
                         incorrect_trial_thresh=5,
                         random_state=41,
                         show_plots=True):
    """
    Same purpose as before, but now the test set is perfectly balanced:
    #correct (test) == #incorrect (test).
    """
    rng = np.random.default_rng(random_state)

    # ---------- load -------------------------------------------------------
    mat = scipy.io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    recs = mat['neural_data'].tolist()

    patient_id   = np.array([int(r.patient_id) for r in recs])
    raw_region   = np.array([r.brain_region for r in recs])
    fr_list      = [np.asarray(r.firing_rates) for r in recs]
    perf_list    = [np.asarray(r.trial_correctness).flatten() for r in recs]

    clean_region = np.char.replace(np.char.replace(raw_region, '_left', ''), '_right', '')

    region_summary = {}

    # ---------- iterate over regions --------------------------------------
    for region in np.unique(clean_region):
        idx_region = np.where(clean_region == region)[0]

        # ---- gather neurons, filter patients by incorrect_trial_thresh ----
        neurons_kept, perf_kept = [], []
        for pid in np.unique(patient_id[idx_region]):
            idxs = idx_region[patient_id[idx_region] == pid]
            if idxs.size == 0:
                continue

            perf = perf_list[idxs[0]]
            if np.sum(perf == 0) < incorrect_trial_thresh:
                continue

            neurons_kept.extend(fr_list[i] for i in idxs)
            perf_kept.append(perf)

        n_neurons = len(neurons_kept)
        if n_neurons < neuron_threshold:
            continue

        # ---- harmonise trial counts --------------------------------------
        min_trials = min(fr.shape[0] for fr in neurons_kept)
        T = neurons_kept[0].shape[1]                       # time bins
        neurons_trim = [fr[:min_trials] for fr in neurons_kept]
        perf_concat  = np.concatenate(perf_kept)[:min_trials]

        data = np.stack(neurons_trim, axis=2).transpose(1, 0, 2)      # (T, trials, neurons)
        _, trials, neurons = data.shape

        correct_trials   = np.where(perf_concat == 1)[0]
        incorrect_trials = np.where(perf_concat == 0)[0]

        if correct_trials.size <= incorrect_trials.size:
            continue  # not enough correct trials to balance test set

        # ------------------------------------------------------------------
        #                ### ⇩  UPDATED  ⇩  (balanced split) ###
        # ------------------------------------------------------------------
        n_test_corr = incorrect_trials.size        # EXACT balance
        test_correct = rng.choice(correct_trials, size=n_test_corr, replace=False)
        train_correct = np.setdiff1d(correct_trials, test_correct)
        # ------------------------------------------------------------------

        # training: only remaining correct trials
        X_train = data[:, train_correct, :].reshape(-1, neurons)
        y_train = np.repeat(np.arange(T), train_correct.size)

        # testing: held‑out correct + all incorrect
        Xc = data[:, test_correct,   :].reshape(-1, neurons)
        yc = np.repeat(np.arange(T), test_correct.size)

        Xi = data[:, incorrect_trials, :].reshape(-1, neurons)
        yi = np.repeat(np.arange(T), incorrect_trials.size)

        clf = GaussianNB()
        clf.fit(X_train, y_train)

        err_c = np.abs(clf.predict(Xc) - yc) * binwidth
        err_i = np.abs(clf.predict(Xi) - yi) * binwidth

        p_val = ttest_ind(err_c, err_i, equal_var=False).pvalue
        mean_c, mean_i = err_c.mean(), err_i.mean()
        sem_c,  sem_i  = sem(err_c),   sem(err_i)

        if show_plots:
            plt.figure(figsize=(3.4, 5))
            plt.bar(['Correct', 'Incorrect'],
                    [mean_c, mean_i],
                    yerr=[sem_c, sem_i],
                    color=['blue', 'red'],
                    capsize=4, alpha=.75)
            y_star = max(mean_c + sem_c, mean_i + sem_i) + 0.02
            plt.plot([0, 0, 1, 1], [y_star, y_star+.02, y_star+.02, y_star], color='k')
            sig = "***" if p_val < 1e-3 else "**" if p_val < 1e-2 \
                  else "*" if p_val < 5e-2 else "ns"
            plt.text(0.5, y_star+.02, sig, ha='center')
            plt.ylabel("Decoding error (s)")
            plt.title(f"{region}\n(n neurons = {neurons}, n trials = {trials})")
            plt.tight_layout()
            plt.show()

        region_summary[region] = dict(mean_correct=mean_c,
                                      mean_incorrect=mean_i,
                                      p=p_val,
                                      n_neurons=neurons,
                                      n_trials=trials)

    return region_summary

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, sem
from sklearn.naive_bayes import GaussianNB


def decode_pooled(mat_file,
                  binwidth=0.1,
                  neuron_threshold=20,
                  incorrect_trial_thresh=5,
                  random_state=41,
                  show_plots=True):
    """
    Decode time (class = time‑bin index) from *all* neurons pooled across regions.
    Patients are kept only if they have ≥ `incorrect_trial_thresh` incorrect trials.
    The test set is perfectly balanced: every incorrect trial plus an equal number
    of held‑out correct trials.

    Parameters
    ----------
    mat_file : str
        Path to the .mat file containing a 'neural_data' struct array. Each element
        must have the fields
            ├─ firing_rates : (nTrials, nTimeBins) array
            ├─ trial_correctness : (nTrials,) array of 1/0
            ├─ patient_id : scalar
            └─ brain_region : str               (ignored here)
    binwidth : float, default 0.1
        Width of each time bin in seconds (used to convert bin errors to seconds).
    neuron_threshold : int, default 20
        Minimum total number of neurons required to run the decoder.
    incorrect_trial_thresh : int, default 5
        Minimum incorrect‑trial count a patient must contribute to be included.
    random_state : int, default 41
        Seed for the NumPy RNG used in balanced sampling.
    show_plots : bool, default True
        Whether to draw the bar plot comparing decoding errors.

    Returns
    -------
    summary : dict
        {
          'mean_correct'   : float,
          'mean_incorrect' : float,
          'p'              : float  (Welch t‑test),
          'n_neurons'      : int,
          'n_trials'       : int
        }
        or an empty dict if the data fail the neuron/trial balance checks.
    """
    rng = np.random.default_rng(random_state)

    # ---------- load -------------------------------------------------------
    mat  = scipy.io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    recs = mat['neural_data'].tolist()

    patient_id  = np.array([int(r.patient_id) for r in recs])
    fr_list     = [np.asarray(r.firing_rates) for r in recs]
    perf_list   = [np.asarray(r.trial_correctness).flatten() for r in recs]

    # ---------- select patients -------------------------------------------
    neurons_kept, perf_kept = [], []
    for pid in np.unique(patient_id):
        idxs = np.where(patient_id == pid)[0]
        if idxs.size == 0:
            continue

        perf = perf_list[idxs[0]]
        if np.sum(perf == 0) < incorrect_trial_thresh:
            continue   # patient does not meet incorrect‑trial threshold

        neurons_kept.extend(fr_list[i] for i in idxs)
        perf_kept.append(perf)

    n_neurons = len(neurons_kept)
    if n_neurons < neuron_threshold:
        # Not enough neurons to run the analysis
        return {}

    # ---------- harmonise trial counts ------------------------------------
    min_trials = min(fr.shape[0] for fr in neurons_kept)
    T          = neurons_kept[0].shape[1]          # number of time bins
    neurons_trim = [fr[:min_trials] for fr in neurons_kept]
    perf_concat  = np.concatenate(perf_kept)[:min_trials]

    # data.shape  = (T, trials, neurons)
    data = np.stack(neurons_trim, axis=2).transpose(1, 0, 2)
    _, trials, neurons = data.shape

    # ---------- build balanced train / test split -------------------------
    correct_trials   = np.where(perf_concat == 1)[0]
    incorrect_trials = np.where(perf_concat == 0)[0]

    if correct_trials.size <= incorrect_trials.size:
        # Not enough correct trials to balance the test set
        return {}

    n_test_corr  = incorrect_trials.size
    test_correct = rng.choice(correct_trials, size=n_test_corr, replace=False)
    train_correct = np.setdiff1d(correct_trials, test_correct)

    # ---------- reshape to sample × feature format ------------------------
    X_train = data[:, train_correct, :].reshape(-1, neurons)
    y_train = np.repeat(np.arange(T), train_correct.size)

    Xc = data[:, test_correct,   :].reshape(-1, neurons)
    yc = np.repeat(np.arange(T), test_correct.size)

    Xi = data[:, incorrect_trials, :].reshape(-1, neurons)
    yi = np.repeat(np.arange(T), incorrect_trials.size)

    # ---------- decoding ---------------------------------------------------
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    err_c = np.abs(clf.predict(Xc) - yc) * binwidth
    err_i = np.abs(clf.predict(Xi) - yi) * binwidth

    p_val  = ttest_ind(err_c, err_i, equal_var=False).pvalue
    mean_c, mean_i = err_c.mean(), err_i.mean()
    sem_c,  sem_i  = sem(err_c), sem(err_i)

    # ---------- optional plot ---------------------------------------------
    if show_plots:
        plt.figure(figsize=(3.0, 4.5))
        plt.bar(['Correct', 'Incorrect'],
                [mean_c, mean_i],
                yerr=[sem_c, sem_i],
                color=['steelblue', 'salmon'],
                alpha=.80, capsize=4)
        y_star = max(mean_c + sem_c, mean_i + sem_i) + 0.02
        plt.plot([0, 0, 1, 1], [y_star, y_star+.02, y_star+.02, y_star], color='k')
        sig = "***" if p_val < 1e-3 else "**" if p_val < 1e-2 \
              else "*" if p_val < 5e-2 else "ns"
        plt.text(0.5, y_star+.025, sig, ha='center')
        plt.ylabel("Decoding error (s)")
        plt.title(f"All regions pooled\n(n neurons = {neurons}, n trials = {trials})")
        plt.tight_layout()
        plt.show()

    # ---------- summary ----------------------------------------------------
    return dict(mean_correct=mean_c,
                mean_incorrect=mean_i,
                p=p_val,
                n_neurons=neurons,
                n_trials=trials)
    
import numpy as np
import scipy.io
from scipy.stats import ttest_ind, sem
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt



import numpy as np
import scipy.io
from scipy.stats import ttest_ind, sem
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


def decode_region_pooled_1000(
    mat_file,
    binwidth=0.1,
    neuron_threshold=0,
    incorrect_trial_thresh=15,
    random_state=41,
    n_iter=100,
    show_plots=True,
):
    """
    Region-wise time decoding with balanced test sets, using repeated
    random splits of the training/test sets.

    For each region:
      - Pool neurons across patients (subject to incorrect_trial_thresh).
      - Ensure at least `neuron_threshold` neurons.
      - On each iteration:
          * Randomly split correct trials into train vs test.
          * Use all incorrect trials in the test set.
          * Train on correct trials only.
          * Compute decoding error for held-out correct and incorrect trials.
      - Aggregate errors across iterations and perform a pooled t-test.

    Additionally, for each region we also compute the *decoding error
    curves* (error as a function of true time bin) for correct and
    incorrect trials, averaged over trials and iterations (like in
    `analyze_all_patients_100`, panel 3).

    Returns
    -------
    region_summary : dict
        region -> {
            # scalar summaries (pooled over all iterations)
            'mean_correct'           : float,
            'mean_incorrect'         : float,
            'p'                      : float,  # pooled t-test on all errors
            'n_neurons'              : int,
            'n_trials'               : int,
            'n_iter'                 : int,

            # per-iteration scalar means
            'mean_correct_per_iter'  : np.ndarray [n_iter],
            'mean_incorrect_per_iter': np.ndarray [n_iter],
            'p_values_per_iter'      : np.ndarray [n_iter],

            # time-resolved decoding error curves (trial-based SEM)
            'time'                   : np.ndarray [T],        # seconds
            'curve_correct_mean'     : np.ndarray [T],
            'curve_incorrect_mean'   : np.ndarray [T],
            'curve_correct_sem'      : np.ndarray [T],
            'curve_incorrect_sem'    : np.ndarray [T],
        }
    """
    rng = np.random.default_rng(random_state)

    # ---------- load -------------------------------------------------------
    mat = scipy.io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    recs = mat["neural_data"].tolist()

    patient_id = np.array([int(r.patient_id) for r in recs])
    raw_region = np.array([r.brain_region for r in recs])
    fr_list = [np.asarray(r.firing_rates) for r in recs]
    perf_list = [np.asarray(r.trial_correctness).flatten() for r in recs]

    # collapse _left/_right   (FIXED: outer replace has 'new' arg)
    clean_region = np.char.replace(
        np.char.replace(raw_region, "_left", ""),
        "_right",
        "",
    )

    region_summary = {}

    # ---------- iterate over regions --------------------------------------
    for region in np.unique(clean_region):
        idx_region = np.where(clean_region == region)[0]

        # ---- gather neurons, filter patients by incorrect_trial_thresh ----
        neurons_kept, perf_kept = [], []
        for pid in np.unique(patient_id[idx_region]):
            idxs = idx_region[patient_id[idx_region] == pid]
            if idxs.size == 0:
                continue

            perf = perf_list[idxs[0]]
            if np.sum(perf == 0) < incorrect_trial_thresh:
                continue

            neurons_kept.extend(fr_list[i] for i in idxs)
            perf_kept.append(perf)

        n_neurons = len(neurons_kept)
        if n_neurons < neuron_threshold:
            continue

        # ---- harmonise trial counts --------------------------------------
        min_trials = min(fr.shape[0] for fr in neurons_kept)
        T = neurons_kept[0].shape[1]  # time bins
        neurons_trim = [fr[:min_trials] for fr in neurons_kept]

        # concat behaviour across kept patients, then truncate to min_trials
        perf_concat = np.concatenate(perf_kept)[:min_trials]

        # data: (T, trials, neurons)
        data = np.stack(neurons_trim, axis=2).transpose(1, 0, 2)
        _, trials, neurons = data.shape

        correct_trials = np.where(perf_concat == 1)[0]
        incorrect_trials = np.where(perf_concat == 0)[0]

        # need strictly more correct than incorrect to have a train set
        if correct_trials.size <= incorrect_trials.size:
            continue

        # -------------------- repeated resampling --------------------------
        all_err_c = []
        all_err_i = []
        mean_c_list = []
        mean_i_list = []
        p_list = []

        # time-resolved collectors (trial-based, like glob_corr_bins/inc)
        region_corr_bins = []  # each: [T, n_test_correct_trials_this_iter]
        region_inc_bins  = []  # each: [T, n_incorrect_trials]

        for _ in range(n_iter):
            # EXACTLY balanced test set on each iteration
            n_test_corr = incorrect_trials.size
            test_correct = rng.choice(correct_trials, size=n_test_corr, replace=True)
            train_correct = np.setdiff1d(correct_trials, test_correct)

            # training: only remaining correct trials
            X_train = data[:, train_correct, :].reshape(-1, neurons)
            y_train = np.repeat(np.arange(T), train_correct.size)

            # testing: held-out correct + all incorrect
            Xc = data[:, test_correct, :].reshape(-1, neurons)
            yc = np.repeat(np.arange(T), test_correct.size)

            Xi = data[:, incorrect_trials, :].reshape(-1, neurons)
            yi = np.repeat(np.arange(T), incorrect_trials.size)

            clf = GaussianNB()
            clf.fit(X_train, y_train)

            pred_c = clf.predict(Xc)
            pred_i = clf.predict(Xi)

            err_c = np.abs(pred_c - yc) * binwidth
            err_i = np.abs(pred_i - yi) * binwidth

            all_err_c.append(err_c)
            all_err_i.append(err_i)

            mean_c = err_c.mean()
            mean_i = err_i.mean()
            mean_c_list.append(mean_c)
            mean_i_list.append(mean_i)

            p_iter = ttest_ind(err_c, err_i, equal_var=False).pvalue
            p_list.append(p_iter)

            # ---------- time-resolved collectors for this iteration --------
            # reshape to [T, n_trials]; keep *all* trials for SEM across trials
            err_c_mat = err_c.reshape(T, -1)
            err_i_mat = err_i.reshape(T, -1)

            region_corr_bins.append(err_c_mat)
            region_inc_bins.append(err_i_mat)

        # concatenate all iterations’ errors (scalar-level)
        all_err_c = np.concatenate(all_err_c)
        all_err_i = np.concatenate(all_err_i)

        mean_c_overall = all_err_c.mean()
        mean_i_overall = all_err_i.mean()
        sem_c_overall = sem(all_err_c)
        sem_i_overall = sem(all_err_i)

        # pooled t-test across all errors
        p_pooled = ttest_ind(all_err_c, all_err_i, equal_var=False).pvalue

        # ---------- aggregate curves across trials & iterations -----------
        if region_corr_bins and region_inc_bins:
            # [T, totalTrials] for this region
            all_corr_mat = np.hstack(region_corr_bins)
            all_inc_mat  = np.hstack(region_inc_bins)

            curve_c_mean = all_corr_mat.mean(axis=1)
            curve_i_mean = all_inc_mat.mean(axis=1)

            curve_c_sem = sem(all_corr_mat, axis=1)
            curve_i_sem = sem(all_inc_mat, axis=1)
        else:
            # fallback (should rarely happen)
            curve_c_mean = np.zeros(T)
            curve_i_mean = np.zeros(T)
            curve_c_sem = np.zeros(T)
            curve_i_sem = np.zeros(T)

        time_axis = np.arange(T) * binwidth

        # ---------- plotting (single summary per region) -------------------
        if show_plots:
            # Bar summary (overall mean error)
            plt.figure(figsize=(3.4, 5))
            plt.bar(
                ["Correct", "Incorrect"],
                [mean_c_overall, mean_i_overall],
                yerr=[sem_c_overall, sem_i_overall],
                color=["blue", "red"],
                capsize=4,
                alpha=0.75,
            )
            y_star = max(
                mean_c_overall + sem_c_overall,
                mean_i_overall + sem_i_overall,
            ) + 0.02
            plt.plot(
                [0, 0, 1, 1],
                [y_star, y_star + 0.02, y_star + 0.02, y_star],
                color="black",
            )
            sig = (
                "***"
                if p_pooled < 1e-3
                else "**"
                if p_pooled < 1e-2
                else "*"
                if p_pooled < 5e-2
                else "ns"
            )
            plt.text(0.5, y_star + 0.02, sig, ha="center")
            plt.ylabel("Decoding error (s)")
            plt.title(
                f"{region}\n(n neurons = {neurons}, "
                f"n trials = {trials}, n_iter = {n_iter})"
            )
            plt.tight_layout()
            plt.show()

            # Time-resolved decoding error curves with SEM (like your panel 3)
            plt.figure(figsize=(6, 4))
            plt.plot(time_axis, curve_c_mean, label="Correct", color="blue")
            plt.fill_between(
                time_axis,
                curve_c_mean - curve_c_sem,
                curve_c_mean + curve_c_sem,
                alpha=0.3,
                color="blue",
            )
            plt.plot(time_axis, curve_i_mean, label="Incorrect", color="red")
            plt.fill_between(
                time_axis,
                curve_i_mean - curve_i_sem,
                curve_i_mean + curve_i_sem,
                alpha=0.3,
                color="red",
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Decoding error (s)")
            plt.title(f"{region} – decoding error vs time")
            plt.legend()
            plt.tight_layout()
            plt.show()

        # ---------- store summary -----------------------------------------
        region_summary[region] = dict(
            # scalar summary
            mean_correct=mean_c_overall,
            mean_incorrect=mean_i_overall,
            p=p_pooled,
            n_neurons=neurons,
            n_trials=trials,
            n_iter=n_iter,
            mean_correct_per_iter=np.asarray(mean_c_list),
            mean_incorrect_per_iter=np.asarray(mean_i_list),
            p_values_per_iter=np.asarray(p_list),
            # curves (trial-based SEM)
            time=time_axis,
            curve_correct_mean=curve_c_mean,
            curve_incorrect_mean=curve_i_mean,
            curve_correct_sem=curve_c_sem,
            curve_incorrect_sem=curve_i_sem,
        )

    return region_summary

if __name__ == "__main__":
    # summary = decode_region_pooled_1000("3sig15_data.mat",
    #                                binwidth=0.1,
    #                                neuron_threshold=1,
    #                                incorrect_trial_thresh=10,
    #                                random_state=20251125,
    #                                show_plots=True)
    
    # from pprint import pprint
    # pprint(summary)
    analyze_all_patients('3sig15_data.mat', 0.1, neuron_threshold=5)
    #analyze_all_patients('Figure 4/3sig15_LIS.mat', 0.1, neuron_threshold=3)
    #analyze_all_patients_100('3sig15_data.mat', 0.1, neuron_threshold=5)