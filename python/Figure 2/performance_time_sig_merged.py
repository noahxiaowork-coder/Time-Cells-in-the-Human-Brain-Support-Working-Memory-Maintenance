#!/usr/bin/env python
"""
Multi-patient naïve-Bayes decoder comparing correct vs incorrect trials.

This script runs TWO complementary analyses:
  (1) Simple balanced: Match total number of test trials (n_correct = n_incorrect)
  (2) Load-matched: Match both number AND load distribution across trial types

For each patient:
  - Train on correct trials only
  - Test on held-out correct trials and incorrect trials
  - Aggregate decoding errors across patients

Outputs for EACH method:
  - Global bar plot (correct vs incorrect, pooled samples)
  - Grand-average decoding error vs time with significance bars
  - Console statistics

Author: Merged version, 2026-01-13
"""

import numpy as np
import scipy.io
from sklearn.naive_bayes import GaussianNB
from scipy.stats import sem, ttest_rel, ttest_ind
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import significance_stars

DIR = ''


def analyze_all_patients_simple_balanced(mat_file_path, binwidth, neuron_threshold):
    np.random.seed(20250710)

    mat_data = scipy.io.loadmat(mat_file_path)
    neural_data = mat_data['neural_data'][0]

    patient_ids = [int(e['patient_id'][0][0]) for e in neural_data]
    firing_rates = [e['firing_rates'] for e in neural_data]
    trial_performance = [e['trial_correctness'] for e in neural_data]

    unique_patient_ids = np.unique(patient_ids)

    all_decoding_errors_correct = []
    all_decoding_errors_incorrect = []
    per_patient_correct_means = []
    per_patient_incorrect_means = []
    global_errors_correct_bins = []
    global_errors_incorrect_bins = []

    print("\n" + "="*70)
    print("ANALYSIS 1: SIMPLE BALANCED (matching total trial counts)")
    print("="*70)
    print("Included patients:")

    for patient_id in unique_patient_ids:

        selected_neurons = [
            (firing_rates[i], trial_performance[i])
            for i, pid in enumerate(patient_ids)
            if pid == patient_id
        ]

        if len(selected_neurons) < neuron_threshold:
            continue

        print(patient_id)

        selected_firing_rates = [np.asarray(n[0]) for n in selected_neurons]
        selected_performance = [np.asarray(n[1]).flatten() for n in selected_neurons]

        reshaped_data = np.stack(selected_firing_rates, axis=2).transpose(1, 0, 2)
        time_bins, trials, neurons = reshaped_data.shape

        combined_performance = np.all(np.stack(selected_performance, axis=0), axis=0)
        correct_trials = np.where(combined_performance == 1)[0]
        incorrect_trials = np.where(combined_performance == 0)[0]

        if len(correct_trials) == 0 or len(incorrect_trials) == 0:
            continue

        n_incorrect = len(incorrect_trials)
        if len(correct_trials) < n_incorrect:
            continue

        test_correct_trials = np.random.choice(correct_trials, n_incorrect, replace=False)
        train_correct_trials = np.setdiff1d(correct_trials, test_correct_trials)

        X_train = reshaped_data[:, train_correct_trials, :].reshape(-1, neurons)
        y_train = np.repeat(np.arange(time_bins), len(train_correct_trials))

        X_test_correct = reshaped_data[:, test_correct_trials, :].reshape(-1, neurons)
        y_test_correct = np.repeat(np.arange(time_bins), len(test_correct_trials))

        X_test_incorrect = reshaped_data[:, incorrect_trials, :].reshape(-1, neurons)
        y_test_incorrect = np.repeat(np.arange(time_bins), len(incorrect_trials))

        clf = GaussianNB()
        clf.fit(X_train, y_train)

        y_pred_correct = clf.predict(X_test_correct)
        y_pred_incorrect = clf.predict(X_test_incorrect)

        decoding_errors_correct = np.abs(y_pred_correct - y_test_correct) * binwidth
        decoding_errors_incorrect = np.abs(y_pred_incorrect - y_test_incorrect) * binwidth

        all_decoding_errors_correct.extend(decoding_errors_correct)
        all_decoding_errors_incorrect.extend(decoding_errors_incorrect)

        per_patient_correct_means.append(decoding_errors_correct.mean())
        per_patient_incorrect_means.append(decoding_errors_incorrect.mean())

        y_pred_correct_rs = y_pred_correct.reshape(time_bins, -1)
        y_test_correct_rs = y_test_correct.reshape(time_bins, -1)
        errors_correct_per_bin = np.abs(y_pred_correct_rs - y_test_correct_rs) * binwidth

        y_pred_incorrect_rs = y_pred_incorrect.reshape(time_bins, -1)
        y_test_incorrect_rs = y_test_incorrect.reshape(time_bins, -1)
        errors_incorrect_per_bin = np.abs(y_pred_incorrect_rs - y_test_incorrect_rs) * binwidth

        global_errors_correct_bins.append(errors_correct_per_bin)
        global_errors_incorrect_bins.append(errors_incorrect_per_bin)

    return {
        'all_errors_correct': all_decoding_errors_correct,
        'all_errors_incorrect': all_decoding_errors_incorrect,
        'per_patient_correct': per_patient_correct_means,
        'per_patient_incorrect': per_patient_incorrect_means,
        'global_errors_correct_bins': global_errors_correct_bins,
        'global_errors_incorrect_bins': global_errors_incorrect_bins,
        'binwidth': binwidth
    }


def analyze_all_patients_load_matched(mat_file_path, binwidth, neuron_threshold):
    np.random.seed(203243)

    mat_data = scipy.io.loadmat(mat_file_path)
    neural_data = mat_data['neural_data'][0]

    subject_ids = [int(e['patient_id'][0][0]) for e in neural_data]
    rates = [e['firing_rates'] for e in neural_data]
    trial_correct = [e['trial_correctness'] for e in neural_data]
    trial_load = [e['trial_load'] for e in neural_data]

    unique_subjects = np.unique(subject_ids)

    all_err_correct = []
    all_err_incorrect = []
    subj_mean_correct = []
    subj_mean_incorrect = []
    global_err_correct_bins = []
    global_err_incorrect_bins = []

    print("\n" + "="*70)
    print("ANALYSIS 2: LOAD-MATCHED BALANCED (matching load distributions)")
    print("="*70)
    print("Included subjects:")

    for sid in unique_subjects:

        subject_items = [
            (rates[i], trial_correct[i], trial_load[i])
            for i, pid in enumerate(subject_ids)
            if pid == sid
        ]

        if len(subject_items) < neuron_threshold:
            continue

        print(sid)

        subj_rates = [np.asarray(it[0]) for it in subject_items]
        subj_performance = [np.asarray(it[1]).flatten() for it in subject_items]
        subj_load_vec = np.asarray(subject_items[0][2]).flatten().astype(int)

        reshaped = np.stack(subj_rates, axis=2).transpose(1, 0, 2)
        time_bins, trials, neurons = reshaped.shape

        combined_perf = np.all(np.stack(subj_performance, axis=0), axis=0)
        corr_idx = np.where(combined_perf == 1)[0]
        inc_idx = np.where(combined_perf == 0)[0]

        if corr_idx.size == 0 or inc_idx.size == 0:
            continue

        rng = np.random.default_rng(4343)

        def split_by_load(indices):
            loads = subj_load_vec[indices]
            return {
                1: indices[loads == 1],
                2: indices[loads == 2],
                3: indices[loads == 3],
            }

        corr_by_load = split_by_load(corr_idx)
        inc_by_load = split_by_load(inc_idx)

        test_corr_blocks = []
        test_inc_blocks = []

        for L in (1, 2, 3):
            inc_L = inc_by_load[L]
            corr_L = corr_by_load[L]
            if inc_L.size == 0 or corr_L.size == 0:
                continue

            n_take = min(inc_L.size, corr_L.size)

            sel_inc = rng.choice(inc_L, size=n_take, replace=False)
            sel_corr = rng.choice(corr_L, size=n_take, replace=False)

            test_inc_blocks.append(sel_inc)
            test_corr_blocks.append(sel_corr)

        if len(test_corr_blocks) == 0:
            continue

        test_corr_idx = np.sort(np.concatenate(test_corr_blocks))
        test_inc_idx = np.sort(np.concatenate(test_inc_blocks))

        train_corr_idx = np.setdiff1d(corr_idx, test_corr_idx)

        X_train = reshaped[:, train_corr_idx, :].reshape(-1, neurons)
        y_train = np.repeat(np.arange(time_bins), len(train_corr_idx))

        X_test_corr = reshaped[:, test_corr_idx, :].reshape(-1, neurons)
        y_test_corr = np.repeat(np.arange(time_bins), len(test_corr_idx))

        X_test_inc = reshaped[:, test_inc_idx, :].reshape(-1, neurons)
        y_test_inc = np.repeat(np.arange(time_bins), len(test_inc_idx))

        clf = GaussianNB()
        clf.fit(X_train, y_train)

        y_pred_corr = clf.predict(X_test_corr)
        y_pred_inc = clf.predict(X_test_inc)

        err_corr = np.abs(y_pred_corr - y_test_corr) * binwidth
        err_inc = np.abs(y_pred_inc - y_test_inc) * binwidth

        all_err_correct.extend(err_corr)
        all_err_incorrect.extend(err_inc)

        subj_mean_correct.append(err_corr.mean())
        subj_mean_incorrect.append(err_inc.mean())

        y_pred_corr_rs = y_pred_corr.reshape(time_bins, -1)
        y_test_corr_rs = y_test_corr.reshape(time_bins, -1)
        err_corr_bin = np.abs(y_pred_corr_rs - y_test_corr_rs) * binwidth

        y_pred_inc_rs = y_pred_inc.reshape(time_bins, -1)
        y_test_inc_rs = y_test_inc.reshape(time_bins, -1)
        err_inc_bin = np.abs(y_pred_inc_rs - y_test_inc_rs) * binwidth

        global_err_correct_bins.append(err_corr_bin)
        global_err_incorrect_bins.append(err_inc_bin)

    return {
        'all_errors_correct': all_err_correct,
        'all_errors_incorrect': all_err_incorrect,
        'per_patient_correct': subj_mean_correct,
        'per_patient_incorrect': subj_mean_incorrect,
        'global_errors_correct_bins': global_err_correct_bins,
        'global_errors_incorrect_bins': global_err_incorrect_bins,
        'binwidth': binwidth
    }


def plot_results(results, title_suffix="", save_suffix=""):
    all_errors_correct = results['all_errors_correct']
    all_errors_incorrect = results['all_errors_incorrect']
    per_patient_correct = results['per_patient_correct']
    per_patient_incorrect = results['per_patient_incorrect']
    global_errors_correct_bins = results['global_errors_correct_bins']
    global_errors_incorrect_bins = results['global_errors_incorrect_bins']
    binwidth = results['binwidth']

    t_stat_raw, p_raw = ttest_ind(all_errors_correct, all_errors_incorrect, equal_var=False)

    mean_raw_correct = np.mean(all_errors_correct)
    mean_raw_incorrect = np.mean(all_errors_incorrect)
    sem_raw_correct = sem(all_errors_correct)
    sem_raw_incorrect = sem(all_errors_incorrect)

    plt.figure(figsize=(4, 6))
    plt.bar(
        ['Correct', 'Incorrect'],
        [mean_raw_correct, mean_raw_incorrect],
        yerr=[sem_raw_correct, sem_raw_incorrect],
        capsize=5,
        alpha=0.6,
        color=['blue', 'red']
    )

    ymax = max(mean_raw_correct + sem_raw_correct, mean_raw_incorrect + sem_raw_incorrect)
    star_y = ymax + 0.02
    plt.plot([0, 0, 1, 1], [ymax, star_y, star_y, ymax], lw=1.3, color='black')

    star = significance_stars(p_raw)
    plt.text(0.5, star_y + 0.005, star, ha='center', va='bottom', fontsize=14)

    plt.ylabel("Mean Decoding Error (s)")
    plt.title(f"Decoding Error – pooled samples{title_suffix}")
    plt.tight_layout()
    if save_suffix:
        plt.savefig(DIR + f"error_bar{save_suffix}.svg", format="svg")
    plt.show()

    t_stat_pt, p_pt = ttest_rel(per_patient_correct, per_patient_incorrect)

    plt.figure(figsize=(4.8, 6))

    for c, i in zip(per_patient_correct, per_patient_incorrect):
        plt.plot([0, 1], [c, i], color='gray', alpha=0.6)
        plt.scatter(0, c, color='gray', alpha=0.8, zorder=10)
        plt.scatter(1, i, color='gray', alpha=0.8, zorder=10)

    plt.hlines(np.mean(per_patient_correct), -0.2, 0.2, color='blue', linewidth=2.5)
    plt.hlines(np.mean(per_patient_incorrect), 0.8, 1.2, color='red', linewidth=2.5)

    ymax2 = max(per_patient_correct + per_patient_incorrect)
    star_y2 = ymax2 + 0.02
    plt.plot([0, 0, 1, 1], [ymax2, star_y2, star_y2, ymax2], lw=1.3, color='black')

    star_pt = significance_stars(p_pt)
    if star_pt == "ns":
        sig_label = f"p = {p_pt:.3f}"
    else:
        sig_label = star_pt

    plt.text(0.5, star_y2 + 0.005, sig_label, ha='center', va='bottom', fontsize=14)

    plt.xticks([0, 1], ['Correct', 'Incorrect'])
    plt.ylabel("Mean Decoding Error (s)")
    plt.title(f"Decoding Error – patient means{title_suffix}")
    plt.tight_layout()
    if save_suffix:
        plt.savefig(DIR + f"error_patient_means{save_suffix}.svg", format="svg")
    plt.show()

    if global_errors_correct_bins and global_errors_incorrect_bins:

        all_corr = np.hstack(global_errors_correct_bins)
        all_inc = np.hstack(global_errors_incorrect_bins)

        mean_corr = np.mean(all_corr, axis=1)
        sem_corr = sem(all_corr, axis=1)

        mean_inc = np.mean(all_inc, axis=1)
        sem_inc = sem(all_inc, axis=1)

        n_bins = mean_corr.size
        time_vec = np.arange(n_bins) * binwidth

        plt.figure(figsize=(6, 4))
        plt.plot(time_vec, mean_corr, label='Correct', color='blue')
        plt.fill_between(time_vec, mean_corr - sem_corr, mean_corr + sem_corr, color='blue', alpha=0.3)

        plt.plot(time_vec, mean_inc, label='Incorrect', color='red')
        plt.fill_between(time_vec, mean_inc - sem_inc, mean_inc + sem_inc, color='red', alpha=0.3)

        plt.xlabel("Time (s)")
        plt.ylabel("Decoding Error (s)")
        plt.title(f"Grand-average decoding error{title_suffix}")
        plt.legend()
        plt.tight_layout()

        pvals = np.full(n_bins, np.nan)
        for t in range(n_bins):
            x = all_corr[t, :]
            y = all_inc[t, :]
            _, pvals[t] = ttest_ind(x, y, equal_var=False, nan_policy='omit')

        top_env = np.maximum(mean_corr + sem_corr, mean_inc + sem_inc)
        pad = 0.03 * (np.nanmax(top_env) - np.nanmin(top_env) + 1e-9)

        alpha = 0.05
        sig_mask = pvals < alpha

        runs = []
        in_run = False
        start = None
        for i, val in enumerate(sig_mask):
            if val and not in_run:
                in_run = True
                start = i
            elif not val and in_run:
                runs.append((start, i - 1))
                in_run = False
        if in_run:
            runs.append((start, len(sig_mask) - 1))

        min_len = 2
        for a, b in runs:
            if (b - a + 1) < min_len:
                continue
            local_top = np.nanmax(top_env[a:b+1])
            y = local_top + pad
            plt.plot([time_vec[a], time_vec[b]], [y, y], color='0.5', linewidth=4.0, solid_capstyle='butt', zorder=10)

        if save_suffix:
            plt.savefig(DIR + f"error_curve{save_suffix}.svg", format="svg")
        plt.show()

    print(f"\n[GLOBAL pooled]  t = {t_stat_raw:.2f}, p = {p_raw:.3e}")
    print(f"[PATIENT means]  t = {t_stat_pt:.2f}, p = {p_pt:.3e}\n")


if __name__ == "__main__":

    mat_file = 'TC.mat'
    binwidth = 0.1
    neuron_threshold = 5

    results_simple = analyze_all_patients_simple_balanced(mat_file, binwidth, neuron_threshold)
    plot_results(results_simple, title_suffix=" (simple balanced)", save_suffix="_simple")

    results_load_matched = analyze_all_patients_load_matched(mat_file, binwidth, neuron_threshold)
    plot_results(results_load_matched, title_suffix=" (load-matched)", save_suffix="_load_matched")

    print("\n" + "="*70)
    print("BOTH ANALYSES COMPLETED")
    print("="*70)
