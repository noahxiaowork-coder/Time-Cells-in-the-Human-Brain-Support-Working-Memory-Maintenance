import numpy as np
from collections import Counter
import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter1d
from sklearn.naive_bayes import GaussianNB
from scipy.stats import sem, ttest_rel, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns   # still optional but left for your styling needs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def analyze_all_patients(mat_file_path, binwidth, neuron_threshold):
    np.random.seed(203243)

    # Load data
    mat_data = scipy.io.loadmat(mat_file_path)
    neural_data = mat_data['neural_data']
    patient_ids       = [int(entry['patient_id'][0][0]) for entry in neural_data[0]]
    firing_rates      = [entry['firing_rates']        for entry in neural_data[0]]
    trial_performance = [entry['trial_correctness']   for entry in neural_data[0]]
    trial_loads       = [entry['trial_load']          for entry in neural_data[0]]   ### NEW

    unique_patient_ids = np.unique(patient_ids)

    all_decoding_errors_correct = []
    all_decoding_errors_incorrect = []
    per_patient_correct_means = []
    per_patient_incorrect_means = []
    global_errors_correct_bins  = []
    global_errors_incorrect_bins = []

    print("included patients:")

    for patient_id in unique_patient_ids:
        # Gather data for this patient
        selected_items = [
            (firing_rates[i], trial_performance[i], trial_loads[i])           ### NEW (include loads)
            for i, pid in enumerate(patient_ids) 
            if pid == patient_id
        ]
        if len(selected_items) < neuron_threshold:
            continue

        print(patient_id)

        selected_firing_rates = [np.array(item[0]) for item in selected_items]
        selected_performance  = [np.array(item[1]).flatten() for item in selected_items]
        patient_load_vec      = np.array(selected_items[0][2]).flatten().astype(int)   ### NEW (per-trial loads)

        smoothed_neural_data = selected_firing_rates

        # [Time_bins x Trials x Neurons]
        reshaped_data = np.stack(smoothed_neural_data, axis=2).transpose(1, 0, 2)
        time_bins, trials, neurons = reshaped_data.shape

        combined_performance = np.all(np.stack(selected_performance, axis=0), axis=0)
        correct_trials_idx   = np.where(combined_performance == 1)[0]
        incorrect_trials_idx = np.where(combined_performance == 0)[0]

        if len(correct_trials_idx) == 0 or len(incorrect_trials_idx) == 0:
            continue

        # ===========================
        # LOAD-MATCHED TEST SELECTION
        # ===========================
        rng = np.random.default_rng(4343)  ### NEW (reproducible)

        # split indices by load within each group
        def by_load(indices):
            Ls = patient_load_vec[indices]
            return {
                1: indices[Ls == 1],
                2: indices[Ls == 2],
                3: indices[Ls == 3]
            }

        corr_by_load = by_load(correct_trials_idx)
        inc_by_load  = by_load(incorrect_trials_idx)

        # target per-load counts: min(incorrect, correct) so both sides match
        test_corr_idxs = []
        test_inc_idxs  = []
        for L in (1, 2, 3):
            inc_L = inc_by_load[L]
            corr_L = corr_by_load[L]
            n_inc = inc_L.size
            n_corr = corr_L.size
            if n_inc == 0 or n_corr == 0:
                # If one group has no trials in this load, skip this load entirely.
                continue
            n_take = min(n_inc, n_corr)

            # sample without replacement on both sides
            sel_inc  = rng.choice(inc_L,  size=n_take, replace=False)
            sel_corr = rng.choice(corr_L, size=n_take, replace=False)

            test_inc_idxs.append(sel_inc)
            test_corr_idxs.append(sel_corr)

        if len(test_corr_idxs) == 0:
            # No overlapping loads with available trials—skip this patient
            continue

        test_correct_trials   = np.sort(np.concatenate(test_corr_idxs))
        test_incorrect_trials = np.sort(np.concatenate(test_inc_idxs))

        # remaining correct trials are used for training
        train_correct_trials = np.setdiff1d(correct_trials_idx, test_correct_trials)

        # Prepare training set (correct trials only)
        X_train = reshaped_data[:, train_correct_trials, :].reshape(-1, neurons)
        y_train = np.repeat(np.arange(time_bins), len(train_correct_trials))

        # Prepare testing sets (load-matched held-out correct + load-matched incorrect)
        X_test_correct = reshaped_data[:, test_correct_trials, :].reshape(-1, neurons)
        y_test_correct = np.repeat(np.arange(time_bins), len(test_correct_trials))

        X_test_incorrect = reshaped_data[:, test_incorrect_trials, :].reshape(-1, neurons)
        y_test_incorrect = np.repeat(np.arange(time_bins), len(test_incorrect_trials))

        # Train classifier
        bayesian_clf = GaussianNB()
        bayesian_clf.fit(X_train, y_train)

        # Predict
        y_pred_correct   = bayesian_clf.predict(X_test_correct)
        y_pred_incorrect = bayesian_clf.predict(X_test_incorrect)

        # Decoding errors
        decoding_errors_correct   = np.abs(y_pred_correct   - y_test_correct)   * binwidth
        decoding_errors_incorrect = np.abs(y_pred_incorrect - y_test_incorrect) * binwidth

        # Aggregate
        all_decoding_errors_correct.extend(decoding_errors_correct)
        all_decoding_errors_incorrect.extend(decoding_errors_incorrect)

        per_patient_correct_means.append(decoding_errors_correct.mean())
        per_patient_incorrect_means.append(decoding_errors_incorrect.mean())

        # Per-time-bin errors (for grand curve)
        y_pred_correct_reshaped   = y_pred_correct.reshape(time_bins, -1)
        y_test_correct_reshaped   = y_test_correct.reshape(time_bins, -1)
        errors_correct_per_bin    = np.abs(y_pred_correct_reshaped - y_test_correct_reshaped) * binwidth

        y_pred_incorrect_reshaped = y_pred_incorrect.reshape(time_bins, -1)
        y_test_incorrect_reshaped = y_test_incorrect.reshape(time_bins, -1)
        errors_incorrect_per_bin  = np.abs(y_pred_incorrect_reshaped - y_test_incorrect_reshaped) * binwidth

        global_errors_correct_bins.append(errors_correct_per_bin)
        global_errors_incorrect_bins.append(errors_incorrect_per_bin)


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

        sig_mask = (pvals < alpha)

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


analyze_all_patients('3sig15_data.mat', 0.1, neuron_threshold=5)