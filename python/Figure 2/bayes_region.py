import numpy as np
import scipy.io
from sklearn.naive_bayes import GaussianNB
from scipy.stats import sem, ttest_rel, ttest_ind, ttest_1samp
import matplotlib.pyplot as plt

DIR = ''


def analyze_region_dropout_train_test(
    mat_file_path,
    binwidth,
    neuron_threshold,
    collapse_lr=True,
    n_iterations=1000,
    load_balanced=True
):
    """Regional dropout analysis comparing full vs dropped model contributions."""

    np.random.seed(20250710)

    region_map = {
        "dorsal_anterior_cingulate_cortex": "DaCC",
        "pre_supplementary_motor_area": "PSMA",
        "hippocampus": "HPC",
        "amygdala": "AMY",
        "ventral_medial_prefrontal_cortex": "vmPFC",
    }

    region_colors = {
        "hippocampus": "#FFD700",                       # gold
        "amygdala": "#00FFFF",                          # cyan
        "pre_supplementary_motor_area": "#FF0000",      # red
        "dorsal_anterior_cingulate_cortex": "#0000FF",  # blue
        "ventral_medial_prefrontal_cortex": "#008000",  # green
    }

    ignore_regions = {"ventral_medial_prefrontal_cortex"}

    mat_data = scipy.io.loadmat(mat_file_path)
    recs = mat_data['neural_data'][0]

    patient_ids   = np.array([int(r['patient_id'][0][0]) for r in recs])
    brain_regions = np.array([r['brain_region'][0] for r in recs])
    firing_rates  = [np.asarray(r['firing_rates']) for r in recs]
    trial_perf    = [np.asarray(r['trial_correctness']).flatten() for r in recs]

    if load_balanced:
        trial_loads = [np.asarray(r['trial_load']).flatten() for r in recs]
    else:
        trial_loads = [None for _ in recs]

    if collapse_lr:
        brain_regions = np.char.replace(
            np.char.replace(brain_regions, '_left', ''),
            '_right', ''
        )

    unique_patient_ids = np.unique(patient_ids)
    unique_regions = np.unique(brain_regions)

    regions_to_use = [
        r for r in unique_regions
        if (r in region_map) and (r not in ignore_regions)
    ]

    print("Regions used for dropout:", regions_to_use)
    print("Patients considered for dropout analysis:")

    patient_data = {}

    for pid in unique_patient_ids:
        sel = np.where(patient_ids == int(pid))[0]
        if sel.size < neuron_threshold:
            continue

        selected_firing_rates = [firing_rates[i] for i in sel]
        selected_perf   = [trial_perf[i] for i in sel]
        selected_regions = brain_regions[sel]
        selected_loads   = [trial_loads[i] for i in sel]

        trial_data = np.stack(selected_firing_rates, axis=1)
        reshaped_data = np.transpose(trial_data, (2, 0, 1))
        time_bins, trials, neurons = reshaped_data.shape

        combined_perf = np.all(np.stack(selected_perf, axis=0), axis=0)
        correct_trials   = np.where(combined_perf == 1)[0]
        incorrect_trials = np.where(combined_perf == 0)[0]

        if len(correct_trials) == 0 or len(incorrect_trials) == 0:
            continue
        if len(correct_trials) < len(incorrect_trials):
            continue

        if load_balanced:
            combined_load = selected_loads[0]
            ok = True
            loads_in_incorrect = np.unique(combined_load[incorrect_trials])
            for lv in loads_in_incorrect:
                n_inc  = np.sum(combined_load[incorrect_trials] == lv)
                n_corr = np.sum(combined_load[correct_trials]   == lv)
                if n_inc > 0 and n_corr < n_inc:
                    ok = False
                    break
            if not ok:
                continue
        else:
            combined_load = None

        print(f"  patient {pid}: total neurons = {neurons}")

        patient_data[int(pid)] = dict(
            reshaped_data   = reshaped_data,
            correct_trials  = correct_trials,
            incorrect_trials= incorrect_trials,
            time_bins       = time_bins,
            neurons         = neurons,
            region_labels   = selected_regions,
            trial_load      = combined_load
        )

    if len(patient_data) == 0:
        print("No patients met inclusion criteria.")
        return

    region_gap_all  = {reg: {} for reg in regions_to_use}
    region_gap_drop = {reg: {} for reg in regions_to_use}

    for region_name in regions_to_use:
        print(f"\n=== Region dropout: {region_name} ===")

        for pid, pdata in patient_data.items():
            reshaped_data  = pdata['reshaped_data']
            correct_trials = pdata['correct_trials']
            incorrect_trials = pdata['incorrect_trials']
            time_bins      = pdata['time_bins']
            neurons        = pdata['neurons']
            region_labels  = pdata['region_labels']
            trial_load_vec = pdata['trial_load']

            if not np.any(region_labels == region_name):
                continue
            if np.sum(region_labels != region_name) < 1:
                continue

            if pid not in region_gap_all[region_name]:
                region_gap_all[region_name][pid]  = []
                region_gap_drop[region_name][pid] = []

            for _ in range(n_iterations):
                n_incorrect = len(incorrect_trials)

                if load_balanced and trial_load_vec is not None:
                    loads_in_incorrect = np.unique(trial_load_vec[incorrect_trials])
                    test_correct_list = []
                    ok = True
                    for lv in loads_in_incorrect:
                        n_inc = np.sum(trial_load_vec[incorrect_trials] == lv)
                        if n_inc == 0:
                            continue
                        corr_candidates = correct_trials[trial_load_vec[correct_trials] == lv]
                        if len(corr_candidates) < n_inc:
                            ok = False
                            break
                        chosen_corr = np.random.choice(corr_candidates, n_inc, replace=False)
                        test_correct_list.append(chosen_corr)
                    if not ok or len(test_correct_list) == 0:
                        continue
                    test_correct_trials = np.concatenate(test_correct_list)
                    train_correct_trials = np.setdiff1d(correct_trials, test_correct_trials)
                else:
                    test_correct_trials = np.random.choice(correct_trials,
                                                           n_incorrect,
                                                           replace=False)
                    train_correct_trials = np.setdiff1d(correct_trials, test_correct_trials)

                if len(train_correct_trials) == 0:
                    continue

                X_train_all = reshaped_data[:, train_correct_trials, :].reshape(-1, neurons)
                y_train = np.repeat(np.arange(time_bins), len(train_correct_trials))

                X_test_cor_all = reshaped_data[:, test_correct_trials, :].reshape(-1, neurons)
                y_test_cor = np.repeat(np.arange(time_bins), len(test_correct_trials))

                X_test_inc_all = reshaped_data[:, incorrect_trials, :].reshape(-1, neurons)
                y_test_inc = np.repeat(np.arange(time_bins), len(incorrect_trials))

                clf_all = GaussianNB()
                clf_all.fit(X_train_all, y_train)

                err_cor_all = np.abs(clf_all.predict(X_test_cor_all) - y_test_cor) * binwidth
                err_inc_all = np.abs(clf_all.predict(X_test_inc_all) - y_test_inc) * binwidth

                G_all = err_inc_all.mean() - err_cor_all.mean()

                keep_mask = (region_labels != region_name)
                n_keep = np.sum(keep_mask)
                if n_keep < 1:
                    continue

                data_drop = reshaped_data[:, :, keep_mask]

                X_train_drop = data_drop[:, train_correct_trials, :].reshape(-1, n_keep)
                X_test_cor_drop = data_drop[:, test_correct_trials, :].reshape(-1, n_keep)
                X_test_inc_drop = data_drop[:, incorrect_trials, :].reshape(-1, n_keep)

                clf_drop = GaussianNB()
                clf_drop.fit(X_train_drop, y_train)

                err_cor_drop = np.abs(clf_drop.predict(X_test_cor_drop) - y_test_cor) * binwidth
                err_inc_drop = np.abs(clf_drop.predict(X_test_inc_drop) - y_test_inc) * binwidth

                G_drop = err_inc_drop.mean() - err_cor_drop.mean()

                region_gap_all[region_name][pid].append(G_all)
                region_gap_drop[region_name][pid].append(G_drop)

    regions_used = []
    reg_gap_pvals = []
    per_region_gap_all_vec  = {}
    per_region_gap_drop_vec = {}
    per_region_contrib_vec  = {}

    for reg in regions_to_use:
        pid_dict_all  = region_gap_all[reg]
        pid_dict_drop = region_gap_drop[reg]
        if len(pid_dict_all) == 0:
            continue

        pids = sorted(pid_dict_all.keys())
        gap_all_vec  = []
        gap_drop_vec = []
        contrib_vec  = []

        for pid in pids:
            gaps_all  = np.asarray(pid_dict_all[pid])
            gaps_drop = np.asarray(pid_dict_drop[pid])
            if gaps_all.size == 0 or gaps_drop.size == 0:
                continue

            gap_all_mean  = gaps_all.mean()
            gap_drop_mean = gaps_drop.mean()
            C_p = gap_all_mean - gap_drop_mean

            gap_all_vec.append(gap_all_mean)
            gap_drop_vec.append(gap_drop_mean)
            contrib_vec.append(C_p)

        if len(contrib_vec) == 0:
            continue

        gap_all_vec  = np.asarray(gap_all_vec)
        gap_drop_vec = np.asarray(gap_drop_vec)
        contrib_vec  = np.asarray(contrib_vec)

        _, p_C_two = ttest_1samp(contrib_vec, 0.0)
        _, p_gap   = ttest_rel(gap_all_vec, gap_drop_vec)

        regions_used.append(reg)
        reg_gap_pvals.append(p_gap)

        per_region_gap_all_vec[reg]  = gap_all_vec
        per_region_gap_drop_vec[reg] = gap_drop_vec
        per_region_contrib_vec[reg]  = contrib_vec

    if len(regions_used) == 0:
        print("No regions had valid patients.")
        return

    regions_used = np.array(regions_used)
    xtick_labels = [region_map.get(r, r) for r in regions_used]

    plt.figure(figsize=(max(6, 2 * len(regions_used)), 6))

    x_centers = []
    for i, reg in enumerate(regions_used):
        base = 3 * i
        x_full = base
        x_drop = base + 1
        x_centers.append(base + 0.5)

        gap_all_vec  = per_region_gap_all_vec[reg]
        gap_drop_vec = per_region_gap_drop_vec[reg]

        color_drop = region_colors.get(reg, "#000000")

        for ga, gd in zip(gap_all_vec, gap_drop_vec):
            plt.plot([x_full, x_drop], [ga, gd], color='lightgray', alpha=0.6)
            plt.scatter(x_full, ga, color='gray', alpha=0.9, zorder=10, s=20)
            plt.scatter(x_drop, gd, color=color_drop, alpha=0.9, zorder=10, s=20)

        plt.hlines(gap_all_vec.mean(),  x_full - 0.2, x_full + 0.2,
                   color='gray', linewidth=2.5)
        plt.hlines(gap_drop_vec.mean(), x_drop - 0.2, x_drop + 0.2,
                   color=color_drop, linewidth=2.5)

        p_gap = reg_gap_pvals[i]
        ymax_reg = max(gap_all_vec.max(), gap_drop_vec.max())
        star_y = ymax_reg + 0.02

        if p_gap < 1e-3:
            sig_gap = "***"
        elif p_gap < 1e-2:
            sig_gap = "**"
        elif p_gap < 0.05:
            sig_gap = "*"
        else:
            sig_gap = "ns"

        plt.text((x_full + x_drop) / 2.0, star_y,
                 sig_gap, ha='center', va='bottom', fontsize=20)

    plt.xticks(x_centers, xtick_labels, rotation=45, ha='right')
    plt.ylabel("Incorrect - Correct Mean Decoding Error (s)")
    plt.tight_layout()
    plt.savefig(DIR + 'Temporal_Decoding_Error_Gap.svg', format='svg')
    plt.show()

    plt.figure(figsize=(max(6, 2 * len(regions_used)), 6))
    x = np.arange(len(regions_used))

    for i, reg in enumerate(regions_used):
        contrib_vec = per_region_contrib_vec[reg]
        color = region_colors.get(reg, "#000000")
        jitter = (np.random.rand(len(contrib_vec)) - 0.5) * 0.2

        t_stat, p_two = ttest_1samp(contrib_vec, 0.0)
        p_one = p_two / 2.0 if t_stat > 0 else 1.0

        print(f"Region {reg}: mean C_p = {contrib_vec.mean():.3f}, "
              f"t = {t_stat:.2f}, p_two = {p_two:.3e}, p_one(C>0) = {p_one:.3e}")

        plt.scatter(x[i] + jitter, contrib_vec,
                    color=color, alpha=0.7, zorder=10, s=20)

        mean_C = contrib_vec.mean()
        sem_C  = sem(contrib_vec)
        plt.hlines(mean_C, x[i] - 0.2, x[i] + 0.2, color=color, linewidth=2.5)
        plt.errorbar(x[i], mean_C, yerr=sem_C, fmt='none',
                     ecolor=color, capsize=4)

    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.xticks(x, xtick_labels, rotation=45, ha='right')
    plt.ylabel("Reduction of Decoding Error Gap (s)")
    plt.title("Reduction of Decoding Error Gap after Regional Drop-out")
    plt.tight_layout()
    plt.savefig(DIR + 'Contribution_Temporal_Decoding_Error_Gap.svg', format='svg')
    plt.show()

analyze_region_dropout_train_test('TC.mat', 0.1, neuron_threshold=5)
