import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
from scipy.stats import ttest_1samp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import significance_stars as star_label, clean_region_name

DIR = ''


def repeated_cv_pseudo_population_per_class_by_region(
    mat_file_path,
    patient_ids=range(1, 22),
    m=0,
    num_windows=0,
    test_per_load=7,
    n_iterations=1000,
    random_state=20250710,
    only_correct=False
):
    """Regional load decoder with actual vs null comparison and confusion matrices."""

    rng = np.random.default_rng(random_state)

    region_map = {
        "dorsal_anterior_cingulate_cortex": "DaCC",
        "pre_supplementary_motor_area": "PSMA",
        "hippocampus": "HPC",
        "amygdala": "AMY",
        "ventral_medial_prefrontal_cortex": "vmPFC",
    }

    region_colors = {
        "hippocampus": "#FFD700",
        "amygdala": "#00FFFF",
        "pre_supplementary_motor_area": "#FF0000",
        "dorsal_anterior_cingulate_cortex": "#0000FF",
        "ventral_medial_prefrontal_cortex": "#008000",
    }

    inverse_region_map = {v: k for k, v in region_map.items()}

    mat_data = scipy.io.loadmat(mat_file_path)
    neural_data = mat_data['neural_data']

    region_patient_neurons = {}
    region_patient_load_counts = {}

    for entry in neural_data[0]:
        pid = int(entry['patient_id'][0][0])
        if pid not in patient_ids:
            continue

        frates = entry['firing_rates']
        loads = entry['trial_load'].flatten().astype(int)
        correct = entry['trial_correctness'].flatten().astype(int)
        tfield = int(entry['time_field'][0][0]) - 1

        region_key = clean_region_name(entry['brain_region'])

        if region_key not in region_map:
            continue

        if only_correct:
            keep_idx = (correct == 1)
            frates = frates[keep_idx, :]
            loads = loads[keep_idx]

        if frates.size == 0:
            continue

        if region_key not in region_patient_neurons:
            region_patient_neurons[region_key] = {}
            region_patient_load_counts[region_key] = {}

        if pid not in region_patient_neurons[region_key]:
            region_patient_neurons[region_key][pid] = []
            region_patient_load_counts[region_key][pid] = {1: 0, 2: 0, 3: 0}

        region_patient_neurons[region_key][pid].append((frates, loads, tfield))

        for ld in (1, 2, 3):
            region_patient_load_counts[region_key][pid][ld] = max(
                region_patient_load_counts[region_key][pid][ld],
                np.sum(loads == ld)
            )

    if len(region_patient_neurons) == 0:
        print("No neurons found for any region with the given filters.")
        return None, None

    def decode_one_region(region_key):
        """Build pseudo-population for one region and run decoding iterations."""
        valid_patients = []
        for pid in patient_ids:
            if pid not in region_patient_neurons.get(region_key, {}):
                continue
            if len(region_patient_neurons[region_key][pid]) < m:
                continue
            if only_correct:
                counts = region_patient_load_counts[region_key][pid]
                if not (counts[1] >= 25 and counts[2] >= 25 and counts[3] >= 25):
                    continue
            valid_patients.append(pid)

        if len(valid_patients) == 0:
            print(f"[{region_map[region_key]}] No patients meet inclusion criteria "
                  f"(m={m}, ≥25 correct trials per load={only_correct}). Skipping.")
            return None, None

        all_neurons = []
        for pid in valid_patients:
            for (frates, loads, tfield) in region_patient_neurons[region_key][pid]:
                start_idx = max(0, tfield - num_windows)
                end_idx = min(frates.shape[1], tfield + num_windows + 1)

                windowed = frates[:, start_idx:end_idx]
                mean_rates = np.mean(windowed, axis=1)

                load1_rates = mean_rates[loads == 1]
                load2_rates = mean_rates[loads == 2]
                load3_rates = mean_rates[loads == 3]

                if len(load1_rates) > 0 and len(load2_rates) > 0 and len(load3_rates) > 0:
                    all_neurons.append({
                        'load1': load1_rates,
                        'load2': load2_rates,
                        'load3': load3_rates
                    })

        if len(all_neurons) == 0:
            print(f"[{region_map[region_key]}] No neurons with all loads. Skipping.")
            return None, None

        min_load1 = min(len(n['load1']) for n in all_neurons)
        min_load2 = min(len(n['load2']) for n in all_neurons)
        min_load3 = min(len(n['load3']) for n in all_neurons)

        if (min_load1 == 0) or (min_load2 == 0) or (min_load3 == 0):
            print(f"[{region_map[region_key]}] Some load has zero global min. Skipping.")
            return None, None

        for neuron in all_neurons:
            neuron['load1'] = resample(neuron['load1'], replace=False,
                                       n_samples=min_load1, random_state=random_state)
            neuron['load2'] = resample(neuron['load2'], replace=False,
                                       n_samples=min_load2, random_state=random_state)
            neuron['load3'] = resample(neuron['load3'], replace=False,
                                       n_samples=min_load3, random_state=random_state)

        num_neurons = len(all_neurons)
        load1_matrix = np.zeros((min_load1, num_neurons), dtype=np.float32)
        load2_matrix = np.zeros((min_load2, num_neurons), dtype=np.float32)
        load3_matrix = np.zeros((min_load3, num_neurons), dtype=np.float32)

        for j, neuron in enumerate(all_neurons):
            load1_matrix[:, j] = neuron['load1']
            load2_matrix[:, j] = neuron['load2']
            load3_matrix[:, j] = neuron['load3']

        print(f"--- [{region_map[region_key]}] Pseudopopulation Info (only_correct={only_correct}) ---")
        print(f"  Number of neurons: {num_neurons}")
        print(f"  Final trials for Load=1: {min_load1}")
        print(f"  Final trials for Load=2: {min_load2}")
        print(f"  Final trials for Load=3: {min_load3}")

        if (test_per_load > min_load1 or
            test_per_load > min_load2 or
            test_per_load > min_load3):
            print(f"[{region_map[region_key]}] Requested test_per_load={test_per_load}, "
                  f"but min load trials are ({min_load1},{min_load2},{min_load3}). Skipping.")
            return None, None

        accs_load1_actual = []
        accs_load2_actual = []
        accs_load3_actual = []
        cm_accumulator = np.zeros((3, 3), dtype=float)

        for _ in range(n_iterations):
            idx1 = rng.permutation(min_load1)
            idx2 = rng.permutation(min_load2)
            idx3 = rng.permutation(min_load3)

            test_idx1 = idx1[:test_per_load]
            train_idx1 = idx1[test_per_load:]
            test_idx2 = idx2[:test_per_load]
            train_idx2 = idx2[test_per_load:]
            test_idx3 = idx3[:test_per_load]
            train_idx3 = idx3[test_per_load:]

            X_train = np.vstack([
                load1_matrix[train_idx1, :],
                load2_matrix[train_idx2, :],
                load3_matrix[train_idx3, :]
            ])
            y_train = np.array(
                [1]*len(train_idx1) +
                [2]*len(train_idx2) +
                [3]*len(train_idx3)
            )

            X_test = np.vstack([
                load1_matrix[test_idx1, :],
                load2_matrix[test_idx2, :],
                load3_matrix[test_idx3, :]
            ])
            y_test = np.array(
                [1]*len(test_idx1) +
                [2]*len(test_idx2) +
                [3]*len(test_idx3)
            )

            clf = SVC(kernel='linear', random_state=random_state)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Per-class accuracy
            mask1 = (y_test == 1)
            mask2 = (y_test == 2)
            mask3 = (y_test == 3)

            accs_load1_actual.append(np.mean(y_pred[mask1] == 1) * 100)
            accs_load2_actual.append(np.mean(y_pred[mask2] == 2) * 100)
            accs_load3_actual.append(np.mean(y_pred[mask3] == 3) * 100)

            cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3])
            cm_accumulator += cm

        if region_key != "ventral_medial_prefrontal_cortex":
            row_sums = cm_accumulator.sum(axis=1, keepdims=True)
            cm_percentage = (cm_accumulator / row_sums) * 100
            labels_cm = np.array([[f"{val:.1f}%" for val in row] for row in cm_percentage])

            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm_percentage,
                annot=labels_cm,
                fmt="",
                cmap="Blues",
                xticklabels=["Load 1", "Load 2", "Load 3"],
                yticklabels=["Load 1", "Load 2", "Load 3"],
                vmin=0,
                vmax=65,
                cbar_kws={'label': 'Percentage (%)'}
            )
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(f"Averaged Confusion Matrix (%) - {region_map[region_key]}")

            path = DIR + 'Correct/' if only_correct else DIR
            plt.savefig(path + f'confusionmatrix_{region_map[region_key]}.svg', format='svg')
            plt.show()

        accs_load1_null = []
        accs_load2_null = []
        accs_load3_null = []

        for _ in range(n_iterations):
            idx1 = rng.permutation(min_load1)
            idx2 = rng.permutation(min_load2)
            idx3 = rng.permutation(min_load3)

            test_idx1 = idx1[:test_per_load]
            train_idx1 = idx1[test_per_load:]
            test_idx2 = idx2[:test_per_load]
            train_idx2 = idx2[test_per_load:]
            test_idx3 = idx3[:test_per_load]
            train_idx3 = idx3[test_per_load:]

            X_train = np.vstack([
                load1_matrix[train_idx1, :],
                load2_matrix[train_idx2, :],
                load3_matrix[train_idx3, :]
            ])
            y_train_real = np.array(
                [1]*len(train_idx1) +
                [2]*len(train_idx2) +
                [3]*len(train_idx3)
            )

            X_test = np.vstack([
                load1_matrix[test_idx1, :],
                load2_matrix[test_idx2, :],
                load3_matrix[test_idx3, :]
            ])
            y_test_real = np.array(
                [1]*len(test_idx1) +
                [2]*len(test_idx2) +
                [3]*len(test_idx3)
            )

            y_train_shuffled = rng.permutation(y_train_real)

            clf = SVC(kernel='linear', random_state=random_state)
            clf.fit(X_train, y_train_shuffled)
            y_pred = clf.predict(X_test)

            mask1 = (y_test_real == 1)
            mask2 = (y_test_real == 2)
            mask3 = (y_test_real == 3)

            accs_load1_null.append(np.mean(y_pred[mask1] == 1) * 100)
            accs_load2_null.append(np.mean(y_pred[mask2] == 2) * 100)
            accs_load3_null.append(np.mean(y_pred[mask3] == 3) * 100)

        overall_actual = [
            (l1 + l2 + l3) / 3.0
            for l1, l2, l3 in zip(accs_load1_actual,
                                  accs_load2_actual,
                                  accs_load3_actual)
        ]
        overall_null = [
            (n1 + n2 + n3) / 3.0
            for n1, n2, n3 in zip(accs_load1_null,
                                  accs_load2_null,
                                  accs_load3_null)
        ]

        df_actual_region = pd.DataFrame({
            'Load': (['Load1']*n_iterations +
                     ['Load2']*n_iterations +
                     ['Load3']*n_iterations +
                     ['Overall']*n_iterations),
            'Accuracy': (accs_load1_actual +
                         accs_load2_actual +
                         accs_load3_actual +
                         overall_actual)
        })

        df_null_region = pd.DataFrame({
            'Load': (['Load1']*n_iterations +
                     ['Load2']*n_iterations +
                     ['Load3']*n_iterations +
                     ['Overall']*n_iterations),
            'Accuracy': (accs_load1_null +
                         accs_load2_null +
                         accs_load3_null +
                         overall_null)
        })

        def print_summary(tag, x):
            print(f"  {tag}: mean={np.mean(x):.2f}%, std={np.std(x):.2f}%")

        print(f"---- [{region_map[region_key]}] Actual distribution summary ----")
        for load_name in ['Load1', 'Load2', 'Load3', 'Overall']:
            vals = df_actual_region[df_actual_region['Load'] == load_name]['Accuracy']
            print_summary(load_name, vals)

        print(f"---- [{region_map[region_key]}] Null distribution summary ----")
        for load_name in ['Load1', 'Load2', 'Load3', 'Overall']:
            vals = df_null_region[df_null_region['Load'] == load_name]['Accuracy']
            print_summary(load_name + "_null", vals)

        print(f"---- [{region_map[region_key]}] Empirical p-values (Actual > Null) ----")
        for load_name in ['Load1', 'Load2', 'Load3', 'Overall']:
            act = df_actual_region[df_actual_region['Load'] == load_name]['Accuracy'].values
            nul = df_null_region[df_null_region['Load'] == load_name]['Accuracy'].values
            observed_mean = np.mean(act)
            p_val = np.mean(nul >= observed_mean)
            print(f"  {load_name}: p = {p_val:.3e} ({star_label(p_val)})")

        return df_actual_region, df_null_region

    all_actual_list = []
    all_null_list = []

    for region_key in sorted(region_patient_neurons.keys()):
        df_act, df_nul = decode_one_region(region_key)
        if df_act is None:
            continue

        region_label = region_map[region_key]

        df_act = df_act.copy()
        df_act['Region'] = region_label
        all_actual_list.append(df_act)

        df_nul = df_nul.copy()
        df_nul['Region'] = region_label
        all_null_list.append(df_nul)

    if len(all_actual_list) == 0:
        print("No regions produced valid decoding results.")
        return None, None

    df_actual_all = pd.concat(all_actual_list, ignore_index=True)
    df_null_all = pd.concat(all_null_list, ignore_index=True)

    load_order = ['Load1', 'Load2', 'Load3']
    region_order_all = sorted(df_actual_all['Region'].unique())
    region_order = [r for r in region_order_all if r != 'vmPFC']

    df_plot = df_actual_all[
        df_actual_all['Load'].isin(load_order) &
        df_actual_all['Region'].isin(region_order)
    ]

    plt.figure(figsize=(max(6, 1.6 * len(region_order)), 6))
    ax = sns.boxplot(
        data=df_plot,
        x='Region',
        y='Accuracy',
        hue='Load',
        order=region_order,
        hue_order=load_order,
        palette={'Load1': 'blue', 'Load2': 'green', 'Load3': 'red'},
        showfliers=False
    )

    plt.axhline(y=33.3, color='gray', linestyle='--')
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.xlabel("Brain Region", fontsize=14)
    plt.title("Decoder accuracy by region and load", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Load", fontsize=10)

    offset = 1.0
    n_load = len(load_order)
    group_width = 0.6

    for i, reg in enumerate(region_order):
        center = i

        for j, load in enumerate(load_order):
            x_center = center - group_width/2 + group_width*(j + 0.5)/n_load

            act_vals = df_actual_all[
                (df_actual_all['Region'] == reg) &
                (df_actual_all['Load'] == load)
            ]['Accuracy'].values

            nul_vals = df_null_all[
                (df_null_all['Region'] == reg) &
                (df_null_all['Load'] == load)
            ]['Accuracy'].values

            if len(act_vals) == 0 or len(nul_vals) == 0:
                continue

            observed_mean = np.mean(act_vals)
            p_val = np.mean(nul_vals >= observed_mean)
            label = star_label(p_val)

            y_top = act_vals.max()
            star_y = y_top + offset

            y_min, y_max = ax.get_ylim()
            if star_y > y_max:
                star_y = y_max - 0.5

            ax.text(x_center, star_y, label,
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold', color='black')

    path = DIR + 'Correct/' if only_correct else DIR
    plt.savefig(path + 'regional_accuracy_by_load.svg', format='svg')
    plt.tight_layout()
    plt.show()

    df_overall_actual = df_actual_all[
        (df_actual_all['Load'] == 'Overall') &
        (df_actual_all['Region'] != 'vmPFC')
    ].copy()
    df_overall_null = df_null_all[
        (df_null_all['Load'] == 'Overall') &
        (df_null_all['Region'] != 'vmPFC')
    ].copy()

    region_order_overall = sorted(df_overall_actual['Region'].unique())

    region_palette = {}
    for reg in region_order_overall:
        full_name = inverse_region_map.get(reg, None)
        if full_name is not None and full_name in region_colors:
            region_palette[reg] = region_colors[full_name]

    plt.figure(figsize=(max(6, 1.6 * len(region_order_overall)), 6))
    ax2 = sns.boxplot(
        data=df_overall_actual,
        x='Region',
        y='Accuracy',
        order=region_order_overall,
        palette=region_palette,
        showfliers=False
    )

    plt.axhline(y=33.3, color='gray', linestyle='--')
    plt.ylabel("Overall accuracy (%)", fontsize=14)
    plt.xlabel("Brain Region", fontsize=14)
    plt.title("Overall decoder accuracy by region", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    offset = 1.0
    for i, reg in enumerate(region_order_overall):
        x_center = i

        act_vals = df_overall_actual[df_overall_actual['Region'] == reg]['Accuracy'].values
        nul_vals = df_overall_null[df_overall_null['Region'] == reg]['Accuracy'].values
        if len(act_vals) == 0 or len(nul_vals) == 0:
            continue

        observed_mean = np.mean(act_vals)
        p_val = np.mean(nul_vals >= observed_mean)
        label = star_label(p_val)

        y_top = act_vals.max()
        star_y = y_top + offset

        y_min, y_max = ax2.get_ylim()
        if star_y > y_max:
            star_y = y_max - 0.5

        ax2.text(x_center, star_y, label,
                 ha='center', va='bottom',
                 fontsize=12, fontweight='bold', color='black')

    plt.savefig(path + 'regional_overall_accuracy.svg', format='svg')
    plt.tight_layout()
    plt.show()

    print(
        df_actual_all
        .groupby(['Region', 'Load'])['Accuracy']
        .quantile([0.25, 0.5, 0.75])
        .unstack()
    )

    return df_actual_all, df_null_all


repeated_cv_pseudo_population_per_class_by_region(
    'TC_raw.mat',
    test_per_load=11,
    only_correct=False,
    random_state=20250710,
    n_iterations=1000
)