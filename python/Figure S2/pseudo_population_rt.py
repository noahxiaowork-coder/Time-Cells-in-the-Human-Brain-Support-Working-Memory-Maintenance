from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.io
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Iterable, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import significance_stars

DIR = ''


def remove_extreme_outliers(rt_values):
    q1 = np.percentile(rt_values, 25)
    q3 = np.percentile(rt_values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return rt_values[(rt_values >= lower_bound) & (rt_values <= upper_bound)]


def Decode_reaction_Time(
    mat_file_path: str,
    patient_ids: Iterable[int] = range(1, 22),
    m: int = 0,
    num_windows: int = 0,
    test_per_class: int = 11,
    n_iterations: int = 1000,
    random_state: Optional[int] = 42,
    plot: bool = True,
):
    rng = np.random.default_rng(random_state)

    quantile = 0.3
    only_correct = True

    mat_data = scipy.io.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
    neural_data = mat_data["neural_data"].flatten()

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
            frates = frates[keep, :]
            rt = rt[keep]

        nan_mask = ~np.isnan(rt)
        frates = frates[nan_mask, :]
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

    valid_patients = [
        pid for pid in patient_ids
        if (pid in patient_dict and len(patient_dict[pid]) >= m)
    ]
    if not valid_patients:
        print("No patients meet the inclusion criteria.")
        return None

    all_neurons = []
    global_fast_rts = []
    global_slow_rts = []

    for pid in valid_patients:
        rt_vec = patient_rt_cache[pid]

        q_low = np.quantile(rt_vec, quantile)
        q_high = np.quantile(rt_vec, 1.0 - quantile)
        fast_idx = rt_vec <= q_low
        slow_idx = rt_vec >= q_high

        if fast_idx.sum() < 2 or slow_idx.sum() < 2:
            continue

        global_fast_rts.append(rt_vec[fast_idx])
        global_slow_rts.append(rt_vec[slow_idx])

        for frates, _, tfield in patient_dict[pid]:
            start_idx = max(0, tfield - num_windows)
            end_idx = min(frates.shape[1], tfield + num_windows + 1)
            mean_rates = frates[:, start_idx:end_idx].mean(axis=1)

            fast_rates = mean_rates[fast_idx]
            slow_rates = mean_rates[slow_idx]

            if fast_rates.size and slow_rates.size:
                all_neurons.append({"fast": fast_rates, "slow": slow_rates})

    if not all_neurons:
        print("No neurons found with both fast and slow trials present.")
        return None

    fast_rt_concat = np.concatenate(global_fast_rts) if global_fast_rts else np.array([])
    slow_rt_concat = np.concatenate(global_slow_rts) if global_slow_rts else np.array([])

    if fast_rt_concat.size == 0 or slow_rt_concat.size == 0:
        print("Fast or slow RT arrays are empty.")
        return None

    fast_rt_filtered = remove_extreme_outliers(fast_rt_concat)
    slow_rt_filtered = remove_extreme_outliers(slow_rt_concat)

    if plot:
        rt_df = pd.DataFrame({
            "RT": np.concatenate([fast_rt_filtered, slow_rt_filtered]),
            "Class": ["Fast"] * len(fast_rt_filtered) + ["Slow"] * len(slow_rt_filtered),
        })

        plt.figure(figsize=(3.5, 6))
        sns.boxplot(
            data=rt_df,
            x="Class",
            y="RT",
            palette=["skyblue", "salmon"],
            width=0.3,
            showfliers=True,
            flierprops=dict(marker="o", markersize=4),
        )
        plt.ylabel("Reaction time (s)")
        plt.title("RT distribution (fast vs. slow quantiles)")
        plt.tight_layout()
        plt.savefig(DIR + 'Reaction_time.svg', format='svg')
        plt.show()

    min_fast = min(len(n["fast"]) for n in all_neurons)
    min_slow = min(len(n["slow"]) for n in all_neurons)

    if test_per_class > min_fast or test_per_class > min_slow:
        print(
            f"Requested test_per_class={test_per_class}, "
            f"but mins are fast={min_fast}, slow={min_slow}."
        )
        return None

    for neuron in all_neurons:
        neuron["fast"] = rng.choice(neuron["fast"], size=min_fast, replace=False)
        neuron["slow"] = rng.choice(neuron["slow"], size=min_slow, replace=False)

    n_neurons = len(all_neurons)
    fast_mat = np.stack([n["fast"] for n in all_neurons], axis=1)
    slow_mat = np.stack([n["slow"] for n in all_neurons], axis=1)

    acc_fast_actual = []
    acc_slow_actual = []
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
        acc_overall_actual.append((y_pred == y_test).mean() * 100)

    cm_pct = cm_accumulator / cm_accumulator.sum(axis=1, keepdims=True) * 100

    acc_fast_null = []
    acc_slow_null = []
    acc_overall_null = []

    for _ in range(n_iterations):
        idx_fast = rng.permutation(min_fast)
        idx_slow = rng.permutation(min_slow)

        test_fast, train_fast = idx_fast[:test_per_class], idx_fast[test_per_class:]
        test_slow, train_slow = idx_slow[:test_per_class], idx_slow[test_per_class:]

        X_train = np.vstack([fast_mat[train_fast], slow_mat[train_slow]])
        y_train = rng.permutation(
            np.array([0] * len(train_fast) + [1] * len(train_slow))
        )
        X_test = np.vstack([fast_mat[test_fast], slow_mat[test_slow]])
        y_test = np.array([0] * len(test_fast) + [1] * len(test_slow))

        clf = SVC(kernel="linear", random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc_fast_null.append((y_pred[y_test == 0] == 0).mean() * 100)
        acc_slow_null.append((y_pred[y_test == 1] == 1).mean() * 100)
        acc_overall_null.append((y_pred == y_test).mean() * 100)

    df_actual = pd.DataFrame({"Accuracy": acc_overall_actual})
    df_null = pd.DataFrame({"Accuracy": acc_overall_null})

    if plot:
        plt.figure(figsize=(4.3, 4))
        sns.heatmap(
            cm_pct,
            annot=np.array([[f"{v:.1f}%" for v in row] for row in cm_pct]),
            fmt="",
            cmap="Blues",
            vmin=0,
            vmax=100,
            xticklabels=["Fast", "Slow"],
            yticklabels=["Fast", "Slow"],
            cbar_kws={"label": "Percentage (%)"},
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Averaged confusion matrix (fast vs. slow)")
        plt.tight_layout()
        plt.savefig(DIR + 'confusion.svg', format='svg')
        plt.show()

        plt.figure(figsize=(3.5, 6))
        sns.boxplot(
            data=df_actual,
            y="Accuracy",
            width=0.6,
            color="skyblue",
            showfliers=True,
            flierprops=dict(marker="o", markersize=4),
        )
        plt.axhline(50, ls="--", c="gray")
        plt.ylim(0, df_actual.Accuracy.max() + 5)
        plt.title("Decoder accuracy (overall)")

        p_perm = np.mean(np.array(acc_overall_null) >= np.mean(acc_overall_actual))
        star = significance_stars(p_perm)
        plt.text(
            0,
            max(acc_overall_actual) + 1.0,
            star,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.savefig(DIR + 'decoding_acc.svg', format='svg')
        plt.show()

    mean_act = np.mean(acc_overall_actual)
    std_act = np.std(acc_overall_actual)
    p_perm = np.mean(np.array(acc_overall_null) >= mean_act)
    print(f"Overall: {mean_act:.2f} ± {std_act:.2f}%    perm p = {p_perm:.3e}")

    return df_actual, df_null


Decode_reaction_Time('TC_raw.mat', random_state=42)
