import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from scipy.stats import sem, ttest_rel
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import significance_stars

DIR = ''


def decode_dataset(mat_filepath, test_size=0.3, random_state=20250710):
    """Decode time bins from neural data using Gaussian Naive Bayes."""
    data = scipy.io.loadmat(mat_filepath)
    neural_data_struct = data['neural_data']

    firing_rates_list = [
        np.array(entry['firing_rates']) for entry in neural_data_struct[0]
    ]

    min_trials = min(fr.shape[0] for fr in firing_rates_list)
    truncated_data = [fr[:min_trials, :] for fr in firing_rates_list]

    reshaped_data = np.stack(truncated_data, axis=2).transpose(1, 0, 2)
    time_bins, trials, neurons = reshaped_data.shape

    total_duration = 2.5
    bin_size = total_duration / time_bins
    print(f"Auto-computed bin size: {bin_size:.4f} sec")

    X_trials = reshaped_data
    y_trials = np.tile(np.arange(time_bins), (trials, 1)).T

    all_trials = np.arange(trials)
    train_trials, test_trials = train_test_split(
        all_trials, test_size=test_size, random_state=random_state
    )

    X_train = X_trials[:, train_trials, :].reshape(-1, neurons)
    y_train = y_trials[:, train_trials].ravel()
    X_test  = X_trials[:, test_trials, :].reshape(-1, neurons)
    y_test  = y_trials[:, test_trials].ravel()

    bayes_clf = GaussianNB()
    bayes_clf.fit(X_train, y_train)
    y_pred = bayes_clf.predict(X_test)

    decoding_errors = bin_size * np.abs(y_pred - y_test)

    unique_bins = np.unique(y_test)
    time_centers = unique_bins * bin_size + bin_size / 2

    return y_test, y_pred, decoding_errors, time_centers, X_test, bayes_clf


time_cells_mat = "TC.mat"

y_test_TC, y_pred_TC, errors_TC, tcenters_TC, X_test_TC, bayes_clf_TC = decode_dataset(
    time_cells_mat, test_size=0.3, random_state=20250710
)

num_shuffles = 1000
unique_bins_TC = np.unique(y_test_TC)

all_bin_means_baseline_TC = []
baseline_shuffle_errors = []

for _ in range(num_shuffles):
    y_test_shuffled = np.random.permutation(y_test_TC)
    y_pred_shuffled = bayes_clf_TC.predict(X_test_TC)
    dec_errors_shuff = 0.2 * 0.5 * np.abs(y_pred_shuffled - y_test_shuffled)

    baseline_shuffle_errors.append(np.mean(dec_errors_shuff))

    bin_means = [
        np.mean(dec_errors_shuff[y_test_TC == b]) for b in unique_bins_TC
    ]
    all_bin_means_baseline_TC.append(bin_means)

baseline_shuffle_errors = np.array(baseline_shuffle_errors)
all_bin_means_baseline_TC = np.array(all_bin_means_baseline_TC)

baseline_mean_TC = all_bin_means_baseline_TC.mean(axis=0)
baseline_sem_TC  = sem(all_bin_means_baseline_TC, axis=0)

mean_errors_TC = []
sem_errors_TC  = []

for b in unique_bins_TC:
    mean_errors_TC.append(np.mean(errors_TC[y_test_TC == b]))
    sem_errors_TC.append(sem(errors_TC[y_test_TC == b]))

mean_errors_TC = np.array(mean_errors_TC)
sem_errors_TC  = np.array(sem_errors_TC)


non_time_cells_mat = "non-TC.mat"

y_test_NTC, y_pred_NTC, errors_NTC, tcenters_NTC, X_test_NTC, bayes_clf_NTC = decode_dataset(
    non_time_cells_mat, test_size=0.3, random_state=20250710
)

unique_bins_NTC = np.unique(y_test_NTC)
all_bin_means_baseline_NTC = []

for _ in range(num_shuffles):
    y_test_shuffled = np.random.permutation(y_test_NTC)
    y_pred_shuffled = bayes_clf_NTC.predict(X_test_NTC)
    dec_errors_shuff = 0.2 * 0.5 * np.abs(y_pred_shuffled - y_test_shuffled)

    bin_means = [
        np.mean(dec_errors_shuff[y_test_NTC == b]) for b in unique_bins_NTC
    ]
    all_bin_means_baseline_NTC.append(bin_means)

all_bin_means_baseline_NTC = np.array(all_bin_means_baseline_NTC)
baseline_mean_NTC = all_bin_means_baseline_NTC.mean(axis=0)
baseline_sem_NTC  = sem(all_bin_means_baseline_NTC, axis=0)

mean_errors_NTC = []
sem_errors_NTC  = []

for b in unique_bins_NTC:
    mean_errors_NTC.append(np.mean(errors_NTC[y_test_NTC == b]))
    sem_errors_NTC.append(sem(errors_NTC[y_test_NTC == b]))

mean_errors_NTC = np.array(mean_errors_NTC)
sem_errors_NTC  = np.array(sem_errors_NTC)


plt.figure(figsize=(10, 6.17))

plt.plot(tcenters_TC, mean_errors_TC, '-o', color='blue', label="Time Cells")
plt.fill_between(tcenters_TC,
                 mean_errors_TC - sem_errors_TC,
                 mean_errors_TC + sem_errors_TC,
                 color='blue', alpha=0.2)

plt.plot(tcenters_NTC, mean_errors_NTC, '-o', color='green', label="Non–Time-Cells")
plt.fill_between(tcenters_NTC,
                 mean_errors_NTC - sem_errors_NTC,
                 mean_errors_NTC + sem_errors_NTC,
                 color='green', alpha=0.2)

plt.plot(tcenters_TC, baseline_mean_TC, '-o', color='grey', label="Shuffled Baseline")
plt.fill_between(tcenters_TC,
                 baseline_mean_TC - baseline_sem_TC,
                 baseline_mean_TC + baseline_sem_TC,
                 color='grey', alpha=0.15)

plt.xlabel("Time (s)")
plt.ylabel("Decoding Error (s)")
plt.title("Decoding Error: Time Cells vs Non–Time-Cells vs Baseline")
plt.legend(loc='upper right', fontsize=12, frameon=False)
plt.xlim(0, 2.5)
plt.tight_layout()
plt.show()


tc_distribution = errors_TC
ntc_distribution = errors_NTC
baseline_distribution = baseline_shuffle_errors

tc_mean, tc_sem = np.mean(tc_distribution), sem(tc_distribution)
ntc_mean, ntc_sem = np.mean(ntc_distribution), sem(ntc_distribution)
baseline_mean, baseline_sem = np.mean(baseline_distribution), sem(baseline_distribution)

group_means  = [tc_mean, ntc_mean, baseline_mean]
group_sems   = [tc_sem, ntc_sem, baseline_sem]
group_labels = ["Time Cells", "Non–Time-Cells", "Original Baseline"]


plt.figure(figsize=(3.5, 8))

plt.bar(group_labels, group_means, yerr=group_sems,
        color=["blue", "green", "grey"], alpha=0.8, capsize=5)

plt.ylabel("Mean Decoding Error (s)", fontsize=16)
plt.title("Comparison of Decoding Error", fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=12)

max_bar_val = max(group_means) + max(group_sems)
plt.ylim([0, max_bar_val + 0.12])


def add_sig_line(ax, x1, x2, y, p_val, color='k', lw=3):
    ax.plot([x1 + 0.1, x2 - 0.1], [y, y], color=color, linewidth=lw)
    ax.text((x1 + x2) * 0.5, y, significance_stars(p_val),
            ha='center', va='bottom', fontsize=12)

pairs        = [(0, 1), (1, 2), (0, 2)]
pair_labels  = [(group_labels[i], group_labels[j]) for i, j in pairs]
all_distribs = [tc_distribution, ntc_distribution, baseline_distribution]

ax = plt.gca()

print("\nPairwise paired t-tests\n" + "-" * 32)
for k, ((i, j), label) in enumerate(zip(pairs, pair_labels), start=1):
    d1, d2 = all_distribs[i], all_distribs[j]
    t_stat, p_val = ttest_rel(d1, d2)

    print(f"{k:>2}. {label[0]} vs {label[1]:<16} "
          f"t = {t_stat:8.3f}, p = {p_val:9.3e} "
          f"[{significance_stars(p_val)}]")

    bar_height = max(group_means[i], group_means[j])
    line_y = bar_height + 0.03 + 0.03 * (k - 1)
    add_sig_line(ax, i, j, line_y, p_val)

plt.tight_layout()
plt.show()
