import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from scipy.stats import sem, ttest_rel, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

DIR = '/Users/xiangxuankong/Desktop/Human TC Support WM/Figures/Figure 1/'

###############################################################################
# 1) Define a helper function to load data, truncate trials, reshape, and decode
###############################################################################
def decode_dataset(mat_filepath, test_size=0.3, random_state=20250710):
    """
    Loads .mat with 'neural_data' (neuron x (trials x time_bins)),
    truncates to min. trial count, reshapes to (time_bins, trials, neurons),
    splits trials for train/test, trains GaussianNB, returns:
      - y_test, y_pred
      - decoding_errors (array of shape = #test samples)
      - time_centers (one value per time bin)
      - X_test (for subsequent shuffle)
      - bayes_clf (trained model)
    """
    data = scipy.io.loadmat(mat_filepath)
    neural_data_struct = data['neural_data']  # shape: (1, n_neurons) structured array

    # Extract firing_rate arrays
    firing_rates_list = [np.array(entry['firing_rates']) for entry in neural_data_struct[0]]
    # Truncate all neurons to min # trials
    min_trials = min(fr.shape[0] for fr in firing_rates_list)
    truncated_data = [fr[:min_trials, :] for fr in firing_rates_list]

    # Stack => shape (min_trials, time_bins, n_neurons), then transpose => (time_bins, trials, neurons)
# After reshaping
    reshaped_data = np.stack(truncated_data, axis=2).transpose(1,0,2)
    time_bins, trials, neurons = reshaped_data.shape

    # Compute the bin size based on total_duration
    total_duration = 2.5   # seconds
    bin_size = total_duration / time_bins

    print(f"Auto-computed bin size: {bin_size:.4f} sec")


    # Keep trial structure for train/test
    X_trials = reshaped_data
    # Label each time bin
    y_trials = np.tile(np.arange(time_bins), (trials,1)).T  # (time_bins, trials)

    # Trial-based split
    all_trials = np.arange(trials)
    train_trials, test_trials = train_test_split(all_trials, test_size=test_size,
                                                 random_state=random_state)

    X_train = X_trials[:, train_trials, :].reshape(-1, neurons)
    y_train = y_trials[:, train_trials].ravel()
    X_test  = X_trials[:, test_trials, :].reshape(-1, neurons)
    y_test  = y_trials[:, test_trials].ravel()

    # Train GaussianNB
    bayes_clf = GaussianNB()
    bayes_clf.fit(X_train, y_train)
    y_pred = bayes_clf.predict(X_test)

    # Decoding error (adapt to your scale as in your original code)
    # For example:  decoding_errors = 0.2 * 0.5 * np.abs(y_pred - y_test)
    decoding_errors = bin_size * np.abs(y_pred - y_test)
    # Example time-centers (adjust to match your bin spacing)

    unique_bins = np.unique(y_test)
    time_centers = unique_bins * bin_size + bin_size / 2  # centers of each bin


    return y_test, y_pred, decoding_errors, time_centers, X_test, bayes_clf

###############################################################################
# 2) Decode TIME-CELL dataset (as before) -> + Baseline shuffle 
###############################################################################
time_cells_mat = "3sig15_data.mat"  # Replace with your .mat filename
y_test_TC, y_pred_TC, errors_TC, tcenters_TC, X_test_TC, bayes_clf_TC = decode_dataset(
    time_cells_mat, test_size=0.3, random_state=20250710
)

# Original baseline: shuffle y_test or shuffle neuron indices
num_shuffles = 1000
unique_bins_TC = np.unique(y_test_TC)
all_bin_means_baseline_TC = []
baseline_shuffle_errors = []
for _ in range(num_shuffles):
    # Shuffle labels or neuron indices as your original code did
    y_test_shuffled = np.random.permutation(y_test_TC)
    y_pred_shuffled = bayes_clf_TC.predict(X_test_TC)
    dec_errors_shuff = 0.2 * 0.5 * np.abs(y_pred_shuffled - y_test_shuffled)

    baseline_shuffle_errors.append(np.mean(dec_errors_shuff))

    # Compute mean error per time bin
    bin_means = []
    for b in unique_bins_TC:
        bin_means.append(np.mean(dec_errors_shuff[y_test_TC == b]))
    all_bin_means_baseline_TC.append(bin_means)

baseline_shuffle_errors = np.array(baseline_shuffle_errors)  # shape: (#shuffles,)
all_bin_means_baseline_TC = np.array(all_bin_means_baseline_TC)
baseline_mean_TC = all_bin_means_baseline_TC.mean(axis=0)   # shape: (#bins,)
baseline_sem_TC  = sem(all_bin_means_baseline_TC, axis=0)

# Compute the average error + SEM for time cells
mean_errors_TC = []
sem_errors_TC = []
for b in unique_bins_TC:
    mean_errors_TC.append(np.mean(errors_TC[y_test_TC == b]))
    sem_errors_TC.append(sem(errors_TC[y_test_TC == b]))
mean_errors_TC = np.array(mean_errors_TC)
sem_errors_TC  = np.array(sem_errors_TC)

###############################################################################
# 3) Decode NON–TIME-CELL dataset -> + Baseline shuffle
###############################################################################
non_time_cells_mat = "3sig15_data_ntc.mat"  # Replace with your .mat filename
y_test_NTC, y_pred_NTC, errors_NTC, tcenters_NTC, X_test_NTC, bayes_clf_NTC = decode_dataset(
    non_time_cells_mat, test_size=0.3, random_state=20250710
)

num_shuffles = 1000
unique_bins_NTC = np.unique(y_test_NTC)
all_bin_means_baseline_NTC = []

for _ in range(num_shuffles):
    y_test_shuffled = np.random.permutation(y_test_NTC)
    y_pred_shuffled = bayes_clf_NTC.predict(X_test_NTC)
    dec_errors_shuff = 0.2 * 0.5 * np.abs(y_pred_shuffled - y_test_shuffled)

    bin_means = []
    for b in unique_bins_NTC:
        bin_means.append(np.mean(dec_errors_shuff[y_test_NTC == b]))
    all_bin_means_baseline_NTC.append(bin_means)

all_bin_means_baseline_NTC = np.array(all_bin_means_baseline_NTC)
baseline_mean_NTC = all_bin_means_baseline_NTC.mean(axis=0)
baseline_sem_NTC  = sem(all_bin_means_baseline_NTC, axis=0)

# Compute the average error + SEM for non–time-cells
mean_errors_NTC = []
sem_errors_NTC = []
for b in unique_bins_NTC:
    mean_errors_NTC.append(np.mean(errors_NTC[y_test_NTC == b]))
    sem_errors_NTC.append(sem(errors_NTC[y_test_NTC == b]))
mean_errors_NTC = np.array(mean_errors_NTC)
sem_errors_NTC  = np.array(sem_errors_NTC)

###############################################################################
# 4) Plot all three sets of decoding errors on one figure
#    (Time Cells, Non–Time Cells, and "Original Baseline" from time cells)
###############################################################################
plt.figure(figsize=(10,6.17))

# Time Cells decoding error
plt.plot(tcenters_TC, mean_errors_TC, '-o', color='blue', label="Time Cells")
plt.fill_between(tcenters_TC,
                 mean_errors_TC - sem_errors_TC,
                 mean_errors_TC + sem_errors_TC,
                 color='blue', alpha=0.2)

# Non–Time-Cells decoding error
plt.plot(tcenters_NTC, mean_errors_NTC, '-o', color='green', label="Non–Time-Cells")
plt.fill_between(tcenters_NTC,
                 mean_errors_NTC - sem_errors_NTC,
                 mean_errors_NTC + sem_errors_NTC,
                 color='green', alpha=0.2)

# Original baseline (from time cells) as a single curve
plt.plot(tcenters_TC, baseline_mean_TC, '-o', color='grey', label="Shuffled Baseline")
plt.fill_between(tcenters_TC,
                 baseline_mean_TC - baseline_sem_TC,
                 baseline_mean_TC + baseline_sem_TC,
                 color='grey', alpha=0.15)

plt.xlabel("Time (s)")
plt.ylabel("Decoding Error(s)")
plt.title("Decoding Error: Time Cells vs Non–Time-Cells vs Original Baseline")
plt.legend(loc='upper right', fontsize=12, frameon=False)

plt.xlim(0, 2.5)
plt.tight_layout()
# plt.savefig(DIR + 'Bayesian_Curve.svg', format = 'svg', bbox_inches="tight")
plt.show()



# 4) Build distributions for each group & compute bar heights + significance
###############################################################################
# Time Cells: we have one array of error values per test sample => shape (#test_samples,)
# Let's store the mean error, but also keep the distribution for stats
tc_distribution = errors_TC  # (#test_samples,) 
tc_mean = np.mean(tc_distribution)
tc_sem  = sem(tc_distribution)

# Non–Time-Cells
ntc_distribution = errors_NTC
ntc_mean = np.mean(ntc_distribution)
ntc_sem  = sem(ntc_distribution)

# Original Baseline: we already have a distribution (#shuffles,) of mean errors
baseline_distribution = baseline_shuffle_errors
baseline_mean = np.mean(baseline_distribution)
baseline_sem  = sem(baseline_distribution)

# Prepare data for bar chart
group_means = [tc_mean, ntc_mean, baseline_mean]
group_sems  = [tc_sem, ntc_sem, baseline_sem]
group_labels = ["Time Cells", "Non–Time-Cells", "Original Baseline"]

###############################################################################
# 5) Create the bar chart
###############################################################################
import matplotlib.pyplot as plt

plt.figure(figsize=(3.5, 8))
bars = plt.bar(group_labels, group_means, yerr=group_sems, 
               color=["blue", "green", "grey"], alpha=0.8, capsize=5)

plt.ylabel("Mean Decoding Error(s)", fontsize=16)
plt.title("Comparison of Decoding Error: Time Cells vs Non–Time-Cells vs Baseline", fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=12)

# Give a bit of headroom for significance lines
max_bar_val = max(group_means) + max(group_sems)
plt.ylim([0, max_bar_val + 0.12])  # slightly smaller top margin


###############################################################################
# 6) Pairwise significance tests & add thick horizontal lines, with new spacing
###############################################################################
###############################################################################
# ---- 6) Pairwise significance tests & console output with p-values  ----
###############################################################################
from itertools import combinations

def significance_from_p(p):
    """Convert p-value into a string: '*', '**', '***', or 'ns'."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

def add_sig_line(ax, x1, x2, y, p_val, color='k', lw=3):
    """Draw a horizontal significance bar plus text above it."""
    ax.plot([x1 + 0.1, x2 - 0.1], [y, y], color=color, linewidth=lw)
    ax.text((x1 + x2) * 0.5, y, significance_from_p(p_val),
            ha='center', va='bottom', color=color, fontsize=12)

# Define the *order* in which you want to show / draw the pairs
pairs          = [(0, 1), (1, 2), (0, 2)]        # (TC–NTC, NTC–BL, TC–BL)
pair_labels    = [(group_labels[i], group_labels[j]) for i, j in pairs]
all_distribs   = [tc_distribution, ntc_distribution, baseline_distribution] 

ax = plt.gca()
max_y = max(group_means) + max(group_sems)        # top of the tallest bar

print("\nPairwise Welch’s t-tests (unequal variances)\n" + "-"*42)
for k, ((g1, g2), label) in enumerate(zip(pairs, pair_labels), start=1):
    d1, d2 = all_distribs[g1], all_distribs[g2]
    t_stat, p_val = ttest_rel(d1, d2)

    # ------ console output ------
    print(f"{k:>2}. {label[0]} vs {label[1]:<16} "
          f"t = {t_stat:8.3f},  p = {p_val:9.3e}  "
          f"[{significance_from_p(p_val)}]")

    # ------ figure annotation ------
    bar_height = max(group_means[g1], group_means[g2])
    line_y     = bar_height + 0.03 + 0.03 * (k-1)         # stagger the bars
    add_sig_line(ax, g1, g2, line_y, p_val)

plt.tight_layout()
# plt.savefig(DIR + 'Bayesian_Bar.svg', format = 'svg', bbox_inches="tight")
plt.show()
