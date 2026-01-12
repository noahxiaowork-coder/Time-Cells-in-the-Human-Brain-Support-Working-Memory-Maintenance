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

import numpy as np
import scipy.io
# from scipy.ndimage import gaussian_filter1d   # keep if you want smoothing
DIR = '/Users/xiangxuankong/Desktop/Human TC Support WM/Figures/Figure 2/'
def separate_trials(mat_file_path, patient_id):
    """
    Load a .mat data file, filter neurons by patient_id, and separate trials into
    correct and incorrect trials with dimensions Trial x Neuron x Time Bin.
    Additionally, return the time fields for the selected patient's neurons.

    Parameters:
    - mat_file_path: str, path to the .mat data file
    - patient_id: int or str, the patient ID to filter neurons.

    Returns:
    - correct_trials: np.ndarray, array of correct trials (Trial x Neuron x Time Bin).
    - incorrect_trials: np.ndarray, array of incorrect trials (Trial x Neuron x Time Bin).
    - time_fields: list of int, time fields for the selected patient's neurons.
    """
    np.random.seed(20250710)

    # Load data
    mat_data = scipy.io.loadmat(mat_file_path)
    neural_data = mat_data['neural_data']
    
    # Extract patient_id, time fields, and data components
    patient_ids = [int(entry['patient_id'][0][0]) for entry in neural_data[0]]
    firing_rates = [entry['firing_rates'] for entry in neural_data[0]]
    trial_performance = [entry['trial_correctness'] for entry in neural_data[0]]  # Correct/Incorrect performance
    time_fields = [int(entry['time_field'][0][0]) - 1 for entry in neural_data[0]]  # Adjust for zero-based indexing

    # Ensure input IDs match the data type
    patient_id = int(patient_id)

    # Filter neurons by patient_id
    selected_neurons = [
        (firing_rates[i], trial_performance[i], time_fields[i]) 
        for i, pid in enumerate(patient_ids) 
        if pid == patient_id
    ]

    if len(selected_neurons) == 0:
        print(f"No data found for patient_id: {patient_id}")
        return None, None, None

    # Process firing rates, trial performance, and time fields
    selected_firing_rates = [np.array(neuron[0]) for neuron in selected_neurons]
    selected_performance = [np.array(neuron[1]).flatten() for neuron in selected_neurons]

    selected_time_fields = [neuron[2] for neuron in selected_neurons]
    print(selected_time_fields[0])

    # Apply Gaussian smoothing along time bins
    # smoothed_neural_data = [gaussian_filter1d(rate, sigma=2, axis=1) for rate in selected_firing_rates]
    smoothed_neural_data = selected_firing_rates
    # Stack data across neurons for trial-level analysis
    trial_data = np.stack(smoothed_neural_data, axis=1)  # Shape: Trial x Neuron x Time Bin
    trial_performance = np.all(np.stack(selected_performance, axis=0), axis=0).astype(int)  # Shape: Trial

    # Separate correct and incorrect trials
    correct_trials = trial_data[trial_performance == 1]  # Correct trials
    incorrect_trials = trial_data[trial_performance == 0]  # Incorrect trials

    return correct_trials, incorrect_trials, selected_time_fields


correct_trials, incorrect_trials, fields= separate_trials('100msTCdata.mat', patient_id=8)

if correct_trials is not None and incorrect_trials is not None:
    print(f"Correct Trials Shape: {correct_trials.shape}")
    print(f"Incorrect Trials Shape: {incorrect_trials.shape}")
    print(fields)

import numpy as np

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

import matplotlib.pyplot as plt

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def compute_cross_correlation_distribution_with_threshold(
    correct_trials, incorrect_trials, time_fields, num_iterations=1000
):
    """
    Compute the distribution of mean cross-correlations by resampling trials from correct trials.
    Also compute the distribution of single trial cross-correlations and mark percentiles on both plots.

    Parameters:
    - correct_trials: np.ndarray, shape (Trials x Neurons x Time Bins), correct trials data.
    - incorrect_trials: np.ndarray, shape (Trials x Neurons x Time Bins), incorrect trials data.
    - time_fields: list of int, indices of time fields for each neuron.
    - num_iterations: int, number of iterations for resampling.

    Returns:
    - mean_correlation_distribution: list, mean cross-correlation values from each iteration.
    - single_trial_distribution: list, single trial cross-correlation values from all sampled correct trials.
    - incorrect_mean_correlation: float, mean cross-correlation of incorrect trials.
    - percentile_below_threshold: float, percentile of correct trial cross-correlations below the threshold.
    """
    n_incorrect = len(incorrect_trials)  # Number of incorrect trials
    mean_correlation_distribution = []
    single_trial_distribution = []

    # Calculate mean cross-correlation for incorrect trials
    incorrect_trial_correlations = [
        process_trial_and_compute_mean_correlation(trial, time_fields) for trial in incorrect_trials
    ]
    incorrect_mean_correlation = np.mean(incorrect_trial_correlations)

    # Add a progress bar with tqdm
    for _ in tqdm(range(num_iterations), desc="Computing cross-correlation distribution"):
        # Randomly sample n_incorrect trials from correct trials
        sampled_trials = correct_trials[np.random.choice(len(correct_trials), n_incorrect, replace=False)]
        
        # Compute mean and single trial cross-correlation across sampled trials
        sampled_correlations = [
            process_trial_and_compute_mean_correlation(trial, time_fields) for trial in sampled_trials
        ]
        mean_correlation = np.mean(sampled_correlations)
        mean_correlation_distribution.append(mean_correlation)
        single_trial_distribution.extend(sampled_correlations)

    # Calculate the percentile of correct trial correlations below the incorrect mean correlation
    percentile_below_threshold_mean = np.mean(
        np.array(mean_correlation_distribution) < incorrect_mean_correlation
    ) * 100
    percentile_below_threshold_single = np.mean(
        np.array(single_trial_distribution) < incorrect_mean_correlation
    ) * 100

    # Plot the mean cross-correlation distribution
    plt.figure(figsize=(10, 6))
    plt.hist(mean_correlation_distribution, bins=30, alpha=0.7, color="blue", edgecolor="black", label="Mean Cross-Correlations")
    plt.axvline(
        incorrect_mean_correlation, color="red", linestyle="--", label=f"Incorrect Mean: {incorrect_mean_correlation:.4f}"
    )
    plt.text(
        incorrect_mean_correlation, plt.ylim()[1] * 0.8,
        f"Percentile: {percentile_below_threshold_mean:.2f}%",
        color="red", fontsize=10, ha="center"
    )
    plt.title("Distribution of Mean Cross-Correlation Values")
    plt.xlabel("Mean Cross-Correlation")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()

    # Plot the single-trial cross-correlation distribution
    plt.figure(figsize=(10, 6))
    plt.hist(single_trial_distribution, bins=50, alpha=0.7, color="green", edgecolor="black", label="Single-Trial Cross-Correlations")
    plt.axvline(
        incorrect_mean_correlation, color="red", linestyle="--", label=f"Incorrect Mean: {incorrect_mean_correlation:.4f}"
    )
    plt.text(
        incorrect_mean_correlation, plt.ylim()[1] * 0.8,
        f"Percentile: {percentile_below_threshold_single:.2f}%",
        color="red", fontsize=10, ha="center"
    )
    plt.title("Distribution of Single-Trial Cross-Correlation Values")
    plt.xlabel("Single-Trial Cross-Correlation")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()

    print(f"Percent of correct mean trials below incorrect mean: {percentile_below_threshold_mean:.2f}%")
    print(f"Percent of correct single trials below incorrect mean: {percentile_below_threshold_single:.2f}%")

    return mean_correlation_distribution, single_trial_distribution, incorrect_mean_correlation, percentile_below_threshold_mean, percentile_below_threshold_single


# # # Perform the operation and display the distribution
# cross_correlation_distribution = compute_cross_correlation_distribution_with_threshold(
#     correct_trials, incorrect_trials, fields, num_iterations=1000
# )

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind        # or mannwhitneyu / ks_2samp


def resample_and_plot_cross_correlation(correct_cross_correlations, incorrect_cross_correlations, min_time_cells, num_iterations):
    """
    Resample the correct trial cross-correlations, compare with incorrect trial mean,
    and plot the distributions.

    Parameters:
    - correct_cross_correlations: list/array-like, cross-correlation values for correct trials.
    - incorrect_cross_correlations: list/array-like, cross-correlation values for incorrect trials.
    - min_time_cells: int, used for labeling the plots.
    - num_iterations: int, number of resampling iterations.
    """

    # Helper function to convert p-value to significance stars
    def significance_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"

    incorrect_mean_correlation = np.mean(incorrect_cross_correlations)
    mean_correlation_distribution = []

    # Resampling
    resampled_values = []
    for _ in tqdm(range(num_iterations), desc="Resampling correct trials"):
        sampled_correlations = np.random.choice(
            correct_cross_correlations, 
            size=len(incorrect_cross_correlations), 
            replace=False
        )
        resampled_values.append(sampled_correlations)
        mean_correlation_distribution.append(np.mean(sampled_correlations))


    resampled_values  = np.array(resampled_values)
    incorrect_values  = np.array(incorrect_cross_correlations)

    # percentile_below_threshold = fraction * 100
    # => p_value = fraction
    percentile_below_threshold = np.mean(np.array(mean_correlation_distribution) < incorrect_mean_correlation) * 100
    p_value = percentile_below_threshold / 100.0  # Convert percent to fraction
    star_label = significance_stars(p_value)

    # --- Plot 1: Distribution of Resampled Mean Cross-Correlation Values ---
    plt.figure(figsize=(8, 5))
    plt.hist(mean_correlation_distribution, bins=30, alpha=0.7, color="blue",
             edgecolor="black", label="Resampled Correct Mean")

    # Red dashed line at the incorrect mean
    plt.axvline(
        x=incorrect_mean_correlation, 
        color="red", 
        linestyle="--", 
        linewidth=3,  # make the line thicker
        ymin=0.0,     # start 20% up the y-axis
        ymax=0.8,     # end 80% up the y-axis
        label="Incorrect Mean"
    )

    # Annotate significance stars near the top
    y_upper = plt.ylim()[1]
    plt.text(
        incorrect_mean_correlation, 0.8 * y_upper,  # Adjust the 0.8 as needed
        star_label,
        color="red", fontsize=15, ha="center"
    )

    plt.title(f"Mean Cross-Correlation Values (Resampled Correct Trials)")
    plt.xlabel("Mean Cross-Correlation")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.5)
    plt.legend()
    plt.savefig(DIR + 'crosscorrelation.svg', format='svg')
    plt.show()

    # --- Plot 2: Distribution of Incorrect Trials ---
    plt.figure(figsize=(5, 8))
    plt.hist(incorrect_cross_correlations, bins=50, alpha=0.7, color="green",
             edgecolor="black", label="Incorrect Trials")
    plt.axvline(incorrect_mean_correlation, color="red", linestyle="--")
    plt.title(f"Distribution of Cross-Correlation Values (Incorrect Trials)\n"
              f"Patients with > {min_time_cells} Time Cells")
    plt.xlabel("Cross-Correlation")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()

    # Compute means
    mean_corr   = np.mean(mean_correlation_distribution)
    mean_incorr = np.mean(incorrect_cross_correlations)

    # Compute histogram bins
    bins = 100
    range_min = min(min(mean_correlation_distribution),
                    min(incorrect_cross_correlations))
    range_max = max(max(mean_correlation_distribution),
                    max(incorrect_cross_correlations))
    bin_edges   = np.linspace(range_min, range_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute histogram counts
    corr_hist,   _ = np.histogram(mean_correlation_distribution, bins=bin_edges)
    incorr_hist, _ = np.histogram(incorrect_cross_correlations,  bins=bin_edges)

    # Peak-normalise
    corr_norm   = corr_hist   / corr_hist.max()
    incorr_norm = incorr_hist / incorr_hist.max()

    # Plotting
    bar_w = bin_edges[1] - bin_edges[0]
    plt.figure(figsize=(8, 5))

    plt.bar(bin_centers, incorr_norm,
            width=bar_w, alpha=0.45, color="red",
            edgecolor="black", label="Incorrect trials", zorder=1)

    plt.bar(bin_centers, corr_norm,
            width=bar_w, alpha=0.45, color="blue",
            edgecolor="black", label="Correct (resampled means)", zorder=2)

    # ── Dashed lines for means ───────────────────────────────────────────────
    plt.axvline(mean_corr, color="blue", linestyle="--", linewidth=1.8, label="Mean (Correct)", zorder=3)
    plt.axvline(mean_incorr, color="red", linestyle="--", linewidth=1.8, label="Mean (Incorrect)", zorder=3)

    # ── Significance bar between the two means ───────────────────────────────
    x1, x2 = mean_corr, mean_incorr
    y_brk = max(np.max(corr_norm), np.max(incorr_norm)) + 0.05

    plt.plot([x1, x1, x2, x2],
            [y_brk - 0.02, y_brk, y_brk, y_brk - 0.02],
            lw=1.5, c="k")

    # Use your predefined significance label (e.g., "***")
    plt.text((x1 + x2) / 2, y_brk + 0.03,
            star_label, ha="center", va="bottom", fontsize=14)

    # ── Cosmetics ────────────────────────────────────────────────────────────
    plt.title("Peak-Normalised Cross-Correlation Distributions\n"
            "(Resampled Correct Means vs. Incorrect Trials)")
    plt.xlabel("Mean Cross-Correlation")
    plt.ylabel("Normalised Frequency (Peak = 1)")
    plt.legend()
    plt.grid(axis="y", alpha=0.4, linestyle="--")
    plt.tight_layout()
    plt.show()


   # --- Plot 4: Peak-Normalized Histograms (Bar Style) ---

    # Histogram parameters
    bins = 50
    range_min = min(min(correct_cross_correlations), min(incorrect_cross_correlations))
    range_max = max(max(correct_cross_correlations), max(incorrect_cross_correlations))
    bin_edges = np.linspace(range_min, range_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute histograms
    correct_hist, _ = np.histogram(correct_cross_correlations, bins=bin_edges)
    incorrect_hist, _ = np.histogram(incorrect_cross_correlations, bins=bin_edges)

    # Normalize by peak
    correct_norm = correct_hist / np.max(correct_hist)
    incorrect_norm = incorrect_hist / np.max(incorrect_hist)

    # Plotting
    bar_width = bin_edges[1] - bin_edges[0]

    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers, correct_norm, width=bar_width, alpha=0.5, edgecolor="black", label="Correct Trials")
    plt.bar(bin_centers, incorrect_norm, width=bar_width, alpha=0.5, edgecolor="black", label="Incorrect Trials")

    plt.xlabel("Mean Cross‑Correlation (single trials)")
    plt.ylabel("Normalized Frequency (Peak = 1)")
    plt.title("Figure 4  |  Peak-Normalized Correct vs. Incorrect Trial Distributions")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()

    # Histogram parameters
    bins = 50
    range_min = min(min(correct_cross_correlations), min(incorrect_cross_correlations))
    range_max = max(max(correct_cross_correlations), max(incorrect_cross_correlations))
    bin_edges = np.linspace(range_min, range_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute histograms
    correct_hist, _ = np.histogram(correct_cross_correlations, bins=bin_edges)
    incorrect_hist, _ = np.histogram(incorrect_cross_correlations, bins=bin_edges)

    # Compute CDFs
    correct_cdf = np.cumsum(correct_hist)
    correct_cdf = correct_cdf / correct_cdf[-1]  # normalize to [0,1]

    incorrect_cdf = np.cumsum(incorrect_hist)
    incorrect_cdf = incorrect_cdf / incorrect_cdf[-1]  # normalize to [0,1]

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, correct_cdf, label="Correct Trials", linewidth=2)
    plt.plot(bin_centers, incorrect_cdf, label="Incorrect Trials", linewidth=2)

    plt.xlabel("Mean Cross‑Correlation (single trials)")
    plt.ylabel("Cumulative Probability")
    plt.title("Figure 4 | CDF of Correct vs. Incorrect Trial Distributions")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Print final summary
    print(f"p-value (fraction of correct means < incorrect mean) = {p_value:.4f}")
    print(f"Significance label: {star_label}")

    # --- Plot: Distribution of All Resampled Cross-Correlation Values ---

    # Flatten the array of resampled subsets
    flattened_resampled = resampled_values.flatten()

    # Compute histograms
    bins = 50
    range_min = min(flattened_resampled.min(), incorrect_values.min())
    range_max = max(flattened_resampled.max(), incorrect_values.max())
    bin_edges = np.linspace(range_min, range_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute histograms
    resampled_hist, _ = np.histogram(flattened_resampled, bins=bin_edges)
    incorrect_hist, _ = np.histogram(incorrect_values, bins=bin_edges)

    # Normalize by peak
    resampled_norm = resampled_hist / np.max(resampled_hist)
    incorrect_norm = incorrect_hist / np.max(incorrect_hist)

    # Plotting
    bar_width = bin_edges[1] - bin_edges[0]
    t, p_value = ttest_ind(flattened_resampled, incorrect_values, alternative='greater') 
    
    print(p_value)

    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers, resampled_norm, width=bar_width, alpha=0.5,
            edgecolor="black", label="Correct Trials (All Resampled)")
    plt.bar(bin_centers, incorrect_norm, width=bar_width, alpha=0.5,
            edgecolor="black", label="Incorrect Trials")

    plt.xlabel("Cross-Correlation (Single Trials from Resampling)")
    plt.ylabel("Normalized Frequency (Peak = 1)")
    plt.title("All Resampled Correct Trials vs. Incorrect Trials")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()

    
    # # ─────────── PLOT 5 – resampled vs incorrect (peak-norm) ───────────
    # bins = 30
    # range_min = min(resampled_values.min(), incorrect_values.min())
    # range_max = max(resampled_values.max(), incorrect_values.max())
    # bin_edges   = np.linspace(range_min, range_max, bins + 1)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # bar_w       = bin_edges[1] - bin_edges[0]

    # # Histograms
    # res_hist, _ = np.histogram(resampled_values, bins=bin_edges)
    # inc_hist, _ = np.histogram(incorrect_values, bins=bin_edges)

    # res_norm = res_hist / res_hist.max()
    # inc_norm = inc_hist / inc_hist.max()

    # mean_res = resampled_values.mean()
    # mean_inc = incorrect_values.mean()

    
    # plt.figure(figsize=(8, 5))
    # plt.bar(bin_centers, inc_norm, width=bar_w,
    #         alpha=.45, color="red",   edgecolor="black",
    #         label="Incorrect trials")
    # plt.bar(bin_centers, res_norm, width=bar_w,
    #         alpha=.45, color="blue",  edgecolor="black",
    #         label="Correct (resampled)")
    # # Dashed mean lines
    # plt.axvline(mean_inc, color="red",  ls="--", lw=1.8, label="Mean incorrect")
    # plt.axvline(mean_res, color="blue", ls="--", lw=1.8, label="Mean resampled")


    # t_stat, p_val   = ttest_ind(resampled_values,
    #                             incorrect_values,
    #                             equal_var=False)     # Welch
    # star_label      = significance_stars(p_val)      # <-- used later


    # # Significance bar between means
    # y_peak = max(res_norm.max(), inc_norm.max()) + .05
    # plt.plot([mean_res, mean_res, mean_inc, mean_inc],
    #          [y_peak-.02, y_peak, y_peak, y_peak-.02], c="k")
    # plt.text((mean_res+mean_inc)/2, y_peak+.03, star_label,
    #          ha="center", va="bottom", fontsize=14)

    # plt.xlabel("Cross-correlation")
    # plt.ylabel("Normalised frequency (peak = 1)")
    # plt.title("Peak-normalised distributions\n(resampled correct vs. incorrect)")
    # plt.legend()
    # plt.grid(axis='y', alpha=.4, ls='--')
    # plt.tight_layout()
    # plt.show()


def analyze_cross_correlation_all_patients(mat_file_path, min_time_cells=5, min_incorrect_trials = 0, num_iterations=1000):
    """
    Loop through all patient IDs in the data, extract their correct and incorrect trials,
    compute cross-correlation values for each trial, and aggregate them into two groups.
    Skip patients with less than a certain number of time cells.

    Parameters:
    - mat_file_path: str, path to the .mat data file.
    - min_time_cells: int, minimum number of time cells required to include a patient's data.
    - num_iterations: int, number of resampling iterations for correct trial distribution.

    Returns:
    - correct_cross_correlations: list, cross-correlation values for all correct trials.
    - incorrect_cross_correlations: list, cross-correlation values for all incorrect trials.
    """
    # Load data
    mat_data = scipy.io.loadmat(mat_file_path)
    neural_data = mat_data['neural_data']

    # Get unique patient IDs
    patient_ids = [int(entry['patient_id'][0][0]) for entry in neural_data[0]]
    unique_patient_ids = np.unique(patient_ids)

    # Initialize lists to store cross-correlation values
    correct_cross_correlations = []
    incorrect_cross_correlations = []
    passing_patient_ids = []

    # Loop through each patient ID
    for patient_id in tqdm(unique_patient_ids, desc="Processing all patients"):
        # Extract correct trials, incorrect trials, and time fields
        correct_trials, incorrect_trials, time_fields = separate_trials(mat_file_path, patient_id)

        if correct_trials is None or incorrect_trials is None:
            continue

        # Skip patients with insufficient time cells
        # if len(time_fields) < min_time_cells or correct_trials.shape[1] < min_time_cells or incorrect_trials.shape[1] < min_time_cells:
        if len(time_fields) < min_time_cells:
            continue

        if incorrect_trials.shape[0] < min_time_cells:
            continue 

        if patient_id not in passing_patient_ids:
            passing_patient_ids.append(patient_id)

        # Compute cross-correlation for correct trials
        for trial in correct_trials:
            corr_value = process_trial_and_compute_mean_correlation(trial, time_fields)
            correct_cross_correlations.append(corr_value)

        # Compute cross-correlation for incorrect trials
        for trial in incorrect_trials:
            corr_value = process_trial_and_compute_mean_correlation(trial, time_fields)
            incorrect_cross_correlations.append(corr_value)

    print(f"Patients passing the criteria: {passing_patient_ids}")

        # --- after the for‑loop that fills the two lists ---
    print(f"Total correct trials pooled:   {len(correct_cross_correlations)}")
    print(f"Total incorrect trials pooled: {len(incorrect_cross_correlations)}")


    # Perform resampling and analysis
    resample_and_plot_cross_correlation(correct_cross_correlations, incorrect_cross_correlations, min_time_cells, num_iterations)

    return correct_cross_correlations, incorrect_cross_correlations

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

analyze_cross_correlation_all_patients("3sig15_data.mat", min_time_cells=5)

def analyze_cross_correlation_per_patient(
    mat_file_path,
    min_time_cells=5,
    min_incorrect_trials=5,
    num_iterations=1000
):
    """
    Run the correct‑vs‑incorrect permutation test separately for each patient
    who has (a) ≥ min_time_cells time‑cells and (b) ≥ min_incorrect_trials
    incorrect trials.

    Returns
    -------
    results : dict
        {patient_id: {"n_correct": int,
                      "n_incorrect": int,
                      "incorrect_mean": float,
                      "p_value": float,
                      "star": str}}
    """
    # ---------------- helpers ---------------- #
    def significance_stars(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    def resample_p(correct_vals, incorrect_vals):
        """One‑sided permutation p(correct_mean < incorrect_mean)."""
        inc_mean = np.mean(incorrect_vals)
        null = [
            np.mean(np.random.choice(correct_vals,
                                     size=len(incorrect_vals),
                                     replace=False))
            for _ in range(num_iterations)
        ]
        p = np.mean(np.array(null) < inc_mean)
        return inc_mean, p, significance_stars(p), null

    # -------------- main loop --------------- #
    results = {}
    mat_data     = scipy.io.loadmat(mat_file_path)
    neural_data  = mat_data["neural_data"]
    all_pids     = np.unique([int(e["patient_id"][0][0]) for e in neural_data[0]])

    for pid in tqdm(all_pids, desc="Per‑patient analysis"):
        # pull the trials for this patient
        correct_trials, incorrect_trials, tf = separate_trials(mat_file_path, pid)
        if correct_trials is None:
            continue

        # inclusion criteria
        if len(tf) < min_time_cells:
            continue
        if incorrect_trials.shape[0] < min_incorrect_trials:
            continue

        # compute a mean cross‑corr for every trial
        c_vals = [process_trial_and_compute_mean_correlation(tr, tf)
                  for tr in correct_trials]
        i_vals = [process_trial_and_compute_mean_correlation(tr, tf)
                  for tr in incorrect_trials]

        # permutation test
        inc_mean, p, stars, null = resample_p(c_vals, i_vals)

        # ----- quick plot (optional – comment out if unwanted) ----- #
        plt.figure(figsize=(6,4))
        plt.hist(null, bins=30, alpha=.7, color="skyblue",
                 edgecolor="black", label="Resampled correct means")
        plt.axvline(inc_mean, color="red", linestyle="--",
                    label=f"Incorrect mean = {inc_mean:.3f}")
        plt.text(inc_mean, plt.ylim()[1]*0.85, stars, ha="center",
                 fontsize=14, color="red")
        plt.title(f"Patient {pid}  (nC={len(c_vals)}, nI={len(i_vals)})")
        plt.xlabel("Mean cross‑correlation");  plt.ylabel("Freq")
        plt.legend();  plt.tight_layout();     plt.show()
        # ----------------------------------------------------------- #

        # save summary
        results[pid] = dict(n_correct=len(c_vals),
                            n_incorrect=len(i_vals),
                            incorrect_mean=inc_mean,
                            p_value=p,
                            star=stars)

        print(f"Patient {pid}: nC={len(c_vals):3d}, nI={len(i_vals):3d}, "
              f"p={p:.4f}  {stars}")

    return results

# correct_cc_values, incorrect_cc_values = analyze_cross_correlation_all_patients('100msTCdata_G2.mat', min_time_cells=5, num_iterations=1000)
# if __name__ == "__main__":
#     region_stats = analyze_cross_corr_by_region("100msTCdata.mat",
#                                                 neuron_threshold=5,
#                                                 min_incorrect_trials=1,
#                                                 min_time_cells=5,
#                                                 num_iterations=1000,
#                                                 collapse_lr=True)
#     from pprint import pprint
#     pprint(region_stats)


def analyze_cross_correlation_all_patients_balanced(
    mat_file_path,
    min_time_cells      = 5,
    min_incorrect_trials= 1,
    num_iterations      = 1000,
):
    """
    • Collect per‑trial mean cross‑correlations for every patient that meets both
      thresholds (≥ min_time_cells time cells AND ≥ min_incorrect_trials incorrect trials).

    • During the permutation test each patient i contributes `n_i` correct‑trial
      values, where `n_i` = #incorrect trials that patient provided.

      → This prevents a patient with many correct trials and only a few incorrect
        trials from dominating the null distribution.

    Returns
    -------
    summary : dict  {patient_id: {"n_correct": int,
                                  "n_incorrect": int,
                                  "incorrect_mean": float}}
    """
    # ---------- helper ---------- #
    def significance_stars(p):
        return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"

    # ---------- gather data ---------- #
    per_patient_correct   = {}   # pid → list of per‑trial corr values
    per_patient_incorrect = {}

    mat_data    = scipy.io.loadmat(mat_file_path)
    neural_data = mat_data["neural_data"]
    all_pids    = np.unique([int(e["patient_id"][0][0]) for e in neural_data[0]])

    for pid in tqdm(all_pids, desc="Loading patients"):
        c_trials, i_trials, tf = separate_trials(mat_file_path, pid)
        if c_trials is None:          # no data
            continue
        if len(tf) < min_time_cells:  # not enough time cells
            continue
        if i_trials.shape[0] < min_incorrect_trials:
            continue

        # per‑trial mean correlations
        c_vals = [process_trial_and_compute_mean_correlation(tr, tf) for tr in c_trials]
        i_vals = [process_trial_and_compute_mean_correlation(tr, tf) for tr in i_trials]

        if len(c_vals) == 0 or len(i_vals) == 0:
            continue   # safety

        per_patient_correct[pid]   = c_vals
        per_patient_incorrect[pid] = i_vals

    if not per_patient_correct:
        print("No patients satisfied the inclusion criteria.")
        return None

    # ---------- global incorrect pool & descriptive stats ---------- #
    incorrect_all = np.concatenate(list(per_patient_incorrect.values()))
    incorrect_mean_global = np.mean(incorrect_all)

    print(f"\nIncluded patients: {sorted(per_patient_correct.keys())}")
    print(f"Total correct trials   (after filtering): {sum(len(v) for v in per_patient_correct.values())}")
    print(f"Total incorrect trials (after filtering): {len(incorrect_all)}")

    # ---------- permutation / “bootstrap” with balanced contribution ---------- #
    null_distribution = []

    for _ in tqdm(range(num_iterations), desc="Balanced resampling"):
        sampled_vals = []

        for pid in per_patient_correct.keys():
            n_incorrect = len(per_patient_incorrect[pid])
            # guarantees balance; if a patient has fewer correct than incorrect,
            # fall back to sampling with replacement.
            replace_flag = n_incorrect > len(per_patient_correct[pid])
            sampled_vals.extend(
                np.random.choice(per_patient_correct[pid],
                                 size=n_incorrect,
                                 replace=replace_flag)
            )

        null_distribution.append(np.mean(sampled_vals))

    null_distribution = np.array(null_distribution)
    p_value = np.mean(null_distribution < incorrect_mean_global)
    star = significance_stars(p_value)

    # ---------- plotting ---------- #
    plt.figure(figsize=(8, 5))
    plt.hist(null_distribution, bins=30, alpha=0.7, color="steelblue",
             edgecolor="black", label="Balanced resampled correct means")
    plt.axvline(incorrect_mean_global, color="red", linestyle="--",
                linewidth=2.5, label=f"Incorrect mean = {incorrect_mean_global:.3f}")
    plt.text(incorrect_mean_global, plt.ylim()[1]*0.85, star,
             ha="center", fontsize=16, color="red")
    plt.xlabel("Mean cross‑correlation");  plt.ylabel("Frequency")
    plt.title("Null distribution with balanced patient contribution")
    plt.legend();  plt.grid(alpha=0.4);  plt.tight_layout();  plt.show()

    print(f"p‑value (one‑sided) = {p_value:.4f}   {star}")

    # ---------- return a tidy summary ---------- #
    summary = {pid: dict(
                    n_correct   = len(per_patient_correct[pid]),
                    n_incorrect = len(per_patient_incorrect[pid]),
                    incorrect_mean = np.mean(per_patient_incorrect[pid]))
               for pid in per_patient_correct.keys()}

    return summary

# analyze_cross_correlation_per_patient('100msTCdata.mat', min_time_cells=10, min_incorrect_trials=4, num_iterations = 1000)

# analyze_cross_correlation_all_patients_balanced('100msTCdata_G2.mat', min_time_cells=5, min_incorrect_trials=2, num_iterations = 1000)