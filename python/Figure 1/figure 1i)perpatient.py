import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from scipy.stats import sem
import matplotlib.pyplot as plt

DIR = '/Users/xiangxuankong/Desktop/Human TC Support WM/Figures/Figure 1/'

###############################################################################
# Helper function for decoding a single patient
###############################################################################
def decode_patient(neural_data_struct, patient_id, total_duration=2.5, test_size=0.3, random_state=20250710):
    """
    Filters neurons for the given patient ID, truncates to min trials,
    reshapes to (time_bins, trials, neurons), trains/test GaussianNB,
    returns mean decoding error for that patient.
    """
    # Extract only neurons for this patient
    all_units = [entry for entry in neural_data_struct[0] if entry['patient_id'][0][0] == patient_id]
    if len(all_units) == 0:
        return None

    # Extract firing rates
    firing_rates_list = [np.array(entry['firing_rates']) for entry in all_units]
    min_trials = min(fr.shape[0] for fr in firing_rates_list)
    truncated_data = [fr[:min_trials, :] for fr in firing_rates_list]

    reshaped_data = np.stack(truncated_data, axis=2).transpose(1,0,2)
    time_bins, trials, neurons = reshaped_data.shape
    bin_size = total_duration / time_bins

    X_trials = reshaped_data
    y_trials = np.tile(np.arange(time_bins), (trials,1)).T

    all_trials = np.arange(trials)
    train_trials, test_trials = train_test_split(all_trials, test_size=test_size,
                                                 random_state=random_state)

    X_train = X_trials[:, train_trials, :].reshape(-1, neurons)
    y_train = y_trials[:, train_trials].ravel()
    X_test  = X_trials[:, test_trials, :].reshape(-1, neurons)
    y_test  = y_trials[:, test_trials].ravel()

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    decoding_errors = bin_size * np.abs(y_pred - y_test)
    mean_error = np.mean(decoding_errors)

    return mean_error
import numpy as np
import scipy.io
from collections import Counter

###############################################################################
# Load both datasets
###############################################################################
time_cells_mat = "3sig15_data.mat"
non_time_cells_mat = "3sig15_data_ntc.mat"

data_TC  = scipy.io.loadmat(time_cells_mat)
data_NTC = scipy.io.loadmat(non_time_cells_mat)

neural_data_TC  = data_TC['neural_data']
neural_data_NTC = data_NTC['neural_data']

###############################################################################
# Count neurons per patient in TIME CELLS
###############################################################################
patient_ids_TC = [entry['patient_id'][0][0] for entry in neural_data_TC[0]]
patient_ids_NTC = [entry['patient_id'][0][0] for entry in neural_data_NTC[0]]

count_TC = Counter(patient_ids_TC)
print("Time-Cell neuron counts per patient:")
print(count_TC)

###############################################################################
# Set minimum time cells per patient
###############################################################################
MIN_TIME_CELLS = 5

# Keep patients with enough time cells AND that exist in non-time-cells too
eligible_patients = [pid for pid, count in count_TC.items()
                     if count >= MIN_TIME_CELLS and pid in patient_ids_NTC]

print(f"\nEligible patients (≥ {MIN_TIME_CELLS} time cells): {eligible_patients}")

###############################################################################
# Then decode as before
###############################################################################
TC_errors_per_patient  = []
NTC_errors_per_patient = []

for pid in eligible_patients:
    mean_err_TC  = decode_patient(neural_data_TC,  pid)
    mean_err_NTC = decode_patient(neural_data_NTC, pid)

    if mean_err_TC is not None and mean_err_NTC is not None:
        TC_errors_per_patient.append(mean_err_TC)
        NTC_errors_per_patient.append(mean_err_NTC)
    else:
        print(f"Skipping patient {pid} (missing data)")

TC_errors_per_patient  = np.array(TC_errors_per_patient)
NTC_errors_per_patient = np.array(NTC_errors_per_patient)

print("\nDecoding Errors (Time Cells):", TC_errors_per_patient)
print("Decoding Errors (Non-Time-Cells):", NTC_errors_per_patient)

###############################################################################
# Dot plot: same as before
###############################################################################
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
# Paired t-test
t_stat, p_pt = ttest_rel(TC_errors_per_patient, NTC_errors_per_patient)
print(f"Paired t-test: t = {t_stat:.3f}, p = {p_pt:.3e}")

# Figure
plt.figure(figsize=(4.8, 6))

# Paired grey dots + connecting lines
for tc, ntc in zip(TC_errors_per_patient, NTC_errors_per_patient):
    plt.plot([0, 1], [tc, ntc], color='gray', alpha=0.6)
    plt.scatter(0, tc, color='gray', alpha=0.8, zorder=10)
    plt.scatter(1, ntc, color='gray', alpha=0.8, zorder=10)

# Overlay mean lines
plt.hlines(np.mean(TC_errors_per_patient), -0.2, 0.2,
           color='blue', linewidth=2.5, zorder=5)
plt.hlines(np.mean(NTC_errors_per_patient), 0.8, 1.2,
           color='green', linewidth=2.5, zorder=5)

# Significance annotation
ymax = max(np.max(TC_errors_per_patient), np.max(NTC_errors_per_patient))
star_y = ymax + 0.02

plt.plot([0, 0, 1, 1], [ymax, star_y, star_y, ymax],
         lw=1.3, color='black')

if p_pt < 1e-3:
    sig_label = "***"
elif p_pt < 1e-2:
    sig_label = "**"
elif p_pt < 0.05:
    sig_label = "*"
else:
    sig_label = f"p = {p_pt:.3f}"

plt.text(0.5, star_y + 0.005, sig_label,
         ha='center', va='bottom', fontsize=14)

# Aesthetics
plt.xticks([0, 1], ['Time Cells', 'Non–Time-Cells'], fontsize=12)
plt.ylabel("Mean Decoding Error (s)", fontsize=12)
plt.title("Per-Patient Decoding Errors", fontsize=14)
plt.xlim(-0.5, 1.5)
plt.tight_layout()

# Save if you want
plt.savefig(DIR + "paired_dotplot_per_patient.svg", format="svg")
plt.show()
