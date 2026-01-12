import numpy as np
import scipy.io
from sklearn.svm import SVC
from sklearn.utils import resample

import matplotlib.pyplot as plt
import seaborn as sns

# For one-sided t-test in SciPy >= 1.9
from scipy.stats import ttest_1samp
import numpy as np
import scipy.io
from sklearn.svm import SVC
from sklearn.utils import resample

import matplotlib.pyplot as plt
import seaborn as sns

# For one-sided t-test in SciPy >= 1.9
from scipy.stats import ttest_1samp
from sklearn.metrics import confusion_matrix


def star_label(p):
    if p < 1e-3:
        return "***"
    elif p < 1e-2:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return f"p = {p:.3f}"



DIR = '/Users/xiangxuankong/Desktop/Human TC Support WM/Figures/Figure 3/'

import numpy as np
import scipy.io
from sklearn.svm import SVC
from sklearn.utils import resample
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def loocv_with_permutation_test(
    mat_file_path,
    patient_ids=range(1, 22),
    m=0,
    num_windows=1,
    n_permutations=500,
    random_state=42,
    only_correct=True
):
    """
    1) Build a pseudo-population (equal # trials per load).
    2) Perform standard LOOCV on the real data to get a per-trial correctness distribution
       for each load (Load=1,2,3).
    3) Shuffle labels (entire y vector) n_permutations times, each time performing LOOCV
       to build a large 'null' distribution of correctness for each load.
    4) Compare real correctness distribution vs. null distribution with a t-test.
    5) Plot only the real correctness distributions as boxplots, add significance stars.
    """

    rng = np.random.default_rng(random_state)

    # 1) Load .mat data
    mat_data = scipy.io.loadmat(mat_file_path)
    neural_data = mat_data['neural_data']

    # 2) Group by patient
    patient_dict = {}
    for entry in neural_data[0]:
        pid = int(entry['patient_id'][0][0])
        frates = entry['firing_rates']  # (num_trials, num_time_bins)
        loads = entry['trial_load'].flatten().astype(int)

        correctness = entry['trial_correctness'].flatten().astype(int)
        tfield = int(entry['time_field'][0][0]) - 1

        # If only_correct == True, filter out incorrect trials
        if only_correct:
            idx_correct = (correctness == 1)
            frates = frates[idx_correct, :]
            loads  = loads[idx_correct]

        if pid not in patient_dict:
            patient_dict[pid] = []
        patient_dict[pid].append((frates, loads, tfield))

    # 3) Filter patients
    valid_patients = []
    for pid in patient_ids:
        if pid in patient_dict and len(patient_dict[pid]) >= m:
            valid_patients.append(pid)
    if len(valid_patients) == 0:
        print(f"No patients meet the minimum {m} time cells.")
        return None

    # 4) Gather valid neurons, each with load1, load2, load3
    all_neurons = []
    for pid in valid_patients:
        for (frates, loads, tfield) in patient_dict[pid]:
            start_idx = max(0, tfield - num_windows)
            end_idx   = min(frates.shape[1], tfield + num_windows + 1)
            
            # average across the selected window
            windowed = frates[:, start_idx:end_idx]
            mean_rates = np.mean(windowed, axis=1)

            load1_rates = mean_rates[loads == 1]
            load2_rates = mean_rates[loads == 2]
            load3_rates = mean_rates[loads == 3]

            # keep only if >=1 trial per load
            if len(load1_rates) > 0 and len(load2_rates) > 0 and len(load3_rates) > 0:
                all_neurons.append({
                    'load1': load1_rates,
                    'load2': load2_rates,
                    'load3': load3_rates
                })

    if len(all_neurons) == 0:
        msg = "No neurons found with all three loads present."
        if only_correct:
            msg = "No neurons found (all loads) with only-correct trials."
        print(msg)
        return None

    # 5) Global min # of trials per load
    min_load1 = min(len(n['load1']) for n in all_neurons)
    min_load2 = min(len(n['load2']) for n in all_neurons)
    min_load3 = min(len(n['load3']) for n in all_neurons)
    if (min_load1 == 0) or (min_load2 == 0) or (min_load3 == 0):
        print("Some load has zero global min. Exiting.")
        return None

    # 6) Downsample each neuron to the global minimum for each load (to keep it balanced)
    for neuron in all_neurons:
        neuron['load1'] = resample(neuron['load1'], replace=False,
                                   n_samples=min_load1, random_state=random_state)
        neuron['load2'] = resample(neuron['load2'], replace=False,
                                   n_samples=min_load2, random_state=random_state)
        neuron['load3'] = resample(neuron['load3'], replace=False,
                                   n_samples=min_load3, random_state=random_state)

    # 7) Build big X matrix (rows=trials, cols=neurons) and big y label vector
    #    We'll stack load1, load2, load3 in that order
    num_neurons = len(all_neurons)
    X_load1 = np.zeros((min_load1, num_neurons), dtype=np.float32)
    X_load2 = np.zeros((min_load2, num_neurons), dtype=np.float32)
    X_load3 = np.zeros((min_load3, num_neurons), dtype=np.float32)

    for j, neuron in enumerate(all_neurons):
        X_load1[:, j] = neuron['load1']
        X_load2[:, j] = neuron['load2']
        X_load3[:, j] = neuron['load3']

    X_all = np.vstack([X_load1, X_load2, X_load3])  # shape = (T, num_neurons)
    y_all = np.array([1]*min_load1 + [2]*min_load2 + [3]*min_load3)

    # Some info
    print(f"--- LOOCV Pseudopopulation Info (only_correct={only_correct}) ---")
    print(f"  Number of neurons: {num_neurons}")
    print(f"  Final trials for Load=1: {min_load1}")
    print(f"  Final trials for Load=2: {min_load2}")
    print(f"  Final trials for Load=3: {min_load3}")
    print(f"  Total trials: {len(y_all)}")

    # --------------------------------------------------------
    # 8) LOOCV on the REAL (unshuffled) dataset
    # --------------------------------------------------------
    T = len(y_all)
    # We'll store predicted labels so we can measure correctness on a per-trial basis
    y_pred_real = np.zeros(T, dtype=int)

    clf = SVC(kernel='linear', random_state=random_state)

    for i in range(T):
        # Leave out the i-th trial
        train_mask = np.ones(T, dtype=bool)
        train_mask[i] = False

        X_train = X_all[train_mask, :]
        y_train = y_all[train_mask]
        X_test  = X_all[i, :].reshape(1, -1)  # 1 row
        y_test  = y_all[i]

        clf.fit(X_train, y_train)
        y_pred_real[i] = clf.predict(X_test)

    # correctness per trial => 1 if correct else 0
    correctness_real = (y_pred_real == y_all).astype(int)

    # Split by load
    real_load1_idx = (y_all == 1)
    real_load2_idx = (y_all == 2)
    real_load3_idx = (y_all == 3)

    real_correct_load1 = correctness_real[real_load1_idx]  # array of 0/1, length = min_load1
    real_correct_load2 = correctness_real[real_load2_idx]  # length = min_load2
    real_correct_load3 = correctness_real[real_load3_idx]  # length = min_load3

    # --------------------------------------------------------
    # 9) Permutation test: do the same LOOCV on shuffled labels
    #    for n_permutations times.
    #    We'll accumulate the correctness (0/1) for each load
    #    across all permutations into large arrays.
    # --------------------------------------------------------
    perm_correct_load1 = []
    perm_correct_load2 = []
    perm_correct_load3 = []

    # We'll re-use the SVC but re-fit for each fold in each permutation
    for _ in range(n_permutations):
        # Shuffle y_all in place (keeping X_all the same).
        # But we want to do a *fresh* shuffle each time. So let's do:
        y_shuf = rng.permutation(y_all)  # same length, same # of 1,2,3, but scrambled order

        y_pred_shuf = np.zeros(T, dtype=int)

        # LOOCV again, but with shuffled labels
        for i in range(T):
            train_mask = np.ones(T, dtype=bool)
            train_mask[i] = False

            X_train = X_all[train_mask, :]
            y_train = y_shuf[train_mask]
            X_test  = X_all[i, :].reshape(1, -1)
            # "true" label for measuring correctness is now y_shuf[i]
            y_test_shuf = y_shuf[i]

            clf.fit(X_train, y_train)
            y_pred_shuf[i] = clf.predict(X_test)

        correctness_shuf = (y_pred_shuf == y_shuf).astype(int)

        # Now separate out which trials are "load1" in the *shuffled* sense
        # i.e. for all i where y_shuf[i] == 1
        idx_shuf_load1 = (y_shuf == 1)
        idx_shuf_load2 = (y_shuf == 2)
        idx_shuf_load3 = (y_shuf == 3)

        # store the 0/1 correctness for these trials
        perm_correct_load1.append(correctness_shuf[idx_shuf_load1])
        perm_correct_load2.append(correctness_shuf[idx_shuf_load2])
        perm_correct_load3.append(correctness_shuf[idx_shuf_load3])

    # Flatten these into single arrays:
    #  - We'll have (n_permutations * min_load1) total entries for load1, etc.
    perm_correct_load1 = np.concatenate(perm_correct_load1, axis=0)
    perm_correct_load2 = np.concatenate(perm_correct_load2, axis=0)
    perm_correct_load3 = np.concatenate(perm_correct_load3, axis=0)

    # --------------------------------------------------------
    # 10) Plot only the REAL correctness as boxplots,
    #     but significance stars reflect comparison vs. permuted set
    # --------------------------------------------------------
    # Build a DataFrame for real data
    df_real = pd.DataFrame({
        'Load': (['Load1'] * len(real_correct_load1)
                + ['Load2'] * len(real_correct_load2)
                + ['Load3'] * len(real_correct_load3)),
        'Correctness': np.concatenate([real_correct_load1,
                                       real_correct_load2,
                                       real_correct_load3])
    })

    # Make the boxplot
    plt.figure(figsize=(6,8))
    sns.boxplot(x='Load', y='Correctness', data=df_real, showfliers=True)
    plt.title(f"Leave-One-Out CV Correctness (Real Labels), n_perm={n_permutations}")
    plt.ylim([-0.1, 1.1])
    plt.ylabel("Correctness (0 or 1)")
    plt.xlabel("Load Condition")

    # Add chance line at 0.33, if you want (for a rough reference in 3-class case):
    plt.axhline(y=0.33, color='gray', linestyle='--')

    # We'll do a significance test for each load: real distribution vs. perm distribution
    # We'll define a small helper to convert p-values to star labels

    # We'll gather p-values for each load
    # Two-sample t-test (the distribution of 0/1 real correctness vs. the distribution of 0/1 perm correctness)
    t1, p1 = ttest_ind(real_correct_load1, perm_correct_load1, equal_var=False)
    t2, p2 = ttest_ind(real_correct_load2, perm_correct_load2, equal_var=False)
    t3, p3 = ttest_ind(real_correct_load3, perm_correct_load3, equal_var=False)


    # t1, p1 = ttest_1samp(real_correct_load1, 0.33)
    # t2, p2 = ttest_1samp(real_correct_load2, 0.33)
    # t3, p3 = ttest_1samp(real_correct_load3, 0.33)

    # Determine star labels
    star1 = star_label(p1)
    star2 = star_label(p2)
    star3 = star_label(p3)

    # Place stars above each box
    # We'll place them at e.g. y ~ 1.02 or so.
    offset = 0.02
    loads = ['Load1','Load2','Load3']
    star_positions = [1.02, 1.02, 1.02]  # just a constant above 1.0

    for i, (star, y_pos) in enumerate(zip([star1, star2, star3], star_positions)):
        plt.text(i, y_pos, star, ha='center', va='bottom', color='black', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Print summary stats
    print("=== Real correctness mean (±std) ===")
    print(f"Load1: {real_correct_load1.mean():.3f} ± {real_correct_load1.std():.3f}, n={len(real_correct_load1)}")
    print(f"Load2: {real_correct_load2.mean():.3f} ± {real_correct_load2.std():.3f}, n={len(real_correct_load2)}")
    print(f"Load3: {real_correct_load3.mean():.3f} ± {real_correct_load3.std():.3f}, n={len(real_correct_load3)}\n")

    print("=== Permutation distributions mean (±std) ===")
    print(f"Load1 perm: {perm_correct_load1.mean():.3f} ± {perm_correct_load1.std():.3f}, n={len(perm_correct_load1)}")
    print(f"Load2 perm: {perm_correct_load2.mean():.3f} ± {perm_correct_load2.std():.3f}, n={len(perm_correct_load2)}")
    print(f"Load3 perm: {perm_correct_load3.mean():.3f} ± {perm_correct_load3.std():.3f}, n={len(perm_correct_load3)}\n")

    print("=== Two-sample t-test (real vs perm) ===")
    print(f"Load1: t={t1:.3f}, p={p1:.3e}")
    print(f"Load2: t={t2:.3f}, p={p2:.3e}")
    print(f"Load3: t={t3:.3f}, p={p3:.3e}")

    return {
        'real_correct_load1': real_correct_load1,
        'real_correct_load2': real_correct_load2,
        'real_correct_load3': real_correct_load3,
        'perm_correct_load1': perm_correct_load1,
        'perm_correct_load2': perm_correct_load2,
        'perm_correct_load3': perm_correct_load3,
        'pvals': [p1, p2, p3]
    }

import numpy as np
import scipy.io
from sklearn.svm import SVC
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, ttest_1samp
def repeated_cv_pseudo_population_per_class_colored_outliers_with_label_shuffling(
    mat_file_path,
    patient_ids=range(1, 22),
    m=0,
    num_windows=0,
    test_per_load=7,
    n_iterations=1000,
    random_state=20250710,
    only_correct=False
):
    """
    ...  (doc-string unchanged)  ...
    """
    rng = np.random.default_rng(random_state)

    # -------------------------------
    # 1) Load .mat data
    # -------------------------------
    mat_data = scipy.io.loadmat(mat_file_path)
    neural_data = mat_data['neural_data']

    # -------------------------------
    # 2) Build per-patient structures
    # -------------------------------
    patient_dict = {}          # pid -> list[ (frates, loads, tfield) ]
    patient_load_counts = {}   # pid -> dict{1: n, 2: n, 3: n}

    for entry in neural_data[0]:
        pid       = int(entry['patient_id'][0][0])
        frates    = entry['firing_rates']                    # (trials × bins)
        loads     = entry['trial_load'].flatten().astype(int)
        correct   = entry['trial_correctness'].flatten().astype(int)
        tfield    = int(entry['time_field'][0][0]) - 1

        # ----- keep only correct trials if requested -----
        if only_correct:
            keep_idx = (correct == 1)
            frates   = frates[keep_idx, :]
            loads    = loads[keep_idx]

        # skip neurons that lost all trials
        if frates.size == 0:
            continue

        # initialise dicts
        if pid not in patient_dict:
            patient_dict[pid]       = []
            patient_load_counts[pid] = {1: 0, 2: 0, 3: 0}

        patient_dict[pid].append((frates, loads, tfield))

        # update per-patient correct-trial counts
        for ld in (1, 2, 3):
            patient_load_counts[pid][ld] = max(
                patient_load_counts[pid][ld],
                np.sum(loads == ld)
            )

    # -------------------------------
    # 3) Patient-level inclusion filter
    # -------------------------------
    valid_patients = []
    for pid in patient_ids:
        # must exist, have ≥ m neurons, and (if only_correct) ≥ 30 correct trials per load
        if pid not in patient_dict:
            continue
        if len(patient_dict[pid]) < m:
            continue
        if only_correct:
            counts = patient_load_counts[pid]
            if not (counts[1] >= 25 and counts[2] >= 25 and counts[3] >= 25):
                continue
        valid_patients.append(pid)

    if len(valid_patients) == 0:
        print("No patients meet the inclusion criteria "
              f"(m={m}, ≥30 correct trials per load = {only_correct}).")
        return None

    # ---- everything below (steps 4-13) remains unchanged ----

    # -------------------------------
    # 4) Gather valid neurons
    # -------------------------------
    all_neurons = []
    for pid in valid_patients:
        for (frates, loads, tfield) in patient_dict[pid]:
            start_idx = max(0, tfield - num_windows)
            end_idx   = min(frates.shape[1], tfield + num_windows + 1)
            
            # average across the selected window
            windowed = frates[:, start_idx:end_idx]
            mean_rates = np.mean(windowed, axis=1)

            load1_rates = mean_rates[loads == 1]
            load2_rates = mean_rates[loads == 2]
            load3_rates = mean_rates[loads == 3]

            # keep only if >=1 trial per load
            if len(load1_rates) > 0 and len(load2_rates) > 0 and len(load3_rates) > 0:
                all_neurons.append({
                    'load1': load1_rates,
                    'load2': load2_rates,
                    'load3': load3_rates
                })

    if len(all_neurons) == 0:
        if only_correct:
            print("No neurons found (all loads) with only-correct trials.")
        else:
            print("No neurons found with all three loads present.")
        return None

    # -------------------------------
    # 5) Global min # of trials per load
    # -------------------------------
    min_load1 = min(len(n['load1']) for n in all_neurons)
    min_load2 = min(len(n['load2']) for n in all_neurons)
    min_load3 = min(len(n['load3']) for n in all_neurons)

    if (min_load1 == 0) or (min_load2 == 0) or (min_load3 == 0):
        print("Some load has zero global min. Exiting.")
        return None

    # -------------------------------
    # 6) Downsample each neuron
    # -------------------------------
    for neuron in all_neurons:
        neuron['load1'] = resample(neuron['load1'], replace=False,
                                   n_samples=min_load1, random_state=random_state)
        neuron['load2'] = resample(neuron['load2'], replace=False,
                                   n_samples=min_load2, random_state=random_state)
        neuron['load3'] = resample(neuron['load3'], replace=False,
                                   n_samples=min_load3, random_state=random_state)

    # -------------------------------
    # 7) Build big matrices
    # -------------------------------
    num_neurons = len(all_neurons)
    load1_matrix = np.zeros((min_load1, num_neurons), dtype=np.float32)
    load2_matrix = np.zeros((min_load2, num_neurons), dtype=np.float32)
    load3_matrix = np.zeros((min_load3, num_neurons), dtype=np.float32)

    for j, neuron in enumerate(all_neurons):
        load1_matrix[:, j] = neuron['load1']
        load2_matrix[:, j] = neuron['load2']
        load3_matrix[:, j] = neuron['load3']

    # Print some basic info
    print(f"--- Pseudopopulation Info (only_correct={only_correct}) ---")
    print(f"  Number of neurons: {num_neurons}")
    print(f"  Final trials for Load=1: {min_load1}")
    print(f"  Final trials for Load=2: {min_load2}")
    print(f"  Final trials for Load=3: {min_load3}")

    # Check if test_per_load is feasible
    if test_per_load > min_load1 or test_per_load > min_load2 or test_per_load > min_load3:
        print(f"Requested test_per_load={test_per_load}, but min load trials are "
              f"({min_load1},{min_load2},{min_load3}). Exiting.")
        return None

    # -------------------------------
    # 8) Repeated sub-sampling (Actual)
    # -------------------------------
    accs_load1_actual = []
    accs_load2_actual = []
    accs_load3_actual = []

    cm_accumulator = np.zeros((3, 3))  # 3 classes (load1,2,3)
    for _ in range(n_iterations):
        # Shuffle each load's trials
        idx1 = rng.permutation(min_load1)
        idx2 = rng.permutation(min_load2)
        idx3 = rng.permutation(min_load3)

        # Split test vs. train
        test_idx1 = idx1[:test_per_load]
        train_idx1 = idx1[test_per_load:]
        test_idx2 = idx2[:test_per_load]
        train_idx2 = idx2[test_per_load:]
        test_idx3 = idx3[:test_per_load]
        train_idx3 = idx3[test_per_load:]

        # Build training sets
        X_train_load1 = load1_matrix[train_idx1, :]
        X_train_load2 = load2_matrix[train_idx2, :]
        X_train_load3 = load3_matrix[train_idx3, :]
        X_train = np.vstack([X_train_load1, X_train_load2, X_train_load3])
        y_train = np.array([1]*len(train_idx1) + [2]*len(train_idx2) + [3]*len(train_idx3))

        # Build test sets
        X_test_load1 = load1_matrix[test_idx1, :]
        X_test_load2 = load2_matrix[test_idx2, :]
        X_test_load3 = load3_matrix[test_idx3, :]
        X_test = np.vstack([X_test_load1, X_test_load2, X_test_load3])
        y_test = np.array([1]*len(test_idx1) + [2]*len(test_idx2) + [3]*len(test_idx3))

        # Train SVM with REAL labels
        clf = SVC(kernel='linear', random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Per-class accuracy
        mask1 = (y_test == 1)
        mask2 = (y_test == 2)
        mask3 = (y_test == 3)

        accs_load1_actual.append(np.mean(y_pred[mask1] == 1))
        accs_load2_actual.append(np.mean(y_pred[mask2] == 2))
        accs_load3_actual.append(np.mean(y_pred[mask3] == 3))

        cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3])
        cm_accumulator += cm

    cm_percentage = (cm_accumulator / cm_accumulator.sum(axis=1, keepdims=True)) * 100
    labels = np.array([[f"{val:.1f}%" for val in row] for row in cm_percentage])

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        cm_percentage, annot=labels, fmt="", cmap="Blues",
        xticklabels=["Load 1", "Load 2", "Load 3"], 
        yticklabels=["Load 1", "Load 2", "Load 3"],
        vmin=0, vmax=65,
        cbar_kws={'label': 'Percentage (%)'}
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Averaged Confusion Matrix (%)")
    if only_correct:
        path = DIR + 'Correct/'
    else:
        path = DIR

    plt.savefig(path + 'confusionmatrix.svg', format = 'svg')
    plt.show()

    # Convert to % 
    accs_load1_actual = [100*a for a in accs_load1_actual]
    accs_load2_actual = [100*a for a in accs_load2_actual]
    accs_load3_actual = [100*a for a in accs_load3_actual]

    # -------------------------------
    # 9) Repeated sub-sampling (Label-Shuffled)
    # -------------------------------
    accs_load1_null = []
    accs_load2_null = []
    accs_load3_null = []

    for _ in range(n_iterations):
        # Same partition approach
        idx1 = rng.permutation(min_load1)
        idx2 = rng.permutation(min_load2)
        idx3 = rng.permutation(min_load3)

        test_idx1 = idx1[:test_per_load]
        train_idx1 = idx1[test_per_load:]
        test_idx2 = idx2[:test_per_load]
        train_idx2 = idx2[test_per_load:]
        test_idx3 = idx3[:test_per_load]
        train_idx3 = idx3[test_per_load:]

        X_train_load1 = load1_matrix[train_idx1, :]
        X_train_load2 = load2_matrix[train_idx2, :]
        X_train_load3 = load3_matrix[train_idx3, :]
        X_train = np.vstack([X_train_load1, X_train_load2, X_train_load3])
        y_train_real = np.array([1]*len(train_idx1) + [2]*len(train_idx2) + [3]*len(train_idx3))

        X_test_load1 = load1_matrix[test_idx1, :]
        X_test_load2 = load2_matrix[test_idx2, :]
        X_test_load3 = load3_matrix[test_idx3, :]
        X_test = np.vstack([X_test_load1, X_test_load2, X_test_load3])
        y_test_real = np.array([1]*len(test_idx1) + [2]*len(test_idx2) + [3]*len(test_idx3))

        # Now shuffle y_train labels (to break real mapping)
        y_train_shuffled = rng.permutation(y_train_real)

        # Train SVM with SHUFFLED labels
        clf = SVC(kernel='linear', random_state=random_state)
        clf.fit(X_train, y_train_shuffled)
        y_pred = clf.predict(X_test)

        # Per-class accuracy w.r.t. REAL test labels
        mask1 = (y_test_real == 1)
        mask2 = (y_test_real == 2)
        mask3 = (y_test_real == 3)

        accs_load1_null.append(np.mean(y_pred[mask1] == 1) * 100)
        accs_load2_null.append(np.mean(y_pred[mask2] == 2) * 100)
        accs_load3_null.append(np.mean(y_pred[mask3] == 3) * 100)

    # -------------------------------
    # 10) Organize into DataFrames
    # -------------------------------
    import pandas as pd

    df_actual = pd.DataFrame({
        'Load':   ['Load1']*n_iterations + ['Load2']*n_iterations + ['Load3']*n_iterations,
        'Accuracy': accs_load1_actual + accs_load2_actual + accs_load3_actual
    })
    df_null = pd.DataFrame({
        'Load':   ['Load1']*n_iterations + ['Load2']*n_iterations + ['Load3']*n_iterations,
        'Accuracy': accs_load1_null + accs_load2_null + accs_load3_null
    })

    # For convenience in analysis:
    df_actual_load1 = df_actual[df_actual['Load'] == 'Load1']['Accuracy']
    df_actual_load2 = df_actual[df_actual['Load'] == 'Load2']['Accuracy']
    df_actual_load3 = df_actual[df_actual['Load'] == 'Load3']['Accuracy']

    df_null_load1 = df_null[df_null['Load'] == 'Load1']['Accuracy']
    df_null_load2 = df_null[df_null['Load'] == 'Load2']['Accuracy']
    df_null_load3 = df_null[df_null['Load'] == 'Load3']['Accuracy']

        # -------------------------------
    # 11) Boxplot for Actual Only
    # -------------------------------
    plt.figure(figsize=(5, 8.1))
    positions = [0, 1, 2]
    width = 0.6

    def plot_one_box(data_series, x_position, color_str):
        sns.boxplot(
            x=[x_position] * len(data_series),
            y=data_series,
            width=width,
            color=color_str,
            showfliers=True,
            flierprops=dict(
                marker='o',
                markerfacecolor=color_str,
                markeredgecolor=color_str,
                markersize=4,
                linestyle='none'
            ),
            boxprops=dict(facecolor=color_str, edgecolor='black'),
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black')
        )

    # Actual distributions only
    plot_one_box(df_actual_load1, positions[0], 'blue')
    plot_one_box(df_actual_load2, positions[1], 'green')
    plot_one_box(df_actual_load3, positions[2], 'red')

    plt.xticks(positions, ['Load1', 'Load2', 'Load3'], fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=16)
    plt.xlabel("Load Condition", fontsize=16)
    plt.title("Accuracy Distribution", fontsize=16)

    # Chance line at ~33%
    plt.axhline(y=33.3, color='gray', linestyle='--')

    overall_max = df_actual['Accuracy'].max()
    plt.ylim([0, overall_max + 5])


    # # -------------------------------
    # # 11 b) Overall accuracy (one box)
    # # -------------------------------
    # # ➊  Compute overall accuracy per iteration
    # overall_actual = [
    #     (l1 + l2 + l3) / 3.0
    #     for l1, l2, l3 in zip(accs_load1_actual,
    #                         accs_load2_actual,
    #                         accs_load3_actual)
    # ]
    # overall_null = [
    #     (n1 + n2 + n3) / 3.0
    #     for n1, n2, n3 in zip(accs_load1_null,
    #                         accs_load2_null,
    #                         accs_load3_null)
    # ]

    # # ➋  Build a DataFrame just for plotting
    # df_overall = pd.DataFrame({'Accuracy': overall_actual})

    # # ➌  Draw the single-box box-plot
    # plt.figure(figsize=(2.8, 6))
    # sns.boxplot(data=df_overall, y='Accuracy',
    #             width=0.6, color='skyblue',
    #             showfliers=True,
    #             flierprops=dict(marker='o', markersize=4,
    #                             markerfacecolor='skyblue',
    #                             markeredgecolor='skyblue'))
    # plt.axhline(33.3, ls='--', c='gray')        # 3-way chance
    # plt.ylabel("Overall accuracy (%)")
    # plt.title("Overall decoder accuracy")
    # plt.xticks([])                               # hide the lone x-tick

    # # ➍  Permutation p-value and star annotation
    # observed_mean = np.mean(overall_actual)
    # p_val = np.mean(np.array(overall_null) >= observed_mean)

    # def significance_stars(p):
    #     if p < 1e-3:   return "***"
    #     if p < 1e-2:   return "**"
    #     if p < 0.05:   return "*"
    #     return "ns"

    # star = significance_stars(p_val)
    # y_star = max(overall_actual) + 1.0
    # plt.text(0, y_star, star, ha='center', va='bottom',
    #         fontsize=16, fontweight='bold')

    # plt.tight_layout()
    # plt.show()

    # # ➎  Optional console read-out
    # print(f"Overall accuracy:  mean={observed_mean:.2f} %")
    # print(f"Permutation p-value (Actual > Null) = {p_val:.3e}")

    # -------------------------------
    # 12) 2-sample t-tests (Actual vs. Null)
    # -------------------------------
 
    offset = 1.0  # vertical offset for the star
    load_pairs = [
        (df_actual_load1, df_null_load1),
        (df_actual_load2, df_null_load2),
        (df_actual_load3, df_null_load3),
    ]
    for i, (actual_vals, null_vals) in enumerate(load_pairs):
        # Empirical p-value based on null distribution
        observed_mean = np.mean(actual_vals)
        p_val = np.mean(null_vals >= observed_mean)
        def star_label(p):
            if p < 1e-3:
                return "***"
            elif p < 1e-2:
                return "**"
            elif p < 0.05:
                return "*"
            else:
                return f"p = {p:.3f}"


        star_label = star_label(p_val)


        # Decide where to place the star
        y_max_box = actual_vals.max()
        star_y = y_max_box + offset
        if star_y > (overall_max + 5):
            star_y = overall_max - 1

        plt.text(i, star_y, star_label,
                ha='center', va='bottom',
                color='black', fontsize=16, fontweight='bold')

    plt.tight_layout()

    if only_correct:
        path = DIR + 'Correct/'
    else:
        path = DIR

    #plt.savefig(path + 'bar.svg', format = 'svg')
    plt.show()

    # -------------------------------
    # 13) Print final stats
    # -------------------------------
    def print_summary(tag, x):
        print(f"{tag} => mean={np.mean(x):.2f}%, std={np.std(x):.2f}%")

    print("---- Actual distribution summary ----")
    print_summary("Load1", df_actual_load1)
    print_summary("Load2", df_actual_load2)
    print_summary("Load3", df_actual_load3)

    print("\n---- Null (label-shuffled) distribution summary ----")
    print_summary("Load1_null", df_null_load1)
    print_summary("Load2_null", df_null_load2)
    print_summary("Load3_null", df_null_load3)

    # Empirical p-values summary
    print("\n---- Empirical p-values ----")
    for load_name, (act, nul) in zip(['Load1', 'Load2', 'Load3'], load_pairs):
        observed_mean = np.mean(act)
        p_val = np.mean(nul >= observed_mean)
        print(f"{load_name}: Empirical p-value = {p_val:.3e}")

    return df_actual, df_null

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
    """
    Regional version of the pseudo-population decoder with confusion matrices.

    For each brain region (after stripping '_left'/'_right'):
        'dorsal_anterior_cingulate_cortex' -> 'DaCC'
        'pre_supplementary_motor_area'     -> 'PSMA'
        'hippocampus'                      -> 'HPC'
        'amygdala'                         -> 'AMY'
        'ventral_medial_prefrontal_cortex' -> 'vmPFC'

    Steps per region:
      - Build pseudo-population restricted to that region's neurons.
      - Run repeated train/test decoding (Actual).
      - Run label-shuffled null decoding.
      - Compute per-load accuracies and overall accuracy per iteration.
      - Plot an averaged confusion matrix (row-normalized %) for the Actual model.

    Produces:
      1) Region × Load boxplot (Load1/2/3) with significance stars (no outliers).
      2) Region-only overall accuracy boxplot with significance stars (no outliers),
         colored by region using region_colors (vmPFC omitted from plots).
      3) For each region: confusion-matrix heatmap of averaged Actual predictions
         (same style as in repeated_cv_pseudo_population_per_class_colored_outliers_with_label_shuffling).

    Returns
    -------
    df_actual_all : pd.DataFrame
        Columns: ['Region', 'Load', 'Accuracy']
        Load in {'Load1','Load2','Load3','Overall'}; Accuracy in %.
    df_null_all   : pd.DataFrame
        Same columns but for label-shuffled null distributions.
    """

    import numpy as np
    import scipy.io
    from sklearn.utils import resample
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    rng = np.random.default_rng(random_state)

    # --------------------------------------------------------
    # 0) Region map & helper to clean region strings
    # --------------------------------------------------------
    region_map = {
        "dorsal_anterior_cingulate_cortex": "DaCC",
        "pre_supplementary_motor_area": "PSMA",
        "hippocampus": "HPC",
        "amygdala": "AMY",
        "ventral_medial_prefrontal_cortex": "vmPFC",
    }

    # Colors for **second** plot (overall region accuracy)
    region_colors = {
        "hippocampus": "#FFD700",
        "amygdala": "#00FFFF",
        "pre_supplementary_motor_area": "#FF0000",
        "dorsal_anterior_cingulate_cortex": "#0000FF",
        "ventral_medial_prefrontal_cortex": "#008000",
    }
    # Inverse map: acronym -> full-name key used in region_colors
    inverse_region_map = {v: k for k, v in region_map.items()}

    def clean_region(entry_region_field):
        """
        Convert MATLAB/NumPy 'brain_region' field into a clean key
        like 'hippocampus', 'amygdala', etc., with '_left'/'_right' stripped.
        """
        import numpy as np

        reg = entry_region_field

        if isinstance(reg, np.ndarray):
            reg = np.squeeze(reg)
            if reg.dtype.kind in ('U', 'S'):
                reg = ''.join(reg.flat)
            else:
                reg = str(reg)
        else:
            reg = str(reg)

        reg = reg.strip().lower()

        for suffix in ('_left', '_right'):
            if reg.endswith(suffix):
                reg = reg[:-len(suffix)]
                break

        return reg

    def star_label(p):
        if p < 1e-3:
            return "***"
        elif p < 1e-2:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"

    # --------------------------------------------------------
    # 1) Load .mat data
    # --------------------------------------------------------
    mat_data = scipy.io.loadmat(mat_file_path)
    neural_data = mat_data['neural_data']

    # --------------------------------------------------------
    # 2) Build per-region, per-patient structures
    # --------------------------------------------------------
    # region_key -> pid -> list[(frates, loads, tfield)]
    region_patient_neurons = {}
    # region_key -> pid -> dict{1: n, 2: n, 3: n} (max correct-trial count per load)
    region_patient_load_counts = {}

    for entry in neural_data[0]:
        pid = int(entry['patient_id'][0][0])
        if pid not in patient_ids:
            continue

        frates = entry['firing_rates']           # (trials × bins)
        loads = entry['trial_load'].flatten().astype(int)
        correct = entry['trial_correctness'].flatten().astype(int)
        tfield = int(entry['time_field'][0][0]) - 1

        region_key = clean_region(entry['brain_region'])

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

    # --------------------------------------------------------
    # Helper: run decoding for one region
    # --------------------------------------------------------
    def decode_one_region(region_key):
        """
        Build pseudo-population for a single region and run repeated
        decoding with label shuffling. Returns (df_actual_region, df_null_region)
        with Load in {'Load1','Load2','Load3','Overall'} and plots
        a confusion matrix for the Actual model.
        """

        # Patient inclusion for this region
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

        # Gather neurons
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
            if only_correct:
                print(f"[{region_map[region_key]}] No neurons with all loads (only_correct=True). Skipping.")
            else:
                print(f"[{region_map[region_key]}] No neurons with all loads. Skipping.")
            return None, None

        # Global min # of trials per load
        min_load1 = min(len(n['load1']) for n in all_neurons)
        min_load2 = min(len(n['load2']) for n in all_neurons)
        min_load3 = min(len(n['load3']) for n in all_neurons)

        if (min_load1 == 0) or (min_load2 == 0) or (min_load3 == 0):
            print(f"[{region_map[region_key]}] Some load has zero global min. Skipping.")
            return None, None

        # Downsample each neuron
        for neuron in all_neurons:
            neuron['load1'] = resample(neuron['load1'], replace=False,
                                       n_samples=min_load1, random_state=random_state)
            neuron['load2'] = resample(neuron['load2'], replace=False,
                                       n_samples=min_load2, random_state=random_state)
            neuron['load3'] = resample(neuron['load3'], replace=False,
                                       n_samples=min_load3, random_state=random_state)

        # Build matrices
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

        # Actual decoding
        accs_load1_actual = []
        accs_load2_actual = []
        accs_load3_actual = []

        # Confusion-matrix accumulator for this region (Actual only)
        cm_accumulator = np.zeros((3, 3), dtype=float)  # classes: 1,2,3

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
                load2_matrix[test_idx2, :] ,
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

            # Per-class accuracy (%)
            mask1 = (y_test == 1)
            mask2 = (y_test == 2)
            mask3 = (y_test == 3)

            accs_load1_actual.append(np.mean(y_pred[mask1] == 1) * 100)
            accs_load2_actual.append(np.mean(y_pred[mask2] == 2) * 100)
            accs_load3_actual.append(np.mean(y_pred[mask3] == 3) * 100)

            # Confusion matrix for this iteration
            cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3])
            cm_accumulator += cm

        # ---- Confusion matrix plot (per region, Actual model) ----
        # Row-normalize to percentages
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

        try:
            if only_correct:
                path = DIR + 'Correct/'
            else:
                path = DIR
            plt.savefig(path + f'confusionmatrix_{region_map[region_key]}.svg', format='svg')
        except NameError:
            # If DIR is not defined, just skip saving.
            pass

        plt.show()

        # Null decoding (label-shuffled)
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

        # Overall per iteration
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

        # Pack into DataFrames (including 'Overall')
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

        # Console summaries
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

    # --------------------------------------------------------
    # 3) Run decoding per region & aggregate
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 4) Figure 1: Region × Load (Load1/2/3) + stars, no outliers, no vmPFC
    # --------------------------------------------------------
    load_order = ['Load1', 'Load2', 'Load3']
    region_order_all = sorted(df_actual_all['Region'].unique())
    # Exclude vmPFC from plots
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
        showfliers=False   # no outliers
    )

    plt.axhline(y=33.3, color='gray', linestyle='--')
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.xlabel("Brain Region", fontsize=14)
    plt.title("Decoder accuracy by region and load", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Load", fontsize=10)

    # Manual star positions (no reliance on ax.artists)
    offset = 1.0
    n_load = len(load_order)
    group_width = 0.6  # horizontal span for the group of boxes per region

    for i, reg in enumerate(region_order):
        center = i  # seaborn puts region i at x=i

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

    try:
        if only_correct:
            path = DIR + 'Correct/'
        else:
            path = DIR
        plt.savefig(path + 'regional_accuracy_by_load.svg', format='svg')
    except NameError:
        pass

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 5) Figure 2: Region-only overall + stars, no outliers, no vmPFC, custom colors
    # --------------------------------------------------------
    df_overall_actual = df_actual_all[
        (df_actual_all['Load'] == 'Overall') &
        (df_actual_all['Region'] != 'vmPFC')
    ].copy()
    df_overall_null = df_null_all[
        (df_null_all['Load'] == 'Overall') &
        (df_null_all['Region'] != 'vmPFC')
    ].copy()

    region_order_overall = sorted(df_overall_actual['Region'].unique())

    # Build palette mapping acronyms -> hex color using inverse_region_map + region_colors
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
        showfliers=False   # no outliers
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

    try:
        if only_correct:
            path = DIR + 'Correct/'
        else:
            path = DIR
        plt.savefig(path + 'regional_overall_accuracy.svg', format='svg')
    except NameError:
        pass

    plt.tight_layout()
    plt.show()


    # For the region × load plot
    print(
        df_actual_all
        .groupby(['Region', 'Load'])['Accuracy']
        .quantile([0.25, 0.5, 0.75])
        .unstack()  # columns: 0.25, 0.5, 0.75
    )

    return df_actual_all, df_null_all

def repeated_cv_pseudo_population_per_class_colored_outliers_with_label_shuffling_Versus(
    mat_file_path,
    patient_ids=range(1, 22),
    m=0,
    num_windows=0,
    test_per_load=7,
    n_iterations=1000,
    random_state=40,
    only_correct=False,
    alpha=0.05,                 # significance threshold for drawing bars
    n_box_perms=10000,          # #permutations for BETWEEN-BOX empirical tests
    mc_correction="bonferroni", # 'bonferroni' or 'none'
    target_pair="Load1 vs Load3",  # NEW: directional hypothesis target
    alt="greater"                 # NEW: 'greater' means target > comparator
):
    """
    Pair-wise alternative to `repeated_cv_pseudo_population_per_class_colored_outliers_with_label_shuffling`.

    For each iteration the function trains three binary linear-SVM decoders:
      • Load1 vs Load2
      • Load2 vs Load3
      • Load1 vs Load3

    Returns two DataFrames (actual & null accuracies) and a box-plot with:
      1) Empirical significance stars above each box (real vs shuffled-label null).
      2) Empirical BETWEEN-BOX significance bars computed by label-permutation
         between accuracy vectors. NEW: one-sided tests that check whether the
         target_pair has higher mean accuracy than each other box (alt='greater').
    """
    import numpy as np
    import scipy.io
    from sklearn.utils import resample
    from sklearn.svm import SVC
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(random_state)

    # ---------- helpers ----------
    def star_label(p):
        if p < 1e-4: return "****"
        if p < 1e-3: return "***"
        if p < 1e-2: return "**"
        if p < 5e-2: return "*"
        return "n.s."

    def add_sig_bar(ax, x1, x2, y, h, p, fontsize=14, lw=1.5):
        """Draw a significance bar from x1 to x2 at base y with height h."""
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=lw, c='black')
        ax.text((x1 + x2) / 2, y + h, star_label(p), ha='center', va='bottom',
                fontsize=fontsize, fontweight='bold')

    def perm_p_value_between_boxes_one_sided(x, y, n_perm=10000, rng=None, alt="greater"):
        """
        Empirical, one-sided permutation p-value for difference in means between
        two accuracy vectors x (target) and y (comparator). Exchange labels across
        pooled sample (exchangeability). (+1 smoothing)

        alt='greater' tests mean(x) > mean(y).
        """
        rng = np.random.default_rng() if rng is None else rng
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n_x = len(x)
        pooled = np.concatenate([x, y])
        obs = x.mean() - y.mean()
        count = 0
        for _ in range(n_perm):
            rng.shuffle(pooled)
            x_ = pooled[:n_x]
            y_ = pooled[n_x:]
            stat = x_.mean() - y_.mean()
            # one-sided tail
            if alt == "greater":
                extreme = (stat >= obs)
            elif alt == "less":
                extreme = (stat <= obs)
            else:
                # fallback to two-sided absolute difference
                extreme = (abs(stat) >= abs(obs))
            if extreme:
                count += 1
        return (count + 1) / (n_perm + 1)

    # -------------------------------
    # 1) Load .mat data
    # -------------------------------
    mat_data = scipy.io.loadmat(mat_file_path)
    neural_data = mat_data['neural_data']

    # -------------------------------
    # 2) Build per-patient structures
    # -------------------------------
    patient_dict = {}
    patient_load_counts = {}

    for entry in neural_data[0]:
        pid       = int(entry['patient_id'][0][0])
        frates    = entry['firing_rates']                    # (trials × bins)
        loads     = entry['trial_load'].flatten().astype(int)
        correct   = entry['trial_correctness'].flatten().astype(int)
        tfield    = int(entry['time_field'][0][0]) - 1

        if only_correct:
            keep_idx = (correct == 1)
            frates   = frates[keep_idx, :]
            loads    = loads[keep_idx]

        if frates.size == 0:
            continue

        if pid not in patient_dict:
            patient_dict[pid]        = []
            patient_load_counts[pid] = {1: 0, 2: 0, 3: 0}

        patient_dict[pid].append((frates, loads, tfield))

        for ld in (1, 2, 3):
            patient_load_counts[pid][ld] = max(
                patient_load_counts[pid][ld],
                int(np.sum(loads == ld))
            )

    # -------------------------------
    # 3) Patient-level inclusion filter
    # -------------------------------
    valid_patients = []
    for pid in patient_ids:
        if pid not in patient_dict:
            continue
        if len(patient_dict[pid]) < m:
            continue
        if only_correct:
            counts = patient_load_counts[pid]
            if not (counts[1] >= 25 and counts[2] >= 25 and counts[3] >= 25):
                continue
        valid_patients.append(pid)

    if len(valid_patients) == 0:
        print("No patients meet the inclusion criteria "
              f"(m={m}, ≥25 correct trials per load = {only_correct}).")
        return None

    # -------------------------------
    # 4) Gather valid neurons
    # -------------------------------
    all_neurons = []
    for pid in valid_patients:
        for (frates, loads, tfield) in patient_dict[pid]:
            start_idx = max(0, tfield - num_windows)
            end_idx   = min(frates.shape[1], tfield + num_windows + 1)

            # average across the selected window
            windowed    = frates[:, start_idx:end_idx]
            mean_rates  = np.mean(windowed, axis=1)

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
        print("No neurons found with all three loads present." if not only_correct
              else "No neurons found (all loads) with only-correct trials.")
        return None

    # -------------------------------
    # 5) Global min # of trials per load
    # -------------------------------
    min_load1 = min(len(n['load1']) for n in all_neurons)
    min_load2 = min(len(n['load2']) for n in all_neurons)
    min_load3 = min(len(n['load3']) for n in all_neurons)

    if (min_load1 == 0) or (min_load2 == 0) or (min_load3 == 0):
        print("Some load has zero global min. Exiting.")
        return None

    # -------------------------------
    # 6) Down-sample each neuron once
    # -------------------------------
    from sklearn.utils import resample as _resample
    for neuron in all_neurons:
        neuron['load1'] = _resample(neuron['load1'], replace=False,
                                    n_samples=min_load1, random_state=random_state)
        neuron['load2'] = _resample(neuron['load2'], replace=False,
                                    n_samples=min_load2, random_state=random_state)
        neuron['load3'] = _resample(neuron['load3'], replace=False,
                                    n_samples=min_load3, random_state=random_state)

    # -------------------------------
    # 7) Build big matrices
    # -------------------------------
    num_neurons   = len(all_neurons)
    load1_matrix  = np.zeros((min_load1, num_neurons), dtype=np.float32)
    load2_matrix  = np.zeros((min_load2, num_neurons), dtype=np.float32)
    load3_matrix  = np.zeros((min_load3, num_neurons), dtype=np.float32)

    for j, neuron in enumerate(all_neurons):
        load1_matrix[:, j] = neuron['load1']
        load2_matrix[:, j] = neuron['load2']
        load3_matrix[:, j] = neuron['load3']

    print(f"--- Pseudopopulation Info (only_correct={only_correct}) ---")
    print(f"  Number of neurons:           {num_neurons}")
    print(f"  Final trials per load: L1={min_load1}, L2={min_load2}, L3={min_load3}")

    if test_per_load > min(min_load1, min_load2, min_load3):
        print(f"Requested test_per_load={test_per_load} exceeds available trials. Exiting.")
        return None

    # ==========================================================
    # 8) Repeated sub-sampling – pair-wise binary decoders
    # ==========================================================
    pair_info = [
        ('Load1','Load2', 1, 2),
        ('Load2','Load3', 2, 3),
        ('Load1','Load3', 1, 3)
    ]

    # Containers: dict[ pair_name_str ] -> list[accuracies]
    acc_real = {p[0]+' vs '+p[1]: [] for p in pair_info}
    acc_null = {p[0]+' vs '+p[1]: [] for p in pair_info}

    for _ in range(n_iterations):
        # ---- draw a fresh train/test split per load (indices) ----
        idx_L1 = rng.permutation(min_load1)
        idx_L2 = rng.permutation(min_load2)
        idx_L3 = rng.permutation(min_load3)

        tst_L1, trn_L1 = idx_L1[:test_per_load], idx_L1[test_per_load:]
        tst_L2, trn_L2 = idx_L2[:test_per_load], idx_L2[test_per_load:]
        tst_L3, trn_L3 = idx_L3[:test_per_load], idx_L3[test_per_load:]

        Xtrain = {
            'Load1': load1_matrix[trn_L1, :],
            'Load2': load2_matrix[trn_L2, :],
            'Load3': load3_matrix[trn_L3, :]
        }
        Xtest = {
            'Load1': load1_matrix[tst_L1, :],
            'Load2': load2_matrix[tst_L2, :],
            'Load3': load3_matrix[tst_L3, :]
        }

        # ---- loop over the three load pairs ----
        for loadA, loadB, _, _ in pair_info:
            pair_name = f"{loadA} vs {loadB}"

            # ---------- real labels ----------
            X_tr = np.vstack([Xtrain[loadA], Xtrain[loadB]])
            y_tr = np.hstack([np.zeros(len(Xtrain[loadA])), np.ones(len(Xtrain[loadB]))])

            X_te = np.vstack([Xtest[loadA], Xtest[loadB]])
            y_te = np.hstack([np.zeros(len(Xtest[loadA])), np.ones(len(Xtest[loadB]))])

            clf = SVC(kernel='linear', random_state=random_state)
            clf.fit(X_tr, y_tr)
            acc_real[pair_name].append(clf.score(X_te, y_te) * 100)

            # ---------- null (label-shuffled on train set) ----------
            y_tr_shuff = rng.permutation(y_tr)
            clf_null   = SVC(kernel='linear', random_state=random_state)
            clf_null.fit(X_tr, y_tr_shuff)
            acc_null[pair_name].append(clf_null.score(X_te, y_te) * 100)

    # ==========================================================
    # 9) Tidy into DataFrames
    # ==========================================================
    rows_real = []
    rows_null = []
    for pair_name in acc_real.keys():
        for v in acc_real[pair_name]:
            rows_real.append({'Pair': pair_name, 'Accuracy': v})
        for v in acc_null[pair_name]:
            rows_null.append({'Pair': pair_name, 'Accuracy': v})

    df_real = pd.DataFrame(rows_real)
    df_null = pd.DataFrame(rows_null)

    # ==========================================================
    # 10) Box-plot with empirical stars (real vs its null)
    #     + NEW: Directional BETWEEN-BOX bars (target > others)
    # ==========================================================
    plt.figure(figsize=(7.5, 7.5))
    palette = ['steelblue', 'seagreen', 'indianred']
    ax = sns.boxplot(data=df_real, x='Pair', y='Accuracy',
                     palette=palette, width=0.6, showfliers=True)

    plt.axhline(50, ls='--', c='gray')  # chance for binary

    ax.set_ylabel("Accuracy (%)", fontsize=16)
    ax.set_xlabel("Pair", fontsize=16)
    ax.set_title(f"Pair-wise decoding ({n_iterations:,} CV iterations)", fontsize=16)
    ax.tick_params(axis='both', labelsize=13)

    # ----- empirical p-value & stars above each box (real vs its null) -----
    ymax = df_real['Accuracy'].max()
    offset = 1.0
    for i, pair_name in enumerate(acc_real.keys()):
        real_vals = np.array(acc_real[pair_name])
        null_vals = np.array(acc_null[pair_name])
        # empirical p: how often null >= mean(real)
        p_emp = (np.sum(null_vals >= real_vals.mean())) / (len(null_vals))
        ax.text(i, ymax + offset, star_label(p_emp),
                ha='center', va='bottom', fontsize=16, fontweight='bold')

        # ----- NEW: Empirical BETWEEN-BOX tests (target real vs mean of other boxes' real) -----
    groups = list(acc_real.keys())  # ['Load1 vs Load2','Load2 vs Load3','Load1 vs Load3']
    if target_pair not in groups:
        raise ValueError(f"target_pair '{target_pair}' not found in groups {groups}")

    idx_map = {g: i for i, g in enumerate(groups)}
    others = [g for g in groups if g != target_pair]

    target_vals = np.array(acc_real[target_pair], dtype=float)

    raw_p = []
    pair_list = []
    for g2 in others:
        # Empirical one-sided test: how often L1vsL3 accuracies fall under the mean of the comparator's real accuracies
        comp_mean = np.mean(np.array(acc_real[g2], dtype=float))
        p = (np.sum(target_vals <= comp_mean)) / (len(target_vals))
        raw_p.append(p)
        pair_list.append((target_pair, g2))

    # Multiple-comparisons correction across the two tests
    if mc_correction.lower() == "bonferroni":
        corr_p = [min(p * len(raw_p), 1.0) for p in raw_p]
    else:
        corr_p = raw_p

    # Draw bars from target to each comparator when significant
    y_base = ymax + offset + 1.6
    h = 1.0
    for k, ((g1, g2), p_corr) in enumerate(zip(pair_list, corr_p)):
        if p_corr < alpha:
            x1, x2 = idx_map[g1], idx_map[g2]
            add_sig_bar(ax, x1, x2, y=y_base + k * (h + 0.25), h=h, p=p_corr)

    plt.tight_layout()
    plt.show()

    # ==========================================================
    # 11) Numeric summaries (empirical)
    # ==========================================================
    print("---- Pair-wise decoding summary (real vs null; empirical p) ----")
    for pair_name in acc_real.keys():
        r = np.array(acc_real[pair_name])
        n_ = np.array(acc_null[pair_name])
        p_emp = (np.sum(n_ >= r.mean())) / (len(n_))
        print(f"{pair_name:11s} | mean={r.mean():6.2f} %, std={r.std():5.2f} %, empirical p={p_emp:.3e}")

    print("\n---- Directional between-box permutation tests ----")
    for (g1, g2), p_raw, p_corr in zip(pair_list, raw_p, corr_p):
        adj_note = " (Bonferroni)" if mc_correction.lower() == "bonferroni" else ""
        direction = ">" if alt == "greater" else "<" if alt == "less" else "!= "
        print(f"{g1}  {direction}  {g2}:  p={p_raw:.3e} | p{adj_note}={p_corr:.3e} | {star_label(p_corr)}")

    print("\n---- Empirical between-box tests: P(L1vsL3 <= mean(other box)) ----")
    for (g1, g2), p_raw, p_corr in zip(pair_list, raw_p, corr_p):
        comp_mean = np.mean(np.array(acc_real[g2], dtype=float))
        direction = ">"  # testing whether target is greater than comparator mean
        adj_note = " (Bonferroni)" if mc_correction.lower() == "bonferroni" else ""
        print(f"{g1} {direction} mean({g2}): p={p_raw:.3e} | p{adj_note}={p_corr:.3e}")


    return df_real, df_null

# repeated_cv_pseudo_population_per_class_colored_outliers_with_label_shuffling('3sig15_data.mat', test_per_load=10, only_correct=False, random_state=20250710, n_iterations= 1000)
#repeated_cv_pseudo_population_per_class_colored_outliers_with_label_shuffling_Versus('3sig15_raw.mat', test_per_load= 11, only_correct=False, random_state=20250710, n_iterations= 1000)
repeated_cv_pseudo_population_per_class_by_region('3sig15_raw.mat', test_per_load= 11, only_correct=False, random_state=20250710, n_iterations= 1000)

#repeated_cv_pseudo_population_per_class_by_region('3sig15_raw.mat', test_per_load= 6, only_correct=True, random_state=20251205, n_iterations= 1000)

# repeated_cv_pseudo_population_per_class_colored_outliers_with_label_shuffling('3sig15_raw.mat', test_per_load= 6, only_correct=True, random_state=20250710, n_iterations= 1000)
#repeated_cv_pseudo_population_per_class_colored_outliers_with_label_shuffling_Versus('3sig15_raw.mat', test_per_load= 6, only_correct=True, random_state=20250710, n_iterations= 1000)
#repeated_cv_pseudo_population_per_class_colored_outliers_with_label_shuffling('3sig15_data.mat', test_per_load= 7, only_correct=False, random_state=42, n_iterations= 1000)
# repeated_cv_pseudo_population_per_class_colored_outliers('100msTCdata.mat')
#loocv_with_permutation_test('100msTCdata.mat')
