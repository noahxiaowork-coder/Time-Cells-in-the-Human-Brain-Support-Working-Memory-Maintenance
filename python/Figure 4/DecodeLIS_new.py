import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.utils import resample
from scipy import stats
from scipy.stats import mannwhitneyu
from itertools import combinations
import numpy as np
DIR = '/Users/xiangxuankong/Desktop/Human TC Support WM/Figures/Figure 4/'


def _matlab_to_str(x):
    """Robustly convert MATLAB-loaded strings/cells/char into a Python str."""
    import numpy as np

    # bytes -> str
    if isinstance(x, (bytes, bytearray)):
        return x.decode('utf-8', 'ignore')

    # already a str
    if isinstance(x, str):
        return x

    # char arrays / string arrays -> collapse to single string
    if isinstance(x, np.ndarray):
        # Empty array
        if x.size == 0:
            return ''
        # Pure text dtypes (unicode/bytes)
        if x.dtype.kind in ('U', 'S'):
            # single scalar char array/string array
            if x.ndim == 0 or x.size == 1:
                return str(x.item())
            # 2D char array (MATLAB char) → join columns in Fortran order
            return ''.join(np.asarray(x).astype(str).ravel(order='F'))
        # Object arrays (cells/strings)
        if x.dtype == object:
            # peel nested layers until we reach a non-array/object
            y = x
            while isinstance(y, np.ndarray) and y.dtype == object and y.size == 1:
                y = y.item()
            # if still array-like, take the first element
            if isinstance(y, np.ndarray) and y.size > 0:
                y = y.ravel()[0]
            return _matlab_to_str(y)

    # MATLAB structured scalar (numpy.void) – try to coerce its contents
    try:
        import numpy as np
        if isinstance(x, np.void):
            # View as object array and recurse
            return _matlab_to_str(np.array(x, dtype=object))
    except Exception:
        pass

    # Fallback
    return str(x)

def decode_last_imageID_load1_multiclass_ttest(
    mat_file_path,
    patient_ids=range(1, 22),
    m=0,
    num_windows=0,
    test_per_class=2,
    n_iterations=1000,
    n_shuffles=1,
    random_state=20250710
):
    """
    Decodes the 'last non-zero image ID' (1..5) within Load=1 trials,
    using a multi-class SVM with repeated sub-sampling. Then uses an
    independent-samples t-test (real vs. null) to determine significance.

    Parameters
    ----------
    mat_file_path : str
        Path to the .mat file containing 'neural_data'.
    patient_ids : iterable
        Which patient IDs to include.
    m : int
        Minimum number of time cells for a patient to be included.
    num_windows : int
        How many time bins on each side of 'time_field' to average over.
    test_per_class : int
        How many test trials for each of the 5 image IDs in each iteration.
    n_iterations : int
        How many times to repeat the sub-sampling cross-validation (real labels).
    n_shuffles : int
        How many times to generate a full null distribution.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    real_accuracies : list of float
        The per-iteration accuracy (in %) for the real labels cross-validation.
    null_accuracies : list of float
        The entire distribution of accuracies (in %) for all null runs.
    p_value : float
        p-value from a two-sample t-test comparing real vs. null distributions.
    """

    rng = np.random.default_rng(random_state)

    # 1) Load .mat data
    mat_data = loadmat(mat_file_path)
    neural_data = mat_data['neural_data']

    # 2) Group by patient
    patient_dict = {}
    for entry in neural_data[0]:
        pid = int(entry['patient_id'][0][0])

        frates  = entry['firing_rates']           # shape: (num_trials, num_time_bins)
        loads   = entry['trial_load'].flatten()   # shape: (num_trials,)
        tfield  = int(entry['time_field'][0][0]) - 1  # make zero-based
        img_ids = entry['trial_imageIDs']         # shape: (num_trials, 3)

        if pid not in patient_dict:
            patient_dict[pid] = []
        patient_dict[pid].append((frates, loads, tfield, img_ids))

    # 3) Filter patients by 'm'
    valid_patients = []
    for pid in patient_ids:
        if pid in patient_dict and len(patient_dict[pid]) >= m:
            valid_patients.append(pid)
    if len(valid_patients) == 0:
        print(f"No patients meet the minimum {m} time cells.")
        return [], [], np.nan

    # 4) Gather valid neurons (only Load=1 trials; last non-zero in 1..5)
    all_neurons = []
    for pid in valid_patients:
        for (frates, loads, tfield, img_ids) in patient_dict[pid]:
            # (A) average the firing rate in [tfield-num_windows, ..., tfield+num_windows]
            start_idx = max(0, tfield - num_windows)
            end_idx   = min(frates.shape[1], tfield + num_windows + 1)
            windowed  = frates[:, start_idx:end_idx]
            mean_rates = np.mean(windowed, axis=1)  # shape: (num_trials,)

            # (B) only keep trials with load=1
            mask_load1 = (loads == 1)

            # (C) find the last non-zero in each row of img_ids
            labels_all = []

            for row in img_ids:
                row_ = row.flatten()
                nonzeros = row_[row_ != 0]
                if len(nonzeros) > 0:
                    labels_all.append(nonzeros[-1])  # last nonzero
                else:
                    labels_all.append(0)  # no image
            labels_all = np.array(labels_all)


            # only keep labels in {1..5}
            mask_valid_img = (labels_all >= 1) & (labels_all <= 5)

            final_mask = mask_load1 & mask_valid_img
            final_rates = mean_rates[final_mask]
            final_labels = labels_all[final_mask]  # each in {1..5}

            # (D) check if it has at least 1 trial for each label in 1..5
            keep_it = True
            for c in [1, 2, 3, 4, 5]:
                if np.sum(final_labels == c) < 1:
                    keep_it = False
                    break

            if keep_it:
                # store data for each of the 5 classes
                neuron_dict = {}
                for c in [1,2,3,4,5]:
                    neuron_dict[f'class{c}'] = final_rates[final_labels == c]
                all_neurons.append(neuron_dict)

    if len(all_neurons) == 0:
        print("No neurons found that have Load=1 trials with all 5 imageIDs present.")
        return [], [], np.nan

    # 5) Find global min # trials per class so we can downsample each neuron
    min_counts = []
    for c in [1,2,3,4,5]:
        key = f'class{c}'
        counts = [len(n[key]) for n in all_neurons]
        min_counts.append(min(counts))
    # Ensure we have enough to do test_per_class
    for c_i, cval in enumerate([1,2,3,4,5], 0):
        if min_counts[c_i] < test_per_class:
            print(f"Cannot proceed: class={cval}, global min (#trials)={min_counts[c_i]} < test_per_class={test_per_class}")
            return [], [], np.nan

    # 6) Downsample each neuron to the global min for each class
    for neuron in all_neurons:
        for c_i, cval in enumerate([1,2,3,4,5], 0):
            key = f'class{cval}'
            neuron[key] = resample(
                neuron[key],
                replace=False,
                n_samples=min_counts[c_i],
                random_state=random_state
            )

    # 7) Build big matrices (rows=trials, cols=neurons) for each class
    num_neurons = len(all_neurons)
    class_matrices = {}
    for c_i, cval in enumerate([1,2,3,4,5], 0):
        key = f'class{cval}'
        mat = np.zeros((min_counts[c_i], num_neurons), dtype=np.float32)
        for j, neuron in enumerate(all_neurons):
            mat[:, j] = neuron[key]
        class_matrices[key] = mat

    # 8) Repeated sub-sampling (Real Labels)
    real_accuracies = []
    for _ in range(n_iterations):
        # Per iteration, sample test_per_class from each class
        X_train_list = []
        y_train_list = []
        X_test_list  = []
        y_test_list  = []

        for cval in [1,2,3,4,5]:
            key = f'class{cval}'
            mat = class_matrices[key]
            n_c = mat.shape[0]

            idx_c = rng.permutation(n_c)
            test_idx_c  = idx_c[:test_per_class]
            train_idx_c = idx_c[test_per_class:]

            X_train_list.append(mat[train_idx_c, :])
            y_train_list.append(np.full(len(train_idx_c), cval))

            X_test_list.append(mat[test_idx_c, :])
            y_test_list.append(np.full(len(test_idx_c), cval))

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_test  = np.vstack(X_test_list)
        y_test  = np.concatenate(y_test_list)

        clf = SVC(kernel='linear', random_state=random_state, decision_function_shape='ovr')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = np.mean(y_pred == y_test) * 100.0
        real_accuracies.append(acc)

    # 9) Generate null distribution: same pipeline but with label-shuffled data
    #    We'll store ALL the per-iteration accuracies in null_accuracies (not just means).
    null_accuracies = []
    for shuffle_i in range(n_shuffles):
        for _ in range(n_iterations):
            X_train_list = []
            y_train_list = []
            X_test_list  = []
            y_test_list  = []

            for cval in [1,2,3,4,5]:
                key = f'class{cval}'
                mat = class_matrices[key]
                n_c = mat.shape[0]

                idx_c = rng.permutation(n_c)
                test_idx_c  = idx_c[:test_per_class]
                train_idx_c = idx_c[test_per_class:]

                X_train_list.append(mat[train_idx_c, :])
                # real label would be cval, but will shuffle below
                y_train_list.append(np.full(len(train_idx_c), cval))

                X_test_list.append(mat[test_idx_c, :])
                y_test_list.append(np.full(len(test_idx_c), cval))

            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)
            X_test  = np.vstack(X_test_list)
            y_test  = np.concatenate(y_test_list)

            # Shuffle y_train in-place
            y_train = rng.permutation(y_train)

            clf_null = SVC(kernel='linear', random_state=random_state, decision_function_shape='ovr')
            clf_null.fit(X_train, y_train)
            y_pred_null = clf_null.predict(X_test)

            acc_null = np.mean(y_pred_null == y_test) * 100.0
            null_accuracies.append(acc_null)

    # 10) Statistical test (t-test) comparing real vs. null
    #     For decoding, we often do a one-sided test: real > null
    #     You can use alternative='greater' in scipy 1.6+.
    #     If older versions, you can do a two-sided and interpret accordingly.


    #     Below is the two-sided version for broad compatibility:
    # t_stat, p_value = stats.ttest_ind(real_accuracies, null_accuracies, alternative="greater")

    #Empirical version
    # real_accuracies = np.array(real_accuracies)
    # null_accuracies = np.array(null_accuracies)

    observed_mean = np.mean(real_accuracies)
    p_value = np.mean(null_accuracies >= observed_mean)

    # If you specifically want "real > null", you can do:
    #    p_value = stats.ttest_ind(real_accuracies, null_accuracies, equal_var=False).pvalue / 2
    #    # but only if t_stat is in the right direction

    # 11) Make a box plot for real_accuracies
    plt.figure(figsize=(5, 8))
    # Using basic matplotlib boxplot (instead of seaborn) to keep it simple
    box = plt.boxplot(real_accuracies, labels=[''], patch_artist=True)

    # Customize box color and outliers
    for element in ['boxes', 'whiskers', 'caps', 'medians']:
        plt.setp(box[element], color='blue')

    # Fill the box with a lighter blue (optional)
    for patch in box['boxes']:
        patch.set(facecolor='lightblue')

    # Set outlier marker (fliers) color to blue
    for flier in box['fliers']:
        flier.set(marker='o', color='blue', alpha=0.8)

    plt.ylabel("Accuracy (%)", fontsize = 20)
    plt.title("Decoding the Single Encoded Image")

    # Draw chance line for 5-way classification = 20%
    plt.axhline(y=20, linestyle='--')
    

    # Optionally add significance annotation
    max_acc = max(real_accuracies) if real_accuracies else 100
    y_star = max_acc + 5  # position above the box

    # Define the significance level and corresponding stars
    if p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'
    else:
        stars = ''

    # Only add text if statistically significant
    if stars:
        plt.text(1, y_star, stars, ha='center', va='bottom', fontsize=20)
    plt.ylim([0, y_star + 5])
    plt.tight_layout()
    plt.savefig(DIR + 'decdoing.svg', format = 'svg')
    plt.show()

    # Print results
    mean_acc = np.mean(real_accuracies)
    sd_acc  = np.std(real_accuracies, ddof=1)
    print(f"Real cross-validation accuracy = {mean_acc:.2f}% ± {sd_acc:.2f}% (mean ± s.d., N={n_iterations})")
    print(f"Null distribution size = {len(null_accuracies)}")
    print(f"t-test p-value = {p_value:.5g}")

    return real_accuracies, null_accuracies, p_value

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.utils import resample
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.utils import resample
from scipy import stats

def decode_last_imageID_load123_multiclass_ttest(
    mat_file_path,
    patient_ids=range(1, 22),
    m=0,
    num_windows=0,
    test_per_class=2,
    n_iterations=1000,
    n_shuffles=1,
    random_state=42
):
    """
    Decodes the 'last non-zero image ID' (1..5) for load=1, 2, 3, 
    using a multi-class SVM trained ONLY on load=1 data. For each iteration:
      - Subsample train/test from load=1 (test_per_class for the test set).
      - Use ALL (downsampled) trials from load=2 and load=3 as test sets.
      - Also generate separate null distributions (shuffle labels in the L1 training set),
        and evaluate on the full L2, L3 sets.
    Finally, returns the real/null accuracies for loads 1..3 and p-values from t-tests.
    """

    rng = np.random.default_rng(random_state)

    # 1) Load .mat data
    mat_data = loadmat(mat_file_path)
    neural_data = mat_data['neural_data']

    # 2) Group by patient
    patient_dict = {}
    for entry in neural_data[0]:
        pid = int(entry['patient_id'][0][0])

        frates  = entry['firing_rates']           # shape: (num_trials, num_time_bins)
        loads   = entry['trial_load'].flatten()   # shape: (num_trials,)
        tfield  = int(entry['time_field'][0][0]) - 1  # zero-based
        img_ids = entry['trial_imageIDs']         # shape: (num_trials, 3)

        if pid not in patient_dict:
            patient_dict[pid] = []
        patient_dict[pid].append((frates, loads, tfield, img_ids))

    # 3) Filter patients by 'm'
    valid_patients = []
    for pid in patient_ids:
        if pid in patient_dict and len(patient_dict[pid]) >= m:
            valid_patients.append(pid)
    if len(valid_patients) == 0:
        print(f"No patients meet the minimum {m} time cells.")
        return [], [], [], [], [], [], {'load1': np.nan, 'load2': np.nan, 'load3': np.nan}

    # -------------------------------------------------------------------------
    # HELPER FUNCTION to extract and average trials for a given load L
    # Returns a dict with 'class1'..'class5' if >=1 trial per class, else None.
    # -------------------------------------------------------------------------
    def get_load_dict(frates, loads, tfield, img_ids, load_value, num_windows):
        # Average firing around tfield
        start_idx = max(0, tfield - num_windows)
        end_idx   = min(frates.shape[1], tfield + num_windows + 1)
        windowed  = frates[:, start_idx:end_idx]
        mean_rates = np.mean(windowed, axis=1)  # shape: (num_trials,)

        # keep only trials with load=load_value
        load_mask = (loads == load_value)

        # find last non-zero image
        labels_all = []
        for row in img_ids:
            nonzeros = row[row != 0]
            if len(nonzeros) > 0:
                labels_all.append(nonzeros[0])  # last nonzero
            else:
                labels_all.append(0)
        labels_all = np.array(labels_all)

        # keep labels in 1..5
        valid_img_mask = (labels_all >= 1) & (labels_all <= 5)
        final_mask = load_mask & valid_img_mask

        final_rates = mean_rates[final_mask]
        final_labels = labels_all[final_mask]

        # check if we have at least 1 trial for each label 1..5
        for c in [1,2,3,4,5]:
            if np.sum(final_labels == c) < 1:
                return None  # doesn't meet requirement

        # build dict
        load_dict = {}
        for c in [1,2,3,4,5]:
            load_dict[f'class{c}'] = final_rates[final_labels == c]
        return load_dict

    # 4) For each valid patient entry, gather data for load=1,2,3 
    #    Keep only if the neuron passes for all 3 loads
    all_neurons = []
    for pid in valid_patients:
        for (frates, loads, tfield, img_ids) in patient_dict[pid]:
            ld1 = get_load_dict(frates, loads, tfield, img_ids, load_value=1, num_windows=num_windows)
            ld2 = get_load_dict(frates, loads, tfield, img_ids, load_value=2, num_windows=num_windows)
            ld3 = get_load_dict(frates, loads, tfield, img_ids, load_value=3, num_windows=num_windows)

            if (ld1 is not None) and (ld2 is not None) and (ld3 is not None):
                neuron_dict = {}
                for c in [1,2,3,4,5]:
                    neuron_dict[f'l1_class{c}'] = ld1[f'class{c}']
                    neuron_dict[f'l2_class{c}'] = ld2[f'class{c}']
                    neuron_dict[f'l3_class{c}'] = ld3[f'class{c}']
                all_neurons.append(neuron_dict)

    if len(all_neurons) == 0:
        print("No neurons found that have >=1 trial for each imageID in {1..5} across loads 1,2,3.")
        return [], [], [], [], [], [], {'load1': np.nan, 'load2': np.nan, 'load3': np.nan}

    # 5) Global min # trials for each load & class => so we can downsample
    min_counts_l1 = []
    min_counts_l2 = []
    min_counts_l3 = []
    for c in [1,2,3,4,5]:
        key_l1 = f'l1_class{c}'
        key_l2 = f'l2_class{c}'
        key_l3 = f'l3_class{c}'

        counts_l1 = [len(n[key_l1]) for n in all_neurons]
        counts_l2 = [len(n[key_l2]) for n in all_neurons]
        counts_l3 = [len(n[key_l3]) for n in all_neurons]

        min_counts_l1.append(min(counts_l1))
        min_counts_l2.append(min(counts_l2))
        min_counts_l3.append(min(counts_l3))

    # Check we have enough for test_per_class in L1
    for c_i, cval in enumerate([1,2,3,4,5], 0):
        if min_counts_l1[c_i] < test_per_class:
            print(f"Cannot proceed: (Load=1) class={cval}, min(#trials)={min_counts_l1[c_i]} < test_per_class={test_per_class}")
            return [], [], [], [], [], [], {'load1': np.nan, 'load2': np.nan, 'load3': np.nan}
        # We don't strictly need to require test_per_class from L2, L3 
        # if we want to use "all" as test. But let's at least check there's >=1:
        if min_counts_l2[c_i] < 1:
            print(f"Cannot proceed: (Load=2) class={cval} has zero trials after filtering.")
            return [], [], [], [], [], [], {'load1': np.nan, 'load2': np.nan, 'load3': np.nan}
        if min_counts_l3[c_i] < 1:
            print(f"Cannot proceed: (Load=3) class={cval} has zero trials after filtering.")
            return [], [], [], [], [], [], {'load1': np.nan, 'load2': np.nan, 'load3': np.nan}

    # 6) Downsample each neuron
    for neuron in all_neurons:
        for c_i, cval in enumerate([1,2,3,4,5], 0):
            k1 = f'l1_class{cval}'
            k2 = f'l2_class{cval}'
            k3 = f'l3_class{cval}'
            neuron[k1] = resample(
                neuron[k1], replace=False, n_samples=min_counts_l1[c_i], random_state=random_state
            )
            neuron[k2] = resample(
                neuron[k2], replace=False, n_samples=min_counts_l2[c_i], random_state=random_state
            )
            neuron[k3] = resample(
                neuron[k3], replace=False, n_samples=min_counts_l3[c_i], random_state=random_state
            )

    # 7) Build big matrices (rows=trials, cols=neurons) for each load
    num_neurons = len(all_neurons)
    class_matrices_l1 = {}
    class_matrices_l2 = {}
    class_matrices_l3 = {}
    for c in [1,2,3,4,5]:
        mat_l1 = np.zeros((0, num_neurons), dtype=np.float32)
        mat_l2 = np.zeros((0, num_neurons), dtype=np.float32)
        mat_l3 = np.zeros((0, num_neurons), dtype=np.float32)

        for j, neuron in enumerate(all_neurons):
            # shape: (nSamples_for_that_class, )
            arr_l1 = neuron[f'l1_class{c}'][:, None]  # make columns for hstack
            arr_l2 = neuron[f'l2_class{c}'][:, None]
            arr_l3 = neuron[f'l3_class{c}'][:, None]

            # We'll add them as columns later. We want rows=trials, cols=neurons
            # Actually easier to build them class-wise then we combine by columns
            # So let's accumulate them properly...
            pass

        # Actually let's do the same approach as the original code, 
        # building each class's matrix with shape (#trials_in_that_class, #neurons)
        n_c_l1 = len(all_neurons[0][f'l1_class{c}'])  # but that varies across neurons if we haven't forced them to be same? We did: min_counts_l1!
        mat_l1 = np.zeros((min_counts_l1[c-1], num_neurons), dtype=np.float32)
        mat_l2 = np.zeros((min_counts_l2[c-1], num_neurons), dtype=np.float32)
        mat_l3 = np.zeros((min_counts_l3[c-1], num_neurons), dtype=np.float32)

        for j, neuron in enumerate(all_neurons):
            mat_l1[:, j] = neuron[f'l1_class{c}']
            mat_l2[:, j] = neuron[f'l2_class{c}']
            mat_l3[:, j] = neuron[f'l3_class{c}']

        class_matrices_l1[f'class{c}'] = mat_l1
        class_matrices_l2[f'class{c}'] = mat_l2
        class_matrices_l3[f'class{c}'] = mat_l3

    # Pre-build the "entire test set" for Load=2 and Load=3: 
    # We'll use ALL trials for each class. That means we just stack them up.
    # We'll do that once outside the iteration, for speed and consistency.
    allX_l2_list = []
    allY_l2_list = []
    allX_l3_list = []
    allY_l3_list = []
    for cval in [1,2,3,4,5]:
        mat2 = class_matrices_l2[f'class{cval}']
        mat3 = class_matrices_l3[f'class{cval}']

        allX_l2_list.append(mat2)
        allY_l2_list.append(np.full(mat2.shape[0], cval))

        allX_l3_list.append(mat3)
        allY_l3_list.append(np.full(mat3.shape[0], cval))

    X_test_l2_full = np.vstack(allX_l2_list)
    y_test_l2_full = np.concatenate(allY_l2_list)
    X_test_l3_full = np.vstack(allX_l3_list)
    y_test_l3_full = np.concatenate(allY_l3_list)

    # 8) Sub-sampling on Load=1 each iteration, but for L2/L3 we use all data
    real_accs_l1 = []
    real_accs_l2 = []
    real_accs_l3 = []

    for _ in range(n_iterations):
        # Build L1 train/test
        X_train_list = []
        y_train_list = []
        X_test_l1_list = []
        y_test_l1_list = []

        for cval in [1,2,3,4,5]:
            mat1 = class_matrices_l1[f'class{cval}']
            n1   = mat1.shape[0]
            idx1 = rng.permutation(n1)

            test_idx1  = idx1[:test_per_class]
            train_idx1 = idx1[test_per_class:]

            X_train_list.append(mat1[train_idx1, :])
            y_train_list.append(np.full(len(train_idx1), cval))

            X_test_l1_list.append(mat1[test_idx1, :])
            y_test_l1_list.append(np.full(len(test_idx1), cval))

        # Combine for training
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)

        # Combine for L1 test
        X_test_l1 = np.vstack(X_test_l1_list)
        y_test_l1 = np.concatenate(y_test_l1_list)

        # For L2, L3, we use the entire set:
        X_test_l2 = X_test_l2_full
        y_test_l2 = y_test_l2_full
        X_test_l3 = X_test_l3_full
        y_test_l3 = y_test_l3_full

        # Train real SVM
        clf = SVC(kernel='linear', random_state=random_state, decision_function_shape='ovr')
        clf.fit(X_train, y_train)

        pred_l1 = clf.predict(X_test_l1)
        pred_l2 = clf.predict(X_test_l2)
        pred_l3 = clf.predict(X_test_l3)

        real_accs_l1.append(np.mean(pred_l1 == y_test_l1) * 100.0)
        real_accs_l2.append(np.mean(pred_l2 == y_test_l2) * 100.0)
        real_accs_l3.append(np.mean(pred_l3 == y_test_l3) * 100.0)

    # 9) Null distribution: shuffle labels in L1 training, test on L1 sub-sample + all L2 + all L3
    null_accs_l1 = []
    null_accs_l2 = []
    null_accs_l3 = []

    for _ in range(n_shuffles):
        for _ in range(n_iterations):
            X_train_list = []
            y_train_list = []
            X_test_l1_list = []
            y_test_l1_list = []

            for cval in [1,2,3,4,5]:
                mat1 = class_matrices_l1[f'class{cval}']
                n1   = mat1.shape[0]
                idx1 = rng.permutation(n1)

                test_idx1  = idx1[:test_per_class]
                train_idx1 = idx1[test_per_class:]

                X_train_list.append(mat1[train_idx1, :])
                y_train_list.append(np.full(len(train_idx1), cval))

                X_test_l1_list.append(mat1[test_idx1, :])
                y_test_l1_list.append(np.full(len(test_idx1), cval))

            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)

            X_test_l1 = np.vstack(X_test_l1_list)
            y_test_l1 = np.concatenate(y_test_l1_list)

            # Full sets for L2, L3
            X_test_l2 = X_test_l2_full
            y_test_l2 = y_test_l2_full
            X_test_l3 = X_test_l3_full
            y_test_l3 = y_test_l3_full

            # Shuffle training labels
            y_train = rng.permutation(y_train)

            clf_null = SVC(kernel='linear', random_state=random_state, decision_function_shape='ovr')
            clf_null.fit(X_train, y_train)

            pred_null_l1 = clf_null.predict(X_test_l1)
            pred_null_l2 = clf_null.predict(X_test_l2)
            pred_null_l3 = clf_null.predict(X_test_l3)

            null_accs_l1.append(np.mean(pred_null_l1 == y_test_l1) * 100.0)
            null_accs_l2.append(np.mean(pred_null_l2 == y_test_l2) * 100.0)
            null_accs_l3.append(np.mean(pred_null_l3 == y_test_l3) * 100.0)


    # 10) t-tests

    observed_mean_l1 = np.mean(real_accs_l1)
    p_val_l1 = np.mean(null_accs_l1 >= observed_mean_l1)

    observed_mean_l2 = np.mean(real_accs_l2)
    p_val_l2 = np.mean(null_accs_l2 >= observed_mean_l2)

    observed_mean_l3 = np.mean(real_accs_l3)
    p_val_l3 = np.mean(null_accs_l3 >= observed_mean_l3)

    # Compare Load 2 vs Load 3 directly
    t_stat_23, p_val_23 = stats.ttest_ind(real_accs_l2, real_accs_l3, alternative="greater")


    p_vals = {
        'load1': p_val_l1,
        'load2': p_val_l2,
        'load3': p_val_l3
    }

    # 11) Plot three boxplots with customized colors
    plt.figure(figsize=(7, 6))
    data_to_plot = [real_accs_l1, real_accs_l2, real_accs_l3]
    box_colors = ['blue', 'green', 'red']
    box = plt.boxplot(data_to_plot, labels=['Load=1', 'Load=2', 'Load=3'], patch_artist=True)

    # Color boxes and fliers
    for patch, flier, color in zip(box['boxes'], box['fliers'], box_colors):
        patch.set_facecolor(color)
        flier.set(marker='o', color=color, alpha=0.5)

    for median in box['medians']:
        median.set(color='black', linewidth=1.5)

    plt.ylabel("Accuracy (%)")
    plt.title("Decoding Last Image")
    plt.axhline(y=20, linestyle='--', color='gray')  # chance level for 5-way

    # Significance annotation
    loads = [1, 2, 3]
    maxvals = [max(real_accs_l1), max(real_accs_l2), max(real_accs_l3)]
    pval_list = [p_val_l1, p_val_l2, p_val_l3]
    offset = 5
    for i, (mv, pv) in enumerate(zip(maxvals, pval_list), start=1):
        y_star = mv + offset
        if pv < 0.001:
            star = '***'
        elif pv < 0.01:
            star = '**'
        elif pv < 0.05:
            star = '*'
        else:
            star = ''
        if star:
            plt.text(i, y_star, star, ha='center', va='bottom', fontsize=20)
        offset += 5

    plt.ylim([0, max(maxvals) + offset + 5])

    # Significance bar between Load 2 and Load 3
    if p_val_23 < 0.05:
        y, h, col = max(maxvals[1], maxvals[2]) + offset + 5, 3, 'k'
        x1, x2 = 2, 3  # Positions of Load 2 and Load 3
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        
        if p_val_23 < 0.001:
            star = '***'
        elif p_val_23 < 0.01:
            star = '**'
        else:
            star = '*'
            
        plt.text((x1 + x2) * .5, y + h + 1, star, ha='center', va='bottom', color=col, fontsize=20)


    plt.tight_layout()
    plt.show()

    print("==== Results ====")
    print(f"Real L1 acc: {np.mean(real_accs_l1):.2f}% ± {np.std(real_accs_l1):.2f}% (N={n_iterations}) | p={p_val_l1:.2g}")
    print(f"Real L2 acc: {np.mean(real_accs_l2):.2f}% ± {np.std(real_accs_l2):.2f}% (N={n_iterations}) | p={p_val_l2:.2g}")
    print(f"Real L3 acc: {np.mean(real_accs_l3):.2f}% ± {np.std(real_accs_l3):.2f}% (N={n_iterations}) | p={p_val_l3:.2g}")

    return (real_accs_l1, real_accs_l2, real_accs_l3,
            null_accs_l1, null_accs_l2, null_accs_l3,
            p_vals)


# custom colors for each region
region_color_map = {
    'HPC': '#FFD700',     # yellow - Hippocampus
    'AMY': '#00FFFF',     # cyan - Amygdala
    'PSMA': '#FF0000',    # red - pre-SMA
    'DaCC': '#0000FF',    # blue - dACC
    'vmPFC': '#008000',   # green - vmPFC
}

def decode_last_imageID_load1_multiclass_ttest_by_region(
    mat_file_path,
    patient_ids=range(1, 22),
    m=0,
    num_windows=0,
    test_per_class=2,
    n_iterations=1000,
    n_shuffles=1,
    random_state=42,
    label_map=None,
    show_plots=True,
):
    """
    Decode the last image ID (1–5) per brain region, plot all regions
    together, and test *between* regions (Welch’s t‑tests).

    Returns
    -------
    results : dict
        {region_acronym: (real_acc list, null_acc list, p_value_vs_null)}
    """
    # ──────────────────────────────────────────────────────────────────────────
    # 0.  Setup
    # ──────────────────────────────────────────────────────────────────────────
    DEFAULT_MAP = {
        'dorsal_anterior_cingulate_cortex': 'DaCC',
        'pre_supplementary_motor_area': 'PSMA',
        'hippocampus': 'HPC',
        'amygdala': 'AMY',
         'ventral_medial_prefrontal_cortex': 'vmPFC',
    }
    if label_map is None:
        label_map = DEFAULT_MAP

    rng = np.random.default_rng(random_state)

    def _strip_lat(region):
        if region.endswith('_left'):
            return region[:-5]
        if region.endswith('_right'):
            return region[:-6]
        return region

    # ──────────────────────────────────────────────────────────────────────────
    # 1.  Load data
    # ──────────────────────────────────────────────────────────────────────────
    mat = loadmat(mat_file_path)
    neural_data = mat['neural_data'][0]

    # ──────────────────────────────────────────────────────────────────────────
    # 2.  Organise neurons by region
    # ──────────────────────────────────────────────────────────────────────────
    region_neurons = {}
    for entry in neural_data:
        pid = int(entry['patient_id'][0][0])
        if pid not in patient_ids:
            continue

        region_raw = entry['brain_region']
        region_raw = _matlab_to_str(region_raw)
        region_base = _strip_lat(region_raw)
        region_key  = label_map.get(region_base, region_base)

        fr = entry['firing_rates']
        loads = entry['trial_load'].ravel()
        tfield = int(entry['time_field'][0][0]) - 1
        img_ids = entry['trial_imageIDs']
        pref_img = int(entry['preferred_image'][0][0])  # NEW

        # mean rate around time_field (± num_windows)
        s, e = max(0, tfield - num_windows), min(fr.shape[1], tfield + num_windows + 1)
        mean_rates = fr[:, s:e].mean(axis=1)

        # label = last non-zero image ID, only Load-1 trials
        labels_all = np.array([row[row != 0][-1] if np.any(row) else 0 for row in img_ids])
        mask = (loads == 1) & (labels_all >= 1) & (labels_all <= 5)
        if not mask.any():
            continue

        rates, labels = mean_rates[mask], labels_all[mask]

        if all((labels == c).sum() >= 1 for c in range(1, 6)):
            neuron_dict = {f'class{c}': rates[labels == c] for c in range(1, 6)}
            neuron_dict['preferred_image'] = pref_img          # NEW
            region_neurons.setdefault(region_key, {}).setdefault(pid, []).append(neuron_dict)

    # ──────────────────────────────────────────────────────────────────────────
    # 3.  Filter patients with ≥ m neurons
    # ──────────────────────────────────────────────────────────────────────────
    for region in list(region_neurons.keys()):
        pts_ok = {pid: cells for pid, cells in region_neurons[region].items()
                  if len(cells) >= m}
        region_neurons[region] = [cell for cells in pts_ok.values() for cell in cells]
        if not region_neurons[region]:
            del region_neurons[region]

    if not region_neurons:
        raise RuntimeError("No region has neurons that satisfy the inclusion criteria.")


    # ──────────────────────────────────────────────────────────────────────────
    # 3b.  Determine which classes exist per region based on preferred_image
    # ──────────────────────────────────────────────────────────────────────────
    region_classes = {}
    for region, neurons in region_neurons.items():
        prefs = {n.get('preferred_image') for n in neurons if 'preferred_image' in n}
        # restrict to valid image IDs 1–5
        valid_classes = sorted(c for c in prefs if isinstance(c, (int, np.integer)) and 1 <= c <= 5)
        # Fallback: if somehow no valid preferred_image found, keep all 1–5
        if not valid_classes:
            valid_classes = list(range(1, 6))
        region_classes[region] = valid_classes


    # ──────────────────────────────────────────────────────────────────────────
    # 4.  Core decoder
    # ──────────────────────────────────────────────────────────────────────────
    def _decode(neuron_list, class_ids):
        """
        neuron_list : list of neuron_dicts for a region
        class_ids   : iterable of ints (subset of {1,…,5}) that this region supports,
                      based on preferred_image.
        """
        class_ids = list(class_ids)
        if len(class_ids) < 2:
            # not meaningful to decode with < 2 classes
            return None

        # Minimum trial count per class across neurons
        mins = [min(len(n[f'class{c}']) for n in neuron_list) for c in class_ids]
        if any(mc < test_per_class for mc in mins):
            return None

        # Down-sample trials per class so all neurons/classes are balanced
        neurons_ds = []
        for n in neuron_list:
            n_ds = {
                f'class{c}': resample(
                    n[f'class{c}'], replace=False,
                    n_samples=mc, random_state=random_state
                )
                for c, mc in zip(class_ids, mins)
            }
            neurons_ds.append(n_ds)

        class_mats = {
            c: np.column_stack([n_ds[f'class{c}'] for n_ds in neurons_ds])
            for c in class_ids
        }

        # real distribution
        real = []
        for _ in range(n_iterations):
            Xtr, ytr, Xts, yts = [], [], [], []
            for c in class_ids:
                idx = rng.permutation(class_mats[c].shape[0])
                ts, tr = idx[:test_per_class], idx[test_per_class:]
                Xtr.append(class_mats[c][tr]); ytr.append(np.full(tr.size, c))
                Xts.append(class_mats[c][ts]); yts.append(np.full(ts.size, c))
            Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
            Xts, yts = np.vstack(Xts), np.concatenate(yts)
            clf = SVC(kernel='linear', decision_function_shape='ovr',
                      random_state=random_state)
            clf.fit(Xtr, ytr)
            real.append((clf.predict(Xts) == yts).mean() * 100)

        # null distribution
        null = []
        for _ in range(n_shuffles):
            for _ in range(n_iterations):
                Xtr, ytr, Xts, yts = [], [], [], []
                for c in class_ids:
                    idx = rng.permutation(class_mats[c].shape[0])
                    ts, tr = idx[:test_per_class], idx[test_per_class:]
                    Xtr.append(class_mats[c][tr]); ytr.append(np.full(tr.size, c))
                    Xts.append(class_mats[c][ts]); yts.append(np.full(ts.size, c))
                Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
                Xts, yts = np.vstack(Xts), np.concatenate(yts)
                ytr = rng.permutation(ytr)           # shuffle labels
                clf = SVC(kernel='linear', decision_function_shape='ovr',
                          random_state=random_state)
                clf.fit(Xtr, ytr)
                null.append((clf.predict(Xts) == yts).mean() * 100)

        p_val = (np.array(null) >= np.mean(real)).mean()
        return real, null, p_val

    # ──────────────────────────────────────────────────────────────────────────
    # 5.  Decode each region
    # ──────────────────────────────────────────────────────────────────────────
    results = {}
    
    for region, neuron_list in region_neurons.items():
        class_ids = region_classes[region]          # NEW: region-specific classes
        res = _decode(neuron_list, class_ids)
        if res is None:
            print(f"[skip] {region}: too few trials after balancing "
                  f"for classes {class_ids}.")
            continue
        real_acc, null_acc, p_val = res
        results[region] = (real_acc, null_acc, p_val, class_ids)  # store classes too
        chance = 100.0 / len(class_ids)
        print(f"{region} (classes {class_ids}, chance={chance:.1f}%): "
              f"{np.mean(real_acc):.2f}% ± {np.std(real_acc):.2f}% "
              f"(N={n_iterations}), p_null={p_val:.4g}")



    # ──────────────────────────────────────────────────────────────────────────
    # 6.  Combined plot + region‑vs‑region stats
    # ──────────────────────────────────────────────────────────────────────────
    if show_plots and results:
        regions = list(results.keys())


        # regions = [r for r in regions if r != 'vmPFC']
        # if not regions:
        #     print("No regions left to plot after excluding vmPFC.")
        #     return results
        
        # Swap HPC and AMY positions if both exist
        if 'HPC' in regions and 'AMY' in regions:
            idx_hpc = regions.index('HPC')
            idx_amy = regions.index('AMY')
            regions[idx_hpc], regions[idx_amy] = regions[idx_amy], regions[idx_hpc]



        acc_lists = [results[r][0] for r in regions]
        p_vs_null = [results[r][2] for r in regions]

        # colours
        colours = [region_color_map.get(r, "#999999") for r in regions]


        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(regions)), 6))
        box = ax.boxplot(acc_lists,
                         patch_artist=True,
                         labels=regions,
                         widths=0.7)

        for patch, c in zip(box['boxes'], colours):
            patch.set_facecolor(c)
            patch.set_edgecolor('black')

        for element in ['whiskers', 'caps', 'medians']:
            plt.setp(box[element], color='black')

        # chance
        ax.axhline(20, ls='--', linewidth=1)

        # within‑region significance vs null
        y_max = max(max(acc) for acc in acc_lists)
        height_step = 5
        y_bar = y_max + height_step
        for x, p in enumerate(p_vs_null, start=1):
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            if stars:
                ax.text(x, y_bar, stars, ha='center', va='bottom', fontweight='bold')
        ax.set_ylim(0, y_bar + height_step)


       
    used_y_levels = set()
    height_pad    = 1.5
    collision_step = 2
    inset = 0.15   # amount to make the bar narrower on each side
    pairwise_pvalues = {}

    # for (i, regA), (j, regB) in combinations(enumerate(regions, start=1), 2):

    for (i, regA), (j, regB) in combinations(enumerate(regions, start=1), 2):
        target_regions = {'HPC', 'AMY'}

        # Determine if it's a HPC/AMY vs other
        if regA in target_regions and regB not in target_regions:
            region_high = regA
            region_low = regB
            idx_high = i
            idx_low = j
        elif regB in target_regions and regA not in target_regions:
            region_high = regB
            region_low = regA
            idx_high = j
            idx_low = i
        else:
            continue  # skip pairs without HPC/AMY vs other

        # Get real accuracies (already from the decoder)
        acc_high = np.array(results[region_high][0])
        acc_low = np.array(results[region_low][0])

        # Compute observed mean of the 'target' region (e.g., HPC/AMY)
        observed_mean = np.mean(acc_high)

        # Use the other region's real accuracies as empirical null
        p_pair = np.mean(acc_low >= observed_mean)
        # Store p-value regardless of significance for reporting
        pairwise_pvalues[f"{region_high} > {region_low}"] = p_pair


        if p_pair >= 0.05:
            continue

        stars = '***' if p_pair < 0.001 else '**' if p_pair < 0.01 else '*'

        # plotting the bar as before
        top_high = np.max(acc_high)
        top_low = np.max(acc_low)
        y0 = max(top_high, top_low) + height_pad - 20

        while y0 in used_y_levels:
            y0 += collision_step
        used_y_levels.add(y0)

        x1 = idx_high + inset
        x2 = idx_low - inset
        bar_height = 0.4

        ax.plot([x1, x1, x2, x2], [y0, y0 + bar_height, y0 + bar_height, y0], lw=1.2, c='black')
        ax.text((idx_high + idx_low) / 2, y0 + bar_height, stars,
                ha='center', va='bottom', fontweight='bold')

    print("\nEmpirical p-values (HPC/AMY > other regions):")
    for comparison, p_val in pairwise_pvalues.items():
        significance = 'n.s.' if p_val >= 0.05 else '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
        print(f"{comparison}: p = {p_val:.4f} ({significance})")


    ax.set_ylabel('Accuracy (%)', fontsize = 20)
    ax.set_title('Last‑image decoding by brain region')
    plt.tight_layout()
    plt.savefig(DIR + 'decdoing_region.svg', format = 'svg')
    plt.show()

    return results

def decode_fast_slow_load1_in_out(
    mat_file_path,
    patient_ids=range(1, 22),
    m=0,
    num_windows=0,
    test_per_class=3,
    n_iterations=1000,
    n_shuffles=1,
    random_state=42,
    make_plots=True,
):
    """
    Decode *fast* vs *slow* RTs for Load-1 trials, run separately for
    probe-IN and probe-OUT conditions.

    Parameters
    ----------
    mat_file_path : str
        .mat file path holding `neural_data`.
    patient_ids : iterable
        Patient IDs to include.
    m : int
        Min. no. of qualifying neurons per patient.
    num_windows : int
        ± #bins around each neuron's `time_field` to average.
    test_per_class : int
        Held-out trials per class (fast / slow) in every iteration.
    n_iterations : int
        Repeated sub-sampling CV iterations (real labels).
    n_shuffles : int
        Full null-distribution repeats.
    random_state : int
        Seed for the NumPy random generator.
    make_plots : bool
        If True, draw a side-by-side box-plot (chance = 50 %).

    Returns
    -------
    results : dict
        Keys = 'in', 'out'.  Each value is a (real, null, p) tuple:
            real_accuracies : list[float]
            null_accuracies : list[float]
            p_value         : float   (permutation, one-tailed: real > null)
    """

    rng = np.random.default_rng(random_state)
    mat   = loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
    ndata = mat["neural_data"]          # 1-D MATLAB struct array

    # ------------------------------------------------------------------ #
    # 1 ▪ organise neurons by patient
    # ------------------------------------------------------------------ #
    patient_dict = {}
    for nd in ndata:
        pid = int(nd.patient_id)
        entry = dict(
            frates=nd.firing_rates,          # (trials × bins)
            loads = nd.trial_load.flatten(), # (trials,)
            tfield=int(nd.time_field) - 1,   # zero-based
            probe=nd.trial_probe_in_out.flatten(), # (trials,)
            rt   = nd.trial_RT.flatten(),    # (trials,)
        )
        patient_dict.setdefault(pid, []).append(entry)

    valid_pids = [
        pid for pid in patient_ids
        if pid in patient_dict and len(patient_dict[pid]) >= m
    ]
    if not valid_pids:
        raise RuntimeError(f"No patients meet m={m} neuron threshold.")

    # Helper ------------------------------------------------------------- #
    def run_one_group(in_or_out):
        """
        Decode fast vs slow within one probe condition.
        in_or_out = 1  (probe-IN)  or  0  (probe-OUT)
        """
        # ------------- a) collect neurons that have both fast & slow ---- #
        neurons = []
        for pid in valid_pids:
            for rec in patient_dict[pid]:
                # Load-1  &  probe mask
                m_load  = rec["loads"] == 1
                m_probe = rec["probe"] == in_or_out
                mask    = m_load & m_probe

                if not np.any(mask):
                    continue

                # Median split on RT (within the selected trials)
                rts = rec["rt"][mask]
                med = np.median(rts)
                lab = np.where(rts < med, 0, 1)   # 0=fast, 1=slow

                # ensure both classes represented
                if (lab == 0).sum() == 0 or (lab == 1).sum() == 0:
                    continue

                # average firing in window around time_field
                f     = rec["frates"][mask]           # (n_trials × n_bins)
                tf    = rec["tfield"]
                lo    = max(0, tf - num_windows)
                hi    = tf + num_windows + 1
                frate = f[:, lo:hi].mean(axis=1)      # (n_trials,)

                neurons.append(dict(rate=frate, label=lab))

        if not neurons:
            print(f"No neurons with both fast & slow trials for probe={in_or_out}.")
            return [], [], np.nan

        # ------------- b) down-sample to global minimum per class ------- #
        min_n = [
            min((n["label"] == c).sum() for n in neurons) for c in (0, 1)
        ]
        if min(min_n) < test_per_class:
            print(f"probe={in_or_out}: min class count {min_n} < test_per_class.")
            return [], [], np.nan

        for n in neurons:
            for c in (0, 1):
                idx = np.where(n["label"] == c)[0]
                sel = rng.choice(idx, size=min_n[c], replace=False)
                n[f"class{c}"] = n["rate"][sel]

        # ------------- c) build class matrices -------------------------- #
        mats = {}
        for c in (0, 1):
            m = np.stack([n[f"class{c}"] for n in neurons], axis=1)
            mats[c] = m.astype(np.float32)            # (trials × neurons)

        # ------------- d) CV loop (real labels) ------------------------- #
        def run_decoder(real_labels=True):
            accs = []
            for _ in range(n_iterations):
                Xtr, ytr, Xte, yte = [], [], [], []
                for c in (0, 1):
                    n_trials = mats[c].shape[0]
                    perm     = rng.permutation(n_trials)
                    te, tr   = perm[:test_per_class], perm[test_per_class:]
                    Xtr.append(mats[c][tr])
                    ytr.append(np.full(tr.size, c))
                    Xte.append(mats[c][te])
                    yte.append(np.full(te.size, c))
                Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
                Xte = np.vstack(Xte); yte = np.concatenate(yte)

                if not real_labels:
                    ytr = rng.permutation(ytr)

                clf = SVC(kernel="linear", random_state=random_state)
                clf.fit(Xtr, ytr)
                accs.append((clf.predict(Xte) == yte).mean() * 100)
            return accs

        real_acc  = run_decoder(real_labels=True)
        null_acc  = []
        for _ in range(n_shuffles):
            null_acc.extend(run_decoder(real_labels=False))

        # permutation p-value (one-tailed: real > null)
        p_perm = np.mean(np.array(null_acc) >= np.mean(real_acc))
        return real_acc, null_acc, p_perm

    # ------------------------------------------------------------------ #
    # 2 ▪ run for probe-IN (1) and probe-OUT (0)
    # ------------------------------------------------------------------ #
    results = {}
    for flag, tag in [(1, "in"), (0, "out")]:
        results[tag] = run_one_group(flag)

    # ------------------------------------------------------------------ #
    # 3 ▪ optional plotting
    # ------------------------------------------------------------------ #
    if make_plots:
        plt.figure(figsize=(6, 7))
        boxes = plt.boxplot(
            [results[k][0] for k in ("in", "out")],
            labels=["probe-IN", "probe-OUT"],
            patch_artist=True,
        )
        for b in boxes["boxes"]:
            b.set(facecolor="lightblue")
        plt.axhline(50, ls="--", lw=1)
        plt.ylabel("Accuracy (%)")
        plt.title("Fast vs Slow RT decoding (Load 1)")
        # significance stars
        y_max = max(max(results["in"][0], default=0),
                    max(results["out"][0], default=0)) + 5
        for i, k in enumerate(("in", "out"), 1):
            p = results[k][2]
            stars = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""
            if stars:
                plt.text(i, y_max, stars, ha="center", va="bottom")
        plt.ylim(0, y_max + 5)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    return results

def decode_last_imageID_load1_multiclass_ttest_by_region_Original(
    mat_file_path,
    patient_ids=range(1, 22),
    m=0,
    num_windows=0,
    test_per_class=2,
    n_iterations=1000,
    n_shuffles=1,
    random_state=42,
    label_map=None,
    show_plots=True,
):
    """
    Decode the last image ID (1–5) per brain region, plot all regions
    together, and test *between* regions (Welch’s t‑tests).

    Returns
    -------
    results : dict
        {region_acronym: (real_acc list, null_acc list, p_value_vs_null)}
    """
    # ──────────────────────────────────────────────────────────────────────────
    # 0.  Setup
    # ──────────────────────────────────────────────────────────────────────────
    DEFAULT_MAP = {
        'dorsal_anterior_cingulate_cortex': 'DaCC',
        'pre_supplementary_motor_area': 'PSMA',
        'hippocampus': 'HPC',
        'amygdala': 'AMY',
         'ventral_medial_prefrontal_cortex': 'vmPFC',
    }
    if label_map is None:
        label_map = DEFAULT_MAP

    rng = np.random.default_rng(random_state)

    def _strip_lat(region):
        if region.endswith('_left'):
            return region[:-5]
        if region.endswith('_right'):
            return region[:-6]
        return region

    # ──────────────────────────────────────────────────────────────────────────
    # 1.  Load data
    # ──────────────────────────────────────────────────────────────────────────
    mat = loadmat(mat_file_path)
    neural_data = mat['neural_data'][0]

    # ──────────────────────────────────────────────────────────────────────────
    # 2.  Organise neurons by region
    # ──────────────────────────────────────────────────────────────────────────
    region_neurons = {}
    for entry in neural_data:
        pid = int(entry['patient_id'][0][0])
        if pid not in patient_ids:
            continue

        region_raw = entry['brain_region']
        region_raw = _matlab_to_str(region_raw)
        region_base = _strip_lat(region_raw)
        region_key  = label_map.get(region_base, region_base)


        fr = entry['firing_rates']
        loads = entry['trial_load'].ravel()
        tfield = int(entry['time_field'][0][0]) - 1
        img_ids = entry['trial_imageIDs']

        # mean rate around time_field (± num_windows)
        s, e = max(0, tfield - num_windows), min(fr.shape[1], tfield + num_windows + 1)
        mean_rates = fr[:, s:e].mean(axis=1)

        # label = last non‑zero image ID, only Load‑1 trials
        labels_all = np.array([row[row != 0][-1] if np.any(row) else 0 for row in img_ids])
        mask = (loads == 1) & (labels_all >= 1) & (labels_all <= 5)
        if not mask.any():
            continue

        rates, labels = mean_rates[mask], labels_all[mask]
        if all((labels == c).sum() >= 1 for c in range(1, 6)):
            neuron_dict = {f'class{c}': rates[labels == c] for c in range(1, 6)}
            region_neurons.setdefault(region_key, {}).setdefault(pid, []).append(neuron_dict)

    # ──────────────────────────────────────────────────────────────────────────
    # 3.  Filter patients with ≥ m neurons
    # ──────────────────────────────────────────────────────────────────────────
    for region in list(region_neurons.keys()):
        pts_ok = {pid: cells for pid, cells in region_neurons[region].items()
                  if len(cells) >= m}
        region_neurons[region] = [cell for cells in pts_ok.values() for cell in cells]
        if not region_neurons[region]:
            del region_neurons[region]

    if not region_neurons:
        raise RuntimeError("No region has neurons that satisfy the inclusion criteria.")

    # ──────────────────────────────────────────────────────────────────────────
    # 4.  Core decoder
    # ──────────────────────────────────────────────────────────────────────────
    def _decode(neuron_list):
        mins = [min(len(n[f'class{c}']) for n in neuron_list) for c in range(1, 6)]
        if any(mc < test_per_class for mc in mins):
            return None
        # down‑sample
        neurons_ds = []
        for n in neuron_list:
            n_ds = {f'class{c}': resample(n[f'class{c}'], replace=False,
                                          n_samples=mc, random_state=random_state)
                    for c, mc in zip(range(1, 6), mins)}
            neurons_ds.append(n_ds)

        class_mats = {c: np.column_stack([n_ds[f'class{c}'] for n_ds in neurons_ds])
                      for c in range(1, 6)}

        # real distribution
        real = []
        for _ in range(n_iterations):
            Xtr, ytr, Xts, yts = [], [], [], []
            for c in range(1, 6):
                idx = rng.permutation(class_mats[c].shape[0])
                ts, tr = idx[:test_per_class], idx[test_per_class:]
                Xtr.append(class_mats[c][tr]); ytr.append(np.full(tr.size, c))
                Xts.append(class_mats[c][ts]); yts.append(np.full(ts.size, c))
            Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
            Xts, yts = np.vstack(Xts), np.concatenate(yts)
            clf = SVC(kernel='linear', decision_function_shape='ovr',
                      random_state=random_state)
            clf.fit(Xtr, ytr)
            real.append((clf.predict(Xts) == yts).mean() * 100)

        # null distribution
        null = []
        for _ in range(n_shuffles):
            for _ in range(n_iterations):
                Xtr, ytr, Xts, yts = [], [], [], []
                for c in range(1, 6):
                    idx = rng.permutation(class_mats[c].shape[0])
                    ts, tr = idx[:test_per_class], idx[test_per_class:]
                    Xtr.append(class_mats[c][tr]); ytr.append(np.full(tr.size, c))
                    Xts.append(class_mats[c][ts]); yts.append(np.full(ts.size, c))
                Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
                Xts, yts = np.vstack(Xts), np.concatenate(yts)
                ytr = rng.permutation(ytr)           # shuffle labels
                clf = SVC(kernel='linear', decision_function_shape='ovr',
                          random_state=random_state)
                clf.fit(Xtr, ytr)
                null.append((clf.predict(Xts) == yts).mean() * 100)

        p_val = (np.array(null) >= np.mean(real)).mean()
        return real, null, p_val

    # ──────────────────────────────────────────────────────────────────────────
    # 5.  Decode each region
    # ──────────────────────────────────────────────────────────────────────────
    results = {}
    
    for region, neuron_list in region_neurons.items():
        res = _decode(neuron_list)
        if res is None:
            print(f"[skip] {region}: too few trials after balancing.")
            continue
        real_acc, null_acc, p_val = res
        results[region] = (real_acc, null_acc, p_val)
        print(f"{region}: {np.mean(real_acc):.2f}% ± {np.std(real_acc):.2f}% "
              f"(N={n_iterations}), p_null={p_val:.4g}")
    


    # ──────────────────────────────────────────────────────────────────────────
    # 6.  Combined plot + region‑vs‑region stats
    # ──────────────────────────────────────────────────────────────────────────
    if show_plots and results:
        regions = list(results.keys())


        # regions = [r for r in regions if r != 'vmPFC']
        # if not regions:
        #     print("No regions left to plot after excluding vmPFC.")
        #     return results
        
        # Swap HPC and AMY positions if both exist
        if 'HPC' in regions and 'AMY' in regions:
            idx_hpc = regions.index('HPC')
            idx_amy = regions.index('AMY')
            regions[idx_hpc], regions[idx_amy] = regions[idx_amy], regions[idx_hpc]



        acc_lists = [results[r][0] for r in regions]
        p_vs_null = [results[r][2] for r in regions]

        # colours
        colours = [region_color_map.get(r, "#999999") for r in regions]


        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(regions)), 6))
        box = ax.boxplot(acc_lists,
                         patch_artist=True,
                         labels=regions,
                         widths=0.7)

        for patch, c in zip(box['boxes'], colours):
            patch.set_facecolor(c)
            patch.set_edgecolor('black')

        for element in ['whiskers', 'caps', 'medians']:
            plt.setp(box[element], color='black')

        # chance
        ax.axhline(20, ls='--', linewidth=1)

        # within‑region significance vs null
        y_max = max(max(acc) for acc in acc_lists)
        height_step = 5
        y_bar = y_max + height_step
        for x, p in enumerate(p_vs_null, start=1):
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            if stars:
                ax.text(x, y_bar, stars, ha='center', va='bottom', fontweight='bold')
        ax.set_ylim(0, y_bar + height_step)


       
    used_y_levels = set()
    height_pad    = 1.5
    collision_step = 2
    inset = 0.15   # amount to make the bar narrower on each side
    pairwise_pvalues = {}

    # for (i, regA), (j, regB) in combinations(enumerate(regions, start=1), 2):

    for (i, regA), (j, regB) in combinations(enumerate(regions, start=1), 2):
        target_regions = {'HPC', 'AMY'}

        # Determine if it's a HPC/AMY vs other
        if regA in target_regions and regB not in target_regions:
            region_high = regA
            region_low = regB
            idx_high = i
            idx_low = j
        elif regB in target_regions and regA not in target_regions:
            region_high = regB
            region_low = regA
            idx_high = j
            idx_low = i
        else:
            continue  # skip pairs without HPC/AMY vs other

        # Get real accuracies (already from the decoder)
        acc_high = np.array(results[region_high][0])
        acc_low = np.array(results[region_low][0])

        # Compute observed mean of the 'target' region (e.g., HPC/AMY)
        observed_mean = np.mean(acc_high)

        # Use the other region's real accuracies as empirical null
        p_pair = np.mean(acc_low >= observed_mean)
        # Store p-value regardless of significance for reporting
        pairwise_pvalues[f"{region_high} > {region_low}"] = p_pair


        if p_pair >= 0.05:
            continue

        stars = '***' if p_pair < 0.001 else '**' if p_pair < 0.01 else '*'

        # plotting the bar as before
        top_high = np.max(acc_high)
        top_low = np.max(acc_low)
        y0 = max(top_high, top_low) + height_pad - 20

        while y0 in used_y_levels:
            y0 += collision_step
        used_y_levels.add(y0)

        x1 = idx_high + inset
        x2 = idx_low - inset
        bar_height = 0.4

        ax.plot([x1, x1, x2, x2], [y0, y0 + bar_height, y0 + bar_height, y0], lw=1.2, c='black')
        ax.text((idx_high + idx_low) / 2, y0 + bar_height, stars,
                ha='center', va='bottom', fontweight='bold')

    print("\nEmpirical p-values (HPC/AMY > other regions):")
    for comparison, p_val in pairwise_pvalues.items():
        significance = 'n.s.' if p_val >= 0.05 else '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
        print(f"{comparison}: p = {p_val:.4f} ({significance})")


    ax.set_ylabel('Accuracy (%)', fontsize = 20)
    ax.set_title('Last‑image decoding by brain region')
    plt.tight_layout()
    plt.savefig(DIR + 'decdoing_region.svg', format = 'svg')
    plt.show()

    return results

# decode_last_imageID_load123_multiclass_ttest("Figure 4/may11_3sig15.mat")

# decode_last_imageID_load1_multiclass_ttest("Refined_win0_exp_raw.mat")
# decode_last_imageID_load1_multiclass_ttest("LIS_may11_Sig2.5G1_0.7_nw2.mat")
# decode_last_imageID_load1_multiclass_ttest_by_region("LIS_may11_Sig2.5G1_0.7_nw2.mat", n_iterations=3000)

# decode_last_imageID_load1_multiclass_ttest("100msTCdata.mat")
# decode_last_imageID_load1_multiclass_ttest("Figure 4/may11_3sig15.mat", random_state=20250710)


def decode_last_imageID_load1_multiclass_ttest_alternative(
    mat_file_path,
    patient_ids=range(1, 22),
    m=0,
    test_per_class=2,
    n_iterations=1000,
    n_shuffles=1,
    random_state=42
):
    """
    Decodes the 'last non-zero image ID' (1..5) within Load=1 trials,
    using a multi-class SVM with repeated sub-sampling.
    Then uses an independent-samples t-test (real vs. null) for significance.

    Differences from the original version:
    1) For each trial, we use the firing rates averaged across the *entire trial*.
    2) We print the global minimum number of trials for each class (1..5).

    Parameters
    ----------
    mat_file_path : str
        Path to the .mat file containing 'neural_data'.
    patient_ids : iterable
        Which patient IDs to include.
    m : int
        Minimum number of time cells for a patient to be included.
    test_per_class : int
        How many test trials for each of the 5 image IDs in each iteration.
    n_iterations : int
        How many times to repeat the sub-sampling cross-validation (real labels).
    n_shuffles : int
        How many times to generate a null distribution of label-shuffled runs.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    real_accuracies : list of float
        The per-iteration accuracy (in %) for the real labels cross-validation.
    null_accuracies : list of float
        The distribution of accuracies (in %) for all label-shuffled runs.
    p_value : float
        p-value from a two-sample t-test comparing real vs. null distributions.
    """

    rng = np.random.default_rng(random_state)

    # 1) Load .mat data
    mat_data = loadmat(mat_file_path)
    neural_data = mat_data['neural_data']

    # 2) Group by patient
    patient_dict = {}
    for entry in neural_data[0]:
        pid = int(entry['patient_id'][0][0])

        # shape: (num_trials, num_time_bins)
        frates = entry['firing_rates']  
        # shape: (num_trials,)
        loads = entry['trial_load'].flatten()  
        # NOTE: 'time_field' is not used for the alternative approach
        img_ids = entry['trial_imageIDs']  # shape: (num_trials, 3)

        if pid not in patient_dict:
            patient_dict[pid] = []
        patient_dict[pid].append((frates, loads, img_ids))

    # 3) Filter by minimum number of time cells (m) if needed
    #    (In the original code, "len(patient_dict[pid]) >= m" was used, 
    #     but you can adapt if "m" was meant differently. We'll keep it as in the original.)
    valid_patients = []
    for pid in patient_ids:
        if pid in patient_dict and len(patient_dict[pid]) >= m:
            valid_patients.append(pid)
    if len(valid_patients) == 0:
        print(f"No patients meet the minimum {m} time cells.")
        return [], [], np.nan

    # 4) Gather valid neurons (only Load=1 trials; last non-zero in 1..5)
    all_neurons = []
    for pid in valid_patients:
        for (frates, loads, img_ids) in patient_dict[pid]:

            # (A) Instead of time-field averaging, average *across the entire trial*
            # shape of frates: (num_trials, num_time_bins)
            mean_rates = np.mean(frates, axis=1)  # shape: (num_trials,)

            # (B) Keep only Load=1 trials
            mask_load1 = (loads == 1)

            # (C) Find the last non-zero in each row of img_ids
            labels_all = []
            for row in img_ids:
                row_ = row.flatten()
                nonzeros = row_[row_ != 0]
                if len(nonzeros) > 0:
                    labels_all.append(nonzeros[-1])  # last nonzero
                else:
                    labels_all.append(0)  # no image
            labels_all = np.array(labels_all)

            # only keep labels in {1..5}
            mask_valid_img = (labels_all >= 1) & (labels_all <= 5)

            final_mask = mask_load1 & mask_valid_img
            final_rates = mean_rates[final_mask]
            final_labels = labels_all[final_mask]  # each in {1..5}

            # (D) check if it has at least 1 trial for each label in 1..5
            keep_it = True
            for c in [1, 2, 3, 4, 5]:
                if np.sum(final_labels == c) < 1:
                    keep_it = False
                    break

            if keep_it:
                # store data for each of the 5 classes
                neuron_dict = {}
                for c in [1, 2, 3, 4, 5]:
                    neuron_dict[f'class{c}'] = final_rates[final_labels == c]
                all_neurons.append(neuron_dict)

    if len(all_neurons) == 0:
        print("No neurons found that have Load=1 trials with all 5 imageIDs present.")
        return [], [], np.nan

    # 5) Find global min # trials per class so we can downsample
    min_counts = []
    for c in [1, 2, 3, 4, 5]:
        key = f'class{c}'
        counts = [len(n[key]) for n in all_neurons]
        min_counts.append(min(counts))

    # Print the global minimum for each of the 5 classes
    print("Global minimum for each class (1..5):", min_counts)

    # Ensure we have enough for test_per_class
    for c_i, cval in enumerate([1, 2, 3, 4, 5], 0):
        if min_counts[c_i] < test_per_class:
            print(f"Cannot proceed: class={cval}, global min (#trials)={min_counts[c_i]} < test_per_class={test_per_class}")
            return [], [], np.nan

    # 6) Downsample each neuron to the global min for each class
    for neuron in all_neurons:
        for c_i, cval in enumerate([1, 2, 3, 4, 5], 0):
            key = f'class{cval}'
            neuron[key] = resample(
                neuron[key],
                replace=False,
                n_samples=min_counts[c_i],
                random_state=random_state
            )

    # 7) Build big matrices (rows=trials, cols=neurons) for each class
    num_neurons = len(all_neurons)
    class_matrices = {}
    for c_i, cval in enumerate([1, 2, 3, 4, 5], 0):
        key = f'class{cval}'
        mat = np.zeros((min_counts[c_i], num_neurons), dtype=np.float32)
        for j, neuron in enumerate(all_neurons):
            mat[:, j] = neuron[key]
        class_matrices[key] = mat

    # 8) Repeated sub-sampling (Real Labels)
    real_accuracies = []
    for _ in range(n_iterations):
        X_train_list = []
        y_train_list = []
        X_test_list  = []
        y_test_list  = []

        for cval in [1, 2, 3, 4, 5]:
            key = f'class{cval}'
            mat = class_matrices[key]
            n_c = mat.shape[0]

            idx_c = rng.permutation(n_c)
            test_idx_c  = idx_c[:test_per_class]
            train_idx_c = idx_c[test_per_class:]

            X_train_list.append(mat[train_idx_c, :])
            y_train_list.append(np.full(len(train_idx_c), cval))

            X_test_list.append(mat[test_idx_c, :])
            y_test_list.append(np.full(len(test_idx_c), cval))

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_test  = np.vstack(X_test_list)
        y_test  = np.concatenate(y_test_list)

        clf = SVC(kernel='linear', random_state=random_state, decision_function_shape='ovr')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = np.mean(y_pred == y_test) * 100.0
        real_accuracies.append(acc)

    # 9) Generate null distribution (label-shuffled)
    null_accuracies = []
    for shuffle_i in range(n_shuffles):
        for _ in range(n_iterations):
            X_train_list = []
            y_train_list = []
            X_test_list  = []
            y_test_list  = []

            for cval in [1, 2, 3, 4, 5]:
                key = f'class{cval}'
                mat = class_matrices[key]
                n_c = mat.shape[0]

                idx_c = rng.permutation(n_c)
                test_idx_c  = idx_c[:test_per_class]
                train_idx_c = idx_c[test_per_class:]

                X_train_list.append(mat[train_idx_c, :])
                y_train_list.append(np.full(len(train_idx_c), cval))

                X_test_list.append(mat[test_idx_c, :])
                y_test_list.append(np.full(len(test_idx_c), cval))

            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)
            X_test  = np.vstack(X_test_list)
            y_test  = np.concatenate(y_test_list)

            # Shuffle labels for training
            y_train = rng.permutation(y_train)

            clf_null = SVC(kernel='linear', random_state=random_state, decision_function_shape='ovr')
            clf_null.fit(X_train, y_train)
            y_pred_null = clf_null.predict(X_test)

            acc_null = np.mean(y_pred_null == y_test) * 100.0
            null_accuracies.append(acc_null)

    # 10) Two-sample t-test (real vs. null)
    # t_stat, p_value = ttest_ind(real_accuracies, null_accuracies, equal_var=False)
    #Empirical
    observed_mean = np.mean(real_accuracies)
    p_value = np.mean(null_accuracies >= observed_mean)

    plt.figure(figsize=(5, 8))

    # Boxplot
    box = plt.boxplot(real_accuracies, labels=['Real'], patch_artist=True)

    # Customize colors
    for element in ['boxes', 'whiskers', 'caps', 'medians']:
        plt.setp(box[element], color='blue')
    for patch in box['boxes']:
        patch.set(facecolor='lightblue')
    for flier in box['fliers']:
        flier.set(marker='o', color='blue', alpha=0.8)

    # Axis labels and title with larger font size
    plt.ylabel("Accuracy (%)", fontsize=20)
    plt.title("Decoding the Single Encoded Image", fontsize=20)

    # Tick labels font size
    plt.tick_params(axis='both', which='major', labelsize=20)

    # Chance line
    plt.axhline(y=20, linestyle='--')

    # Significance stars
    max_acc = max(real_accuracies) if real_accuracies else 100
    y_star = max_acc + 5
    if p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'
    else:
        stars = ''

    if stars:
        plt.text(1, y_star, stars, ha='center', va='bottom', fontsize=20)

    plt.ylim([0, y_star + 5])
    plt.tight_layout()
    plt.savefig(DIR + 'concept_decode.svg', format='svg')
    plt.show()


    mean_acc = np.mean(real_accuracies)
    std_acc  = np.std(real_accuracies)
    print(f"Real cross-validation accuracy = {mean_acc:.2f}% ± {std_acc:.2f}% (mean ± std, N={n_iterations})")
    print(f"Null distribution size = {len(null_accuracies)}")
    print(f"t-test p-value = {p_value:.5g}")




    print("Unique accuracies:", np.unique(real_accuracies))
    print("Q1, median, Q3:", np.percentile(real_accuracies, [25, 50, 75]))

    return real_accuracies, null_accuracies, p_value


real_accuracies, null_acc, p_value = decode_last_imageID_load1_multiclass_ttest("Nov27_Nov18.mat", random_state=20250710, num_windows=0)

concept_accuracies, null, p = decode_last_imageID_load1_multiclass_ttest_alternative("100msCCdata_global.mat", random_state=42, n_iterations=1000)

# # p_value_concept_cue = stats.ttest_ind(real_accuracies, concept_accuracies, alternative='greater')

# # print(p_value_concept_cue)


import numpy as np
import matplotlib.pyplot as plt

def perm_test_greater(a, b, n_perm=10000, random_state=None):
    """
    One-sided permutation test for H1: mean(a) > mean(b)
    Returns (empirical p-value, observed mean difference).
    """
    rng = np.random.default_rng(random_state)

    a = np.asarray(a)
    b = np.asarray(b)

    d_obs = np.mean(a) - np.mean(b)

    concat = np.concatenate([a, b])
    n_a = len(a)

    count = 0
    for _ in range(n_perm):
        rng.shuffle(concat)  # in-place
        d_new = np.mean(concat[:n_a]) - np.mean(concat[n_a:])
        if d_new >= d_obs:        # one-sided: a > b
            count += 1

    # bias-corrected estimate so p is never exactly 0 or 1
    p_value = (count + 1) / (n_perm + 1)
    return p_value, d_obs

p_perm2 = perm_test_greater(real_accuracies, concept_accuracies)
print(p_perm2)   # (p_value, d_obs)

from scipy.stats import mannwhitneyu, ks_2samp

# 1) Rank-sum / Mann–Whitney (A > B?)
u_stat, p_mw = mannwhitneyu(real_accuracies,
                            concept_accuracies,
                            alternative='greater')
print("Mann–Whitney (real > concept): U =", u_stat, "  p =", p_mw)

# 2) Test for any distributional difference (not just shift)
ks_stat, p_ks = ks_2samp(real_accuracies,
                         concept_accuracies,
                         alternative='two-sided')
print("KS test (any distributional difference): D =", ks_stat, "  p =", p_ks)


# ---- Plot the two distributions ----
plt.figure(figsize=(6, 4))

plt.hist(real_accuracies, bins=20, alpha=0.5, label='real_accuracies', density=True)
plt.hist(concept_accuracies, bins=20, alpha=0.5, label='concept_accuracies', density=True)

plt.xlabel('Accuracy')
plt.ylabel('Density')
plt.title('Distributions of real_accuracies vs concept_accuracies')
plt.legend()
plt.tight_layout()
plt.show()


#real_accuracies, null_accuracies, p_value = decode_last_imageID_load1_multiclass_ttest_alternative("Figure 4/concept_Global.mat")

# decode_last_imageID_load1_multiclass_ttest("3sig15_data.mat", random_state=20250710, num_windows=3)
#decode_last_imageID_load1_multiclass_ttest_by_region("Nov27_Nov18.mat", random_state=20250710, n_iterations=10000, num_windows=0)
