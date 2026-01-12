from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.utils import resample
from sklearn.svm import SVC
from scipy.stats import ttest_ind

DIR = '/Users/xiangxuankong/Desktop/Human TC Support WM/Figures/Figure S4/'

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

    return real_accuracies, null_accuracies, p_value

#real_accuracies, null_accuracies, p_value = decode_last_imageID_load1_multiclass_ttest_alternative("Figure 4/concept_Global.mat")
real_accuracies, null_accuracies, p_value = decode_last_imageID_load1_multiclass_ttest_alternative("100msCCdata_global.mat")