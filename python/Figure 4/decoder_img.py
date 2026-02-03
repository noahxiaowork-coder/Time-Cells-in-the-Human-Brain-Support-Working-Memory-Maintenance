import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.utils import resample
from scipy.stats import mannwhitneyu

DIR = ''


def _matlab_to_str(x):
    """Convert MATLAB strings to Python str."""
    import numpy as np

    if isinstance(x, (bytes, bytearray)):
        return x.decode('utf-8', 'ignore')
    if isinstance(x, str):
        return x

    if isinstance(x, np.ndarray):
        if x.size == 0:
            return ''
        if x.dtype.kind in ('U', 'S'):
            if x.ndim == 0 or x.size == 1:
                return str(x.item())
            return ''.join(np.asarray(x).astype(str).ravel(order='F'))
        if x.dtype == object:
            y = x
            while isinstance(y, np.ndarray) and y.dtype == object and y.size == 1:
                y = y.item()
            if isinstance(y, np.ndarray) and y.size > 0:
                y = y.ravel()[0]
            return _matlab_to_str(y)

    try:
        import numpy as np
        if isinstance(x, np.void):
            return _matlab_to_str(np.array(x, dtype=object))
    except Exception:
        pass

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
    """Decode last image ID from Load=1 trials using linear SVM."""
    rng = np.random.default_rng(random_state)

    mat_data = loadmat(mat_file_path)
    neural_data = mat_data['neural_data']

    patient_dict = {}
    for entry in neural_data[0]:
        pid = int(entry['patient_id'][0][0])

        frates  = entry['firing_rates']
        loads   = entry['trial_load'].flatten()
        tfield  = int(entry['time_field'][0][0]) - 1
        img_ids = entry['trial_imageIDs']

        patient_dict.setdefault(pid, []).append((frates, loads, tfield, img_ids))

    valid_patients = []
    for pid in patient_ids:
        if pid in patient_dict and len(patient_dict[pid]) >= m:
            valid_patients.append(pid)
    if len(valid_patients) == 0:
        print(f"No patients meet the minimum {m} time cells.")
        return [], [], np.nan

    all_neurons = []
    for pid in valid_patients:
        for (frates, loads, tfield, img_ids) in patient_dict[pid]:
            start_idx = max(0, tfield - num_windows)
            end_idx   = min(frates.shape[1], tfield + num_windows + 1)
            windowed  = frates[:, start_idx:end_idx]
            mean_rates = np.mean(windowed, axis=1)

            mask_load1 = (loads == 1)

            labels_all = []
            for row in img_ids:
                row_ = row.flatten()
                nonzeros = row_[row_ != 0]
                labels_all.append(nonzeros[-1] if len(nonzeros) > 0 else 0)
            labels_all = np.array(labels_all)

            mask_valid_img = (labels_all >= 1) & (labels_all <= 5)

            final_mask = mask_load1 & mask_valid_img
            final_rates = mean_rates[final_mask]
            final_labels = labels_all[final_mask]

            keep_it = True
            for c in [1, 2, 3, 4, 5]:
                if np.sum(final_labels == c) < 1:
                    keep_it = False
                    break

            if keep_it:
                neuron_dict = {f'class{c}': final_rates[final_labels == c] for c in [1, 2, 3, 4, 5]}
                all_neurons.append(neuron_dict)

    if len(all_neurons) == 0:
        print("No neurons found with Load=1 trials containing all 5 imageIDs.")
        return [], [], np.nan

    min_counts = []
    for c in [1, 2, 3, 4, 5]:
        key = f'class{c}'
        counts = [len(n[key]) for n in all_neurons]
        min_counts.append(min(counts))

    for c_i, cval in enumerate([1, 2, 3, 4, 5], 0):
        if min_counts[c_i] < test_per_class:
            print(f"Cannot proceed: class={cval}, min trials={min_counts[c_i]} < test_per_class={test_per_class}")
            return [], [], np.nan

    for neuron in all_neurons:
        for c_i, cval in enumerate([1, 2, 3, 4, 5], 0):
            key = f'class{cval}'
            neuron[key] = resample(
                neuron[key],
                replace=False,
                n_samples=min_counts[c_i],
                random_state=random_state
            )

    num_neurons = len(all_neurons)
    class_matrices = {}
    for c_i, cval in enumerate([1, 2, 3, 4, 5], 0):
        key = f'class{cval}'
        mat = np.zeros((min_counts[c_i], num_neurons), dtype=np.float32)
        for j, neuron in enumerate(all_neurons):
            mat[:, j] = neuron[key]
        class_matrices[key] = mat

    real_accuracies = []
    for _ in range(n_iterations):
        X_train_list, y_train_list = [], []
        X_test_list,  y_test_list  = [], []

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

        real_accuracies.append(np.mean(y_pred == y_test) * 100.0)

    null_accuracies = []
    for _ in range(n_shuffles):
        for _ in range(n_iterations):
            X_train_list, y_train_list = [], []
            X_test_list,  y_test_list  = [], []

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

            y_train = rng.permutation(y_train)

            clf_null = SVC(kernel='linear', random_state=random_state, decision_function_shape='ovr')
            clf_null.fit(X_train, y_train)
            y_pred_null = clf_null.predict(X_test)

            null_accuracies.append(np.mean(y_pred_null == y_test) * 100.0)

    observed_mean = np.mean(real_accuracies)
    p_value = np.mean(np.array(null_accuracies) >= observed_mean)

    plt.figure(figsize=(5, 8))
    box = plt.boxplot(real_accuracies, labels=[''], patch_artist=True)

    for element in ['boxes', 'whiskers', 'caps', 'medians']:
        plt.setp(box[element], color='blue')
    for patch in box['boxes']:
        patch.set(facecolor='lightblue')
    for flier in box['fliers']:
        flier.set(marker='o', color='blue', alpha=0.8)

    plt.ylabel("Accuracy (%)", fontsize=20)
    plt.title("Decoding the Single Encoded Image")
    plt.axhline(y=20, linestyle='--')

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
    plt.savefig(DIR + 'decdoing.svg', format='svg')
    plt.show()

    mean_acc = np.mean(real_accuracies)
    sd_acc  = np.std(real_accuracies, ddof=1)
    print(f"Real cross-validation accuracy = {mean_acc:.2f}% ± {sd_acc:.2f}% (mean ± s.d., N={n_iterations})")
    print(f"Null distribution size = {len(null_accuracies)}")
    print(f"Empirical p-value (null >= mean(real)) = {p_value:.5g}")

    return real_accuracies, null_accuracies, p_value


def decode_last_imageID_load1_multiclass_ttest_alternative(
    mat_file_path,
    patient_ids=range(1, 22),
    m=0,
    test_per_class=2,
    n_iterations=1000,
    n_shuffles=1,
    random_state=42
):
    """Decode image ID using mean firing across full trial."""
    rng = np.random.default_rng(random_state)

    mat_data = loadmat(mat_file_path)
    neural_data = mat_data['neural_data']

    patient_dict = {}
    for entry in neural_data[0]:
        pid = int(entry['patient_id'][0][0])
        frates = entry['firing_rates']
        loads = entry['trial_load'].flatten()
        img_ids = entry['trial_imageIDs']
        patient_dict.setdefault(pid, []).append((frates, loads, img_ids))

    valid_patients = []
    for pid in patient_ids:
        if pid in patient_dict and len(patient_dict[pid]) >= m:
            valid_patients.append(pid)
    if len(valid_patients) == 0:
        print(f"No patients meet the minimum {m} time cells.")
        return [], [], np.nan

    all_neurons = []
    for pid in valid_patients:
        for (frates, loads, img_ids) in patient_dict[pid]:
            mean_rates = np.mean(frates, axis=1)

            mask_load1 = (loads == 1)

            labels_all = []
            for row in img_ids:
                row_ = row.flatten()
                nonzeros = row_[row_ != 0]
                labels_all.append(nonzeros[-1] if len(nonzeros) > 0 else 0)
            labels_all = np.array(labels_all)

            mask_valid_img = (labels_all >= 1) & (labels_all <= 5)
            final_mask = mask_load1 & mask_valid_img
            final_rates = mean_rates[final_mask]
            final_labels = labels_all[final_mask]

            keep_it = True
            for c in [1, 2, 3, 4, 5]:
                if np.sum(final_labels == c) < 1:
                    keep_it = False
                    break

            if keep_it:
                neuron_dict = {f'class{c}': final_rates[final_labels == c] for c in [1, 2, 3, 4, 5]}
                all_neurons.append(neuron_dict)

    if len(all_neurons) == 0:
        print("No neurons found with Load=1 trials containing all 5 imageIDs.")
        return [], [], np.nan

    min_counts = []
    for c in [1, 2, 3, 4, 5]:
        key = f'class{c}'
        counts = [len(n[key]) for n in all_neurons]
        min_counts.append(min(counts))

    for c_i, cval in enumerate([1, 2, 3, 4, 5], 0):
        if min_counts[c_i] < test_per_class:
            print(f"Cannot proceed: class={cval}, min trials={min_counts[c_i]} < test_per_class={test_per_class}")
            return [], [], np.nan

    for neuron in all_neurons:
        for c_i, cval in enumerate([1, 2, 3, 4, 5], 0):
            key = f'class{cval}'
            neuron[key] = resample(
                neuron[key],
                replace=False,
                n_samples=min_counts[c_i],
                random_state=random_state
            )

    num_neurons = len(all_neurons)
    class_matrices = {}
    for c_i, cval in enumerate([1, 2, 3, 4, 5], 0):
        key = f'class{cval}'
        mat = np.zeros((min_counts[c_i], num_neurons), dtype=np.float32)
        for j, neuron in enumerate(all_neurons):
            mat[:, j] = neuron[key]
        class_matrices[key] = mat

    real_accuracies = []
    for _ in range(n_iterations):
        X_train_list, y_train_list = [], []
        X_test_list,  y_test_list  = [], []

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

        real_accuracies.append(np.mean(y_pred == y_test) * 100.0)

    null_accuracies = []
    for _ in range(n_shuffles):
        for _ in range(n_iterations):
            X_train_list, y_train_list = [], []
            X_test_list,  y_test_list  = [], []

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

            y_train = rng.permutation(y_train)

            clf_null = SVC(kernel='linear', random_state=random_state, decision_function_shape='ovr')
            clf_null.fit(X_train, y_train)
            y_pred_null = clf_null.predict(X_test)

            null_accuracies.append(np.mean(y_pred_null == y_test) * 100.0)

    observed_mean = np.mean(real_accuracies)
    p_value = np.mean(np.array(null_accuracies) >= observed_mean)

    plt.figure(figsize=(5, 8))
    box = plt.boxplot(real_accuracies, labels=['Real'], patch_artist=True)

    for element in ['boxes', 'whiskers', 'caps', 'medians']:
        plt.setp(box[element], color='blue')
    for patch in box['boxes']:
        patch.set(facecolor='lightblue')
    for flier in box['fliers']:
        flier.set(marker='o', color='blue', alpha=0.8)

    plt.ylabel("Accuracy (%)", fontsize=20)
    plt.title("Decoding the Single Encoded Image", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.axhline(y=20, linestyle='--')

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
    print(f"Empirical p-value (null >= mean(real)) = {p_value:.5g}")

    return real_accuracies, null_accuracies, p_value


def perm_test_greater(a, b, n_perm=10000, random_state=None):
    """One-sided permutation test for mean(a) > mean(b)."""
    rng = np.random.default_rng(random_state)

    a = np.asarray(a)
    b = np.asarray(b)

    d_obs = np.mean(a) - np.mean(b)

    concat = np.concatenate([a, b])
    n_a = len(a)

    count = 0
    for _ in range(n_perm):
        rng.shuffle(concat)
        d_new = np.mean(concat[:n_a]) - np.mean(concat[n_a:])
        if d_new >= d_obs:
            count += 1

    p_value = (count + 1) / (n_perm + 1)
    return p_value, d_obs


real_accuracies, null_acc, p_value = decode_last_imageID_load1_multiclass_ttest(
    "cue_specific_TC.mat",
    random_state=20250710,
    num_windows=0
)

concept_accuracies, null, p = decode_last_imageID_load1_multiclass_ttest_alternative(
    "CC_zscore.mat",
    random_state=42,
    n_iterations=1000
)

p_perm2 = perm_test_greater(real_accuracies, concept_accuracies)
print("Permutation test (real > concept):", p_perm2)

u_stat, p_mw = mannwhitneyu(
    real_accuracies,
    concept_accuracies,
    alternative='greater'
)
print("Mann–Whitney (real > concept): U =", u_stat, "  p =", p_mw)
