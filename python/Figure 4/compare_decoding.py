import numpy as np
from scipy.io import loadmat
from scipy.stats import mannwhitneyu
from sklearn.svm import SVC
from sklearn.utils import resample


def decode_compare_two_celltypes_excluding_overlap(
    mat_file_A,
    mat_file_B,
    patient_ids=range(1, 22),
    m=0,
    num_windows_A=0,
    num_windows_B=0,
    feature_mode_A="timefield",
    feature_mode_B="wholetrial",
    test_per_class=2,
    n_iterations=1000,
    n_shuffles=200,
    random_state=42,
):
    """
    Compare decoding performance between two cell types from separate .mat files,
    excluding overlapping neurons (same patient_id and unit_id in both files).

    Parameters:
        feature_mode_A/B: "timefield" (uses time field window) or "wholetrial" (uses entire trial)
        n_shuffles: number of label-shuffled iterations for null distribution (use >100 for stable p-values)

    Returns:
        Dictionary with kept neuron counts and decoding results for both cell types A and B
    """

    rng = np.random.default_rng(random_state)

    def _get_field(entry, name):
        """Robustly access MATLAB struct fields."""
        return entry[name]

    def _iter_neurons(mat_path):
        """Yield (patient_id, unit_id) and entry for each neuron."""
        mat = loadmat(mat_path)
        neural_data = mat["neural_data"][0]
        for entry in neural_data:
            pid = int(_get_field(entry, "patient_id")[0][0])
            uid = int(_get_field(entry, "unit_id")[0][0])
            yield (pid, uid), entry

    def _extract_dataset(mat_path, feature_mode, num_windows):
        """
        Build dictionary mapping neuron_key -> trial data.
        Computes per-trial mean firing rates based on feature_mode.
        """
        neuron_dict = {}
        for (pid, uid), entry in _iter_neurons(mat_path):
            if pid not in patient_ids:
                continue

            frates = _get_field(entry, "firing_rates")
            loads = _get_field(entry, "trial_load").flatten()
            img_ids = _get_field(entry, "trial_imageIDs")

            # Extract features based on mode
            if feature_mode == "timefield":
                tfield = int(_get_field(entry, "time_field")[0][0]) - 1
                s = max(0, tfield - num_windows)
                e = min(frates.shape[1], tfield + num_windows + 1)
                mean_rates = frates[:, s:e].mean(axis=1)
            elif feature_mode == "wholetrial":
                mean_rates = frates.mean(axis=1)
            else:
                raise ValueError(f"Unknown feature_mode={feature_mode}. Use 'timefield' or 'wholetrial'.")

            neuron_dict[(pid, uid)] = dict(
                pid=pid,
                uid=uid,
                mean_rates=mean_rates,
                loads=loads,
                img_ids=img_ids,
            )
        return neuron_dict

    def _last_nonzero_labels(img_ids):
        """Extract last non-zero label from each trial's image IDs."""
        labels = []
        for row in img_ids:
            row_ = row.flatten()
            nz = row_[row_ != 0]
            labels.append(int(nz[-1]) if nz.size > 0 else 0)
        return np.asarray(labels)

    def _build_neuron_class_trials(neuron_records):
        """
        Create list of neurons formatted as {'class1': rates, ..., 'class5': rates}.
        Only uses load==1 trials with labels 1-5, requires ≥1 trial per class.
        """
        neurons = []
        
        # Group by patient for minimum neuron filter
        by_patient = {}
        for key, rec in neuron_records.items():
            by_patient.setdefault(rec["pid"], []).append(key)

        valid_pids = [pid for pid in patient_ids if pid in by_patient and len(by_patient[pid]) >= m]
        if not valid_pids:
            return []

        for pid in valid_pids:
            for key in by_patient[pid]:
                rec = neuron_records[key]
                mean_rates = rec["mean_rates"]
                loads = rec["loads"]
                labels_all = _last_nonzero_labels(rec["img_ids"])

                mask = (loads == 1) & (labels_all >= 1) & (labels_all <= 5)
                if not np.any(mask):
                    continue

                rates = mean_rates[mask]
                labels = labels_all[mask]

                # Require at least 1 trial per class
                if any((labels == c).sum() < 1 for c in (1, 2, 3, 4, 5)):
                    continue

                neurons.append({f"class{c}": rates[labels == c] for c in (1, 2, 3, 4, 5)})

        return neurons

    def _decode_5way(neurons):
        """
        Run 5-way decoding with label-shuffled null distribution.
        
        Returns:
            real_accs: list of accuracies from real labels
            null_means: list of mean accuracies from shuffled labels
            p_vs_null: probability that null mean ≥ observed mean
        """
        if len(neurons) == 0:
            return [], [], np.nan

        # Find global minimum trials per class
        mins = []
        for c in (1, 2, 3, 4, 5):
            mins.append(min(len(n[f"class{c}"]) for n in neurons))

        if any(mc < test_per_class for mc in mins):
            return [], [], np.nan

        # Downsample each neuron to global minimum
        neurons_ds = []
        for n in neurons:
            nd = {}
            for c, mc in zip((1, 2, 3, 4, 5), mins):
                seed = int(rng.integers(0, 2**31 - 1))
                nd[f"class{c}"] = resample(n[f"class{c}"], replace=False, n_samples=mc, random_state=seed)
            neurons_ds.append(nd)

        # Build class matrices (trials × neurons)
        class_mats = {}
        for c in (1, 2, 3, 4, 5):
            class_mats[c] = np.column_stack([nd[f"class{c}"] for nd in neurons_ds]).astype(np.float32)

        def _run_one(real_labels=True):
            """Run n_iterations of train/test splits."""
            accs = []
            for _ in range(n_iterations):
                Xtr, ytr, Xte, yte = [], [], [], []
                for c in (1, 2, 3, 4, 5):
                    mat = class_mats[c]
                    idx = rng.permutation(mat.shape[0])
                    te = idx[:test_per_class]
                    tr = idx[test_per_class:]
                    Xtr.append(mat[tr])
                    ytr.append(np.full(tr.size, c))
                    Xte.append(mat[te])
                    yte.append(np.full(te.size, c))
                    
                Xtr = np.vstack(Xtr)
                ytr = np.concatenate(ytr)
                Xte = np.vstack(Xte)
                yte = np.concatenate(yte)

                if not real_labels:
                    ytr = rng.permutation(ytr)

                clf = SVC(kernel="linear", decision_function_shape="ovr", random_state=0)
                clf.fit(Xtr, ytr)
                accs.append((clf.predict(Xte) == yte).mean() * 100.0)
            return accs

        # Run actual decoding
        real_accs = _run_one(real_labels=True)

        # Run null distribution (shuffled labels)
        null_means = []
        for _ in range(n_shuffles):
            null_accs = _run_one(real_labels=False)
            null_means.append(float(np.mean(null_accs)))

        observed_mean = float(np.mean(real_accs))
        p_vs_null = float(np.mean(np.asarray(null_means) >= observed_mean))

        return real_accs, null_means, p_vs_null

    # Load both datasets
    data_A = _extract_dataset(mat_file_A, feature_mode_A, num_windows_A)
    data_B = _extract_dataset(mat_file_B, feature_mode_B, num_windows_B)

    keys_A = set(data_A.keys())
    keys_B = set(data_B.keys())
    overlap = keys_A & keys_B

    # Remove overlapping neurons from both
    for k in overlap:
        data_A.pop(k, None)
        data_B.pop(k, None)

    # Build neuron trial lists
    neurons_A = _build_neuron_class_trials(data_A)
    neurons_B = _build_neuron_class_trials(data_B)

    # Decode each cell type
    real_A, null_means_A, pA = _decode_5way(neurons_A)
    real_B, null_means_B, pB = _decode_5way(neurons_B)

    # Mann-Whitney U test for comparison
    if len(real_A) and len(real_B):
        u, p_mw = mannwhitneyu(real_A, real_B, alternative="greater")
    else:
        p_mw = np.nan

    out = {
        "kept_counts": {
            "A_neurons_kept_after_overlap": len(data_A),
            "B_neurons_kept_after_overlap": len(data_B),
            "overlap_removed": len(overlap),
            "A_neurons_used_in_decoder": len(neurons_A),
            "B_neurons_used_in_decoder": len(neurons_B),
        },
        "A": {
            "real_accuracies": real_A,
            "null_means": null_means_A,
            "p_vs_null": pA,
            "mean_real": float(np.mean(real_A)) if len(real_A) else np.nan,
            "mean_null": float(np.mean(null_means_A)) if len(null_means_A) else np.nan,
        },
        "B": {
            "real_accuracies": real_B,
            "null_means": null_means_B,
            "p_vs_null": pB,
            "mean_real": float(np.mean(real_B)) if len(real_B) else np.nan,
            "mean_null": float(np.mean(null_means_B)) if len(null_means_B) else np.nan,
        },
        "comparison": {
            "p_A_greater_B_mannwhitney": p_mw,
        }
    }

    # Print summary
    print("=== Overlap handling ===")
    print(f"Removed overlap neurons: {len(overlap)}")
    print(f"A kept: {len(data_A)} | A used in decoder: {len(neurons_A)}")
    print(f"B kept: {len(data_B)} | B used in decoder: {len(neurons_B)}")
    print("\n=== Decoding results ===")
    print(f"A mean acc: {out['A']['mean_real']:.2f}% | p_vs_null: {out['A']['p_vs_null']:.4g}")
    print(f"B mean acc: {out['B']['mean_real']:.2f}% | p_vs_null: {out['B']['p_vs_null']:.4g}")
    print(f"Mann-Whitney U p(A>B): {p_mw:.4g}")

    return out


# Run comparison
out = decode_compare_two_celltypes_excluding_overlap(
    mat_file_A="cue_specific_TC.mat",
    mat_file_B="CC_zscore.mat",
    feature_mode_A="timefield",
    num_windows_A=0,
    feature_mode_B="wholetrial",
    n_iterations=1000,
    n_shuffles=1,
    test_per_class=2,
    random_state=20250710,
)