import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter1d
from sklearn.naive_bayes import GaussianNB
from scipy.stats import sem, ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns

def _significance_stars(p: float) -> str:
    """Return significance stars for a *p*‑value."""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

def analyze_rt_fast_slow_decoding_errors(mat_file_path: str, *,
                                         binwidth: float = 0.1,
                                         neuron_threshold: int = 5,
                                         quantile: float = 0.5,
                                         min_trials: int = 100,
                                         random_state: int = 42,
                                         plot: bool = True,
                                         trial_level_analysis: bool = False,
                                         correct_only: bool = False):

    """Train on middle‑80 % of trials, test on top/bottom 10 % RT.

    Parameters
    ----------
    mat_file_path : str
        Path to neural_data MAT file.
    binwidth : float
        Bin size (s) used when converting bin indices to seconds.
    neuron_threshold : int
        Minimum # neurons per patient to include.
    quantile : float, default 0.10
        Fraction of trials at each RT tail to hold out for *testing*.
    min_trials : int, default 109
        Exclude patients with fewer trials than this after NaN removal.
    random_state : int
        RNG seed for reproducibility.
    plot : bool
        If True, show summary bar chart with per‑patient dots.
    """

    rng = np.random.default_rng(random_state)

    # ------ load MATLAB struct ------
    m = scipy.io.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
    nd = m["neural_data"].flatten()

    patient_ids_all = [int(e.patient_id) for e in nd]
    firing_rates_all = [e.firing_rates for e in nd]
    rt_all = [e.trial_RT for e in nd]

    unique_pids = np.unique(patient_ids_all)

    # global aggregators
    err_fast_all, err_slow_all = [], []
    per_patient_fast_means, per_patient_slow_means = [], []

    print("Included patients:")
    for pid in unique_pids:
        # gather neuron entries for this patient
        idx_neurons = [i for i, p in enumerate(patient_ids_all) if p == pid]
        if len(idx_neurons) < neuron_threshold:
            continue

        fr_list = [np.asarray(firing_rates_all[i]) for i in idx_neurons]
        rt_vec = np.asarray(rt_all[idx_neurons[0]]).astype(float)
        load_vec = np.asarray(nd[idx_neurons[0]].trial_load, dtype=int)

        non_nan = ~np.isnan(rt_vec)
        


        if correct_only:
            correct_vec = np.asarray(nd[idx_neurons[0]].trial_correctness, dtype=bool)
            non_nan &= correct_vec

        if non_nan.sum() < min_trials:
            continue

        rt_vec = rt_vec[non_nan]
        load_vec = load_vec[non_nan]
        fr_list = [fr[non_nan] for fr in fr_list]


        fast_trials_all, slow_trials_all, train_trials_all = [], [], []

        for load_val in np.unique(load_vec):
            load_mask = load_vec == load_val
            rt_sub = rt_vec[load_mask]

            if len(rt_sub) < 4:
                continue  # too few trials

            q_low = np.quantile(rt_sub, quantile)
            q_high = np.quantile(rt_sub, 1 - quantile)

            fast_idx = np.where(load_mask & (rt_vec <= q_low))[0]
            slow_idx = np.where(load_mask & (rt_vec >= q_high))[0]
            train_idx = np.where(load_mask & (rt_vec > q_low) & (rt_vec < q_high))[0]

            fast_trials_all.append(fast_idx)
            slow_trials_all.append(slow_idx)
            train_trials_all.append(train_idx)

        # Concatenate across loads
        fast_trials = np.concatenate(fast_trials_all)
        slow_trials = np.concatenate(slow_trials_all)
        train_trials = np.concatenate(train_trials_all)


        if len(fast_trials) == 0 or len(slow_trials) == 0 or len(train_trials) == 0:
            continue

        # stack neurons: [time_bins × trials × neurons]
        fr_stack = np.stack(fr_list, axis=2).transpose(1, 0, 2)  # time_bins, trials, neurons
        time_bins, trials, neurons = fr_stack.shape

        # training matrix
        X_train = fr_stack[:, train_trials, :].reshape(-1, neurons)
        y_train = np.repeat(np.arange(time_bins), len(train_trials))

        # testing matrices
        X_fast = fr_stack[:, fast_trials, :].reshape(-1, neurons)
        y_fast = np.repeat(np.arange(time_bins), len(fast_trials))
        X_slow = fr_stack[:, slow_trials, :].reshape(-1, neurons)
        y_slow = np.repeat(np.arange(time_bins), len(slow_trials))

        # train classifier
        clf = GaussianNB()
        clf.fit(X_train, y_train)

        # predict & errors
        pred_fast = clf.predict(X_fast)
        pred_slow = clf.predict(X_slow)

        if trial_level_analysis:
            err_fast = np.mean(np.abs(pred_fast.reshape(time_bins, -1) - y_fast.reshape(time_bins, -1)), axis=0) * binwidth
            err_slow = np.mean(np.abs(pred_slow.reshape(time_bins, -1) - y_slow.reshape(time_bins, -1)), axis=0) * binwidth
        else:
            err_fast = np.abs(pred_fast - y_fast) * binwidth
            err_slow = np.abs(pred_slow - y_slow) * binwidth


        err_fast_all.extend(err_fast)
        err_slow_all.extend(err_slow)
        per_patient_fast_means.append(err_fast.mean())
        per_patient_slow_means.append(err_slow.mean())
        print(pid)


    t_stat, p_val = ttest_rel(err_fast_all, err_slow_all)
    print(f"Paired t‑test (Fast vs Slow): t = {t_stat:.3f}, p = {p_val:.3e}")

    if not plot:
        return err_fast_all, err_slow_all

    # bar chart with per‑patient dots
    mean_fast, mean_slow = np.mean(err_fast_all), np.mean(err_slow_all)
    sem_fast, sem_slow = sem(err_fast_all), sem(err_slow_all)
    labels = ["Fast", "Slow"]
    means = [mean_fast, mean_slow]
    errors = [sem_fast, sem_slow]

    plt.figure(figsize=(4, 6))
    bars = plt.bar(labels, means, yerr=errors, capsize=5, alpha=0.5,
                   color=["royalblue", "firebrick"])

    max_y = max(means) + max(errors) * 1.5
    plt.plot([0, 0, 1, 1], [max_y, max_y + 0.02, max_y + 0.02, max_y],
             color="black", linewidth=1.5)

    star = _significance_stars(p_val)
    plt.text(0.5, max_y + 0.02, star, ha="center", fontsize=14)

    # per‑patient lines
    for f, s in zip(per_patient_fast_means, per_patient_slow_means):
        plt.plot([0, 1], [f, s], color="gray", alpha=0.5)
        plt.scatter(0, f, color="gray", alpha=0.5, zorder=10)
        plt.scatter(1, s, color="gray", alpha=0.5, zorder=10)

    plt.ylabel("Mean Decoding Error (s)")
    plt.title("Decoding Error — Fast vs Slow RT")
    plt.tight_layout()
    plt.show()

    return err_fast_all, err_slow_all

analyze_rt_fast_slow_decoding_errors('final.mat', trial_level_analysis= False, correct_only= True)
# analyze_rt_fast_slow_decoding_errors_correctonly('100msTCdata_G2RT.mat')