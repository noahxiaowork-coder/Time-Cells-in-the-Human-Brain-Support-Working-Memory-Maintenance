# ---------------------------------------------------------------
# Integrated per-patient concept- vs time-cell decoder pipeline
# ---------------------------------------------------------------
import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.utils import resample
from collections import defaultdict


DIR = '/Users/xiangxuankong/Desktop/Human TC Support WM/Figures/Figure 5/'
import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC
from collections import defaultdict, Counter


def _last_nonzero_label(img_row):
    """Extract last non-zero entry (1-based IDs 1‥5)."""
    nz = img_row[img_row != 0]
    return nz[-1] if nz.size else 0


def _count_preferred_images(entries, valid_ids=range(1, 6)):
    """
    Count neurons per preferred_image ID in a list of entries.
    Assumes each entry has a field 'preferred_image' that is scalar-like.
    """
    prefs = []
    for e in entries:
        # robust extraction: handles shapes like (1,1), (1,), etc.
        try:
            v = int(np.asarray(e['preferred_image']).ravel()[0])
            prefs.append(v)
        except Exception:
            # skip if anything is off
            continue

    counts = Counter(prefs)
    # ensure all image IDs are present in the dict
    return {img_id: counts.get(img_id, 0) for img_id in valid_ids}


def _format_pref_counts(counts):
    """Format preferred-image counts dict into a compact string."""
    return ", ".join(f"img{img_id}={counts[img_id]}" 
                     for img_id in sorted(counts.keys()))


def _build_patient_data(mat_data, pid, kind="concept", num_windows=0, entries=None):
    """
    Assemble one patient’s trial×neuron matrix and label vector.

    kind='concept' → whole-trial mean firing (ignores time_field)
    kind='time'    → mean in [time_field-num_windows … time_field+num_windows]

    If `entries` is provided, it should be the subset of mat_data['neural_data'][0]
    for this patient; otherwise this function will compute it.
    """
    if entries is None:
        entries = [e for e in mat_data['neural_data'][0]
                   if int(e['patient_id'][0][0]) == pid]

    if not entries:
        return None  # patient absent

    # Trial-level metadata (identical across neurons)
    loads   = entries[0]['trial_load'].flatten()
    img_ids = entries[0]['trial_imageIDs']

    labels_all = np.array([_last_nonzero_label(row) for row in img_ids])
    mask = (loads == 1) & (labels_all >= 1) & (labels_all <= 5)
    keep_idx = np.where(mask)[0]
    if keep_idx.size == 0:
        return None

    y = labels_all[keep_idx]
    X_cols = []

    for e in entries:
        fr = e['firing_rates']  # (trials, time_bins)

        if kind == "concept":
            feat = fr.mean(axis=1)                      # whole-trial mean
        elif kind == "time":
            tf = int(e['time_field'][0][0]) - 1         # 0-based index
            lo = max(0, tf - num_windows)
            hi = min(fr.shape[1], tf + num_windows + 1)
            feat = fr[:, lo:hi].mean(axis=1)
        else:
            raise ValueError("kind must be 'concept' or 'time'")

        X_cols.append(feat[keep_idx])

    X = np.vstack(X_cols).T.astype(np.float32)          # trials × neurons
    return X, y


import numpy as np
from collections import defaultdict
from scipy.io import loadmat, savemat   # NEW

def decode_per_patient(concept_mat,
                       time_mat,
                       patient_ids=range(1, 22),
                       min_concept_cells=0,     # NEW
                       min_time_cells=0,        # NEW
                       num_windows=0,
                       test_per_class=2,
                       n_iterations=1000,
                       random_state=42,
                       return_preds=False
                       ):

    rng = np.random.default_rng(random_state)
    concept_data = loadmat(concept_mat)
    time_data    = loadmat(time_mat)

    results = defaultdict(dict)

    # NEW: global dicts to save per-decoder correctness across patients
    concept_correct = {}
    time_correct    = {}

    for pid in patient_ids:
        # --- patient-specific entries for both datasets --------------------
        c_entries = [e for e in concept_data['neural_data'][0]
                     if int(e['patient_id'][0][0]) == pid]
        t_entries = [e for e in time_data['neural_data'][0]
                     if int(e['patient_id'][0][0]) == pid]

        if not c_entries or not t_entries:
            continue  # patient not present in one of the datasets

        # --- build feature matrices ----------------------------------------
        cdata = _build_patient_data(concept_data, pid, "concept",
                                    num_windows=num_windows,
                                    entries=c_entries)
        tdata = _build_patient_data(time_data,    pid, "time",
                                    num_windows=num_windows,
                                    entries=t_entries)

        if cdata is None or tdata is None:
            continue  # no valid trials after masking

        Xc, y  = cdata              # concept features & labels
        Xt, _  = tdata              # time-cell features (labels identical)

        # number of neurons (cells) contributing to decoding
        n_concept_cells = Xc.shape[1]
        n_time_cells    = Xt.shape[1]

        # --- NEW FILTER: enforce minimum cells per patient -----------------
        if (n_concept_cells < min_concept_cells) or (n_time_cells < min_time_cells):
            continue

        # preferred-image distributions for this patient
        c_pref_counts = _count_preferred_images(c_entries)
        t_pref_counts = _count_preferred_images(t_entries)

        # classes present
        cls, cls_counts = np.unique(y, return_counts=True)
        if np.any(cls_counts < test_per_class):
            continue  # not enough trials for balanced split

        # storage
        acc_c, acc_t, acc_union, synergy_pct = [], [], [], []
        if return_preds:
            preds_c, preds_t, y_all = [], [], []

        # NEW: correctness matrices (will be stacked after loop)
        corr_c_iters = []   # list of (n_test_trials,) for concept
        corr_t_iters = []   # list of (n_test_trials,) for time

        for _ in range(n_iterations):
            train_idx, test_idx = [], []

            # balanced per-class split
            for c in cls:
                idx_c = np.where(y == c)[0]
                idx_c = rng.permutation(idx_c)
                test_c  = idx_c[:test_per_class]
                train_c = idx_c[test_per_class:]
                train_idx.extend(train_c)
                test_idx.extend(test_c)

            train_idx = np.array(train_idx)
            test_idx  = np.array(test_idx)

            # --- train two SVMs with identical train/test indices -----------
            svm_c = SVC(kernel="linear", decision_function_shape="ovr",
                        random_state=random_state)
            svm_t = SVC(kernel="linear", decision_function_shape="ovr",
                        random_state=random_state)
            svm_c.fit(Xc[train_idx], y[train_idx])
            svm_t.fit(Xt[train_idx], y[train_idx])

            ypred_c = svm_c.predict(Xc[test_idx])
            ypred_t = svm_t.predict(Xt[test_idx])
            y_true  = y[test_idx]

            if return_preds:
                preds_c.extend(ypred_c)
                preds_t.extend(ypred_t)
                y_all.extend(y_true)

            # case masks
            both_correct = (ypred_c == y_true) & (ypred_t == y_true)
            one_correct  = ((ypred_c == y_true) ^ (ypred_t == y_true))

            # metrics
            acc_c.append((ypred_c == y_true).mean()*100)
            acc_t.append((ypred_t == y_true).mean()*100)
            acc_union.append((both_correct | one_correct).mean()*100)
            synergy_pct.append(both_correct.mean()*100)  # % both-correct

            # NEW: correctness vectors per iteration (1/0 for each test trial)
            correct_c = (ypred_c == y_true).astype(np.int8)
            correct_t = (ypred_t == y_true).astype(np.int8)
            corr_c_iters.append(correct_c)
            corr_t_iters.append(correct_t)

        # save arrays
        results[pid]["concept_acc"]   = np.array(acc_c)
        results[pid]["time_acc"]      = np.array(acc_t)
        results[pid]["union_acc"]     = np.array(acc_union)
        results[pid]["synergy_pct"]   = np.array(synergy_pct)

        if return_preds:
            results[pid]['pred_c'] = np.array(preds_c, dtype=np.int16)
            results[pid]['pred_t'] = np.array(preds_t, dtype=np.int16)
            results[pid]['y_true'] = np.array(y_all,  dtype=np.int16)

        # NEW: stack correctness into matrices (n_iterations x n_test_trials)
        corr_c_mat = np.vstack(corr_c_iters)  # shape: (n_iterations, n_tests)
        corr_t_mat = np.vstack(corr_t_iters)

        results[pid]['correct_c'] = corr_c_mat
        results[pid]['correct_t'] = corr_t_mat

        # NEW: add to dicts for saving to .mat (key per patient)
        concept_correct[f'pid_{pid:02d}'] = corr_c_mat
        time_correct[f'pid_{pid:02d}']    = corr_t_mat

        # quick text summary per patient
        uc_mean  = results[pid]["union_acc"].mean()
        syn_mean = results[pid]["synergy_pct"].mean()

        print(
            f"Patient {pid:02d}: union acc = {uc_mean:5.2f}%  "
            f"(both-correct ≈ {syn_mean:4.2f}% of tests)  "
            f"[concept cells: {n_concept_cells}, time cells: {n_time_cells}]"
        )
        print(
            f"    Concept cells by preferred_image: "
            f"{_format_pref_counts(c_pref_counts)}"
        )
        print(
            f"    Time/cue cells by preferred_image: "
            f"{_format_pref_counts(t_pref_counts)}"
        )

    # NEW: save two .mat files after all patients are processed
    # Each contains variables 'pid_01', 'pid_02', ... with correctness matrices
    if concept_correct:
        savemat('concept_correctness.mat', concept_correct)
    if time_correct:
        savemat('time_correctness.mat',   time_correct)

    return results

results = decode_per_patient(
    concept_mat="100msCCdata_global.mat",
    time_mat    ="Nov27_Nov18.mat",
    patient_ids = range(1, 22),
    num_windows = 0,          # ± 0 → use exact time_field bin
    test_per_class = 2,
    n_iterations   = 1000,
    random_state   = 42,
    return_preds = True
)


import pickle
# Save
with open("Figure 4/decode_results_strict.pkl", "wb") as f:
    pickle.dump(results, f)
# # Load later
# with open("Figure 4/decode_results.pkl", "rb") as f:
#     results = pickle.load(f)

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_first_iteration_correctness(results,
                                     n_iterations=1000,
                                     save_dir=None,
                                     patients=None):
    """
    For each patient, take the FIRST iteration's test set and plot
    binary correctness (1 = correct, 0 = incorrect) for concept vs time cells.

    x-axis: test trial index *within that first iteration*.
    """
    if patients is None:
        # Use all patients that actually have results
        patients = sorted(results.keys())

    for pid in patients:
        if pid not in results:
            continue

        r = results[pid]

        # Skip patients that don't have prediction arrays
        if not all(k in r for k in ['pred_c', 'pred_t', 'y_true']):
            continue

        y_all   = r['y_true']
        pred_c_all = r['pred_c']
        pred_t_all = r['pred_t']

        # How many test trials per iteration?
        n_total = len(y_all)
        if n_total == 0:
            continue

        n_test_per_iter = n_total // n_iterations
        if n_test_per_iter == 0:
            continue

        # Slice corresponding to the FIRST iteration
        y_true  = y_all[:n_test_per_iter]
        pred_c  = pred_c_all[:n_test_per_iter]
        pred_t  = pred_t_all[:n_test_per_iter]

        # Binary correctness (1 = correct, 0 = incorrect)
        correct_c = (pred_c == y_true).astype(int)
        correct_t = (pred_t == y_true).astype(int)

        x = np.arange(n_test_per_iter)

        plt.figure(figsize=(7, 3))
        plt.step(x, correct_c, where='mid', label='Concept decoder', linewidth=1.5)
        plt.step(x, correct_t, where='mid', label='Time decoder', linestyle='--', linewidth=1.5)

        plt.yticks([0, 1], ['Incorrect', 'Correct'])
        plt.ylim(-0.1, 1.1)
        plt.xlabel('Test trial index (first iteration)')
        plt.ylabel('Correct (1) / Incorrect (0)')
        plt.title(f'Patient {pid:02d}')
        plt.legend()
        plt.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f'patient_{pid:02d}_iter1_correctness.svg')
            plt.savefig(out_path, format = 'svg')
            plt.close()
        else:
            plt.show()

plot_first_iteration_correctness(results, n_iterations=1000, save_dir=DIR)

# --------------------------------------------------------------------
#  Post-analysis: per-patient stats, permutation tests, complementarity
# --------------------------------------------------------------------
import numpy as np

def permutation_pvalue(values, chance, n_perm=10_000, rng=None):
    """
    One-sided permutation test:
    H0: mean(values) == chance
    HA: mean(values)  > chance
    Uses sign-flipping of (values - chance).
    """
    if rng is None:
        rng = np.random.default_rng()
    diffs = values - chance
    obs   = diffs.mean()
    perm_means = []
    for _ in range(n_perm):
        flips = rng.choice([-1, 1], size=len(diffs))
        perm_means.append((diffs * flips).mean())
    perm_means = np.array(perm_means)
    p = np.mean(perm_means >= obs)   # greater-than because HA is ">"
    return p, obs + chance           # return p-value and observed mean

def summarize_results(results):
    print("Patient |  Concept (mean±SD) |   Time (mean±SD) |  Union (mean±SD) "
          "|  p_concept |  p_time |  p_union | "
          "UniqC%  UniqT%  Both%  → Complementarity%")
    print("─"*128)

    for pid, dat in results.items():
        c = dat["concept_acc"]
        t = dat["time_acc"]
        u = dat["union_acc"]
        b = dat["synergy_pct"]

        # means & SDs
        mc, sc = c.mean(), c.std()
        mt, st = t.mean(), t.std()
        mu, su = u.mean(), u.std()

        # permutation tests
        pc, _ = permutation_pvalue(c, chance=20)
        pt, _ = permutation_pvalue(t, chance=20)
        pu, _ = permutation_pvalue(u, chance=36)

        # complementarity
        uniq_c = mc - b.mean()
        uniq_t = mt - b.mean()
        compl  = uniq_c + uniq_t      # % trials with exactly one correct
        # print in the same units (percentage of trials)
        print(f"  {pid:02d}   | "
              f"{mc:5.1f}±{sc:4.1f} | "
              f"{mt:5.1f}±{st:4.1f} | "
              f"{mu:5.1f}±{su:4.1f} | "
              f"{pc:8.4f} | {pt:7.4f} | {pu:7.4f} | "
              f"{uniq_c:6.2f} {uniq_t:6.2f} {b.mean():6.2f} "
              f"→ {compl:6.2f}")

# ---- call it -------------------------------------------------------
summarize_results(results)

# import pandas as pd, seaborn as sns, matplotlib.pyplot as plt

# # reshape results------------------------------------
# rows = []
# for pid, d in results.items():
#     both  = d['synergy_pct'].mean()
#     c_only = d['concept_acc'].mean() - both
#     t_only = d['time_acc'].mean()    - both
#     none   = 100 - (both + c_only + t_only)
#     rows.append(dict(pid=pid, both=both, c_only=c_only, t_only=t_only, none=none))

# df = pd.DataFrame(rows).melt('pid', var_name='Outcome', value_name='Pct')
# order = df.groupby('pid').apply(lambda x: 100-x.loc[x.Outcome.eq('none'), 'Pct'].iloc[0]
#                                 ).sort_values(ascending=False).index
# palette = {'both':'#238b45', 'c_only':'#6baed6', 't_only':'#fdb863', 'none':'#aaaaaa'}

# # plot------------------------------------------------
# plt.figure(figsize=(8,3))
# sns.barplot(data=df, x='pid', y='Pct', hue='Outcome',
#             order=order, hue_order=['both','c_only','t_only','none'],
#             palette=palette, edgecolor='k', linewidth=.4)
# plt.ylabel('% test trials')
# plt.xlabel('Patient')
# plt.legend(title='', bbox_to_anchor=(1,1))
# sns.despine(); plt.tight_layout();plt.savefig(DIR + 'sthelse.svg', format = 'svg');plt.show()



import numpy as np
from scipy.stats import binomtest, ttest_1samp, wilcoxon


# ---------- 1. Joint correctness patterns per patient ----------

def joint_outcomes_for_patient(res_patient):
    """
    Build counts of correctness patterns for a single patient.

    CC      = both decoders correct
    C_only  = concept correct, time wrong
    T_only  = time correct, concept wrong
    II      = both wrong
    """
    p_c = res_patient['pred_c']
    p_t = res_patient['pred_t']
    y   = res_patient['y_true']

    both_correct = (p_c == y) & (p_t == y)
    c_only       = (p_c == y) & (p_t != y)
    t_only       = (p_c != y) & (p_t == y)
    both_wrong   = (p_c != y) & (p_t != y)

    counts = {
        "CC": int(both_correct.sum()),
        "C_only": int(c_only.sum()),
        "T_only": int(t_only.sum()),
        "II": int(both_wrong.sum()),
    }
    return counts


# ---------- 2. Conditional probabilities ----------

def conditional_probs(counts):
    """
    Compute:
        p_T_given_Cwrong = P(Time correct | Concept wrong)
        p_C_given_Twrong = P(Concept correct | Time wrong)
    Returns (p_T_given_Cwrong, p_C_given_Twrong), possibly np.nan if denominator=0.
    """
    N_T_only = counts["T_only"]
    N_C_only = counts["C_only"]
    N_II     = counts["II"]

    # Time correct given Concept wrong
    denom_Cwrong = N_T_only + N_II
    p_T_given_Cwrong = N_T_only / denom_Cwrong if denom_Cwrong > 0 else np.nan

    # Concept correct given Time wrong
    denom_Twrong = N_C_only + N_II
    p_C_given_Twrong = N_C_only / denom_Twrong if denom_Twrong > 0 else np.nan

    return p_T_given_Cwrong, p_C_given_Twrong


# ---------- 3. Binomial tests for complementarity per patient ----------

def test_complementarity(counts, p0=0.5):
    """
    For each direction, run a one-sided binomial test:
        H0: p = p0   vs   H1: p > p0

    Returns (test_T_given_Cwrong, test_C_given_Twrong),
    each is either a BinomTestResult or None if there are no error trials.
    """
    N_T_only = counts["T_only"]
    N_C_only = counts["C_only"]
    N_II     = counts["II"]

    # Time correct given Concept wrong
    n_Cwrong = N_T_only + N_II
    if n_Cwrong > 0:
        test_T_given_Cwrong = binomtest(
            N_T_only,
            n_Cwrong,
            p=p0,
            alternative='greater'
        )
    else:
        test_T_given_Cwrong = None

    # Concept correct given Time wrong
    n_Twrong = N_C_only + N_II
    if n_Twrong > 0:
        test_C_given_Twrong = binomtest(
            N_C_only,
            n_Twrong,
            p=p0,
            alternative='greater'
        )
    else:
        test_C_given_Twrong = None

    return test_T_given_Cwrong, test_C_given_Twrong


# ---------- 4. Wrap per-patient stats ----------

def compute_per_patient_complementarity(results):
    """
    Iterate over all patients in the 'results' dict produced by decode_per_patient
    and compute:

        - counts (CC, C_only, T_only, II)
        - conditional probabilities
        - binomial tests

    Returns a dict keyed by pid with all these stats.
    """
    per_patient = {}

    for pid, res in results.items():
        counts = joint_outcomes_for_patient(res)
        p_T_given_Cwrong, p_C_given_Twrong = conditional_probs(counts)
        test_T, test_C = test_complementarity(counts, p0=0.5)

        per_patient[pid] = {
            "counts": counts,
            "p_T_given_Cwrong": p_T_given_Cwrong,
            "p_C_given_Twrong": p_C_given_Twrong,
            "test_T_given_Cwrong": test_T,
            "test_C_given_Twrong": test_C,
        }

    return per_patient


# ---------- 5. Group-level tests across patients ----------

def group_level_complementarity_tests(per_patient_stats):
    """
    Use patient-level conditional probabilities as data points and test vs 0.5.

    Returns a dict with arrays and scipy test results.
    """
    p_T_list = []
    p_C_list = []

    for pid, st in per_patient_stats.items():
        if not np.isnan(st["p_T_given_Cwrong"]):
            p_T_list.append(st["p_T_given_Cwrong"])
        if not np.isnan(st["p_C_given_Twrong"]):
            p_C_list.append(st["p_C_given_Twrong"])

    p_T_arr = np.array(p_T_list)
    p_C_arr = np.array(p_C_list)

    out = {
        "p_T_given_Cwrong": p_T_arr,
        "p_C_given_Twrong": p_C_arr,
        "tests": {}
    }

    if p_T_arr.size > 0:
        out["tests"]["T_ttest"] = ttest_1samp(p_T_arr, 0.5, alternative='greater')
        out["tests"]["T_wilcoxon"] = wilcoxon(p_T_arr - 0.5, alternative='greater')
    else:
        out["tests"]["T_ttest"] = None
        out["tests"]["T_wilcoxon"] = None

    if p_C_arr.size > 0:
        out["tests"]["C_ttest"] = ttest_1samp(p_C_arr, 0.5, alternative='greater')
        out["tests"]["C_wilcoxon"] = wilcoxon(p_C_arr - 0.5, alternative='greater')
    else:
        out["tests"]["C_ttest"] = None
        out["tests"]["C_wilcoxon"] = None

    return out


# ---------- 6. Synergy: union gain per iteration & group-level ----------

def union_gain_per_patient(res_patient):
    """
    For a single patient: union gain = union_acc - max(concept_acc, time_acc)
    per iteration. Returns the gain array.
    """
    ca = np.asarray(res_patient["concept_acc"])
    ta = np.asarray(res_patient["time_acc"])
    ua = np.asarray(res_patient["union_acc"])

    best_single = np.maximum(ca, ta)
    gain = ua - best_single   # per-iteration gain in percentage points
    return gain


def compute_union_gain_all_patients(results):
    """
    Compute per-patient mean union gain and group-level test vs 0.
    """
    mean_gains = {}
    gain_list = []

    for pid, res in results.items():
        gain = union_gain_per_patient(res)
        mean_gain = float(np.mean(gain))
        mean_gains[pid] = mean_gain
        gain_list.append(mean_gain)

    gain_arr = np.array(gain_list)

    # Test whether mean gain across patients is > 0
    if gain_arr.size > 0:
        ttest_gain = ttest_1samp(gain_arr, 0.0, alternative='greater')
        wilcoxon_gain = wilcoxon(gain_arr, alternative='greater')
    else:
        ttest_gain = None
        wilcoxon_gain = None

    return {
        "per_patient_mean_gain": mean_gains,
        "gain_array": gain_arr,
        "ttest_gain": ttest_gain,
        "wilcoxon_gain": wilcoxon_gain,
    }


# ---------- 7. Run everything and print a concise summary ----------

per_patient_stats = compute_per_patient_complementarity(results)
group_stats = group_level_complementarity_tests(per_patient_stats)
union_gain_stats = compute_union_gain_all_patients(results)

print("\n=== Per-patient complementarity (examples) ===")
for pid in sorted(per_patient_stats.keys()):
    st = per_patient_stats[pid]
    c = st["counts"]
    print(f"\nPatient {pid:02d}")
    print(f"  Counts: CC={c['CC']}, C_only={c['C_only']}, T_only={c['T_only']}, II={c['II']}")
    print(f"  p(T | C wrong) = {st['p_T_given_Cwrong']:.3f}   "
          f"p(C | T wrong) = {st['p_C_given_Twrong']:.3f}")

    if st["test_T_given_Cwrong"] is not None:
        tT = st["test_T_given_Cwrong"]
        ciT = tT.proportion_ci()
        print(f"    Binom T|Cwrong: p={tT.pvalue:.3g}, CI=({ciT.low:.3f}, {ciT.high:.3f})")
    if st["test_C_given_Twrong"] is not None:
        tC = st["test_C_given_Twrong"]
        ciC = tC.proportion_ci()
        print(f"    Binom C|Twrong: p={tC.pvalue:.3g}, CI=({ciC.low:.3f}, {ciC.high:.3f})")

print("\n=== Group-level complementarity ===")
pT = group_stats["p_T_given_Cwrong"]
pC = group_stats["p_C_given_Twrong"]
print(f"  N patients (T|C wrong): {pT.size}, mean={np.nanmean(pT):.3f}")
print(f"  N patients (C|T wrong): {pC.size}, mean={np.nanmean(pC):.3f}")
print("  T|Cwrong t-test vs 0.5:", group_stats["tests"]["T_ttest"])
print("  T|Cwrong Wilcoxon vs 0.5:", group_stats["tests"]["T_wilcoxon"])
print("  C|Twrong t-test vs 0.5:", group_stats["tests"]["C_ttest"])
print("  C|Twrong Wilcoxon vs 0.5:", group_stats["tests"]["C_wilcoxon"])

print("\n=== Union-accuracy synergy (union gain) ===")
mg = union_gain_stats["per_patient_mean_gain"]
for pid in sorted(mg.keys()):
    print(f"  Patient {pid:02d}: mean union gain = {mg[pid]:.3f} percentage points")
print("  Group mean union gain =", np.mean(union_gain_stats["gain_array"]))
print("  t-test gain>0:", union_gain_stats["ttest_gain"])
print("  Wilcoxon gain>0:", union_gain_stats["wilcoxon_gain"])




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

acc = []
for pid, d in results.items():
    for lbl, arr in [('Concept', d['concept_acc']),
                     ('Time',    d['time_acc']),
                     ('Union',   d['union_acc'])]:
        acc.extend([dict(pid=pid, dec=lbl, val=v) for v in arr])
df = pd.DataFrame(acc)

plt.figure(figsize=(4,3))
sns.violinplot(data=df, x='dec', y='val', inner=None, palette='Set2')
sns.boxplot(data=df, x='dec', y='val', width=.15, showcaps=False,
            boxprops={'zorder':3}, showfliers=False, whiskerprops={'linewidth':0})
sns.stripplot(data=df.groupby(['pid','dec']).val.mean().reset_index(),
              x='dec', y='val', hue='pid', palette='gray', dodge=True, marker='o',
              size=3, linewidth=0, alpha=.8, legend=False)
plt.ylabel('Accuracy (%)'); plt.xlabel('')
sns.despine(); plt.tight_layout();plt.savefig(DIR + 'sth.svg', format = 'svg');plt.show()



import numpy as np
pts = [(d['union_acc'].mean(), d['synergy_pct'].mean()) for d in results.values()]
u, s = np.array(pts).T

plt.figure(figsize=(3.2,3.2))
plt.scatter(u, s, c='k')
plt.plot([u.min()-1, u.max()+1], [u.min()-1, u.max()+1], '--', color='0.6')
plt.xlabel('Union accuracy (%)')
plt.ylabel('Synergy (both correct) (%)')
sns.despine(); 
plt.tight_layout();
plt.savefig(DIR + 'something.svg', format = 'svg');
plt.show()



# from sklearn.metrics import mutual_info_score
# import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
# sns.set(style='white')

# def interaction_information(c_pred, t_pred, y_true):
#     joint = [f'{c}_{t}' for c, t in zip(c_pred, t_pred)]
#     ii_nat = (mutual_info_score(c_pred, y_true) +
#               mutual_info_score(t_pred, y_true) -
#               mutual_info_score(joint,  y_true))
#     return ii_nat / np.log(2)             # convert nats → bits

# rows = [{'Patient': pid,
#          'II_bits': interaction_information(d['pred_c'],
#                                             d['pred_t'],
#                                             d['y_true'])}
#         for pid, d in results.items()]

# df = pd.DataFrame(rows)

# plt.figure(figsize=(6,2.5))
# sns.barplot(data=df, x='Patient', y='II_bits',
#             palette='vlag', edgecolor='k')
# plt.axhline(0, color='k', lw=.8)
# plt.ylabel('Interaction information (bits)')
# plt.xlabel('Patient')
# sns.despine(); plt.tight_layout(); 
# plt.savefig(DIR + 'mutualinfo.svg', format = 'svg');
# plt.show()


# # Per-patient correctness time series: Concept vs Time
# for pid, d in results.items():
#     if not all(k in d for k in ['pred_c', 'pred_t', 'y_true']):
#         continue  # safety check in case return_preds=False for some reason

#     y_true = d['y_true']
#     pred_c = d['pred_c']
#     pred_t = d['pred_t']

#     # 0/1 correctness vectors
#     corr_c = (pred_c == y_true).astype(int)
#     corr_t = (pred_t == y_true).astype(int)

#     # Optional: rolling mean if you want smoother curves (comment out if not needed)
#     # window = 50
#     # if len(corr_c) >= window:
#     #     ker = np.ones(window) / window
#     #     corr_c_smooth = np.convolve(corr_c, ker, mode='valid')
#     #     corr_t_smooth = np.convolve(corr_t, ker, mode='valid')
#     #     x_c = np.arange(len(corr_c_smooth))
#     #     x_t = np.arange(len(corr_t_smooth))
#     # else:
#     #     corr_c_smooth, corr_t_smooth = corr_c, corr_t
#     #     x_c = x_t = np.arange(len(corr_c))

#     # Basic 0/1 curves over test-trial index
#     fig, ax = plt.subplots(figsize=(8, 2.5))

#     x = np.arange(len(corr_c))
#     ax.plot(x, corr_c, lw=0.8, label='Concept decoder')
#     ax.plot(x, corr_t, lw=0.8, label='Time decoder')

#     # If you prefer smoothed curves, use these instead and comment out the two lines above:
#     # ax.plot(x_c, corr_c_smooth, lw=1.2, label='Concept decoder (rolling mean)')
#     # ax.plot(x_t, corr_t_smooth, lw=1.2, label='Time decoder (rolling mean)')

#     ax.set_ylim(-0.1, 1.1)
#     ax.set_yticks([0, 1])
#     ax.set_yticklabels(['Incorrect', 'Correct'])
#     ax.set_xlabel('Test trial index')
#     ax.set_ylabel('Prediction correctness')
#     ax.set_title(f'Patient {pid:02d}: concept vs time decoder correctness')
#     ax.legend(loc='upper right', frameon=False)

#     plt.tight_layout()
#     plt.savefig(DIR + f'patient_{pid:02d}_trialwise_correctness.svg',
#                 format='svg')
#     plt.close(fig)



import numpy as np
from scipy.stats import ttest_1samp

# ------------------------------------------------------
# Helper: one-sided t-test wrapper
# ------------------------------------------------------
def ttest_1samp_onesided(x, popmean=0.0, alternative='greater'):
    """
    One-sided t-test based on scipy.stats.ttest_1samp (which is two-sided).

    alternative: 'greater' → H1: mean(x) > popmean
                 'less'    → H1: mean(x) < popmean
                 'two-sided'
    Returns: t_stat, p_value
    """
    x = np.asarray(x)
    t_stat, p_two = ttest_1samp(x, popmean)
    if alternative == 'greater':
        if np.isnan(t_stat):
            return t_stat, np.nan
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    elif alternative == 'less':
        if np.isnan(t_stat):
            return t_stat, np.nan
        p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
    else:
        p_one = p_two
    return t_stat, p_one


# ------------------------------------------------------
# Main analysis: complementarity between decoders
# ------------------------------------------------------
def analyze_decoder_complementarity(results):
    """
    results: dict from decode_per_patient(..., return_preds=True)

    Computes:
      - per-patient mean concept, time, union accuracies
      - Δ = union - max(concept, time)
      - trial-wise correctness correlation between decoders
      - proportion of "only one decoder correct" trials

    Prints group-level stats and returns a dict of per-patient arrays.
    """

    pids = sorted(results.keys())

    concept_mean = []
    time_mean    = []
    union_mean   = []
    synergy_mean = []

    delta_union_vs_best = []   # union_mean - max(concept_mean, time_mean)
    corr_correct        = []   # trial-wise correctness correlation
    frac_one_correct    = []   # fraction of trials with exactly one decoder correct

    for pid in pids:
        r = results[pid]

        # --- mean accuracies per patient (across bootstrap iterations) ---
        c_acc = np.asarray(r['concept_acc'])
        t_acc = np.asarray(r['time_acc'])
        u_acc = np.asarray(r['union_acc'])
        s_acc = np.asarray(r['synergy_pct'])

        cm = c_acc.mean()
        tm = t_acc.mean()
        um = u_acc.mean()
        sm = s_acc.mean()

        concept_mean.append(cm)
        time_mean.append(tm)
        union_mean.append(um)
        synergy_mean.append(sm)

        # Δ union − best single decoder
        best_single = max(cm, tm)
        delta_union_vs_best.append(um - best_single)

        # --- trial-wise correctness for correlation and "one-correct" fraction ---
        # These are over *all* test trials from all iterations for this patient
        y_true  = np.asarray(r['y_true'])
        pred_c  = np.asarray(r['pred_c'])
        pred_t  = np.asarray(r['pred_t'])

        correct_c = (pred_c == y_true).astype(int)
        correct_t = (pred_t == y_true).astype(int)

        # If either is constant (all-correct or all-wrong), correlation is undefined
        if correct_c.std() == 0 or correct_t.std() == 0:
            rho = np.nan
        else:
            rho = np.corrcoef(correct_c, correct_t)[0, 1]

        corr_correct.append(rho)

        # fraction of trials where exactly one decoder is correct
        one_correct = ((correct_c == 1) ^ (correct_t == 1)).astype(int)
        frac_one = one_correct.mean()
        frac_one_correct.append(frac_one)

    concept_mean    = np.array(concept_mean)
    time_mean       = np.array(time_mean)
    union_mean      = np.array(union_mean)
    synergy_mean    = np.array(synergy_mean)
    delta_union_vs_best = np.array(delta_union_vs_best)
    corr_correct    = np.array(corr_correct)
    frac_one_correct = np.array(frac_one_correct)

    # Drop NaNs in correlation if any
    valid_corr = ~np.isnan(corr_correct)

    # --------------------------------------------------
    # 1) Δ union − best single decoder > 0 ?
    # --------------------------------------------------
    t_delta, p_delta = ttest_1samp_onesided(delta_union_vs_best,
                                            popmean=0.0,
                                            alternative='greater')
    mean_delta = delta_union_vs_best.mean()
    sem_delta  = delta_union_vs_best.std(ddof=1) / np.sqrt(len(delta_union_vs_best))

    print("--------------------------------------------------")
    print("Δ union − best single decoder (per patient)")
    print("--------------------------------------------------")
    print(f"n patients = {len(delta_union_vs_best)}")
    print(f"mean Δ = {mean_delta:.3f} ± {sem_delta:.3f} %-points (mean ± SEM)")
    print(f"one-sided t-test H1: Δ > 0 → t = {t_delta:.3f}, p = {p_delta:.3e}")
    print()

    # --------------------------------------------------
    # 2) Trial-wise correctness correlation < 0 ?
    # --------------------------------------------------
    t_corr, p_corr = ttest_1samp_onesided(corr_correct[valid_corr],
                                          popmean=0.0,
                                          alternative='less')
    mean_corr = corr_correct[valid_corr].mean()
    sem_corr  = corr_correct[valid_corr].std(ddof=1) / np.sqrt(valid_corr.sum())

    print("--------------------------------------------------")
    print("Trial-wise correctness correlation (concept vs time)")
    print("--------------------------------------------------")
    print(f"n patients (with non-constant correctness) = {valid_corr.sum()}")
    print(f"mean ρ = {mean_corr:.3f} ± {sem_corr:.3f} (mean ± SEM)")
    print(f"one-sided t-test H1: ρ < 0 → t = {t_corr:.3f}, p = {p_corr:.3e}")
    print()

    # --------------------------------------------------
    # 3) Proportion of 'only one correct' trials > 0 ?
    # --------------------------------------------------
    t_one, p_one = ttest_1samp_onesided(frac_one_correct,
                                        popmean=0.0,
                                        alternative='greater')
    mean_one = frac_one_correct.mean()
    sem_one  = frac_one_correct.std(ddof=1) / np.sqrt(len(frac_one_correct))

    print("--------------------------------------------------")
    print("Fraction of trials with exactly one decoder correct")
    print("--------------------------------------------------")
    print(f"n patients = {len(frac_one_correct)}")
    print(f"mean fraction = {mean_one:.3f} ± {sem_one:.3f} (mean ± SEM)")
    print(f"one-sided t-test H1: fraction > 0 → t = {t_one:.3f}, p = {p_one:.3e}")
    print()

    # Also return everything in case you want to plot / reuse
    return dict(
        pids=np.array(pids),
        concept_mean=concept_mean,
        time_mean=time_mean,
        union_mean=union_mean,
        synergy_mean=synergy_mean,
        delta_union_vs_best=delta_union_vs_best,
        corr_correct=corr_correct,
        frac_one_correct=frac_one_correct,
    )


# ------------------------------------------------------
# Run the analysis
# ------------------------------------------------------
stats_out = analyze_decoder_complementarity(results)


import numpy as np
from scipy.stats import ttest_rel

def compute_synergy_null(results, n_iterations=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    stats = {}

    for pid, r in results.items():

        # skip if missing predictions
        if not all(k in r for k in ['pred_c', 'pred_t', 'y_true']):
            continue

        y_all  = r['y_true']
        pc_all = r['pred_c']
        pt_all = r['pred_t']

        n_total = len(y_all)
        n_test = n_total // n_iterations
        if n_test == 0:
            continue

        obs_synergy  = []
        null_synergy = []

        for it in range(n_iterations):

            s = it * n_test
            e = (it+1) * n_test

            y  = y_all[s:e]
            pc = pc_all[s:e]
            pt = pt_all[s:e]

            c = (pc == y).astype(int)
            t = (pt == y).astype(int)

            # observed synergy
            obs = np.mean((c == 1) & (t == 1))
            obs_synergy.append(obs)

            # null synergy: scramble time correctness
            t_scr = rng.permutation(t)
            null = np.mean((c == 1) & (t_scr == 1))
            null_synergy.append(null)

        obs_synergy  = np.array(obs_synergy)
        null_synergy = np.array(null_synergy)

        # paired t-test: observed < null
        stat, pval = ttest_rel(obs_synergy, null_synergy, alternative='less')

        stats[pid] = {
            "observed": obs_synergy,
            "null": null_synergy,
            "tstat": stat,
            "pval": pval
        }

        print(f"Patient {pid:02d}: t={stat:.3f}, p={pval:.3g}")

    return stats


compute_synergy_null(results)