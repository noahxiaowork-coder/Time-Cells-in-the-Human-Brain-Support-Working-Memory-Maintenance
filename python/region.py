import numpy as np
import pandas as pd
import scipy.io
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import sem
from pathlib import Path
import sys

# Add current directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent))
from utils import significance_stars

DIR = ''

region_colors = {
    "hippocampus": "#FFD700",
    "amygdala": "#00FFFF",
    "pre_supplementary_motor_area": "#FF0000",
    "dorsal_anterior_cingulate_cortex": "#0000FF",
    "ventral_medial_prefrontal_cortex": "#008000",
}


def plot_per_patient_proportions_from_csv(
    time_cell_mat,
    counts_csv="cell_counts_by_patient_region.csv",
    jitter=0.08,
    seed=20250710,
    alpha=0.05,
    num_scrambles=500,
):
    region_map = {
        "dorsal_anterior_cingulate_cortex": "DaCC",
        "pre_supplementary_motor_area": "PSMA",
        "hippocampus": "HPC",
        "amygdala": "AMY",
        "ventral_medial_prefrontal_cortex": "vmPFC",
    }
    abbrev_to_substr = {v: k for k, v in region_map.items()}

    null_info_counts = {
        "amygdala": (10.158, 0.001996),
        "dorsal_anterior_cingulate_cortex": (7.488, 0.03992),
        "hippocampus": (7.91, 0.021956),
        "pre_supplementary_motor_area": (10.354, 0.001996),
        "ventral_medial_prefrontal_cortex": (1.37, 0.003992),
    }
    # This is manually filled in by the results processed in MATLAB.

    tc_neurons = scipy.io.loadmat(time_cell_mat)["neural_data"][0]
    df_counts = pd.read_csv(counts_csv)

    total_cell_dict = {
        (
            abbrev_to_substr[str(r.brain_region)]
            if str(r.brain_region) in abbrev_to_substr
            else r.brain_region,
            str(r.patient_id),
        ): int(r.total_cells)
        for _, r in df_counts.iterrows()
    }

    region_total_cells = {}
    for substr in region_map:
        pats_in_region = {
            pat_id for (reg, pat_id) in total_cell_dict if reg == substr
        }
        region_total_cells[substr] = sum(
            total_cell_dict[(substr, pat)] for pat in pats_in_region
        )

    time_cell_dict = defaultdict(int)
    for n in tc_neurons:
        raw_region = (
            n["brain_region"][0]
            .lower()
            .replace("_left", "")
            .replace("_right", "")
        )
        patient_id = str(n["patient_id"][0])
        for substr in region_map:
            if substr in raw_region:
                time_cell_dict[(substr, patient_id)] += 1
                break

    region_patient_props = defaultdict(list)
    region_order = list(region_map)
    region_labels = [region_map[s] for s in region_order]

    patients_all = set(p for (_, p) in total_cell_dict.keys())
    for pat in patients_all:
        for substr in region_order:
            tot_cnt = total_cell_dict.get((substr, pat), 0)
            tc_cnt = time_cell_dict.get((substr, pat), 0)
            if tot_cnt > 0:
                region_patient_props[substr].append(tc_cnt / tot_cnt)

    means_percent = np.array(
        [np.mean(region_patient_props.get(s, [0.0])) * 100 for s in region_order],
        dtype=float,
    )
    sem_percent = np.array(
        [
            sem(region_patient_props.get(s, [])) * 100
            if len(region_patient_props.get(s, [])) > 1
            else 0.0
            for s in region_order
        ],
        dtype=float,
    )

    dacc_substr = "dorsal_anterior_cingulate_cortex"
    dacc_full_patients = []
    for (substr, pat_id), tot_cnt in total_cell_dict.items():
        if substr != dacc_substr or tot_cnt <= 0:
            continue
        tc_cnt = time_cell_dict.get((substr, pat_id), 0)
        if tc_cnt == tot_cnt and tc_cnt > 0:
            dacc_full_patients.append((pat_id, tc_cnt, tot_cnt))

    if dacc_full_patients:
        print("\nPatients with 100% DaCC time cells:")
        for pat_id, tc_cnt, tot_cnt in dacc_full_patients:
            print(f"  patient {pat_id}: {tc_cnt}/{tot_cnt} DaCC cells (100.0%)")
    else:
        print("\nNo patients with 100% DaCC time cells found.")

    patient_overall_props = []
    for pat in patients_all:
        tot_cells_pat = 0
        tc_cells_pat = 0
        for substr in region_order:
            tot_cells_pat += total_cell_dict.get((substr, pat), 0)
            tc_cells_pat += time_cell_dict.get((substr, pat), 0)
        if tot_cells_pat > 0:
            patient_overall_props.append(tc_cells_pat / tot_cells_pat)

    if patient_overall_props:
        overall_mean = np.mean(patient_overall_props) * 100
        if len(patient_overall_props) > 1:
            overall_sem = sem(patient_overall_props) * 100
        else:
            overall_sem = 0.0

        print("\nAcross patients (all regions combined):")
        print(
            f"  Mean proportion of time cells = {overall_mean:.2f}% ± {overall_sem:.2f}% (mean ± SEM)"
        )
    else:
        print("\nNo per-patient overall proportions could be computed.")

    rng = np.random.default_rng(seed)
    x = np.arange(len(region_labels))

    fig, ax = plt.subplots(figsize=(4.4, 6.8))

    ax.bar(
        x,
        means_percent,
        yerr=sem_percent,
        capsize=4,
        color=[region_colors[s] for s in region_order],
        ecolor="black",
        alpha=0.8,
        zorder=1,
    )

    for xi, pct, se in zip(x, means_percent, sem_percent):
        ax.text(
            xi,
            pct + se + 0.8,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    for idx, substr in enumerate(region_order):
        vals = region_patient_props.get(substr, [])
        if not vals:
            continue
        jitter_x = x[idx] + rng.uniform(-jitter, jitter, size=len(vals))
        ax.scatter(
            jitter_x,
            np.array(vals) * 100,
            color=region_colors[substr],
            edgecolor="k",
            alpha=0.3,
            zorder=3,
        )

    null_means_percent = {}
    for substr in region_order:
        nm_count, _ = null_info_counts[substr]
        denom = region_total_cells.get(substr, 0)
        if denom > 0:
            null_means_percent[substr] = (nm_count / denom) * 100.0
        else:
            null_means_percent[substr] = np.nan

    for substr in region_order:
        nm_pct = null_means_percent[substr]
        if not np.isnan(nm_pct):
            ax.axhline(
                y=nm_pct,
                linestyle="--",
                linewidth=1.3,
                color=region_colors[substr],
                alpha=0.9,
                zorder=0,
            )

    for idx, substr in enumerate(region_order):
        _, p_emp = null_info_counts[substr]
        star = significance_stars(p_emp)
        if star == "ns":
            star = ""  # Don't show 'ns' on this plot
        if star:
            ax.text(
                x[idx],
                means_percent[idx] + sem_percent[idx] + 2.0,
                star,
                ha="center",
                va="bottom",
                fontsize=12,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(region_labels, rotation=45, ha="right", fontsize=12)
    ax.set_ylabel("Proportion of Time Cells (%)", fontsize=14)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=12)
    plt.tight_layout()

    print("\nSummary of Time Cells per Region:")
    for substr in region_order:
        pats_in_region = {
            pat_id for (reg, pat_id) in total_cell_dict if reg == substr
        }
        total_cells_in_region = region_total_cells[substr]
        time_cells_in_region = sum(
            time_cell_dict.get((substr, pat), 0) for pat in pats_in_region
        )
        if total_cells_in_region > 0:
            percent_obs = (
                time_cells_in_region / total_cells_in_region
            ) * 100
            nm_count, p_emp = null_info_counts[substr]
            nm_pct = null_means_percent[substr]
            print(
                f"{region_map[substr]:<6}: {time_cells_in_region}/{total_cells_in_region} "
                f"({percent_obs:.2f}%) | NullMean={nm_count:.3f} (→ {nm_pct:.3f}%), EmpP={p_emp:.5f}"
            )
        else:
            nm_count, p_emp = null_info_counts[substr]
            print(
                f"{region_map[substr]:<6}: No data | NullMean={nm_count:.3f} (→ n/a), EmpP={p_emp:.5f}"
            )

    plt.savefig(DIR + "Region.svg", format="svg", bbox_inches="tight")
    plt.show()


plot_per_patient_proportions_from_csv(
    "cue_specific_TC.mat",
    "cell_counts_by_patient_region.csv",
)
