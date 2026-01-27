# NWB Sternberg Single-Unit Analysis Pipeline

This repository provides a MATLAB pipeline for importing and analyzing NWB-format human single-unit recordings from the Sternberg working memory task.

The main entry point is:

```
NWB_SB_import_main.m
```

This script loads NWB data, initializes dependencies, extracts single units, and calls downstream analysis functions such as time-cell detection. 

The dataset used in the original analysis is available at:

> [https://www.nature.com/articles/s41597-024-02943-8](https://www.nature.com/articles/s41597-024-02943-8)

---

## 1. System Requirements

### Operating System

* macOS 12+ (tested)
* Linux Ubuntu 20.04+ (expected to work)
* Windows 10/11 (expected to work; path separators handled automatically)

### Software Dependencies

#### MATLAB

* MATLAB R2022b or newer
  (Tested on R2023a / R2024a)

Required MATLAB toolboxes:

* Statistics and Machine Learning Toolbox
* Signal Processing Toolbox
* Parallel Computing Toolbox (optional but recommended for speed)

#### MatNWB

* **matnwb v2.6+**
* Download from:
  [https://github.com/NeurodataWithoutBorders/matnwb](https://github.com/NeurodataWithoutBorders/matnwb)

The script automatically runs `generateCore()` if the NWB API is not yet initialized.

#### Repository Code

* This repository (including the `helpers/` directory and NWB utility functions)
* Functions such as:

  * `NWB_importFromFolder`
  * `NWB_SB_extractUnits`
  * `Find_cue_cells`
  * `NWB_calcSelective_SB`
  * `create_neural_data`

must be present in MATLAB path.

### Data Requirements

* NWB files downloaded from the public dataset:

  * Kyzar et al., Scientific Data (2024)
* Disk space requirement:

  * ~20–40 GB depending on selected subjects and waveform extraction.

### Non-standard Hardware

* No special hardware required.
* Recommended:

  * ≥32 GB RAM for full dataset loading
  * SSD storage for faster I/O

---

## 2. Installation Guide

### Step 1 — Install MATLAB

Install MATLAB R2022b or newer with the required toolboxes.

### Step 2 — Install MatNWB

Clone or download matnwb:

```bash
git clone https://github.com/NeurodataWithoutBorders/matnwb.git
```

Place it somewhere permanent (e.g., `/Users/yourname/matnwb`).

### Step 3 — Download Dataset

Download the NWB dataset from:

> [https://www.nature.com/articles/s41597-024-02943-8](https://www.nature.com/articles/s41597-024-02943-8)

Extract the NWB files into a local folder, for example:

```
/Users/yourname/Data/Human_TC_WM/
```

### Step 4 — Configure Paths

Edit the following section in `NWB_SB_import_main.m`:

```matlab
paths.baseData = '/your/path/to/NWB/data';
paths.nwb_sb   = paths.baseData;
paths.nwb_sc   = paths.baseData;
paths.code     = '/your/path/to/this/repository';
paths.matnwb   = '/your/path/to/matnwb';
```

Ensure all paths are correct.

### Step 5 — Verify Installation

Launch MATLAB and run:

```matlab
NWB_SB_import_main
```

On first run, matnwb will automatically generate core NWB classes.

---

## 3. Demo

### Demo Objective

Load Sternberg task data, extract single units, and detect cue-related / time-related neurons.

### Demo Instructions

In `NWB_SB_import_main.m`, use a small subset for fast testing:

```matlab
taskFlag = 2;        % Sternberg
importRange = 1:2;  % Small demo subset
```

Run:

```matlab
NWB_SB_import_main
```

The script will:

1. Load NWB files.
2. Extract all single units.
3. Run:

   ```matlab
   [neural_data, time_cell_info, unit_stats] = Find_cue_cells( ...
       nwbAll_sb, all_units_sb, 0.100, false);
   ```

### Expected Output

In MATLAB workspace:

* `neural_data`
  Structured neural feature matrix used for downstream decoding and plotting.
* `time_cell_info`
  Metadata describing detected time cells / cue cells.
* `unit_stats`
  Per-unit statistics and selectivity metrics.

Console output:

* Status messages for loading, extraction, and analysis steps.

### Expected Runtime (Demo)

On a normal desktop (8–12 cores, SSD, 32 GB RAM):

* Import subjects: ~1–3 minutes
* Unit extraction: ~1 minutes
* Time cell detection: ~40–60 minutes

Total demo runtime: **~5–10 minutes**

---

## 4. Instructions for Use

### Main Control Script

`NWB_SB_import_main.m` is the main control script.
You typically modify only:

```matlab
taskFlag     % 1 = Screening, 2 = Sternberg, 3 = Both
importRange  % Subject indices
paths.*      % Local paths
```

All other analysis functions are called from this script.

---

### Core Analysis Functions

The essential functions are:

#### 1. Data Import

```matlab
[nwbAll_sb, importLog_sb] = NWB_importFromFolder(...)
```

Loads NWB files into MATLAB objects.

#### 2. Single-Unit Extraction

```matlab
all_units_sb = NWB_SB_extractUnits(nwbAll_sb, load_all_waveforms);
```

Extracts spike times and waveforms.

#### 3. Time Cell Identification

* `NWB_calcSelective_SB`
* Identifies time cells 

#### 4. Neural Data Construction

```matlab
create_neural_data(...)
```

Generates structured data used by plotting and decoding functions.

Many downstream plotting and decoding scripts depend on this output.



### Reproducibility Notes

* Randomized procedures (if enabled in downstream scripts) should set fixed seeds for reproducibility.
* MATLAB version and matnwb version should be recorded when publishing results.
