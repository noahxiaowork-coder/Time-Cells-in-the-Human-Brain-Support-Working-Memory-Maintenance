function region_stats = NWB_runTimeCellNulls(nwbAll, all_units, bin_width_analysis, useSpikeInPeak, num_null_iter)
%NWB_runTimeCellNulls  Observed time-cell detection + nulls with per-trial circular shifts
% and shuffled image labels (LOAD-1 ONLY).
%
% region_stats = NWB_runTimeCellNulls(nwbAll, all_units, binW, useSpikeInPeak, num_null_iter)
%
% INPUTS
%   nwbAll             – {1×S} NWB session objects
%   all_units          – struct array with fields: subject_id, unit_id,
%                        session_count, spike_times  (and optionally electrodes)
%   bin_width_analysis – scalar seconds (e.g., 0.100 for 100 ms)
%   useSpikeInPeak     – logical; apply spike-in-peak rule (±2 bins, ≥max(4,70% trials))
%   num_null_iter      – integer; number of null iterations (default 500)
%
% OUTPUT
%   region_stats – struct with:
%       .regions             : sanitized region names (suffix _left/_right removed)
%       .observed_counts     : counts per region (observed)
%       .p_values            : empirical p-values (null ≥ observed, +1 smoothing)
%       .null_ge_observed    : number of iterations where null ≥ observed
%       .num_null_iter       : N
%       .note                : description of p-value computation
%
% Notes:
%   • Detection uses ONLY load-1 (Enc1 only) trials for binning and stats.
%   • Null generation shifts spikes ONLY within load-1 maintenance windows
%     and ALSO randomizes (permutes) the image labels across those load-1 trials.

if nargin < 5 || isempty(num_null_iter), num_null_iter = 500; end

%% ---------------------------- Parameters --------------------------------
rng(42);                              % deterministic permutations for detection
alphaLim         = 0.05;
num_permutations = 1000;
totalBins        = round(2.5 / bin_width_analysis);   % 25 for 100 ms default
min_trial_count  = 7;

% Gaussian kernel (row vector)
gKer = GaussianKernal(3, 1.5); gKer = gKer(:).';

% Spike-in-peak half-width (bins)
spikeWinHalfWidth = 2;

num_units = numel(all_units);

%% -------------------------- Observed detection --------------------------
fprintf('=== Observed detection ===\n');
[time_cell_info_obs, unit_region, unit_tsM_load1_cache, unit_imgIDs_load1_cache] = detect_time_cells_observed( ...
    nwbAll, all_units, bin_width_analysis, totalBins, min_trial_count, ...
    useSpikeInPeak, spikeWinHalfWidth, gKer, alphaLim, num_permutations);

[region_list, observed_counts] = tally_by_region(time_cell_info_obs, all_units, unit_region);
print_region_counts('Observed', region_list, observed_counts);

%% -------------------------- Null shift caches (load-1 only) -------------
% Build per-neuron caches mapping spikes to load-1 trial windows.
Tmaint = totalBins * bin_width_analysis;
shift_cache = cell(num_units,1);

% Optional: pre-draw per-neuron bin shifts (compact); OFF by default
make_predrawn = false;             % set true to pre-draw shifts
predrawn_shifts = cell(num_units,1);

for iU = 1:num_units
    SU  = all_units(iU);
    tsM_load1 = unit_tsM_load1_cache{iU};   % load-1 maintenance starts only
    shift_cache{iU} = build_shift_cache_for_unit_load1(SU.spike_times, tsM_load1, Tmaint);
    if make_predrawn && ~isempty(tsM_load1)
        nT = numel(tsM_load1);
        predrawn_shifts{iU} = uint16(randi(totalBins, nT, num_null_iter) - 1);  % bin shifts 0..totalBins-1
    end
end

%% -------------------------- Null iterations -----------------------------
fprintf('\n=== %d null iterations (shift within load-1 trials + shuffle image labels) ===\n', num_null_iter);
null_ge_obs = zeros(size(observed_counts));

for it = 1:num_null_iter
    tStart = tic;

    % Build surrogate spikes + shuffled image labels for all units
    all_units_null = all_units;  % shallow copy
    imgIDs_override = cell(num_units,1);

    for iU = 1:num_units
        C = shift_cache{iU};
        tsM_load1 = unit_tsM_load1_cache{iU};
        imgIDs_l1 = unit_imgIDs_load1_cache{iU};

        % Set shuffled image labels (permute only across load-1 trials)
        if ~isempty(imgIDs_l1)
            imgIDs_override{iU} = imgIDs_l1(randperm(numel(imgIDs_l1)));
        else
            imgIDs_override{iU} = [];
        end

        % Shift spikes only if we have load-1 trials
        if C.is_empty
            continue;
        end

        st = all_units(iU).spike_times(:);

        % Per-iteration shifts (one per load-1 trial)
        if make_predrawn
            s = double(predrawn_shifts{iU}(:, it)) * bin_width_analysis;  % seconds
        else
            s = rand(C.nTrials,1) * Tmaint;                               % seconds
        end

        % Vectorized application to all in-window (load-1) spikes
        tr        = double(C.trial_id_of_in_idx);    % per-spike -> trial id
        new_rel   = mod(C.rel_times_in + s(tr), Tmaint);
        new_times = C.t0_by_trial(tr) + new_rel;

        st_null = st;
        st_null(C.in_idx) = new_times;               % overwrite only load-1 spikes
        % Spikes outside load-1 windows remain unchanged

        all_units_null(iU).spike_times = st_null;
    end

    % Detect on null (uses ONLY load-1 trials) with shuffled labels
    [tc_null, ~] = detect_time_cells_observed( ...
        nwbAll, all_units_null, bin_width_analysis, totalBins, min_trial_count, ...
        useSpikeInPeak, spikeWinHalfWidth, gKer, alphaLim, num_permutations, true, imgIDs_override);

    % Tally in the same region order
    [~, null_counts] = tally_by_region(tc_null, all_units, unit_region, region_list);

    % Update exceedance tally
    null_ge_obs = null_ge_obs + (null_counts(:) >= observed_counts(:));

    % Print per-iteration summary + runtime
    tSec = toc(tStart);
    label = sprintf('Iter %3d', it);
    print_region_counts(label, region_list, null_counts, tSec);
end

% Empirical p-values (+1 pseudo-count)
p_vals = (null_ge_obs + 1) ./ (num_null_iter + 1);

%% -------------------------- Final summary -------------------------------
fprintf('\n=== Final region_stats ===\n');
print_region_counts('Observed', region_list, observed_counts);
print_region_counts('Null >= Obs (counts)', region_list, null_ge_obs);
fprintf('Empirical p-values:\n');
for i = 1:numel(region_list)
    fprintf('  %-24s  p = %.6f\n', region_list{i}, p_vals(i));
end

region_stats = struct( ...
    'regions',            {region_list}, ...
    'observed_counts',    observed_counts, ...
    'p_values',           p_vals, ...
    'null_ge_observed',   null_ge_obs, ...
    'num_null_iter',      num_null_iter, ...
    'note', 'p = (1 + #null >= observed) / (N + 1); spikes circularly shifted within load-1 windows; load-1 image labels permuted per iteration.');

end  % --------------------------- END MAIN -------------------------------


%% ===================== Detection on observed (helper) ===================
function [time_cell_info, unit_region, unit_tsM_load1_cache, unit_imgIDs_load1_cache] = detect_time_cells_observed( ...
    nwbAll, all_units, binW, totalBins, min_trial_count, ...
    useSpikeInPeak, spikeWinHalfWidth, gKer, alphaLim, num_permutations, quiet, imgIDs_override)
% If imgIDs_override (cell num_units×1) is provided, and contains a vector at iU,
% those image IDs are used for that unit’s load-1 trials (must match length).

if nargin < 12, quiet = false; end
if nargin < 13, imgIDs_override = []; end

num_units = numel(all_units);

time_cell_rows = {};
ptr = 0;

unit_region              = cell(num_units,1);
unit_tsM_load1_cache     = cell(num_units,1);
unit_imgIDs_load1_cache  = cell(num_units,1);

for iU = 1:num_units
    SU = all_units(iU);
    subject_id = SU.subject_id;
    cellID     = SU.unit_id;

    % if ~quiet
    %     fprintf('Processing unit %d/%d  [Subject %d  Unit %d]\n', ...
    %             iU, num_units, subject_id, cellID);
    % end

    % ---- Fetch trial metadata ONCE per unit/session -------------------
    sess = nwbAll{SU.session_count};
    vd   = sess.intervals_trials.vectordata;

    enc1 = vd.get('loadsEnc1_PicIDs').data.load();
    enc2 = vd.get('loadsEnc2_PicIDs').data.load();
    enc3 = vd.get('loadsEnc3_PicIDs').data.load();
    tsM  = vd.get('timestamps_Maintenance').data.load();

    % Region (sanitized)
    unit_region{iU} = get_unit_region_sanitized(sess, SU);

    % Keep single-load (enc1 only) — these are the ONLY trials we use
    singleMask = (enc1 > 0) & (enc2 == 0) & (enc3 == 0);
    if ~any(singleMask)
        unit_tsM_load1_cache{iU}    = [];
        unit_imgIDs_load1_cache{iU} = [];
        continue;
    end

    imgIDs_single = enc1(singleMask);
    tsM_single    = tsM(singleMask);

    % If override provided (for null), use it (length must match)
    if ~isempty(imgIDs_override) && numel(imgIDs_override) >= iU && ~isempty(imgIDs_override{iU})
        if numel(imgIDs_override{iU}) ~= numel(imgIDs_single)
            error('imgIDs_override length mismatch for unit %d.', iU);
        end
        imgIDs_single = imgIDs_override{iU}(:);
    end

    unit_tsM_load1_cache{iU}    = tsM_single(:);    % cache for null shifts
    unit_imgIDs_load1_cache{iU} = enc1(singleMask); % ORIGINAL order (used only to form shuffles)

    uImgs         = unique(imgIDs_single);
    nImgs         = numel(uImgs);

    % ---- Precompute trial×bin spike matrix for ALL load-1 trials ------
    [spkM_all, Z_all] = binSmoothZ_allTrials( ...
        SU.spike_times, tsM_single(:), binW, totalBins, gKer);

    % ---- Per-image stats containers -----------------------------------
    is_locked  = false(1,nImgs);
    peakAmp    = -inf(1,nImgs);
    peakBin    = zeros(1,nImgs);

    % ---- Loop over images ---------------------------------------------
    for k = 1:nImgs
        thisImg   = uImgs(k);
        trialMask = (imgIDs_single == thisImg);
        nTr       = sum(trialMask);
        if nTr < min_trial_count,  continue;  end

        % Slice precomputed matrices (load-1 only)
        spkMat = spkM_all(trialMask, :);          % raw counts (trials×bins)
        Z      = Z_all(trialMask, :);             % smoothed + zscored

        % Peak for dominance
        [pk, pkBin] = max(mean(Z,1));
        peakAmp(k)  = pk;
        peakBin(k)  = pkBin;

        % Spike presence gating
        if useSpikeInPeak
            win = max(1,pkBin-spikeWinHalfWidth) : min(totalBins,pkBin+spikeWinHalfWidth);
            trialsWithSpike = any(spkMat(:,win) > 0, 2);
            if sum(trialsWithSpike) < max(4, ceil(0.7*nTr)), continue; end
        else
            if sum(sum(spkMat,2) > 0) < 3, continue; end
        end

        % ---- Permutation significance (fast row-wise circular shifts) --
        p = permTest_timeLocked_fast(Z, num_permutations);
        if p >= alphaLim, continue; end

        is_locked(k) = true;
    end

    % ---- Global peak-dominance gate -----------------------------------
    if any(is_locked)
        [~, prefIdx] = max(peakAmp);
        others = true(1,nImgs); others(prefIdx) = false;
        if is_locked(prefIdx) && all(peakAmp(prefIdx) > peakAmp(others))
            ptr = ptr + 1;
            time_cell_rows{ptr,1} = [SU.subject_id, SU.unit_id, uImgs(prefIdx), peakBin(prefIdx)];
        end
    end
end

if ptr > 0
    time_cell_info = vertcat(time_cell_rows{:});
else
    time_cell_info = zeros(0,4);
end
end


%% ===================== Region & spike helpers ==========================
function cache = build_shift_cache_for_unit_load1(spike_times, tsM_load1, Tmaint)
% Precompute membership & relative times for spikes that fall in load-1 windows.
% All other spikes (non-load-1 trials) are left untouched.
cache.is_empty = isempty(tsM_load1);
if cache.is_empty
    cache.in_idx = [];
    cache.trial_id_of_in_idx = [];
    cache.rel_times_in = [];
    cache.t0_by_trial = [];
    cache.nTrials = 0;
    return;
end

st = spike_times(:);
nT = numel(tsM_load1);
in_mask = false(size(st));
trial_id_of_in_idx = zeros(0,1,'uint32');
rel_times_in = zeros(0,1);

for tr = 1:nT
    t0 = tsM_load1(tr);
    m = (st >= t0) & (st < t0 + Tmaint);
    if any(m)
        idx = find(m);
        in_mask(idx) = true;
        trial_id_of_in_idx = [trial_id_of_in_idx; repmat(uint32(tr), numel(idx), 1)]; %#ok<AGROW>
        rel_times_in = [rel_times_in; st(idx) - t0]; %#ok<AGROW>
    end
end

cache.in_idx  = find(in_mask);             % indices of spikes in load-1 windows
cache.trial_id_of_in_idx = trial_id_of_in_idx;
cache.rel_times_in = rel_times_in;
cache.t0_by_trial  = tsM_load1(:);
cache.nTrials = nT;
end

function reg = get_unit_region_sanitized(sess, SU)
try
    regStr = sess.general_extracellular_ephys_electrodes ...
                 .vectordata.get('location').data.load(SU.electrodes);
    reg0 = regStr{:};
catch
    reg0 = 'unknown';
end
% Remove trailing _left/_right (case-insensitive)
reg = regexprep(reg0, '(_left|_right)$', '', 'ignorecase');
end


%% ===================== Binning & permutation helpers ===================
function [spkMat, Z] = binSmoothZ_allTrials(spikeTimes, tsMaint_vec, binW, totalBins, gKer)
% Build trial×bin spike counts (raw) and smoothed+zscored matrix for given trials.
nT    = numel(tsMaint_vec);
T     = totalBins * binW;
edges = (0:totalBins) * binW;   % relative edges [0..T]
spkMat = zeros(nT, totalBins, 'double');

% Using histcounts per trial for robustness; avoids inner bin loops
for tr = 1:nT
    t0 = tsMaint_vec(tr);
    trSpk = spikeTimes(spikeTimes >= t0 & spikeTimes < t0 + T) - t0;
    spkMat(tr,:) = histcounts(trSpk, edges);
end

Z = conv2(spkMat, gKer, 'same');
Z = zscore(Z, 0, 2);
end

function p = permTest_timeLocked_fast(Z, nPerm)
% Permutation test using row-wise circular shifts on Z (trials×bins).
obs = max(mean(Z,1));
[nTr, nB] = size(Z);
null = zeros(nPerm,1);

rowIdxMat = repmat((1:nTr).', 1, nB);
baseCols  = 1:nB;

for pIdx = 1:nPerm
    s = randi(nB, nTr, 1) - 1;
    colIdxMat = 1 + mod(bsxfun(@plus, baseCols-1, s), nB);
    P = Z(sub2ind([nTr nB], rowIdxMat, colIdxMat));
    null(pIdx) = max(mean(P,1));
end

p = mean(null >= obs);
end

function g = GaussianKernal(widthBins, sigmaBins)
x = -widthBins:widthBins;
g = exp(-(x.^2) / (2*sigmaBins^2));
g = g / sum(g);
end


%% ===================== Tally & printing helpers ========================
function [region_list, counts] = tally_by_region(time_cell_info, all_units, unit_region, region_list_in)
% Count # detected time cells per sanitized region.
if nargin < 4
    if isempty(time_cell_info)
        region_list = {};
        counts = [];
        return;
    end
    key2region = containers.Map('KeyType','char','ValueType','char');
    for iU = 1:numel(all_units)
        key2region(sprintf('S%d_U%d', all_units(iU).subject_id, all_units(iU).unit_id)) = unit_region{iU};
    end
    regs = cell(size(time_cell_info,1),1);
    for r = 1:size(time_cell_info,1)
        pid = time_cell_info(r,1); uid = time_cell_info(r,2);
        regs{r} = key2region(sprintf('S%d_U%d', pid, uid));
    end
    [region_list, ~, ic] = unique(regs);
    counts = accumarray(ic, 1, [numel(region_list) 1]);
else
    region_list = region_list_in(:);
    counts = zeros(numel(region_list),1);
    if isempty(time_cell_info), return; end
    key2region = containers.Map('KeyType','char','ValueType','char');
    for iU = 1:numel(all_units)
        key2region(sprintf('S%d_U%d', all_units(iU).subject_id, all_units(iU).unit_id)) = unit_region{iU};
    end
    for r = 1:size(time_cell_info,1)
        pid = time_cell_info(r,1); uid = time_cell_info(r,2);
        reg = key2region(sprintf('S%d_U%d', pid, uid));
        idx = find(strcmp(region_list, reg), 1);
        if ~isempty(idx), counts(idx) = counts(idx) + 1; end
    end
end
end

function print_region_counts(label, region_list, counts, iter_runtime_sec)
if nargin < 4 || isempty(iter_runtime_sec)
    fprintf('%-20s : ', [label ' counts']);
else
    fprintf('%-20s : ', sprintf('%s (%.3f s)', label, iter_runtime_sec));
end

if isempty(region_list)
    fprintf('[no regions]\n');
    return;
end

for i = 1:numel(region_list)
    if i > 1, fprintf(' | '); end
    fprintf('%s=%d', region_list{i}, counts(i));
end
fprintf('\n');
end
