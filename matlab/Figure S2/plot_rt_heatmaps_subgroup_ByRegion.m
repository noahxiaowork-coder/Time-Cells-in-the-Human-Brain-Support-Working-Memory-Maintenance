function out = plot_rt_heatmaps_subgroup_ByRegion(neural_data, bin_width, use_correct, pctThresh, varargin)
%PLOT_RT_HEATMAPS_SUBGROUP_BYREGION  PSTH heat-maps with load-wise fast/slow split, grouped by region.
%
% Similar to plot_rt_heatmaps_subgroup, but:
%   * neurons are grouped by brain region (laterality stripped)
%   * optional ExcludeVentral, MinUnits
%   * separate heatmaps per region (All, Fast, Slow)
%
% OUTPUT:
%   out.regions        : string array of region names (lowercase, laterality stripped)
%   out.regionCounts   : number of neurons per region (after filtering)
%   out.orderPerRegion : cell array; each cell is indices into neural_data for that region (sorted by peak)
%   out.figAll         : cell array of figure handles for "All" trials per region
%   out.figFast        : cell array of figure handles for "Fast" trials per region
%   out.figSlow        : cell array of figure handles for "Slow" trials per region
%   out.time_edges     : time bin edges
%   out.time_centers   : time bin centers

if nargin < 3 || isempty(use_correct), use_correct = true; end
if nargin < 4 || isempty(pctThresh),   pctThresh = 25;     end

% ---- options ----
p = inputParser;
p.addParameter('ExcludeVentral', true,  @(b)islogical(b)&&isscalar(b)); % drop regions starting with 'ventral'
p.addParameter('MinUnits',       0,    @(x)isnumeric(x)&&isscalar(x)&&x>=0);
p.addParameter('FontSize',       16,   @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('FigWidth',       420,  @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('FigHeight',      648,  @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.parse(varargin{:});
opt = p.Results;

% ---- acronym map for titles ----
acronym = containers.Map( ...
    {'hippocampus','amygdala','dorsal_anterior_cingulate_cortex', ...
     'pre_supplementary_motor_area','ventral_medial_prefrontal_cortex'}, ...
    {'HPC','AMY','DaCC','PSMA','vmPFC'});

% helper: strip only a trailing laterality suffix
stripLat = @(s) regexprep(lower(string(s)), '_(left|right)$', '');

%% ------------ BASIC CHECKS & TIME AXIS -------------------------------
num_neurons = numel(neural_data);
assert(num_neurons > 0, 'No neurons found in neural_data.');

num_bins     = size(neural_data(1).firing_rates, 2);
time_edges   = (0:num_bins) * bin_width;                 % includes final edge (e.g., 2.5)
time_centers = time_edges(1:end-1) + diff(time_edges)/2; % bin centers

% ---- region per neuron (laterality stripped) ----
regionRaw  = strings(num_neurons,1);
keepNeuron = true(num_neurons,1);
for n = 1:num_neurons
    if isfield(neural_data, 'brain_region') && ~isempty(neural_data(n).brain_region)
        reg = stripLat(neural_data(n).brain_region);
    else
        reg = "unknown";
    end
    regionRaw(n) = reg;
end
if opt.ExcludeVentral
    keepNeuron = keepNeuron & ~startsWith(regionRaw,"ventral",'IgnoreCase',true);
end

% If everything got excluded, bail early
if ~any(keepNeuron)
    warning('No neurons to plot after ExcludeVentral filtering.');
    out = struct();
    return;
end

all_psth  = zeros(num_neurons, num_bins);
fast_psth = zeros(num_neurons, num_bins);
slow_psth = zeros(num_neurons, num_bins);
peak_bins = zeros(num_neurons, 1);

%% ------------ PATIENT-ID NORMALISATION -------------------------------
if isfield(neural_data, 'patient_id')
    raw_pid = {neural_data.patient_id};
    if all(cellfun(@isnumeric, raw_pid))
        pid_vec     = cell2mat(raw_pid);
        patient_ids = unique(pid_vec);
    else
        pid_vec     = string(raw_pid);
        patient_ids = unique(pid_vec);
    end
else
    pid_vec  = ones(1, num_neurons);
    [neural_data.patient_id] = deal(1);
    patient_ids = 1;
end

%% ------------ PERCENTILE PER LOAD (per patient) -----------------------
percentiles = nan(numel(patient_ids), 3, 2); % [:,:,1]=low, [:,:,2]=high

for pIdx = 1:numel(patient_ids)
    pid = patient_ids(pIdx);
    if isnumeric(pid)
        idx = find(pid_vec == pid, 1, 'first');
    else
        idx = find(strcmp(pid_vec, pid), 1, 'first');
    end

    rt   = neural_data(idx).trial_RT;
    corr = neural_data(idx).trial_correctness == 1;
    load = neural_data(idx).trial_load;

    keep = ~isnan(rt);
    if use_correct, keep = keep & corr; end

    for L = 1:3
        msk = (load == L) & keep;
        if nnz(msk) < 5, continue; end
        percentiles(pIdx, L, 1) = prctile(rt(msk), pctThresh);
        percentiles(pIdx, L, 2) = prctile(rt(msk), 100 - pctThresh);
    end
end

%% ------------ BUILD PSTHs (per neuron) -------------------------------
for n = 1:num_neurons
    if ~keepNeuron(n)
        all_psth(n,:)  = NaN;
        fast_psth(n,:) = NaN;
        slow_psth(n,:) = NaN;
        peak_bins(n)   = NaN;
        continue;
    end

    fr   = neural_data(n).firing_rates;  % trials × bins
    rt   = neural_data(n).trial_RT;
    corr = neural_data(n).trial_correctness == 1;
    load = neural_data(n).trial_load;
    pid  = pid_vec(n);

    if isnumeric(pid)
        pIdx = find(patient_ids == pid);
    else
        pIdx = find(strcmp(patient_ids, pid));
    end

    fast_mask = false(size(rt));
    slow_mask = false(size(rt));
    keep_mask = ~isnan(rt);
    if use_correct, keep_mask = keep_mask & corr; end

    for L = 1:3
        pLow  = percentiles(pIdx, L, 1);
        pHigh = percentiles(pIdx, L, 2);
        if isnan(pLow) || isnan(pHigh), continue; end

        msk = (load == L) & keep_mask;
        fast_mask(msk) = fast_mask(msk) | (rt(msk) <= pLow);
        slow_mask(msk) = slow_mask(msk) | (rt(msk) >= pHigh);
    end

    all_psth(n,:)  = mean(fr(keep_mask,:), 1, 'omitnan');
    fast_psth(n,:) = mean(fr(fast_mask,:), 1, 'omitnan');
    slow_psth(n,:) = mean(fr(slow_mask,:), 1, 'omitnan');

    [~, peak_bins(n)] = max(all_psth(n,:));
end

%% ------------ GROUP BY REGION -----------------------------------------
regsKept = regionRaw(keepNeuron);
uniqRegs = unique(regsKept, 'stable');

if isempty(uniqRegs)
    warning('No non-ventral regions found to plot.');
    out = struct();
    return;
end

% counts per region (after neuron-level filtering)
counts = zeros(numel(uniqRegs),1);
for r = 1:numel(uniqRegs)
    counts(r) = sum(regionRaw == uniqRegs(r) & keepNeuron);
end

% apply MinUnits
keepReg = counts >= opt.MinUnits;
uniqRegs = uniqRegs(keepReg);
counts   = counts(keepReg);

if isempty(uniqRegs)
    warning('No regions meet MinUnits threshold.');
    out = struct();
    return;
end

nR = numel(uniqRegs);
orderPerRegion = cell(nR,1);
figAll  = cell(nR,1);
figFast = cell(nR,1);
figSlow = cell(nR,1);

%% ------------ SORT, NORMALISE, PLOT PER REGION ------------------------
for r = 1:nR
    reg = uniqRegs(r);

    % neuron indices for this region
    maskR = (regionRaw == reg) & keepNeuron & any(isfinite(all_psth),2);
    idxR  = find(maskR);
    if isempty(idxR), continue; end

    % sort by peak bin (within-region)
    [~, ordLocal] = sort(peak_bins(idxR), 'ascend', 'MissingPlacement','last');
    idxR_sorted = idxR(ordLocal);
    orderPerRegion{r} = idxR_sorted;

    A = all_psth(idxR_sorted,:);
    F = fast_psth(idxR_sorted,:);
    S = slow_psth(idxR_sorted,:);

    [A_n, F_n, S_n] = rowwise_minmax_normalize_multi(A, F, S);

    key = char(reg);
    if isKey(acronym, key)
        regLabel = acronym(key);
    else
        regLabel = upper(key);
    end

    ttlAll  = sprintf('%s — All Trials (correct=%d)', regLabel, use_correct);
    ttlFast = sprintf('%s — Fast Trials (\\x2264 %dth pct per load)', regLabel, pctThresh);
    ttlSlow = sprintf('%s — Slow Trials (\\x2265 %dth pct per load)', regLabel, 100-pctThresh);

    figAll{r}  = makePlotRegion(A_n,  ttlAll);
    figFast{r} = makePlotRegion(F_n,  ttlFast);
    figSlow{r} = makePlotRegion(S_n,  ttlSlow);
end

%% ------------ OUTPUT STRUCT -------------------------------------------
out = struct( ...
    'regions',        {uniqRegs}, ...
    'regionCounts',   counts, ...
    'orderPerRegion', {orderPerRegion}, ...
    'figAll',         {figAll}, ...
    'figFast',        {figFast}, ...
    'figSlow',        {figSlow}, ...
    'time_edges',     time_edges, ...
    'time_centers',   time_centers);

%% ------------ Nested plot helper --------------------------------------
    function hFig = makePlotRegion(mat, ttl)
        hFig = figure('Name', ['RT subgroup: ' ttl], ...
                      'Position',[80 90 opt.FigWidth opt.FigHeight], ...
                      'Color','w');
        imagesc(mat, [0 1]);
        colormap(parula);
        cb = colorbar; set(cb, 'FontSize', opt.FontSize);

        % Ensure complete ticks: include the final edge (e.g., 2.5)
        xt_vals   = 0:0.5:time_edges(end);       % 0,0.5,...,2.5
        xt_idx    = arrayfun(@(t) find_closest_bin(time_centers, t), xt_vals);
        xt_labels = arrayfun(@(t) sprintf('%.1f', t), xt_vals, 'UniformOutput', false);

        set(gca, 'TickDir','out', ...
                 'XTick', xt_idx, ...
                 'XTickLabel', xt_labels, ...
                 'FontSize', opt.FontSize);
        xlabel('Time (s)', 'FontSize', opt.FontSize);
        ylabel('Time Cells', 'FontSize', opt.FontSize);
        title(['Time Cells: ' ttl ' (rowwise norm)'], ...
              'FontSize', opt.FontSize, 'Interpreter','none');
    end
end

%% ------------ Shared helpers (same as your original) ------------------
function [varargout] = rowwise_minmax_normalize_multi(varargin)
    nArrays = nargin;
    if nArrays < 2, error('Provide at least two matrices to jointly normalize.'); end
    nRows = size(varargin{1}, 1);
    for k = 2:nArrays
        if size(varargin{k}, 1) ~= nRows
            error('All input matrices must have the same number of rows.');
        end
    end
    varargout = cell(1, nArrays);
    for k = 1:nArrays, varargout{k} = zeros(size(varargin{k})); end
    for r = 1:nRows
        combined_row = [];
        for k = 1:nArrays, combined_row = [combined_row, varargin{k}(r, :)]; end %#ok<AGROW>
        if all(isnan(combined_row))
            for k = 1:nArrays, varargout{k}(r, :) = 0; end
            continue;
        end
        min_val = min(combined_row, [], 'omitnan');
        max_val = max(combined_row, [], 'omitnan');
        if max_val > min_val
            for k = 1:nArrays
                row_k = varargin{k}(r, :);
                row_k(isnan(row_k)) = min_val;
                varargout{k}(r, :) = (row_k - min_val) / (max_val - min_val);
            end
        else
            for k = 1:nArrays, varargout{k}(r, :) = 0.5; end
        end
    end
end

function idx = find_closest_bin(time_centers, target_time)
    [~, idx] = min(abs(time_centers - target_time));
end
