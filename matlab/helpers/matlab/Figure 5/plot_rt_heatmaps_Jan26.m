function plot_rt_heatmaps(neural_data_file, bin_width, use_correct)
%PLOT_RT_HEATMAPS  PSTH heat-maps split by RT speed (Fast vs Slow).
%
% Fast/Slow labels are assigned **within each (Load×Probe) subgroup**:
%   Load 1-In, Load 1-Out, …, Load 3-Out   (6 total).
% Quartiles use only correct trials if USE_CORRECT = true (default).
%
% INPUTS
%   neural_data_file – .mat file holding a `neural_data` struct array
%   bin_width        – PSTH bin width (s)
%   use_correct      – logical; keep only correct trials (default = true)
% -------------------------------------------------------------------------

if nargin < 3 || isempty(use_correct), use_correct = true; end

% ---------- LOAD ----------------------------------------------------------

S = builtin('load', neural_data_file, 'neural_data');   % always hits the built-in
neural_data = S.neural_data;


num_neurons = numel(neural_data);
assert(num_neurons > 0, 'No neurons found in neural_data.');

num_bins     = size(neural_data(1).firing_rates, 2);
time_edges   = (0:num_bins) * bin_width;
time_centers = time_edges(1:end-1) + diff(time_edges)/2;

all_psth  = zeros(num_neurons, num_bins);
fast_psth = zeros(num_neurons, num_bins);
slow_psth = zeros(num_neurons, num_bins);
peak_bins = zeros(num_neurons, 1);

% ---------- PATIENT IDs ---------------------------------------------------
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

% ---------- PRE-COMPUTE QUARTILES PER PATIENT × SUBGROUP ------------------
% Dimensions:  patient × load (1:3) × probe(1=out,2=in)
q_lo =  nan(numel(patient_ids), 3, 2);
q_hi =  nan(numel(patient_ids), 3, 2);

for p = 1:numel(patient_ids)
    pid = patient_ids(p);

    % find first neuron of this patient (all have identical trial vectors)
    if isnumeric(pid)
        idx = find(pid_vec == pid, 1, 'first');
    else
        idx = find(strcmp(pid_vec, pid), 1, 'first');
    end
    if isempty(idx), error('No neuron with patient_id %s found.', string(pid)); end

    rt_vec   = neural_data(idx).trial_RT;
    load_vec = neural_data(idx).trial_load;
    probe_vec= neural_data(idx).trial_probe_in_out;
    corr_vec = neural_data(idx).trial_correctness == 1;

    base_mask = ~isnan(rt_vec);
    if use_correct, base_mask = base_mask & corr_vec; end

    for L = 1:3
        for pr = 0:1    % 0 = out, 1 = in
            m = base_mask & load_vec == L & probe_vec == pr;
            if nnz(m) >= 2
                rt_sub        = rt_vec(m);
                q_lo(p,L,pr+1) = prctile(rt_sub, 50);
                q_hi(p,L,pr+1) = prctile(rt_sub, 50);
            end
        end
    end
end

% ---------- BUILD PSTHs ---------------------------------------------------
for n = 1:num_neurons
    fr   = neural_data(n).firing_rates;           % trials × bins
    rt   = neural_data(n).trial_RT;
    load = neural_data(n).trial_load;
    probe= neural_data(n).trial_probe_in_out;
    corr = neural_data(n).trial_correctness == 1;
    brain_region = neural_data(n).brain_region;
    pid  = pid_vec(n);

    % trial-level base mask
    mask_ok = ~isnan(rt);
    if use_correct, mask_ok = mask_ok & corr; end

    fast_mask = false(size(rt));
    slow_mask = false(size(rt));

    % retrieve quartile table for this patient
    if isnumeric(pid)
        pIdx = find(patient_ids == pid);
    else
        pIdx = find(strcmp(patient_ids, pid));
    end

    for L = 1:3
        for pr = 0:1
            sub = mask_ok & load == L & probe == pr;
            if ~any(sub), continue; end

            q1 = q_lo(pIdx, L, pr+1);
            q3 = q_hi(pIdx, L, pr+1);
            if isnan(q1) || isnan(q3), continue; end

            fast_mask(sub) = rt(sub) <  q1;   % strict
            slow_mask(sub) = rt(sub) >  q3;
        end
    end

    % PSTHs (omit NaNs; handles neurons with no fast/slow trials)
    all_psth(n,:)  = mean(fr(mask_ok,:),        1, 'omitnan');
    fast_psth(n,:) = mean(fr(fast_mask,:),      1, 'omitnan');
    slow_psth(n,:) = mean(fr(slow_mask,:),      1, 'omitnan');

    [~, peak_bins(n)] = max(all_psth(n,:));
end

% ---------- SORT, NORMALISE, PLOT ----------------------------------------
[~, order] = sort(peak_bins);
[sAll,sFast,sSlow] = deal(all_psth(order,:), fast_psth(order,:), slow_psth(order,:));
[sAllN,sFastN,sSlowN] = rowwise_minmax_normalize_multi(sAll, sFast, sSlow);

makePlot(sAllN,  'All Correct Trials');
makePlot(sFastN, 'Fast Trials (<50 % RT)');
makePlot(sSlowN, 'Slow Trials (>50 % RT)');

% ---------- NESTED PLOTTER -----------------------------------------------
    function makePlot(mat, ttl)
        figure; imagesc(mat, [0 1]); colormap(parula); colorbar;
        xt = 0:0.5:time_centers(end);
        set(gca,'TickDir','out',...
            'XTick',arrayfun(@(t)find_closest_bin(time_centers,t),xt),...
            'XTickLabel',arrayfun(@(t)sprintf('%.1f',t),xt,'UniformOutput',false));
        xlabel('Time (s)'); ylabel('Neuron (sorted by peak)');
        title(['Time Cells: ' ttl ' (Joint Norm)']);
    end
end



function [varargout] = rowwise_minmax_normalize_multi(varargin)
%ROWWISE_MINMAX_NORMALIZE_MULTI
%   For each row (i.e. neuron), find the min and max across *all* columns
%   from *all* input matrices combined. Then normalize each matrix’s row 
%   to [0,1] using that global min and max for that row.
%
%   Usage:
%     [A_norm, B_norm, C_norm, ...] = rowwise_minmax_normalize_multi(A, B, C, ...)
% 
%   Each of A, B, C, etc. must have the same number of rows. The function 
%   returns the same number of output arguments as input arguments.

    nArrays = nargin;
    if nArrays < 2
        error('Provide at least two matrices to jointly normalize.');
    end

    % Check that all inputs have the same number of rows
    nRows = size(varargin{1}, 1);
    for k = 2:nArrays
        if size(varargin{k}, 1) ~= nRows
            error('All input matrices must have the same number of rows.');
        end
    end

    % Initialize outputs
    varargout = cell(1, nArrays);
    for k = 1:nArrays
        varargout{k} = zeros(size(varargin{k}));
    end

    % Process row by row
    for r = 1:nRows
        % Combine all rows r across all inputs
        combined_row = [];
        for k = 1:nArrays
            combined_row = [combined_row, varargin{k}(r, :)];
        end

        % Handle all-NaN or constant cases
        if all(isnan(combined_row))
            % All NaNs => just copy as-is
            for k = 1:nArrays
                varargout{k}(r, :) = combined_row(1,1)*0; %#ok
            end
            continue;
        end

        min_val = min(combined_row, [], 'omitnan');
        max_val = max(combined_row, [], 'omitnan');

        if max_val > min_val
            % Normal case
            for k = 1:nArrays
                row_k = varargin{k}(r, :);
                % Replace any NaNs with the min_val (or handle differently if desired)
                row_k(isnan(row_k)) = min_val;
                varargout{k}(r, :) = (row_k - min_val) / (max_val - min_val);
            end
        else
            % If max_val == min_val (constant row), set to 0.5
            for k = 1:nArrays
                varargout{k}(r, :) = 0.5;
            end
        end
    end
end


function idx = find_closest_bin(time_centers, target_time)
%FIND_CLOSEST_BIN Finds the bin index in time_centers closest to target_time.
    [~, idx] = min(abs(time_centers - target_time));
end
 
