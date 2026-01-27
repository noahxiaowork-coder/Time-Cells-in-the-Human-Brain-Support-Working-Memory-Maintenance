function plot_rt_heatmaps_subgroup(neural_data, bin_width, use_correct)

if nargin < 3 || isempty(use_correct), use_correct = true; end

num_neurons = numel(neural_data);
assert(num_neurons > 0, 'No neurons found in neural_data.');

num_bins     = size(neural_data(1).firing_rates, 2);
time_edges   = (0:num_bins) * bin_width;
time_centers = time_edges(1:end-1) + diff(time_edges)/2;

all_psth  = zeros(num_neurons, num_bins);
fast_psth = zeros(num_neurons, num_bins);
slow_psth = zeros(num_neurons, num_bins);
peak_bins = zeros(num_neurons, 1);

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

percentiles = nan(numel(patient_ids), 3, 2);

for p = 1:numel(patient_ids)
    pid = patient_ids(p);
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
        percentiles(p, L, 1) = prctile(rt(msk), 30);
        percentiles(p, L, 2) = prctile(rt(msk), 70);
    end
end

for n = 1:num_neurons
    fr   = neural_data(n).firing_rates;
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

[~, order] = sort(peak_bins,'ascend','MissingPlacement','last');
sFast = fast_psth(order,:);
sSlow = slow_psth(order,:);
[sFastN, sSlowN] = rowwise_minmax_normalize_multi(sFast, sSlow);

makePlot(sFastN, 'Fast Trials');
makePlot(sSlowN, 'Slow Trials');

    function makePlot(mat, ttl)
        figure('Name', ['RT subgroup: ' ttl], 'Position',[80 90 420 648], 'Color','w');
        imagesc(mat, [0 1]);
        colormap(parula);
        cb = colorbar; set(cb, 'FontSize', 16);

        xt_vals   = 0:0.5:time_edges(end);
        xt_idx    = arrayfun(@(t) find_closest_bin(time_centers, t), xt_vals);
        xt_labels = arrayfun(@(t) sprintf('%.1f', t), xt_vals, 'UniformOutput', false);

        set(gca, 'TickDir','out', ...
                 'XTick', xt_idx, ...
                 'XTickLabel', xt_labels, ...
                 'FontSize', 16);
        xlabel('Time (s)', 'FontSize', 16);
        ylabel('Time Cells', 'FontSize', 16);
        title(['Time Cells: ' ttl ], 'FontSize', 16, 'Interpreter','none');
    end
end

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
