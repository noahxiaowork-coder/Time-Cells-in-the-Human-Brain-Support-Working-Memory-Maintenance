function plot_load_heatmaps_Correctonly(neural_data_file, bin_width)
%PLOT_LOAD_HEATMAPS Loads neural_data and generates four heatmaps:
% 1) All trials
% 2) Load 1 trials only
% 3) Load 2 trials only
% 4) Load 3 trials only
%
% Each row (neuron) is min-max normalized to [0,1], but crucially, the same
% neuron is normalized by a single min and max across all conditions.

    % -------------------- LOAD DATA --------------------
    load(neural_data_file, 'neural_data');
    num_neurons = length(neural_data);
    if num_neurons == 0
        error('No neurons found in neural_data.');
    end

    % Determine number of bins and create time vectors
    num_bins = size(neural_data(1).firing_rates, 2);
    time_edges = (0:num_bins) * bin_width;
    time_centers = time_edges(1:end-1) + diff(time_edges)/2;

    % Preallocate arrays
    all_psth   = zeros(num_neurons, num_bins);
    load1_psth = zeros(num_neurons, num_bins);
    load2_psth = zeros(num_neurons, num_bins);
    load3_psth = zeros(num_neurons, num_bins);
    peak_bins  = zeros(num_neurons, 1);

    % -------------------- COMPUTE PSTHS --------------------
    for i = 1:num_neurons
        fr        = neural_data(i).firing_rates;      % [trials x bins]
        load_vals = neural_data(i).trial_load;        % [trials x 1], values 1, 2, or 3
        correctness = neural_data(i).trial_correctness;

        % Separate trials by load
        load1_trials = (load_vals == 1 & correctness == 1);
        load2_trials = (load_vals == 2 & correctness == 1);
        load3_trials = (load_vals == 3 & correctness == 1);

        % Compute average firing rates
        avg_fr_all    = mean(fr, 1);
        avg_fr_load1  = mean(fr(load1_trials, :), 1, 'omitnan');
        avg_fr_load2  = mean(fr(load2_trials, :), 1, 'omitnan');
        avg_fr_load3  = mean(fr(load3_trials, :), 1, 'omitnan');

        % Find peak bin from all trials PSTH
        [~, peak_bin] = max(avg_fr_all);
        peak_bins(i)  = peak_bin;

        % Store in arrays
        all_psth(i, :)   = avg_fr_all;
        load1_psth(i, :) = avg_fr_load1;
        load2_psth(i, :) = avg_fr_load2;
        load3_psth(i, :) = avg_fr_load3;
    end

    % -------------------- SORT NEURONS BY PEAK --------------------
    [~, sort_idx] = sort(peak_bins);
    sorted_all_psth   = all_psth(sort_idx, :);
    sorted_load1_psth = load1_psth(sort_idx, :);
    sorted_load2_psth = load2_psth(sort_idx, :);
    sorted_load3_psth = load3_psth(sort_idx, :);

    % -------------------- JOINT NORMALIZATION --------------------
    % We want the same neuron to be normalized by the same (min, max) across
    % all four PSTHs (All, Load1, Load2, Load3).
    [sorted_all_psth_norm, ...
     sorted_load1_psth_norm, ...
     sorted_load2_psth_norm, ...
     sorted_load3_psth_norm] = rowwise_minmax_normalize_multi( ...
         sorted_all_psth, ...
         sorted_load1_psth, ...
         sorted_load2_psth, ...
         sorted_load3_psth);

    % -------------------- PLOTTING --------------------
    cmap = parula;

    % Determine x-axis ticks and labels
    desired_tick_step = 0.5; % seconds between ticks
    max_time = time_centers(end);
    time_ticks = 0:desired_tick_step:max_time;
    xticks = arrayfun(@(t) find_closest_bin(time_centers, t), time_ticks);
    xticklabels = arrayfun(@(t) sprintf('%.1f', t), time_ticks, 'UniformOutput', false);

    % Plot all trials heatmap
    figure;
    imagesc(sorted_all_psth_norm, [0 1]); % color scale [0,1]
    colormap(cmap); colorbar;
    xlabel('Time (s)'); ylabel('Neuron (sorted by peak)');
    title('Time Cells: All Trials (Joint Normalized)');
    set(gca, 'TickDir', 'out');
    set(gca, 'XTick', xticks, 'XTickLabel', xticklabels);

    % Plot load 1 trials
    figure;
    imagesc(sorted_load1_psth_norm, [0 1]);
    colormap(cmap); colorbar;
    xlabel('Time (s)'); ylabel('Neuron (sorted by all-trials peak)');
    title('Time Cells: Load 1 Trials (Joint Normalized)');
    set(gca, 'TickDir', 'out');
    set(gca, 'XTick', xticks, 'XTickLabel', xticklabels);

    % Plot load 2 trials
    figure;
    imagesc(sorted_load2_psth_norm, [0 1]);
    colormap(cmap); colorbar;
    xlabel('Time (s)'); ylabel('Neuron (sorted by all-trials peak)');
    title('Time Cells: Load 2 Trials (Joint Normalized)');
    set(gca, 'TickDir', 'out');
    set(gca, 'XTick', xticks, 'XTickLabel', xticklabels);

    % Plot load 3 trials
    figure;
    imagesc(sorted_load3_psth_norm, [0 1]);
    colormap(cmap); colorbar;
    xlabel('Time (s)'); ylabel('Neuron (sorted by all-trials peak)');
    title('Time Cells: Load 3 Trials (Joint Normalized)');
    set(gca, 'TickDir', 'out');
    set(gca, 'XTick', xticks, 'XTickLabel', xticklabels);

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
