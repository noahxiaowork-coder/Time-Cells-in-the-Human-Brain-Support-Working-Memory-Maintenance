function plot_load_heatmaps(nwbAll, all_units, neural_data_file, ...
                                bin_width, use_correct)
%PLOT_LOAD_HEATMAPS_RAW  Time-cell heat-maps for the 2.5-s Maintenance epoch
%                        (All / Load-1 / Load-2 / Load-3).
%
% INPUTS
%   nwbAll            – {1×S} cell array of NWB session objects
%   all_units         – struct array with fields: subject_id, unit_id,
%                       session_count, spike_times
%   neural_data_file  – MAT file that holds the neural_data struct array
%                       (used only for trial metadata: load & correctness)
%   bin_width         – PSTH bin size (s)  (e.g., 0.100 for 100 ms)
%   use_correct       – logical (default = false).  If true, keep only
%                       trials where neural_data.trial_correctness == 1
%
% OUTPUT
%   Generates four figures (All, L1, L2, L3), each jointly min-max
%   normalised across conditions per neuron.

% -------------------------------------------------------------- set-up ---
if nargin < 5, use_correct = false; end
maintDur   = 2.5;                          % 2.5-s Maintenance period
nBins      = round(maintDur / bin_width);  % e.g., 25 bins for 100 ms

load(neural_data_file,'neural_data');
nNeurons   = numel(neural_data);
assert(nNeurons > 0, 'No neurons found in neural_data.');

% --- Gaussian kernel ----------------------------------------------------
gKernel = GaussianKernal(0.3 / bin_width, 1.5);

% Pre-allocate PSTH matrices
all_psth   = nan(nNeurons, nBins);
load1_psth = nan(nNeurons, nBins);
load2_psth = nan(nNeurons, nBins);
load3_psth = nan(nNeurons, nBins);
peak_bins  = nan(nNeurons, 1);

% ---------------------------------------------------------- main loop ----
for i = 1:nNeurons
    nd  = neural_data(i);
    pid = nd.patient_id;  uid = nd.unit_id;

    idx = find([all_units.subject_id] == pid & ...
               [all_units.unit_id]    == uid, 1);
    if isempty(idx), continue; end
    SU   = all_units(idx);
    sess = nwbAll{SU.session_count};

    tsMai = sess.intervals_trials. ...
                vectordata.get('timestamps_Maintenance').data.load();
    nTrials = numel(tsMai);
    if nTrials == 0, continue; end

    load_vals = nd.trial_load(:);
    corr_vals = nd.trial_correctness(:);

    if use_correct
        keepOK = (corr_vals == 1);
    else
        keepOK = true(size(load_vals));
    end

    fr = nan(nTrials, nBins);
    for t = 1:nTrials
        if ~keepOK(t), continue; end
        edges = tsMai(t) : bin_width : tsMai(t) + maintDur;
        if numel(edges) ~= nBins + 1, continue; end
        tmp = histcounts(SU.spike_times,edges)./bin_width;
        fr(t,:) = conv(tmp, gKernel, 'same');
    end

    load1_idx =  load_vals == 1 & keepOK;
    load2_idx =  load_vals == 2 & keepOK;
    load3_idx =  load_vals == 3 & keepOK;

    avg_all   = mean(fr,                 1, 'omitnan');
    avg_L1    = mean(fr(load1_idx, :),   1, 'omitnan');
    avg_L2    = mean(fr(load2_idx, :),   1, 'omitnan');
    avg_L3    = mean(fr(load3_idx, :),   1, 'omitnan');

    all_psth(i,:)   = avg_all;
    load1_psth(i,:) = avg_L1;
    load2_psth(i,:) = avg_L2;
    load3_psth(i,:) = avg_L3;

    [~, peak_bins(i)] = max(avg_all);
end

[~, sort_idx] = sort(peak_bins,'ascend','MissingPlacement','last');
all_psth   = all_psth(sort_idx,  :);
load1_psth = load1_psth(sort_idx,:);
load2_psth = load2_psth(sort_idx,:);
load3_psth = load3_psth(sort_idx,:);

all_psth_n = all_psth;
[load1_psth_n, load2_psth_n, load3_psth_n] = ...
    rowwise_minmax_normalize_multi(load1_psth, ...
                                   load2_psth, load3_psth);

cmap = parula;
time_centers = (0:nBins-1)*bin_width + bin_width/2;
tickStep = 0.5;
timeTicks = 0:tickStep:maintDur;
xticks = arrayfun(@(t) find_closest_bin(time_centers,t), timeTicks);
xticklabels = arrayfun(@(t) sprintf('%.1f',t), timeTicks,'uni',0);

makeFig(all_psth_n,   'All Trials',   cmap);
makeFig(load1_psth_n, 'Load-1 Trials',cmap);
makeFig(load2_psth_n, 'Load-2 Trials',cmap);
makeFig(load3_psth_n, 'Load-3 Trials',cmap);

disp('Done – Maintenance-only load heat-maps generated.');

% =======================================================================
% -------------------------- nested helpers -----------------------------
    function makeFig(mat, ttl, cmap)
        figure('Name', ttl, 'Units','pixels','Position',[100 100 400 648]);
        imagesc(mat,[0 1]); colormap(cmap); colorbar;
        xlabel('Time (s)', 'FontSize', 16);
        ylabel('Time Cells', 'FontSize', 16);
        title(sprintf('Time Cells: %s (joint norm.)',ttl), 'FontSize', 16);
        set(gca,'TickDir','out', ...
                'XTick',xticks,'XTickLabel',xticklabels, ...
                'FontSize', 16);
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