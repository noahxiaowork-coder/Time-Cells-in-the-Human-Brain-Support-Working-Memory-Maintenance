function plot_time_cell_heatmaps(neural_data_file, bin_width)
%PLOT_TIME_CELL_HEATMAPS  Visualise time‑cell PSTHs as heatmaps.
%   ▸ One heatmap for ALL trials.
%   ▸ Four additional heatmaps for each trial QUARTER (Q1–Q4).
%
% Normalisation strategy
% ──────────────────────
% Every neuron's min/max is taken **only** from its ALL‑trials PSTH.  The
% same min/max are then applied to its quarter PSTHs so that colour scales
% are directly comparable across all five heatmaps.  After scaling, values
% >1 are clipped to 1 and <0 are clipped to 0.
%
% Sorting strategy
% ────────────────
% Neurons are ordered by the location of their peak bin in the ALL‑trials
% PSTH; this ordering is used for all five panels.

    % ---------------------- Load Data ----------------------------------- %
    load(neural_data_file, 'neural_data');
    num_neurons = length(neural_data);
    if num_neurons == 0
        error('No neurons found in neural_data.');
    end

    % ------------------ Create Time Axes -------------------------------- %
    num_bins     = size(neural_data(1).firing_rates, 2);
    time_edges   = (0:num_bins) * bin_width;          % bin edges in seconds
    time_centers = time_edges(1:end-1) + diff(time_edges)/2; % bin centres

    % --------------- Preallocate PSTH Arrays ---------------------------- %
    all_psth = zeros(num_neurons, num_bins);
    q1_psth  = zeros(num_neurons, num_bins);
    q2_psth  = zeros(num_neurons, num_bins);
    q3_psth  = zeros(num_neurons, num_bins);
    q4_psth  = zeros(num_neurons, num_bins);

    peak_bins_all = zeros(num_neurons, 1);  % for sorting

    % ---------------- Compute PSTHs & Peak Bins ------------------------- %
    for i = 1:num_neurons
        fr = neural_data(i).firing_rates;            % [trials × bins]

        % Quarter indices -------------------------------------------------
        num_trials = size(fr,1);
        qsize      = floor(num_trials/4);
        q1_idx = 1:qsize;
        q2_idx = qsize+1:2*qsize;
        q3_idx = 2*qsize+1:3*qsize;
        q4_idx = 3*qsize+1:num_trials;               % absorbs remainder

        % Average PSTHs --------------------------------------------------- %
        avg_q1  = mean(fr(q1_idx,:), 1);
        avg_q2  = mean(fr(q2_idx,:), 1);
        avg_q3  = mean(fr(q3_idx,:), 1);
        avg_q4  = mean(fr(q4_idx,:), 1);
        avg_all = mean(fr,            1);

        % Peak bin for sorting ------------------------------------------- %
        [~, peak_all] = max(avg_all);
        peak_bins_all(i) = peak_all;

        % Store ----------------------------------------------------------- %
        all_psth(i,:) = avg_all;
        q1_psth(i,:)  = avg_q1;
        q2_psth(i,:)  = avg_q2;
        q3_psth(i,:)  = avg_q3;
        q4_psth(i,:)  = avg_q4;
    end

    % ---------------- Sort by peak bin (ALL trials) --------------------- %
    [~, sort_idx] = sort(peak_bins_all, 'ascend');
    all_psth = all_psth(sort_idx,:);
    q1_psth  = q1_psth(sort_idx,:);
    q2_psth  = q2_psth(sort_idx,:);
    q3_psth  = q3_psth(sort_idx,:);
    q4_psth  = q4_psth(sort_idx,:);

    % ---------------- Row‑wise normalisation (shared min/max) ----------- %
    [all_psth, q1_psth, q2_psth, q3_psth, q4_psth] = ...
        reference_normalise(all_psth, q1_psth, q2_psth, q3_psth, q4_psth);

    % -------------------- Plot Parameters ------------------------------- %
    cmap = parula;
    desired_tick_step = 0.5;                % seconds between x‑axis ticks
    time_ticks  = 0:desired_tick_step:time_centers(end);
    xticks      = arrayfun(@(t) find_closest_bin(time_centers,t), time_ticks);
    xticklabels = arrayfun(@(t) sprintf('%.1f',t), time_ticks,'UniformOutput',false);

    % ------------------ Plotting ---------------------------------------- %
    figure_names = {'All Trials','Quarter 1','Quarter 2','Quarter 3','Quarter 4'};
    psth_mats    = { all_psth,   q1_psth,    q2_psth,   q3_psth,   q4_psth };

    for f = 1:numel(psth_mats)
        figure('Name',['Time Cells: ' figure_names{f}], ...
               'Position',[100+400*(f-1), 100, 370, 500]);
        imagesc(psth_mats{f}, [0 1]);
        colormap(cmap);
        colorbar;
        xlabel('Time (s)');
        ylabel('Time Cells');
        title(figure_names{f});
        set(gca,'TickDir','out','XTick',xticks,'XTickLabel',xticklabels);
    end
end

% ====================================================================== %
% Helper functions
% ====================================================================== %

function [all_norm, q1_norm, q2_norm, q3_norm, q4_norm] = ...
         reference_normalise(all_mat, q1_mat, q2_mat, q3_mat, q4_mat)
%REFERENCE_NORMALISE  Min‑max scale rows of multiple matrices using the
%                     min/max of the reference matrix (all_mat).
% Values >1 are clipped to 1; values <0 are clipped to 0.

    min_vals = min(all_mat, [], 2);
    max_vals = max(all_mat, [], 2);
    range_vals = max_vals - min_vals;

    mats = {all_mat, q1_mat, q2_mat, q3_mat, q4_mat};
    for m = 1:numel(mats)
        mat = mats{m};
        for r = 1:size(mat,1)
            if range_vals(r) > 0
                mat(r,:) = (mat(r,:) - min_vals(r)) / range_vals(r);
            else
                mat(r,:) = 0.5;      % constant row
            end
            % Clip out‑of‑range values
            mat(r, mat(r,:) < 0) = 0;
            mat(r, mat(r,:) > 1) = 1;
        end
        mats{m} = mat;
    end
    [all_norm, q1_norm, q2_norm, q3_norm, q4_norm] = mats{:};
end

function idx = find_closest_bin(time_centers, target_time)
%FIND_CLOSEST_BIN  Return index of time_centers closest to target_time.
    [~, idx] = min(abs(time_centers - target_time));
end
