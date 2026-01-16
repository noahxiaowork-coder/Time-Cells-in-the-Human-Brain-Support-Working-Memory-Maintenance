function [sig_cells, areasSternberg, time_fields, patient_time_cell_count, ...
          brain_region_time_cell_count, time_cell_info] = NWB_calcSelective_SB(nwbAll, all_units)
%NWB_CALCSELECTIVE_SB Identifies time cells from the Sternberg task dataset
%using a smoothing + z-scoring approach and a permutation test.
%
%Outputs:
%   sig_cells                - Struct containing a logical vector marking which units are time cells
%   areasSternberg           - Brain areas corresponding to each unit
%   time_fields              - The "peak" time bin for each time cell (0 if not a time cell)
%   patient_time_cell_count  - Counts of time cells per patient
%   brain_region_time_cell_count - Counts of time cells per brain region
%   time_cell_info           - [patient_id, unit_id, max_fr_bin] for time cells

    % --- Parameter Setup ---
    alphaLim          = 0.05;
    num_permutations  = 1000;       % For permutation testing
    bin_width_analysis = 0.1;       % Binning for time-cell analysis
    total_bins_analysis = 25; 
    bin_width_psth    = 0.1;        % Binning for final PSTH
    total_bins_psth   = 25;
    window_size       = 5;          % Smoothing window size
    edges             = linspace(0, total_bins_psth * bin_width_psth, total_bins_psth + 1);  
    rng(42)

    % gaussian_kernel = GaussianKernal(1, 2.5);
    gaussian_kernel = GaussianKernal(3, 1.5);


    % --- Pre-allocate / Initialize ---
    areasSternberg = cell(length(all_units), 1);
    brain_regions  = cell(length(all_units), 1);
    time_cells     = zeros(length(all_units), 1);
    time_fields    = zeros(length(all_units), 1);
    psth_matrix    = [];
    peak_locations = [];
    cell_brain_regions = {};
    time_cell_info = [];

    % Cell array storing spike times (relative to trial) for each *detected* time field
    psth_per_time_field = cell(total_bins_analysis, 1);

    % --- Main Loop: Evaluate Each Unit ---
    for i = 1:length(all_units)
        SU           = all_units(i);
        subject_id   = SU.subject_id;
        cellID       = SU.unit_id;
        brain_area   = nwbAll{SU.session_count} ...
                         .general_extracellular_ephys_electrodes ...
                         .vectordata.get('location').data.load(SU.electrodes);

        areasSternberg{i} = brain_area{:};
        brain_regions{i}  = brain_area{:};

        fprintf('Processing (%d/%d): Session SBID %d, Unit %d\n', ...
                i, length(all_units), subject_id, cellID);

        % -- Maintenance Timestamps --
        tsMaint     = num2cell(nwbAll{SU.session_count} ...
                               .intervals_trials.vectordata ...
                               .get('timestamps_Maintenance').data.load());
        stim        = cell2struct(tsMaint, {'tsMaint'}, 2);

        % -- Compute Spike Counts per Bin per Trial --
        spks_per_trial_tc = zeros(length(stim), total_bins_analysis);
        for k = 1:length(stim)
            for b = 1:total_bins_analysis
                bin_start = stim(k).tsMaint + (b-1) * bin_width_analysis;
                bin_end   = bin_start + bin_width_analysis;
                spks_per_trial_tc(k, b) = sum(SU.spike_times >= bin_start & SU.spike_times < bin_end);
            end
        end

        % -- Smooth and Z-Score Each Trial --
        for tr = 1:size(spks_per_trial_tc,1)
            trial_counts  = spks_per_trial_tc(tr,:);
            smoothed = conv(trial_counts, gaussian_kernel, 'same');
            %smoothed      = smooth(trial_counts, window_size, 'moving');
            spks_per_trial_tc(tr,:) = zscore(smoothed);
        end

        % -- Average Firing Vector Across Trials --
        fr_per_bin_all_tc = mean(spks_per_trial_tc, 1);

        % -- Permutation Testing to Check for Time Cell --
        p_value = time_cell_analysis_time_level(fr_per_bin_all_tc, SU.spike_times, ...
                                                stim, bin_width_analysis, ...
                                                total_bins_analysis, ...
                                                num_permutations, window_size);

        if p_value < alphaLim
            time_cells(i) = 1;
            [~, max_fr_bin] = max(fr_per_bin_all_tc);
            time_fields(i)   = max_fr_bin;

            fprintf('| Time Cell -> Max Z-scored FR: %.2f, p: %.4f, Time Field: %d\n', ...
                     max(fr_per_bin_all_tc), p_value, max_fr_bin);

            % -- Store time cell info [patient_id, unit_id, time_field] --
            time_cell_info = [time_cell_info; subject_id, cellID, max_fr_bin]; %#ok<AGROW>

            % -- Build PSTH for Identified Time Cell --
            spike_times_relative = [];
            for k = 1:length(stim)
                t_start = stim(k).tsMaint;
                t_end   = stim(k).tsMaint + total_bins_analysis * bin_width_analysis;
                trial_spikes = SU.spike_times(SU.spike_times >= t_start & SU.spike_times < t_end) - t_start;
                spike_times_relative = [spike_times_relative; trial_spikes]; %#ok<AGROW>
            end

            % Concatenate all spike times that correspond to the same time bin index
            if isempty(psth_per_time_field{max_fr_bin})
                psth_per_time_field{max_fr_bin} = spike_times_relative;
            else
                psth_per_time_field{max_fr_bin} = ...
                    [psth_per_time_field{max_fr_bin}; spike_times_relative];
            end

            % -- Add to PSTH Matrix (for heatmap or future analysis) --
            counts = histcounts(spike_times_relative, edges);
            counts = counts / (length(spike_times_relative) * bin_width_psth);
            psth_matrix    = [psth_matrix; counts]; %#ok<AGROW>
            [~, peak_bin]  = max(counts);
            peak_locations = [peak_locations; peak_bin]; %#ok<AGROW>
            cell_brain_regions = [cell_brain_regions; brain_area{:}]; %#ok<AGROW>
        end
    end

    % --- Print Summary ---
    total_time_cells = sum(time_cells);
    fprintf('Total Time Cells: %d/%d (%.2f%%)\n', ...
             total_time_cells, length(all_units), ...
             100 * total_time_cells / length(all_units));

    % --- Collect Per-Patient Counts ---
    unique_patient_ids = unique([all_units.subject_id]);
    patient_time_cell_count = zeros(length(unique_patient_ids), 1);
    for i = 1:length(unique_patient_ids)
        pid = unique_patient_ids(i);
        patient_time_cell_count(i) = sum([all_units.subject_id] == pid & time_cells' == 1);
        fprintf('Patient ID %d: %d time cells\n', pid, patient_time_cell_count(i));
    end

    % --- Collect Brain Region Counts ---
    unique_brain_regions = unique(brain_regions);
    brain_region_time_cell_count = zeros(length(unique_brain_regions), 1);
    for i = 1:length(unique_brain_regions)
        br = unique_brain_regions{i};
        brain_region_time_cell_count(i) = sum(strcmp(brain_regions, br) & (time_cells == 1));
        fprintf('Brain Region %s: %d time cells\n', br, brain_region_time_cell_count(i));
    end

    % --- Prepare and Return Results ---
    sig_cells.time_cells = time_cells;

end % main function


%% Helper Functions

function units_filtered = applyRateFilter(units, rateThreshold)
% Applies a simple global rate filter if rateThreshold > 0
    if isempty(rateThreshold) || rateThreshold <= 0
        units_filtered = units;
        return;
    end
    
    keepMask = false(size(units));
    for i = 1:length(units)
        stimes = units(i).spike_times;
        globalRate = length(stimes) / (max(stimes) - min(stimes));
        keepMask(i) = (globalRate >= rateThreshold);
    end
    units_filtered = units(keepMask);
end


function p_value = time_cell_analysis_time_level(fr_per_bin_all_tc, spike_times, ...
                                                 stim, bin_width, total_bins, ...
                                                 num_permutations, window_size)


    gaussian_kernel = GaussianKernal(3, 1.5);
    [~, max_fr_bin] = max(fr_per_bin_all_tc);
    observed_fr     = fr_per_bin_all_tc(max_fr_bin);

    permuted_max_fr = zeros(num_permutations, 1);

    for perm = 1:num_permutations
        % --- Random shift within the maintenance period for each trial ---
        perm_spike_times = [];
        for k = 1:length(stim)
            t_start = stim(k).tsMaint;
            t_end   = stim(k).tsMaint + total_bins * bin_width;

            trial_spikes = spike_times(spike_times >= t_start & spike_times < t_end) - t_start;
            shift_amount = (randi(total_bins) - 1) * bin_width;
            perm_trial_spikes = mod(trial_spikes + shift_amount, total_bins * bin_width);

            % Re-add the trial offset
            perm_spike_times = [perm_spike_times; (perm_trial_spikes + t_start)]; %#ok<AGROW>
        end

        % --- Re-bin the permuted spike times ---
        perm_spks_mat = zeros(length(stim), total_bins);
        for k = 1:length(stim)
            for b = 1:total_bins
                bin_s = stim(k).tsMaint + (b-1) * bin_width;
                bin_e = bin_s + bin_width;
                perm_spks_mat(k,b) = sum(perm_spike_times >= bin_s & ...
                                         perm_spike_times < bin_e);
            end
        end

        % --- Smooth + Z-score each trial ---
        for tr = 1:size(perm_spks_mat,1)
            trial_counts       = perm_spks_mat(tr,:);
            %smoothed           = smooth(trial_counts, window_size, 'moving');
            smoothed = conv(trial_counts, gaussian_kernel, 'same');
            perm_spks_mat(tr,:)= zscore(smoothed);
        end

        perm_firing = mean(perm_spks_mat, 1);
        permuted_max_fr(perm) = max(perm_firing);
    end

    % --- p-value as fraction of permutations >= observed metric ---
    p_value = sum(permuted_max_fr >= observed_fr) / num_permutations;
end


