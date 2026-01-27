function [sig_cells, areasSternberg] = NWB_calcSelective_SB_concept(nwbAll, all_units)
%GENERATE_IMAGE_HEATMAPS Generates heatmaps of concept cells' firing rates
% for each image during encoding session 1.
disp("begin")
concept_cells_list = [
    2, 1, 34, "/stimulus/templates/StimulusTemplates/image_34";
    2, 2, 28, "/stimulus/templates/StimulusTemplates/image_28";
    2, 13, 59, "/stimulus/templates/StimulusTemplates/image_59";
    2, 21, 59, "/stimulus/templates/StimulusTemplates/image_59";
    3, 11, 23, "/stimulus/templates/StimulusTemplates/image_23";
    3, 18, 2, "/stimulus/templates/StimulusTemplates/image_2";
    3, 28, 21, "/stimulus/templates/StimulusTemplates/image_21";
    4, 3, 57, "/stimulus/templates/StimulusTemplates/image_57";
    4, 4, 57, "/stimulus/templates/StimulusTemplates/image_57";
    4, 7, 57, "/stimulus/templates/StimulusTemplates/image_57";
    4, 8, 57, "/stimulus/templates/StimulusTemplates/image_57";
    4, 13, 15, "/stimulus/templates/StimulusTemplates/image_15";
    4, 21, 56, "/stimulus/templates/StimulusTemplates/image_56";
    4, 40, 56, "/stimulus/templates/StimulusTemplates/image_56";
    4, 53, 56, "/stimulus/templates/StimulusTemplates/image_56";
    4, 54, 56, "/stimulus/templates/StimulusTemplates/image_56";
    4, 56, 65, "/stimulus/templates/StimulusTemplates/image_65";
    4, 58, 65, "/stimulus/templates/StimulusTemplates/image_65";
    4, 63, 29, "/stimulus/templates/StimulusTemplates/image_29";
    4, 65, 57, "/stimulus/templates/StimulusTemplates/image_57";
    5, 24, 45, "/stimulus/templates/StimulusTemplates/image_45";
    5, 48, 47, "/stimulus/templates/StimulusTemplates/image_47";
    5, 49, 33, "/stimulus/templates/StimulusTemplates/image_33";
    7, 3, 38, "/stimulus/templates/StimulusTemplates/image_38";
    7, 4, 38, "/stimulus/templates/StimulusTemplates/image_38";
    7, 5, 38, "/stimulus/templates/StimulusTemplates/image_38";
    7, 6, 38, "/stimulus/templates/StimulusTemplates/image_38";
    7, 9, 38, "/stimulus/templates/StimulusTemplates/image_38";
    7, 16, 31, "/stimulus/templates/StimulusTemplates/image_31";
    8, 5, 8, "/stimulus/templates/StimulusTemplates/image_8";
    8, 40, 45, "/stimulus/templates/StimulusTemplates/image_45";
    8, 41, 45, "/stimulus/templates/StimulusTemplates/image_45";
    8, 44, 37, "/stimulus/templates/StimulusTemplates/image_37";
    8, 53, 45, "/stimulus/templates/StimulusTemplates/image_45";
    9, 1, 5, "/stimulus/templates/StimulusTemplates/image_5";
    9, 7, 5, "/stimulus/templates/StimulusTemplates/image_5";
    9, 10, 5, "/stimulus/templates/StimulusTemplates/image_5";
    9, 12, 5, "/stimulus/templates/StimulusTemplates/image_5";
    9, 13, 17, "/stimulus/templates/StimulusTemplates/image_17";
    9, 17, 5, "/stimulus/templates/StimulusTemplates/image_5";
    9, 27, 17, "/stimulus/templates/StimulusTemplates/image_17";
    10, 18, 44, "/stimulus/templates/StimulusTemplates/image_44";
    11, 34, 59, "/stimulus/templates/StimulusTemplates/image_59";
    11, 35, 25, "/stimulus/templates/StimulusTemplates/image_25";
    11, 37, 63, "/stimulus/templates/StimulusTemplates/image_63";
    11, 45, 63, "/stimulus/templates/StimulusTemplates/image_63";
    11, 46, 32, "/stimulus/templates/StimulusTemplates/image_32";
    11, 62, 63, "/stimulus/templates/StimulusTemplates/image_63";
    11, 64, 32, "/stimulus/templates/StimulusTemplates/image_32";
    12, 13, 4, "/stimulus/templates/StimulusTemplates/image_4";
    12, 14, 21, "/stimulus/templates/StimulusTemplates/image_21";
    12, 15, 19, "/stimulus/templates/StimulusTemplates/image_19";
    12, 20, 14, "/stimulus/templates/StimulusTemplates/image_14";
    13, 7, 23, "/stimulus/templates/StimulusTemplates/image_23";
    13, 8, 54, "/stimulus/templates/StimulusTemplates/image_54";
    13, 13, 63, "/stimulus/templates/StimulusTemplates/image_63";
    13, 15, 54, "/stimulus/templates/StimulusTemplates/image_54";
    13, 18, 63, "/stimulus/templates/StimulusTemplates/image_63";
    13, 21, 63, "/stimulus/templates/StimulusTemplates/image_63";
    13, 22, 63, "/stimulus/templates/StimulusTemplates/image_63";
    13, 23, 5, "/stimulus/templates/StimulusTemplates/image_5";
    13, 25, 63, "/stimulus/templates/StimulusTemplates/image_63";
    13, 26, 63, "/stimulus/templates/StimulusTemplates/image_63";
    13, 27, 23, "/stimulus/templates/StimulusTemplates/image_23";
    13, 29, 63, "/stimulus/templates/StimulusTemplates/image_63";
    13, 30, 23, "/stimulus/templates/StimulusTemplates/image_23";
    13, 31, 5, "/stimulus/templates/StimulusTemplates/image_5";
    13, 32, 5, "/stimulus/templates/StimulusTemplates/image_5";
    13, 35, 63, "/stimulus/templates/StimulusTemplates/image_63";
    13, 38, 5, "/stimulus/templates/StimulusTemplates/image_5";
    14, 5, 7, "/stimulus/templates/StimulusTemplates/image_7";
    14, 12, 4, "/stimulus/templates/StimulusTemplates/image_4";
    14, 13, 2, "/stimulus/templates/StimulusTemplates/image_2";
    14, 14, 2, "/stimulus/templates/StimulusTemplates/image_2";
    14, 15, 2, "/stimulus/templates/StimulusTemplates/image_2";
    14, 16, 7, "/stimulus/templates/StimulusTemplates/image_7";
    14, 17, 3, "/stimulus/templates/StimulusTemplates/image_3";
    14, 18, 25, "/stimulus/templates/StimulusTemplates/image_25";
    14, 19, 7, "/stimulus/templates/StimulusTemplates/image_7";
    14, 20, 25, "/stimulus/templates/StimulusTemplates/image_25";
    14, 21, 3, "/stimulus/templates/StimulusTemplates/image_3";
    14, 22, 2, "/stimulus/templates/StimulusTemplates/image_2";
    14, 25, 4, "/stimulus/templates/StimulusTemplates/image_4";
    14, 26, 4, "/stimulus/templates/StimulusTemplates/image_4";
    14, 27, 4, "/stimulus/templates/StimulusTemplates/image_4";
    14, 28, 4, "/stimulus/templates/StimulusTemplates/image_4";
    14, 29, 7, "/stimulus/templates/StimulusTemplates/image_7";
    14, 35, 25, "/stimulus/templates/StimulusTemplates/image_25";
    14, 37, 7, "/stimulus/templates/StimulusTemplates/image_7";
    14, 39, 7, "/stimulus/templates/StimulusTemplates/image_7";
    14, 53, 25, "/stimulus/templates/StimulusTemplates/image_25";
    15, 16, 57, "/stimulus/templates/StimulusTemplates/image_57";
    15, 31, 57, "/stimulus/templates/StimulusTemplates/image_57";
    16, 1, 23, "/stimulus/templates/StimulusTemplates/image_23";
    16, 2, 23, "/stimulus/templates/StimulusTemplates/image_23";
    16, 4, 23, "/stimulus/templates/StimulusTemplates/image_23";
    16, 8, 29, "/stimulus/templates/StimulusTemplates/image_29";
    16, 12, 42, "/stimulus/templates/StimulusTemplates/image_42";
    16, 14, 23, "/stimulus/templates/StimulusTemplates/image_23";
    16, 16, 42, "/stimulus/templates/StimulusTemplates/image_42";
    16, 18, 5, "/stimulus/templates/StimulusTemplates/image_5";
    16, 21, 23, "/stimulus/templates/StimulusTemplates/image_23";
    16, 31, 24, "/stimulus/templates/StimulusTemplates/image_24";
    16, 33, 42, "/stimulus/templates/StimulusTemplates/image_42";
    18, 2, 11, "/stimulus/templates/StimulusTemplates/image_11";
    18, 4, 11, "/stimulus/templates/StimulusTemplates/image_11";
    18, 7, 11, "/stimulus/templates/StimulusTemplates/image_11";
    18, 9, 62, "/stimulus/templates/StimulusTemplates/image_62";
    18, 10, 52, "/stimulus/templates/StimulusTemplates/image_52";
    18, 20, 41, "/stimulus/templates/StimulusTemplates/image_41";
    18, 26, 52, "/stimulus/templates/StimulusTemplates/image_52";
    18, 29, 11, "/stimulus/templates/StimulusTemplates/image_11";
    18, 34, 62, "/stimulus/templates/StimulusTemplates/image_62";
    18, 36, 62, "/stimulus/templates/StimulusTemplates/image_62";
    18, 57, 18, "/stimulus/templates/StimulusTemplates/image_18";
    19, 7, 15, "/stimulus/templates/StimulusTemplates/image_15";
    21, 2, 16, "/stimulus/templates/StimulusTemplates/image_16";
    21, 3, 49, "/stimulus/templates/StimulusTemplates/image_49";
    21, 6, 65, "/stimulus/templates/StimulusTemplates/image_65";
];

encodingPhase = 'Encoding1';
timestampKey = 'timestamps_Encoding1';
picIDKey = 'loadsEnc1_PicIDs';
time_fields = [];
p_values_pref = [];
p_values_nonPref = [];
patient_unit_info = [];

% Define analysis parameters
bin_width_analysis = 0.1;  % Bin width for time cell analysis
total_bins_analysis = 10;  % Total number of bins for 1 second period with 0.2 bin width
num_permutations = 1000;  % Number of permutations for significance testing

% Loop over the provided concept cells list
for row = 1:size(concept_cells_list, 1)
    subject_id = double(concept_cells_list(row, 1));
    unit_id = double(concept_cells_list(row, 2));
    pref_imageID = double(concept_cells_list(row, 3));

    % Find the corresponding unit in all_units
    for i = 1:length(all_units)
        SU = all_units(i);
        if double(SU.subject_id) == subject_id && double(SU.unit_id) == unit_id
            % Load stimulus timestamps and IDs for ENC1
            if isKey(nwbAll{SU.session_count}.intervals_trials.vectordata, timestampKey) && ...
               isKey(nwbAll{SU.session_count}.intervals_trials.vectordata, picIDKey)

                tsEnc = num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get(timestampKey).data.load());
                ID_Enc = num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get(picIDKey).data.load());

                stim = cell2struct(horzcat(tsEnc, ID_Enc), {['ts' encodingPhase], ['id' encodingPhase]}, 2);

                % Collect firing rates for trials with images other than the preferred image
                trials_nonPref = find([stim.(['id' encodingPhase])] ~= pref_imageID);
                trials_pref = find([stim.(['id' encodingPhase])] == pref_imageID);

                % Skip cells that have not seen their preferred image in ENC1
                if isempty(trials_pref)
                    continue;
                end

                % Bin the data into 0.2s for time cell analysis
                spks_per_trial_pref = zeros(length(trials_pref), total_bins_analysis);
                spks_per_trial_nonPref = zeros(length(trials_nonPref), total_bins_analysis);

                for k = 1:length(trials_pref)
                    for b = 1:total_bins_analysis
                        bin_start = stim(trials_pref(k)).(['ts' encodingPhase]) + (b-1) * bin_width_analysis;
                        bin_end = bin_start + bin_width_analysis;
                        spks_per_trial_pref(k, b) = sum(SU.spike_times >= bin_start & SU.spike_times < bin_end);
                    end
                end

                for k = 1:length(trials_nonPref)
                    for b = 1:total_bins_analysis
                        bin_start = stim(trials_nonPref(k)).(['ts' encodingPhase]) + (b-1) * bin_width_analysis;
                        bin_end = bin_start + bin_width_analysis;
                        spks_per_trial_nonPref(k, b) = sum(SU.spike_times >= bin_start & SU.spike_times < bin_end);
                    end
                end

                % Calculate average firing rate per bin across all trials for analysis
                fr_per_bin_all_pref = mean(spks_per_trial_pref, 1);
                fr_per_bin_all_nonPref = mean(spks_per_trial_nonPref, 1);

                % Perform time cell analysis for preferred trials
                p_value_pref = time_cell_analysis_time_level(fr_per_bin_all_pref, SU.spike_times, stim(trials_pref), bin_width_analysis, total_bins_analysis, num_permutations, encodingPhase);

                % Perform time cell analysis for non-preferred trials
                p_value_nonPref = time_cell_analysis_time_level(fr_per_bin_all_nonPref, SU.spike_times, stim(trials_nonPref), bin_width_analysis, total_bins_analysis, num_permutations, encodingPhase);

                % Store the results
                [~, max_fr_bin_pref] = max(fr_per_bin_all_pref);
                time_fields = [time_fields; max_fr_bin_pref];
                p_values_pref = [p_values_pref; p_value_pref];
                p_values_nonPref = [p_values_nonPref; p_value_nonPref];
                patient_unit_info = [patient_unit_info; {subject_id, unit_id, pref_imageID}];

                fprintf('Subject: %d, Unit: %d, Preferred Image: %d, p-value (Preferred): %.4f, p-value (Non-Preferred): %.4f\n', subject_id, unit_id, pref_imageID, p_value_pref, p_value_nonPref);
            end
            break; % Exit the loop once the unit is found and processed
        end
    end
end

% Display the results
fprintf('Time Cell Analysis Results:\n');
fprintf('Patient ID | Unit ID | Preferred Image ID | Time Field | p-value (Preferred) | p-value (Non-Preferred)\n');
for k = 1:length(p_values_pref)
    fprintf('%9d | %7d | %20d | %10d | %.4f | %.4f\n', patient_unit_info{k, 1}, patient_unit_info{k, 2}, patient_unit_info{k, 3}, time_fields(k), p_values_pref(k), p_values_nonPref(k));
end

sig_cells = 0;
areasSternberg = 0;
end

function p_value = time_cell_analysis_time_level(fr_per_bin_all_tc, spike_times, stim, bin_width, total_bins, num_permutations, encodingPhase)
%TIME_CELL_ANALYSIS_TIME_LEVEL Performs time cell analysis using permutation testing with time-level shifting

    % Identify the bin with the highest average firing rate
    [~, max_fr_bin] = max(fr_per_bin_all_tc);
    observed_fr = fr_per_bin_all_tc(max_fr_bin);

    % Initialize array to hold firing rates from permutations
    permuted_firing_rates = zeros(num_permutations, 1);

    % Perform permutations to calculate significance
    for perm = 1:num_permutations
        % Generate random shifts and apply circular shift to spike data
        perm_spike_times = [];
        for k = 1:length(stim)
            trial_spikes = spike_times(spike_times >= stim(k).(['ts' encodingPhase]) & spike_times < (stim(k).(['ts' encodingPhase]) + total_bins * bin_width)) - stim(k).(['ts' encodingPhase]);
            shift_amount = (randi(total_bins) - 1) * bin_width; % Random shift within the encoding period
            perm_trial_spikes = mod(trial_spikes + shift_amount, total_bins * bin_width);
            perm_spike_times = [perm_spike_times; perm_trial_spikes + stim(k).(['ts' encodingPhase])];
        end

        % Re-bin the permuted spike times
        perm_spks_per_trial_tc = zeros(length(stim), total_bins);
        for k = 1:length(stim)
            for b = 1:total_bins
                bin_start = stim(k).(['ts' encodingPhase]) + (b-1) * bin_width;
                bin_end = bin_start + bin_width;
                perm_spks_per_trial_tc(k, b) = sum(perm_spike_times >= bin_start & perm_spike_times < bin_end);
            end
        end
        
        % Calculate firing rates for time cell analysis
        perm_fr_per_bin_all_tc = mean(perm_spks_per_trial_tc, 1);
        permuted_firing_rates(perm) = max(perm_fr_per_bin_all_tc);  % Record the highest firing rate in the permutation
    end

    % Calculate p-value
    p_value = sum(permuted_firing_rates >= observed_fr) / num_permutations;
end



