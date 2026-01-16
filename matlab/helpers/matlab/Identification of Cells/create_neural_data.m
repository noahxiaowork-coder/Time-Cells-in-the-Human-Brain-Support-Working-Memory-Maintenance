function neural_data = create_neural_data(nwbAll, all_units, time_cell_info_file, bin_width_analysis, use_z_score, use_smooth)
% Load time_cell_info
data = load(time_cell_info_file, 'time_cell_info');
time_cell_info = data.time_cell_info;

total_bins_analysis = 2.5 / bin_width_analysis;
gaussian_kernel = GaussianKernal(0.3 / bin_width_analysis, 1.5);  % Gaussian kernel for smoothing

neural_data = struct('patient_id', {}, 'unit_id', {}, 'time_field', {}, ...
    'firing_rates', {}, 'trial_correctness', {}, 'brain_region', {}, ...
    'trial_imageIDs', {}, 'trial_load', {}, 'trial_RT', {}, 'trial_probe_in_out', {});

for idx = 1:size(time_cell_info,1)
    patient_id = time_cell_info(idx,1);
    unit_id = time_cell_info(idx,2);
    time_field = time_cell_info(idx,3);

    unit_match = ([all_units.subject_id] == patient_id & [all_units.unit_id] == unit_id);
    if ~any(unit_match)
        warning('Unit with patient_id=%d and unit_id=%d not found in all_units.', patient_id, unit_id);
        continue;
    end
    SU = all_units(unit_match);

    tsMaint = nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Maintenance').data.load();
    response_accuracy = nwbAll{SU.session_count}.intervals_trials.vectordata.get('response_accuracy').data.load();
    tsProbe = nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Probe').data.load();
    tsResp  = nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Response').data.load();

    trial_RT = tsResp - tsProbe;  % raw reaction time (s)

    vd = nwbAll{SU.session_count}.intervals_trials.vectordata;
    probe_in_out = vd.get('probe_in_out').data.load();

    ID_Enc1 = nwbAll{SU.session_count}.intervals_trials.vectordata.get('loadsEnc1_PicIDs').data.load();
    ID_Enc2 = nwbAll{SU.session_count}.intervals_trials.vectordata.get('loadsEnc2_PicIDs').data.load();
    ID_Enc3 = nwbAll{SU.session_count}.intervals_trials.vectordata.get('loadsEnc3_PicIDs').data.load();

    stim = struct('tsMaint', num2cell(tsMaint), 'response_accuracy', num2cell(response_accuracy));
    correct_vector = double([stim.response_accuracy] == 1);
    trial_imageIDs = [ID_Enc1(:), ID_Enc2(:), ID_Enc3(:)];
    trial_load = sum(trial_imageIDs ~= 0, 2);

    brain_area = nwbAll{SU.session_count}.general_extracellular_ephys_electrodes.vectordata.get('location').data.load(SU.electrodes);
    brain_region = brain_area{:};

    % Bin spikes per trial
    spks_per_trial_tc = zeros(length(stim), total_bins_analysis);
    for tr = 1:length(stim)
        for b = 1:total_bins_analysis
            bin_start = stim(tr).tsMaint + (b-1)*bin_width_analysis;
            bin_end = bin_start + bin_width_analysis;
            spks_per_trial_tc(tr, b) = sum(SU.spike_times >= bin_start & SU.spike_times < bin_end);
        end
    end

    % Smooth and Z-score per trial
    for tr = 1:size(spks_per_trial_tc,1)
        trial_counts = spks_per_trial_tc(tr,:);
        if use_smooth
            trial_counts_smoothed = conv(trial_counts, gaussian_kernel, 'same');
        else
            trial_counts_smoothed = trial_counts;
        end
        if use_z_score
            spks_per_trial_tc(tr,:) = zscore(trial_counts_smoothed);
        else
            spks_per_trial_tc(tr,:) = trial_counts_smoothed;
        end
    end

    nd.patient_id = patient_id;
    nd.unit_id = unit_id;
    nd.time_field = time_field;
    nd.firing_rates = spks_per_trial_tc;
    nd.trial_correctness = correct_vector(:);
    nd.brain_region = brain_region;
    nd.trial_imageIDs = trial_imageIDs;
    nd.trial_load = trial_load;
    nd.trial_RT = trial_RT(:);
    nd.trial_probe_in_out = probe_in_out(:);
    neural_data(end+1) = nd; %#ok<AGROW>
end
end
