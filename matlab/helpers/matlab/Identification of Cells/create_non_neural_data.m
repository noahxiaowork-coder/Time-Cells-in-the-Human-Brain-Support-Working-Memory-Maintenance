function neural_data = create_non_neural_data(nwbAll, all_units, time_cell_info_file, bin_width_analysis)
% Non-time-cell units processing

data = load(time_cell_info_file, 'time_cell_info');
time_cell_info = data.time_cell_info;

time_cell_units = time_cell_info(:, 1:2);
all_unit_ids = [[all_units.subject_id]', [all_units.unit_id]'];
is_time_cell = ismember(all_unit_ids, time_cell_units, 'rows');
non_time_cell_units = all_units(~is_time_cell);
aboveRate = ones(length(non_time_cell_units), 1);

    for i = 1:length(non_time_cell_units)
        globalRate = length(non_time_cell_units(i).spike_times) / ...
            (max(non_time_cell_units(i).spike_times) - min(non_time_cell_units(i).spike_times));
        if globalRate < rateFilter
            aboveRate(i) = 0;
        end
    end

non_time_cell_units = non_time_cell_units(logical(aboveRate));

% Output struct
neural_data = struct('patient_id', {}, 'unit_id', {}, 'time_field', {}, ...
                     'firing_rates', {}, 'trial_correctness', {}, ...
                     'brain_region', {}, 'trial_imageIDs', {}, ...
                     'trial_load', {}, 'trial_RT', {}, ...
                     'trial_probe_in_out', {});

% Binning/smoothing kernel
total_bins_analysis = 2.5 / bin_width_analysis;
gaussian_kernel = GaussianKernal(0.3 / bin_width_analysis, 1.5);

for idx = 1:length(non_time_cell_units)
    SU = non_time_cell_units(idx);
    patient_id = SU.subject_id;
    unit_id = SU.unit_id;
    time_field = NaN;

    tsMaint = nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Maintenance').data.load();
    response_accuracy = nwbAll{SU.session_count}.intervals_trials.vectordata.get('response_accuracy').data.load();
    tsProbe = nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Probe').data.load();
    tsResp  = nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Response').data.load();
    trial_RT = tsResp - tsProbe;
    vd = nwbAll{SU.session_count}.intervals_trials.vectordata;
    probe_in_out = vd.get('probe_in_out').data.load();

    ID_Enc1 = vd.get('loadsEnc1_PicIDs').data.load();
    ID_Enc2 = vd.get('loadsEnc2_PicIDs').data.load();
    ID_Enc3 = vd.get('loadsEnc3_PicIDs').data.load();

    stim = struct('tsMaint', num2cell(tsMaint), 'response_accuracy', num2cell(response_accuracy));
    correct_vector = double([stim.response_accuracy] == 1);
    trial_imageIDs = [ID_Enc1(:), ID_Enc2(:), ID_Enc3(:)];
    trial_load = sum(trial_imageIDs ~= 0, 2);

    brain_area = nwbAll{SU.session_count}.general_extracellular_ephys_electrodes.vectordata.get('location').data.load(SU.electrodes);
    brain_region = brain_area{:};

    spks_per_trial = zeros(length(stim), total_bins_analysis);
    for tr = 1:length(stim)
        for b = 1:total_bins_analysis
            bin_start = stim(tr).tsMaint + (b-1)*bin_width_analysis;
            bin_end = bin_start + bin_width_analysis;
            spks_per_trial(tr, b) = sum(SU.spike_times >= bin_start & SU.spike_times < bin_end);
        end
    end

    for tr = 1:size(spks_per_trial, 1)
        counts = spks_per_trial(tr,:);

        counts = conv(counts, gaussian_kernel, 'same');

        counts = zscore(counts);

        spks_per_trial(tr,:) = counts;
    end

    % Store fields per unit
    nd.patient_id = patient_id;
    nd.unit_id = unit_id;
    nd.time_field = time_field;
    nd.firing_rates = spks_per_trial;
    nd.trial_correctness = correct_vector(:);
    nd.brain_region = brain_region;
    nd.trial_imageIDs = trial_imageIDs;
    nd.trial_load = trial_load;
    nd.trial_RT = trial_RT(:);
    nd.trial_probe_in_out = probe_in_out(:);
    neural_data(end+1) = nd;
end
end
