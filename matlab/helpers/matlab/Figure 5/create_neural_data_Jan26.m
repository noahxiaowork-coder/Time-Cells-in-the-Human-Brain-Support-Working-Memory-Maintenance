function neural_data = create_neural_data(nwbAll, all_units, time_cell_info_file, bin_width_analysis, use_z_score, use_smooth)
%CREATENEURAL_DATA Loads time cell info and outputs neural_data structure.
% Each entry of neural_data corresponds to a neuron (identified as a time cell),
% and includes:
%   - patient_id
%   - unit_id
%   - time_field
%   - firing_rates (trials x bins) Z-scored firing rates
%   - trial_correctness (trials x 1) binary vector: 1 = correct, 0 = incorrect
%   - brain_region (string)
%   - trial_imageIDs (trials x 3) matrix of image IDs
%   - trial_load (trials x 1) integer load based on how many image IDs are nonzero

    % Load time_cell_info
    data = load(time_cell_info_file, 'time_cell_info');
    time_cell_info = data.time_cell_info; 
    % time_cell_info: Nx3 matrix [patient_id, unit_id, time_field]

    total_bins_analysis = 2.5 / bin_width_analysis;
    % gaussian_sigma = 2.5;
    % kernel_size = round(gaussian_kernel_size * gaussian_sigma);
    % x = -kernel_size:kernel_size;
    % gaussian_kernel = exp(-(x.^2) / (2 * gaussian_sigma^2));
    % gaussian_kernel = gaussian_kernel / sum(gaussian_kernel);

    gaussian_kernel = GaussianKernal(0.3 / bin_width_analysis, 1.5);

    % gaussian_kernel = 
    % gaussian_sigma_bins = 0.5;  % 50 ms SD
    % gaussian_kernel = GaussianKernal(gaussian_sigma_bins, 5);  % ±5σ

    % 
    % if isfield(params, 'rateFilter') && params.rateFilter > 0
    %     rateFilter = params.rateFilter;
    % else
    %     rateFilter = [];
    % end
    % 
    % % Filter units by global firing rate if required
    % aboveRate = ones(length(all_units), 1);
    % if ~isempty(rateFilter)
    %     for i = 1:length(all_units)
    %         globalRate = length(all_units(i).spike_times) / ...
    %             (max(all_units(i).spike_times) - min(all_units(i).spike_times));
    %         if globalRate < rateFilter
    %             aboveRate(i) = 0;
    %         end
    %     end
    % end
    % all_units = all_units(logical(aboveRate));

    % Initialize output structure
    neural_data = struct('patient_id', {}, 'unit_id', {}, 'time_field', {}, ...
                     'firing_rates', {}, 'trial_correctness', {},     ...
                     'brain_region', {}, 'trial_imageIDs', {},        ...
                     'trial_load', {},  'trial_RT', {}, 'trial_probe_in_out', {});              %  ◄─ new field


    % For each time cell in time_cell_info
    for idx = 1:size(time_cell_info,1)
        patient_id = time_cell_info(idx,1);
        unit_id = time_cell_info(idx,2);
        time_field = time_cell_info(idx,3);

        % Find corresponding unit
        unit_match = ([all_units.subject_id] == patient_id & [all_units.unit_id] == unit_id);
        if ~any(unit_match)
            warning('Unit with patient_id=%d and unit_id=%d not found in all_units.', patient_id, unit_id);
            continue;
        end
        SU = all_units(unit_match);

        % Load maintenance timestamps and response accuracy
        tsMaint = nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Maintenance').data.load();
        response_accuracy = nwbAll{SU.session_count}.intervals_trials.vectordata.get('response_accuracy').data.load();

        % --‑ timestamps for probe onset and the subject’s key‑press -------------
        tsProbe = nwbAll{SU.session_count}.intervals_trials. ...
                         vectordata.get('timestamps_Probe').data.load();          %  (# trials × 1)
        tsResp  = nwbAll{SU.session_count}.intervals_trials. ...
                         vectordata.get('timestamps_Response').data.load();       %  (# trials × 1)

        % --‑ raw reaction time in seconds ---------------------------------------
        trial_RT = tsResp - tsProbe;         % keep NaN if either stamp is missing

        % ► load probe‑membership flag (1 = “in memory set”, 0 = “out”)
        % ── probe‑membership flag  (1 = “in memory set”, 0 = “out”) ─────────────
        vd = nwbAll{SU.session_count}.intervals_trials.vectordata;
        probe_in_out = vd.get('probe_in_out').data.load();      % (# trials × 1)
        % --- option B: if the column is missing, reconstruct on the fly ---------



        % Load image IDs for the three encoding images
        ID_Enc1 = nwbAll{SU.session_count}.intervals_trials.vectordata.get('loadsEnc1_PicIDs').data.load();
        ID_Enc2 = nwbAll{SU.session_count}.intervals_trials.vectordata.get('loadsEnc2_PicIDs').data.load();
        ID_Enc3 = nwbAll{SU.session_count}.intervals_trials.vectordata.get('loadsEnc3_PicIDs').data.load();

        % Construct stim struct with timestamps and response accuracy
        stim = struct('tsMaint', num2cell(tsMaint), 'response_accuracy', num2cell(response_accuracy));

        % Extract correct/incorrect trials as binary vector
        correct_vector = double([stim.response_accuracy] == 1);  % 1 for correct, 0 for incorrect

        % Combine image IDs into one matrix (num_trials x 3)
        trial_imageIDs = [ID_Enc1(:), ID_Enc2(:), ID_Enc3(:)];

        % Compute trial load based on how many image IDs are nonzero per trial
        trial_load = sum(trial_imageIDs ~= 0, 2);

        % Get brain region info
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

        % Smooth and Z-score each trial
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

        % Store into neural_data
        nd.patient_id = patient_id;
        nd.unit_id = unit_id;
        nd.time_field = time_field; 
        nd.firing_rates = spks_per_trial_tc; % [trials x bins]
        nd.trial_correctness = correct_vector(:); % [trials x 1]
        nd.brain_region = brain_region; % brain region string
        nd.trial_imageIDs = trial_imageIDs; % [trials x 3] image IDs
        nd.trial_load = trial_load; % [trials x 1]
        nd.trial_RT        = trial_RT(:);    % [trials × 1] reaction time (s)
        nd.trial_probe_in_out = probe_in_out(:);   % [trials × 1] 1 = IN, 0 = OUT
        neural_data(end+1) = nd; %#ok<AGROW>
    end
end


% function neural_data = create_neural_data(nwbAll, all_units, time_cell_info_file, bin_width_analysis, use_z_score, use_smooth)
% 
%     % === (1) ENFORCE detection grid: 0.1 s bins over 0–2.5 s ===
%     if abs(bin_width_analysis - 0.1) > 1e-12
%         warning('bin_width_analysis=%.3f; forcing to 0.1 s to match detection.', bin_width_analysis);
%         bin_width_analysis = 0.1;
%     end
%     total_bins_analysis = round(2.5 / bin_width_analysis);
%     assert(total_bins_analysis == 25, 'Expected 25 bins (0.1 s × 2.5 s).');
% 
%     % === (2) Kernel: same shape as detection, L1-normalized ===
%     gaussian_kernel = GaussianKernal(0.3 / bin_width_analysis, 1.5); % sigma=0.3 s => 3 bins when bin=0.1
%     gaussian_kernel = gaussian_kernel ./ sum(gaussian_kernel);       % <— normalize
% 
%     % Load time_cell_info: [patient_id, unit_id, time_field] (time_field is 1..25)
%     S = load(time_cell_info_file, 'time_cell_info');
%     time_cell_info = S.time_cell_info;
% 
%     neural_data = struct('patient_id',{},'unit_id',{},'time_field',{}, ...
%                          'firing_rates',{},'trial_correctness',{}, ...
%                          'brain_region',{},'trial_imageIDs',{}, ...
%                          'trial_load',{},'trial_RT',{},'trial_probe_in_out',{}, ...
%                          'bin_width',{},'total_bins',{},'kernel_sigma_bins',{}, ...
%                          'kernel_norm_sum',{},'valid_trial_mask',{});  % <— metadata
% 
%     for idx = 1:size(time_cell_info,1)
%         patient_id = time_cell_info(idx,1);
%         unit_id    = time_cell_info(idx,2);
%         time_field = time_cell_info(idx,3);               % 1..25 on 0.1 s grid
%         assert(1 <= time_field && time_field <= 25, 'time_field out of range.');
% 
%         m = ([all_units.subject_id]==patient_id & [all_units.unit_id]==unit_id);
%         if ~any(m), warning('Unit p=%d u=%d not found; skipping.',patient_id,unit_id); continue; end
%         SU = all_units(m);
% 
%         vd = nwbAll{SU.session_count}.intervals_trials.vectordata;
%         tsMaint = vd.get('timestamps_Maintenance').data.load();
%         respAcc = vd.get('response_accuracy').data.load();
% 
%         % Optional extras
%         tsProbe = vd.get('timestamps_Probe').data.load();
%         tsResp  = vd.get('timestamps_Response').data.load();
%         trial_RT = tsResp - tsProbe;
%         probe_in_out = vd.get('probe_in_out').data.load();
% 
%         % Image IDs & load
%         ID_Enc1 = vd.get('loadsEnc1_PicIDs').data.load();
%         ID_Enc2 = vd.get('loadsEnc2_PicIDs').data.load();
%         ID_Enc3 = vd.get('loadsEnc3_PicIDs').data.load();
%         trial_imageIDs = [ID_Enc1(:), ID_Enc2(:), ID_Enc3(:)];
%         trial_load = sum(trial_imageIDs ~= 0, 2);
% 
% 
%          brain_area = nwbAll{SU.session_count}.general_extracellular_ephys_electrodes.vectordata.get('location').data.load(SU.electrodes);
%          brain_region = brain_area{:};
% 
% 
%         % === (3) VALID TRIAL MASK identical to detection ===
%         valid = isfinite(tsMaint) & tsMaint > 0;
%         tsMaint = tsMaint(valid);
%         respAcc = respAcc(valid);
%         trial_RT = trial_RT(valid);
%         probe_in_out = probe_in_out(valid);
%         trial_imageIDs = trial_imageIDs(valid,:);
%         trial_load = trial_load(valid);
% 
%         correct_vector = double(respAcc == 1);
% 
%         % === Bin, smooth, (optionally) z-score — detection-consistent ===
%         duration = 2.5;
%         nTrials = numel(tsMaint);
%         spks_per_trial_tc = zeros(nTrials, total_bins_analysis);
% 
%         for tr = 1:nTrials
%             % edges per detection grid
%             t0 = tsMaint(tr);
%             for b = 1:total_bins_analysis
%                 bin_start = t0 + (b-1)*bin_width_analysis;
%                 bin_end   = bin_start + bin_width_analysis;
%                 spks_per_trial_tc(tr,b) = sum(SU.spike_times >= bin_start & SU.spike_times < bin_end);
%             end
%         end
% 
%         % Smooth (L1-normalized kernel), then row-wise z-score with zero-variance guard
%         for tr = 1:nTrials
%             row = spks_per_trial_tc(tr,:);
%             if use_smooth, row = conv(row, gaussian_kernel, 'same'); end
%             if use_z_score
%                 mu = mean(row); sd = std(row);
%                 if ~isfinite(sd) || sd==0, sd = 1; end
%                 row = (row - mu) / sd;
%             end
%             spks_per_trial_tc(tr,:) = row;
%         end
% 
%         % === (5) Save with metadata so downstream code can assert identity ===
%         nd.patient_id          = patient_id;
%         nd.unit_id             = unit_id;
%         nd.time_field          = time_field;                    % index 1..25
%         nd.firing_rates        = spks_per_trial_tc;             % trials × bins (z-scored if use_z_score)
%         nd.trial_correctness   = correct_vector(:);
%         nd.brain_region        = brain_region;
%         nd.trial_imageIDs      = trial_imageIDs;
%         nd.trial_load          = trial_load;
%         nd.trial_RT            = trial_RT(:);
%         nd.trial_probe_in_out  = probe_in_out(:);
% 
%         % metadata for reproducibility
%         nd.bin_width           = bin_width_analysis;            % should be 0.1
%         nd.total_bins          = total_bins_analysis;           % should be 25
%         nd.kernel_sigma_bins   = 0.3 / bin_width_analysis;      % == 3 when bin=0.1
%         nd.kernel_norm_sum     = sum(gaussian_kernel);          % should be 1
%         nd.valid_trial_mask    = valid(:);                      % w.r.t. original trial list
% 
%         neural_data(end+1) = nd; %#ok<AGROW>
%     end
% end
