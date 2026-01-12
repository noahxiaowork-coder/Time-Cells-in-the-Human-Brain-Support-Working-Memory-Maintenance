function neural_data = create_concept_neural_data(nwbAll, all_units, bin_width_analysis, use_z_score)
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
    rng(42)
    alphaLim = 0.05;
    total_bins_analysis = 2.5 / bin_width_analysis;
    % gaussian_sigma = 2;
    % kernel_size = round(5 * gaussian_sigma);
    % x = -kernel_size:kernel_size;
    % gaussian_kernel = exp(-(x.^2) / (2 * gaussian_sigma^2));
    % gaussian_kernel = gaussian_kernel / sum(gaussian_kernel);

    gaussian_kernel = GaussianKernal(0.3 / bin_width_analysis, 1.5);
    % Filter units by global firing rate if required
    aboveRate = ones(length(all_units), 1);


    % Initialize output structure
    neural_data = struct('patient_id', {}, 'unit_id', {}, ...
                         'firing_rates', {}, 'trial_correctness', {}, 'brain_region', {}, ...
                         'trial_load', {}, 'trial_imageIDs', {}, 'preferred_image', {});

    % For each time cell in time_cell_info
    for i = 1:length(all_units)
        SU           = all_units(i);
        subject_id   = SU.subject_id;
        cellID       = SU.unit_id;

        fprintf('Processing: (%d/%d) Session SBID %d, Unit %d ',i,length(all_units),subject_id,cellID)
        
        % Loading stim timestamps and loads
        tsFix = num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_FixationCross').data.load());
        tsEnc1 = num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Encoding1').data.load());
        tsEnc2 = num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Encoding2').data.load());
        tsEnc3 = num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Encoding3').data.load());
        tsMaint = num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Maintenance').data.load());
        tsProbe = num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Probe').data.load());
        ID_Enc1 = num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get('loadsEnc1_PicIDs').data.load());
        ID_Enc2 = num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get('loadsEnc2_PicIDs').data.load());
        ID_Enc3 = num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get('loadsEnc3_PicIDs').data.load());
        ID_Probe = num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get('loadsProbe_PicIDs').data.load());
    
        stim = cell2struct(...
            horzcat(tsFix,tsEnc1,tsEnc2,tsEnc3,tsMaint, tsProbe,...
            ID_Enc1,ID_Enc2,ID_Enc3,ID_Probe),...
            {'tsFix','tsEnc1','tsEnc2','tsEnc3','tsMaint','tsProbe',...
            'idEnc1','idEnc2','idEnc3','idProbe'},2);
        
        [idUnique, ~, ic] = unique([stim.idEnc1]);    
        uniqueCounts = histcounts(ic,'BinMethod','integers');
        
        HzEnc1 = NaN(length(stim),1);
        signalDelay = 0.2; % Delay of stimulus onset to effect. 
        stimOffset = 1; % Time past stimulus onset. End of picture presentation.
        for k = 1:length(HzEnc1)
            periodFilter = (SU.spike_times>(stim(k).tsEnc1+signalDelay)) & (SU.spike_times<(stim(k).tsEnc1+stimOffset));
            singleTrialSpikes = SU.spike_times(periodFilter);
            trialRate = length(singleTrialSpikes)/(stimOffset-signalDelay); % Firing rate across testing period.
            HzEnc1(k) = trialRate;
        end
        
        % Finding image with maximum response. Used in stage 2 test. 
        nUniqueStim = length(unique([stim.idEnc1]));
        mResp=nan(nUniqueStim,1);
        for k=1:nUniqueStim
            mResp(k) = mean(HzEnc1([stim.idEnc1]==k));
        end
        idMaxHz = find(mResp==max(mResp),1); %Preferred Image ID 
        id_Trial_maxOnly = [stim.idEnc1]; 
        id_Trial_maxOnly([stim.idEnc1]~=idMaxHz) = -1;

        
        %% Significance Tests: Concept Cells
        % Count spikes in 200-1000ms window following stimulus onset
        % following the first encoding period. Use a 1-way ANOVA followed by a
        % t-test of the maximal response versus the non-selective responses.
  
        % Note that using a 1-way anova with two groups simplifies to a t-test
        Is_concept = false;
        p_ANOVA = 1; p_bANOVA = 1;%#ok<NASGU> % Preset as '1' to allow for paramsSB.plotAlways.
        p_ANOVA = anovan(HzEnc1,{string([stim.idEnc1])}, 'display','off','model', 'linear','alpha',alphaLim,'varnames','picID');
        if p_ANOVA < alphaLim % First test: 1-way ANOVA
            p_bANOVA = anovan(HzEnc1,{string(id_Trial_maxOnly)}, 'display','off','model', 'linear','alpha',alphaLim,'varnames','picID');
            if p_bANOVA < alphaLim % Second Test: Binarized 1-way ANOVA (simplifies to t-test)
                
            fprintf('| Concept -> SID %d, Unit %d p1:%.2f p2:%.2f',SU.subject_id,SU.unit_id,p_ANOVA,p_bANOVA)
            Is_concept = true;
            end
        end
        
        if Is_concept


        % ---------------------------------------------------------------
        %  extra test: preferred-image time-locking in load-1 trials
        % ---------------------------------------------------------------
        prefParams.binWidth = bin_width_analysis;   % reuse the global var
        prefParams.alpha    = alphaLim;
        
        [isTimeLocked, tField, tlStats] = detectPreferredTime( ...
                SU, stim, idMaxHz, bin_width_analysis, alphaLim, ...
                'imgWin',[signalDelay stimOffset], ...   % 0.2-1.0 s by default
                'minTrials', 8, ...
                'gaussSig', 3, 'gaussAmp', 1.5, ...
                'nPerm', 1000);
        
        if isTimeLocked
            nd.cell_type        = 'concept_timeLocked';
            nd.time_field_enc1  = tField;
            nd.tl_stats         = tlStats;   % keep full stats for QC
        else
            nd.cell_type        = 'concept';
        end


           % Load maintenance timestamps and response accuracy
        tsMaint = nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Maintenance').data.load();
        response_accuracy = nwbAll{SU.session_count}.intervals_trials.vectordata.get('response_accuracy').data.load();

        % Construct stim struct with timestamps and response accuracy
        stim = struct('tsMaint', num2cell(tsMaint), 'response_accuracy', num2cell(response_accuracy));
        
        ID_Enc1 = nwbAll{SU.session_count}.intervals_trials.vectordata.get('loadsEnc1_PicIDs').data.load();
        ID_Enc2 = nwbAll{SU.session_count}.intervals_trials.vectordata.get('loadsEnc2_PicIDs').data.load();
        ID_Enc3 = nwbAll{SU.session_count}.intervals_trials.vectordata.get('loadsEnc3_PicIDs').data.load();
        
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

        % ---------- Smooth each trial first --------------------------------------
        for tr = 1:size(spks_per_trial_tc,1)
            % 1‑D convolution only along the time‑bin dimension
            spks_per_trial_tc(tr,:) = conv( ...
                spks_per_trial_tc(tr,:), gaussian_kernel, 'same');
        end
        

        % ---------- Z‑score using grand mean & SE --------------------------------
        if use_z_score
            % Grand mean across *all* trials & bins
            grandMean = mean(spks_per_trial_tc(:));
        
            % Standard error of the mean (SE)
            Nobs      = numel(spks_per_trial_tc);           % trials × bins
            grandStd  = std(spks_per_trial_tc(:));          % SD across all values
            grandSE   = grandStd / sqrt(Nobs);              % SE = SD / √N
        
            % Guard against division‑by‑zero (e.g., perfectly constant matrix)
            if grandSE == 0
                warning('SE is zero for unit %d – skipping z‑scoring.', cellID);
            else
                spks_per_trial_tc = (spks_per_trial_tc - grandMean) / grandSE;
            end
        end


        % Store into neural_data
        nd.patient_id = subject_id;
        nd.unit_id = cellID;
        nd.firing_rates = spks_per_trial_tc; % [trials x bins]
        nd.trial_correctness = correct_vector(:); % [trials x 1]
        nd.brain_region = brain_region; % brain region string
        nd.trial_load = trial_load; % [trials x 1]
        nd.trial_imageIDs = trial_imageIDs; % [trials x 3] image IDs
        nd.preferred_image = idMaxHz;
        neural_data(end+1) = nd; %#ok<AGROW>
        end 
       
    end

end

function [tf, timeField, stats] = detectPreferredTime( ...
            SU, stim, preferredID, binWidth, alphaLim, varargin)
%D E T E C T P R E F E R R E D T I M E
%  Returns
%    tf         – logical, true if unit passes the time-lock test
%    timeField  – peak bin (1-based) of the averaged, z-scored PSTH
%    stats      – struct with p-value, nTrials, etc.

% ---------------------------------------------------------------------
% Optional params
p = inputParser;
p.addParameter('imgWin',   [0 1]);   % image window [s] relative to tsEnc1
p.addParameter('minTrials', 10);     % skip if < this number of trials
p.addParameter('gaussSig',  3);      % kernel σ in #bins
p.addParameter('gaussAmp',  1.5);
p.addParameter('nPerm',     1e3);    % permutations
p.parse(varargin{:});
P = p.Results;

imgWinDur       = diff(P.imgWin);          % usually 1 s
totalBins       = imgWinDur / binWidth;    % e.g. 10 for 0.1 s bins
gaussianKernel  = GaussianKernal(P.gaussSig, P.gaussAmp);

% ---------------------------------------------------------------------
% -------- 1)  pick trials: load 1 **and** preferred image ------------
trialMask  = ([stim.trial_load]' == 1) & ([stim.idEnc1]' == preferredID);
if nnz(trialMask) < P.minTrials
    [tf,timeField,stats] = deal(false,NaN,struct('reason','tooFewTrials'));
    return
end
trialIdx   = find(trialMask);
nTrials    = numel(trialIdx);

% -------- 2)  bin spikes per trial within imgWin ---------------------
spksMat = zeros(nTrials, totalBins);
for k = 1:nTrials
    tr   = trialIdx(k);
    t0   = stim(tr).tsEnc1 + P.imgWin(1);
    for b = 1:totalBins
        binStart           = t0 + (b-1)*binWidth;
        binEnd             = binStart + binWidth;
        spksMat(k,b)       = sum(SU.spike_times >= binStart & SU.spike_times < binEnd);
    end
    % smooth + z-score this trial
    spksMat(k,:) = zscore(conv(spksMat(k,:), gaussianKernel, 'same'));
end

meanPSTH  = mean(spksMat,1);
[~,timeField] = max(meanPSTH);
observed  = meanPSTH(timeField);

% -------- 3) permutation: circularly shift each trial -----------------
permMax   = zeros(P.nPerm,1);
edges     = [0:totalBins]*binWidth;   %#ok<NBRAK>
for perm = 1:P.nPerm
    permMat = zeros(size(spksMat));
    for k = 1:nTrials
        shift   = randi(totalBins)-1;             % 0 … totalBins-1
        permMat(k,:) = circshift(spksMat(k,:), shift, 2);
    end
    permMax(perm) = max(mean(permMat,1));
end
pVal = mean(permMax >= observed);

tf    = pVal < alphaLim;
stats = struct('p',pVal,'nTrials',nTrials,'obs',observed,'permMax',permMax);
end
