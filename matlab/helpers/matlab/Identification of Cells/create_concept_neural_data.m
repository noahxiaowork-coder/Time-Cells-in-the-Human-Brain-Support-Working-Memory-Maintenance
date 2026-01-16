function neural_data = create_concept_neural_data(nwbAll, all_units, bin_width_analysis, use_z_score)
% Builds neural_data for concept cells; includes preferred-image time-lock test.
rng(42)
alphaLim = 0.05;
total_bins_analysis = 2.5 / bin_width_analysis;
gaussian_kernel = GaussianKernal(0.3 / bin_width_analysis, 1.5);  % smoothing kernel
aboveRate = ones(length(all_units), 1); %#ok<NASGU>

neural_data = struct('patient_id', {}, 'unit_id', {}, ...
                     'firing_rates', {}, 'trial_correctness', {}, 'brain_region', {}, ...
                     'trial_load', {}, 'trial_imageIDs', {}, 'preferred_image', {});

for i = 1:length(all_units)
    SU         = all_units(i);
    subject_id = SU.subject_id;
    cellID     = SU.unit_id;

    fprintf('Processing: (%d/%d) Session SBID %d, Unit %d ',i,length(all_units),subject_id,cellID)

    tsFix   = num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_FixationCross').data.load());
    tsEnc1  = num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Encoding1').data.load());
    tsEnc2  = num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Encoding2').data.load());
    tsEnc3  = num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Encoding3').data.load());
    tsMaint = num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Maintenance').data.load());
    tsProbe = num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Probe').data.load());
    ID_Enc1 = num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get('loadsEnc1_PicIDs').data.load());
    ID_Enc2 = num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get('loadsEnc2_PicIDs').data.load());
    ID_Enc3 = num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get('loadsEnc3_PicIDs').data.load());
    ID_Probe= num2cell(nwbAll{SU.session_count}.intervals_trials.vectordata.get('loadsProbe_PicIDs').data.load());

    stim = cell2struct(horzcat(tsFix,tsEnc1,tsEnc2,tsEnc3,tsMaint,tsProbe,ID_Enc1,ID_Enc2,ID_Enc3,ID_Probe), ...
                       {'tsFix','tsEnc1','tsEnc2','tsEnc3','tsMaint','tsProbe','idEnc1','idEnc2','idEnc3','idProbe'},2);

    [~, ~, ic] = unique([stim.idEnc1]);
    HzEnc1 = NaN(length(stim),1);
    signalDelay = 0.2;
    stimOffset  = 1;
    for k = 1:length(HzEnc1)
        periodFilter = (SU.spike_times>(stim(k).tsEnc1+signalDelay)) & (SU.spike_times<(stim(k).tsEnc1+stimOffset));
        singleTrialSpikes = SU.spike_times(periodFilter);
        HzEnc1(k) = length(singleTrialSpikes)/(stimOffset-signalDelay);
    end

    nUniqueStim = length(unique([stim.idEnc1]));
    mResp = nan(nUniqueStim,1);
    for k = 1:nUniqueStim
        mResp(k) = mean(HzEnc1([stim.idEnc1]==k));
    end
    idMaxHz = find(mResp==max(mResp),1);
    id_Trial_maxOnly = [stim.idEnc1];
    id_Trial_maxOnly([stim.idEnc1]~=idMaxHz) = -1;

    Is_concept = false;
    p_ANOVA = anovan(HzEnc1,{string([stim.idEnc1])}, 'display','off','model','linear','alpha',alphaLim,'varnames','picID');
    if p_ANOVA < alphaLim
        p_bANOVA = anovan(HzEnc1,{string(id_Trial_maxOnly)}, 'display','off','model','linear','alpha',alphaLim,'varnames','picID');
        if p_bANOVA < alphaLim
            fprintf('| Concept -> SID %d, Unit %d p1:%.2f p2:%.2f',SU.subject_id,SU.unit_id,p_ANOVA,p_bANOVA)
            Is_concept = true;
        end
    end

    if Is_concept
        % Add trial_load required by detectPreferredTime
        tmpIDs = [[stim.idEnc1]' [stim.idEnc2]' [stim.idEnc3]'];
        tmpLoad = sum(tmpIDs~=0,2);
        for t = 1:numel(stim), stim(t).trial_load = tmpLoad(t); end

        prefParams.binWidth = bin_width_analysis; %#ok<NASGU>
        prefParams.alpha    = alphaLim; %#ok<NASGU>

        [isTimeLocked, tField, tlStats] = detectPreferredTime( ...
            SU, stim, idMaxHz, bin_width_analysis, alphaLim, ...
            'imgWin',[signalDelay stimOffset], 'minTrials', 8, 'gaussSig', 3, 'gaussAmp', 1.5, 'nPerm', 1000);

        if isTimeLocked
            nd.cell_type       = 'concept_timeLocked';
            nd.time_field_enc1 = tField;
            nd.tl_stats        = tlStats;
        else
            nd.cell_type       = 'concept';
        end

        tsMaint2 = nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Maintenance').data.load();
        response_accuracy = nwbAll{SU.session_count}.intervals_trials.vectordata.get('response_accuracy').data.load();
        stim2 = struct('tsMaint', num2cell(tsMaint2), 'response_accuracy', num2cell(response_accuracy));
        ID1 = nwbAll{SU.session_count}.intervals_trials.vectordata.get('loadsEnc1_PicIDs').data.load();
        ID2 = nwbAll{SU.session_count}.intervals_trials.vectordata.get('loadsEnc2_PicIDs').data.load();
        ID3 = nwbAll{SU.session_count}.intervals_trials.vectordata.get('loadsEnc3_PicIDs').data.load();

        correct_vector = double([stim2.response_accuracy] == 1);
        trial_imageIDs = [ID1(:), ID2(:), ID3(:)];
        trial_load = sum(trial_imageIDs ~= 0, 2);

        brain_area = nwbAll{SU.session_count}.general_extracellular_ephys_electrodes.vectordata.get('location').data.load(SU.electrodes);
        brain_region = brain_area{:};

        spks_per_trial_tc = zeros(length(stim2), total_bins_analysis);
        for tr = 1:length(stim2)
            for b = 1:total_bins_analysis
                bin_start = stim2(tr).tsMaint + (b-1)*bin_width_analysis;
                bin_end = bin_start + bin_width_analysis;
                spks_per_trial_tc(tr, b) = sum(SU.spike_times >= bin_start & SU.spike_times < bin_end);
            end
        end

        for tr = 1:size(spks_per_trial_tc,1)
            spks_per_trial_tc(tr,:) = conv(spks_per_trial_tc(tr,:), gaussian_kernel, 'same');
        end

        if use_z_score
            grandMean = mean(spks_per_trial_tc(:));
            Nobs      = numel(spks_per_trial_tc);
            grandStd  = std(spks_per_trial_tc(:));
            grandSE   = grandStd / sqrt(Nobs);
            if grandSE ~= 0
                spks_per_trial_tc = (spks_per_trial_tc - grandMean) / grandSE;
            else
                warning('SE is zero for unit %d – skipping z-scoring.', cellID);
            end
        end

        nd.patient_id = subject_id;
        nd.unit_id = cellID;
        nd.firing_rates = spks_per_trial_tc;
        nd.trial_correctness = correct_vector(:);
        nd.brain_region = brain_region;
        nd.trial_load = trial_load;
        nd.trial_imageIDs = trial_imageIDs;
        nd.preferred_image = idMaxHz;
        neural_data(end+1) = nd; %#ok<AGROW>
    end
end
end

function [tf, timeField, stats] = detectPreferredTime(SU, stim, preferredID, binWidth, alphaLim, varargin)
% Preferred-image time-lock detection.
p = inputParser;
p.addParameter('imgWin',   [0 1]);
p.addParameter('minTrials', 10);
p.addParameter('gaussSig',  3);
p.addParameter('gaussAmp',  1.5);
p.addParameter('nPerm',     1e3);
p.parse(varargin{:});
P = p.Results;

imgWinDur      = diff(P.imgWin);
totalBins      = imgWinDur / binWidth;
gaussianKernel = GaussianKernal(P.gaussSig, P.gaussAmp);

trialMask = ([stim.trial_load]' == 1) & ([stim.idEnc1]' == preferredID);
if nnz(trialMask) < P.minTrials
    [tf,timeField,stats] = deal(false,NaN,struct('reason','tooFewTrials'));
    return
end
trialIdx = find(trialMask);
nTrials  = numel(trialIdx);

spksMat = zeros(nTrials, totalBins);
for k = 1:nTrials
    tr = trialIdx(k);
    t0 = stim(tr).tsEnc1 + P.imgWin(1);
    for b = 1:totalBins
        binStart = t0 + (b-1)*binWidth;
        binEnd   = binStart + binWidth;
        spksMat(k,b) = sum(SU.spike_times >= binStart & SU.spike_times < binEnd);
    end
    spksMat(k,:) = zscore(conv(spksMat(k,:), gaussianKernel, 'same'));
end

meanPSTH = mean(spksMat,1);
[~,timeField] = max(meanPSTH);
observed = meanPSTH(timeField);

permMax = zeros(P.nPerm,1);
for perm = 1:P.nPerm
    permMat = zeros(size(spksMat));
    for k = 1:nTrials
        shift = randi(totalBins)-1;
        permMat(k,:) = circshift(spksMat(k,:), shift, 2);
    end
    permMax(perm) = max(mean(permMat,1));
end
pVal = mean(permMax >= observed);

tf    = pVal < alphaLim;
stats = struct('p',pVal,'nTrials',nTrials,'obs',observed,'permMax',permMax);
end
