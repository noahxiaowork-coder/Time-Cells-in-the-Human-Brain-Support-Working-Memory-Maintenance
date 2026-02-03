function [neural_data, time_cell_info] = NWB_calcSelective_SB_Load1ImageSpecificmay11_80( ...
        nwbAll, all_units, params, bin_width_analysis, preprocess)
%BUILDNEURALDATA_LOAD1IMAGESPECIFIC  Detect load‑1 image‑specific time cells
%using **only a training subset (80 %)** of the single‑load trials. The
%remaining 20 % are held‑out and returned in separate test‑set fields.
%
% INPUTS
%   nwbAll            – {1×S} cell of NWB session objects
%   all_units         – struct array with fields: subject_id, unit_id,
%                       session_count, spike_times  (and optionally electrodes)
%   params            – struct; optional field  .rateFilter (Hz)
%   bin_width_analysis– scalar  (s)  e.g. 0.100 for 100 ms bins
%   preprocess        – logical (default = true)
%                       true  → smooth + z‑score each trial (recommended)
%                       false → keep raw binned spike counts
%
% OUTPUTS
%   neural_data   – struct array, one element per time cell, with **train**
%                   and **test** sub‑fields:
%                       .firing_rates_train / _test
%                       .trial_correctness_train / _test
%                       .trial_load_train / _test
%                       .trial_imageIDs_train / _test
%   time_cell_info– numeric matrix [patient, unit, preferredImg, peakBin]
%
% The detection logic matches the May‑10‑2025 Gaussian‑smoothed pipeline,
% except that significance, spike‑in‑peak, and dominance tests are run
% **only on the training trials (80 %)**.
%
% -------------------------------------------------------------------------
% Author:  <your‑name>   Date: 11‑May‑2025  (split‑set revision)
% -------------------------------------------------------------------------

%% ---------------------------- Parameters --------------------------------
rng(42)                              % deterministic permutations

% Detection thresholds
alphaLim         = 0.05;
num_permutations = 1000;
trainFrac        = 0.80;             % ← NEW: train/test split fraction
min_trial_count  = 7;               % applied **after** split

total_bins       = round(2.5 / bin_width_analysis);   % 25 for 100‑ms bins

% Gaussian smoothing
gaussSigmaBins   = 2.5;   % σ in bins
gaussRadiusFact  = 1;     % truncate at ±2 σ   (k = 2 is your preference)
gKer             = makeGaussKernel(gaussSigmaBins, gaussRadiusFact);

% Spike‑in‑peak rule
spikeWinHalfWidth = 2;    % ±2 bins
useSpikeInPeak    = true;

% Optional firing‑rate filter
rateFilter = [];
if isfield(params,'rateFilter') && params.rateFilter > 0
    rateFilter = params.rateFilter;
end

%% ------------------------- Optional rate filter -------------------------
% if ~isempty(rateFilter)
%     keep = arrayfun(@(u)  numel(u.spike_times)/(max(u.spike_times)-min(u.spike_times)) ...
%                                >= rateFilter, all_units);
%     all_units = all_units(keep);
% end
num_units = numel(all_units);

%% ---------------------- Containers for train/test masks -----------------
unitTrialSplit(num_units) = struct('trainMask',[],'testMask',[]);  % pre‑allocate

%% -------------------------- Time‑cell search ----------------------------
time_cell_info = [];

for iU = 1:num_units
    SU = all_units(iU);
    subject_id = SU.subject_id;
    cellID     = SU.unit_id;

    fprintf('Processing unit %d/%d  [Subject %d  Unit %d]\n', ...
            iU, num_units, subject_id, cellID);

    % ---- Fetch trial metadata -----------------------------------------
    sess   = nwbAll{SU.session_count};
    enc1   = sess.intervals_trials.vectordata.get('loadsEnc1_PicIDs').data.load();
    enc2   = sess.intervals_trials.vectordata.get('loadsEnc2_PicIDs').data.load();
    enc3   = sess.intervals_trials.vectordata.get('loadsEnc3_PicIDs').data.load();
    tsM    = sess.intervals_trials.vectordata.get('timestamps_Maintenance').data.load();

    % Keep single‑load trials (enc1 only)
    singleMask = (enc1 > 0) & (enc2 == 0) & (enc3 == 0);
    if ~any(singleMask)
        unitTrialSplit(iU).trainMask = false(size(tsM));
        unitTrialSplit(iU).testMask  = false(size(tsM));
        continue;
    end

    imgIDs   = enc1(singleMask);
    tStruct  = struct('tsMaint', num2cell(tsM(singleMask)));
    singleIdx = find(singleMask);     % indices into all trials for single‑load
    uImgs    = unique(imgIDs);
    nImgs    = numel(uImgs);

    % Containers for global dominance gate
    is_locked  = false(1,nImgs);
    peakAmp    = -inf(1,nImgs);
    peakBin    = zeros(1,nImgs);

    % Masks for train/test at the *unit* level
    trainMaskUnit = false(size(tsM));
    testMaskUnit  = false(size(tsM));

    % ---- Loop over images ---------------------------------------------
    for k = 1:nImgs
        thisImg   = uImgs(k);
        imgIdxRel = find(imgIDs == thisImg);      % indices *within* singleMask
        imgIdxAbs = singleIdx(imgIdxRel);         % absolute trial indices
        nTr       = numel(imgIdxAbs);

        % Split into train/test ----------------------------------------
        perm     = randperm(nTr);
        nTrain   = max(min_trial_count, floor(trainFrac * nTr));
        if nTr < nTrain
            % not enough trials even before split; skip this image
            continue;
        end
        trainAbsIdx = imgIdxAbs(perm(1:nTrain));
        testAbsIdx  = imgIdxAbs(perm(nTrain+1:end));

        % Update unit‑level masks
        trainMaskUnit(trainAbsIdx) = true;
        testMaskUnit(testAbsIdx)   = true;

        % Build trial struct arrays for train set
        trials_train = tStruct(imgIdxRel(perm(1:nTrain)));

        % ---- Bin, smooth, z‑score per trial ---------------------------
        [frAvg, spkMat] = binGaussZscore(SU.spike_times, trials_train, ...
                                         bin_width_analysis, total_bins, gKer);

        % Store peak for global dominance test
        [pk, pkBin] = max(frAvg);
        peakAmp(k)  = pk;
        peakBin(k)  = pkBin;

        % ---- Permutation significance ---------------------------------
        p = permTest_timeLocked(frAvg, SU.spike_times, trials_train, ...
                                bin_width_analysis, total_bins, ...
                                num_permutations, gKer);

        if p >= alphaLim
            continue;
        end

        % ---- Spike‑in‑peak consistency (optional) ---------------------
        if useSpikeInPeak
            win = max(1,pkBin-spikeWinHalfWidth) : ...
                  min(total_bins,pkBin+spikeWinHalfWidth);
            trialsWithSpike = any(spkMat(:,win) > 0, 2);
            if sum(trialsWithSpike) < max(4, ceil(0.7*nTrain))
                continue;
            end
        end

        % ---- Image passes all within‑image gates ----------------------
        is_locked(k) = true;
    end  % image loop

    % Save train/test masks for later assembly
    unitTrialSplit(iU).trainMask = trainMaskUnit;
    unitTrialSplit(iU).testMask  = testMaskUnit;

    % ---- Global peak‑dominance gate -----------------------------------
    [~, prefIdx] = max(peakAmp);
    if is_locked(prefIdx) && ...
       all(peakAmp(prefIdx) > peakAmp(setdiff(1:nImgs,prefIdx)))

        time_cell_info = [time_cell_info; ...
                          SU.subject_id, SU.unit_id, ...
                          uImgs(prefIdx), peakBin(prefIdx)]; %#ok<AGROW>
    end
end  % unit loop

fprintf('Detected %d load‑1 image‑specific time cells.\n', size(time_cell_info,1));

%% -------------------- Build neural_data struct -------------------------
neural_data = struct('patient_id',{},'unit_id',{},'preferred_image',{}, ...
                     'time_field',{}, ...
                     'firing_rates_train',{},'firing_rates_test',{}, ...
                     'trial_correctness_train',{},'trial_correctness_test',{}, ...
                     'brain_region',{}, ...
                     'trial_imageIDs_train',{},'trial_imageIDs_test',{}, ...
                     'trial_load_train',{},'trial_load_test',{});
for r = 1:size(time_cell_info,1)
    pid   = time_cell_info(r,1);
    uid   = time_cell_info(r,2);
    pref  = time_cell_info(r,3);
    tfbin = time_cell_info(r,4);

    SUidx = find([all_units.subject_id]==pid & [all_units.unit_id]==uid, 1);
    if isempty(SUidx)
        warning('Unit %d/%d missing after rate filter.',pid,uid);
        continue;
    end
    SU = all_units(SUidx);

    % ---- Pull trial‑level info ---------------------------------------
    sess = nwbAll{SU.session_count};
    tsM  = sess.intervals_trials.vectordata.get('timestamps_Maintenance') ...
                .data.load();
    resp = sess.intervals_trials.vectordata.get('response_accuracy') ...
                .data.load();
    ID1  = sess.intervals_trials.vectordata.get('loadsEnc1_PicIDs') ...
                .data.load();
    ID2  = sess.intervals_trials.vectordata.get('loadsEnc2_PicIDs') ...
                .data.load();
    ID3  = sess.intervals_trials.vectordata.get('loadsEnc3_PicIDs') ...
                .data.load();

    correctVec = double(resp == 1);
    trialImgs  = [ID1(:), ID2(:), ID3(:)];
    trialLoad  = sum(trialImgs ~= 0, 2);

    % ---- Bin spikes for **all** trials -------------------------------
    nT  = numel(tsM);
    spkM = zeros(nT, total_bins);
    for tr = 1:nT
        for b = 1:total_bins
            t0 = tsM(tr) + (b-1)*bin_width_analysis;
            t1 = t0 + bin_width_analysis;
            spkM(tr,b) = sum(SU.spike_times >= t0 & SU.spike_times < t1);
        end
    end

    % ---- Optional preprocessing --------------------------------------
    if preprocess
        for tr = 1:nT
            sm = conv(spkM(tr,:), gKer, 'same');
            spkM(tr,:) = zscore(sm);
        end
    end

    % ---- Retrieve this unit's train/test masks -----------------------
    trainMask = unitTrialSplit(SUidx).trainMask;
    testMask  = unitTrialSplit(SUidx).testMask;

    % Sanity: ensure disjoint & cover only single‑load trials -----------
    if any(trainMask & testMask)
        error('Overlap in train/test masks for unit %d.', uid);
    end

    % ---- Brain region (if available) ---------------------------------
    try
        regStr = sess.general_extracellular_ephys_electrodes ...
                     .vectordata.get('location').data.load(SU.electrodes);
        brainRegion = regStr{:};
    catch
        brainRegion = 'unknown';
    end

    % ---- Assemble struct ---------------------------------------------
    nd.patient_id                 = pid;
    nd.unit_id                    = uid;
    nd.preferred_image            = pref;
    nd.time_field                 = tfbin;

    nd.firing_rates_train         = spkM(trainMask,:);
    nd.firing_rates_test          = spkM(testMask,:);
    nd.trial_correctness_train    = correctVec(trainMask);
    nd.trial_correctness_test     = correctVec(testMask);
    nd.trial_load_train           = trialLoad(trainMask);
    nd.trial_load_test            = trialLoad(testMask);
    nd.trial_imageIDs_train       = trialImgs(trainMask,:);
    nd.trial_imageIDs_test        = trialImgs(testMask,:);

    nd.brain_region               = brainRegion;

    neural_data(end+1) = nd; %#ok<AGROW>
end

fprintf('Created neural_data for %d time cells (preprocess=%d).\n', ...
        numel(neural_data), preprocess);

end  % --------------------------- END MAIN -------------------------------


%% ===================== Helper sub‑functions ============================
function gKer = makeGaussKernel(sigmaBins, radiusFactor)
    radius = ceil(radiusFactor * sigmaBins);
    x      = -radius:radius;
    gKer   = exp(-0.5 * (x./sigmaBins).^2);
    gKer   = gKer ./ sum(gKer);
end

% Bin, smooth, z‑score for a set of trials
function [frAvg, spkMat] = binGaussZscore(spikeTimes, trials, ...
                                binW, totalBins, gKer)
    nT     = numel(trials);
    spkMat = zeros(nT, totalBins);

    for k = 1:nT
        t0 = trials(k).tsMaint;
        for b = 1:totalBins
            bs = t0 + (b-1)*binW;
            be = bs + binW;
            spkMat(k,b) = sum(spikeTimes >= bs & spikeTimes < be);
        end
    end
    for k = 1:nT
        spkMat(k,:) = zscore(conv(spkMat(k,:), gKer, 'same'));
    end
    frAvg = mean(spkMat,1);
end

% Permutation test (time‑level)
function p = permTest_timeLocked(frAvg, spikeTimes, trials, ...
                                 binW, totalBins, nPerm, gKer)
    obs  = max(frAvg);
    null = zeros(nPerm,1);

    for pIdx = 1:nPerm
        permSpk = [];
        for k = 1:numel(trials)
            tRef = trials(k).tsMaint;
            T    = totalBins * binW;
            trS  = spikeTimes(spikeTimes >= tRef & spikeTimes < tRef+T) - tRef;
            shift  = (randi(totalBins)-1)*binW;
            permTr = mod(trS + shift, T);
            permSpk = [permSpk; permTr + tRef]; %#ok<AGROW>
        end

        permMat = zeros(numel(trials), totalBins);
        for k = 1:numel(trials)
            tRef = trials(k).tsMaint;
            for b = 1:totalBins
                bs = tRef + (b-1)*binW;
                be = bs + binW;
                permMat(k,b) = sum(permSpk >= bs & permSpk < be);
            end
        end
        for k = 1:size(permMat,1)
            permMat(k,:) = zscore(conv(permMat(k,:), gKer, 'same'));
        end
        null(pIdx) = max(mean(permMat,1));
    end
    p = mean(null >= obs);
end
