function [neural_data, time_cell_info] = NWB_calcSelective_SB_Load1ImageSpecificmay11_Null( ...
        nwbAll, all_units, bin_width_analysis, preprocess, useSpikeInPeak, nullShuffle, varargin)
%BUILDNEURALDATA_LOAD1IMAGESPECIFIC_STRICT  Detect load-1 image-specific time cells.
%
% INPUTS
%   nwbAll              – {1×S} NWB session objects
%   all_units           – struct array with fields: subject_id, unit_id,
%                         session_count, spike_times (and optionally electrodes)
%   bin_width_analysis  – scalar (s), e.g. 0.100 for 100 ms bins
%   preprocess          – logical; output firing rates are smoothed if true
%   useSpikeInPeak      – logical; apply spike-in-peak consistency gates if true
%   nullShuffle         – logical; if true, build a null by per-trial circular shifts
%                         within the maintenance window and scramble load-1 image labels
%
% PARAMS (name/value, optional; defaults shown)
%   'Alpha'                 , 0.05
%   'NumPermutations'       , 1000
%   'MaintainDuration'      , 2.5           % seconds
%   'MinTrialsPerImage'     , 7
%   'SpikeWinHalfWidth'     , 1             % half-width (bins) for consistency gate
%   'FracTrialsRequired'    , 0.7           % fraction of trials with ≥1 spike in peak window
%   'MinTrialsRequired'     , 5             % minimum absolute trial count in peak window
%   'RngSeed'               , 42
%   'UseImageSelectivity'   , false         % ⬅ toggle additional selectivity gate
%   'ImageSelectivityMin'   , 0.4           % z-units threshold for selectivity (single bin)
%
% OUTPUTS
%   neural_data     – struct array (one per detected time cell)
%   time_cell_info  – numeric matrix [patient, unit, preferredImg, peakBin]
%
% Notes:
% * Detection uses smoothed + per-trial z-scored traces (`frAvg`) and a max-stat permutation test.
% * Spike-in-peak gate uses RAW counts within ± SpikeWinHalfWidth (in **bins**) around the peak.
% * Optional image selectivity gate (single 100 ms bin) is applied only if exactly one image is significant.
% * Null mode (nullShuffle=true) perturbs detection only (shifts spikes within maintenance; shuffles labels).
% -------------------------------------------------------------------------

if nargin < 6 || isempty(nullShuffle), nullShuffle = false; end

% ---------------------------- Parameters ---------------------------------
p = inputParser;
p.addParameter('Alpha',               0.05);
p.addParameter('NumPermutations',     1000);
p.addParameter('MaintainDuration',    2.5);
p.addParameter('MinTrialsPerImage',   8);
p.addParameter('SpikeWinHalfWidth',   1);
p.addParameter('FracTrialsRequired',  0.5);
p.addParameter('MinTrialsRequired',   3);
p.addParameter('RngSeed',             42);
p.addParameter('UseImageSelectivity', false);   % NEW: toggle
p.addParameter('ImageSelectivityMin', 0.5);     % NEW: z-units (single-bin)
p.parse(varargin{:});
prm = p.Results;
rng(prm.RngSeed);

alphaLim         = prm.Alpha;
num_permutations = prm.NumPermutations;
total_bins       = round(prm.MaintainDuration / bin_width_analysis);
min_trial_count  = prm.MinTrialsPerImage;

% Smoothing kernel used during detection (assumes user-defined GaussianKernal)
gKer = GaussianKernal(3, 1.5);

spikeWinHalfWidth = prm.SpikeWinHalfWidth;
num_units = numel(all_units);

% -------------------------- Time-cell search -----------------------------
time_cell_info = [];

for iU = 1:num_units
    SU = all_units(iU);

    fprintf('Processing unit %d/%d  [Subject %d  Unit %d]\n', ...
            iU, num_units, SU.subject_id, SU.unit_id);

    % ---- Fetch trial metadata -----------------------------------------
    sess = nwbAll{SU.session_count};
    enc1 = sess.intervals_trials.vectordata.get('loadsEnc1_PicIDs').data.load();
    enc2 = sess.intervals_trials.vectordata.get('loadsEnc2_PicIDs').data.load();
    enc3 = sess.intervals_trials.vectordata.get('loadsEnc3_PicIDs').data.load();
    tsM  = sess.intervals_trials.vectordata.get('timestamps_Maintenance').data.load();

    % Keep single-load trials (enc1 only)
    singleMask = (enc1 > 0) & (enc2 == 0) & (enc3 == 0);
    if ~any(singleMask),  continue;  end

    imgIDs   = enc1(singleMask);               % load-1 image IDs (subset)
    tStruct  = struct('tsMaint', num2cell(tsM(singleMask)));
    spikeTimes_det = SU.spike_times;           % detection spikes (possibly shuffled)

    % ---- Null shuffle (optional) --------------------------------------
    % Diagnostics: how much did NullMode change?
% ---- Null shuffle (optional) --------------------------------------
% ---- Null shuffle (optional) --------------------------------------
if nullShuffle
    T = total_bins * bin_width_analysis;          % maintenance duration (s)

    % ===== 1) Continuous circular shifts for ALL trials (loads 1–3) =====
    tsM_all = sess.intervals_trials.vectordata.get('timestamps_Maintenance').data.load();
    numTrialsAll = numel(tsM_all);

    totalSpikesInWin = 0; movedSpikes = 0; shifts_applied = zeros(numTrialsAll,1);
    for tr = 1:numTrialsAll
        t0 = tsM_all(tr); t1 = t0 + T;
        inWin = (spikeTimes_det >= t0) & (spikeTimes_det < t1);

        rel = spikeTimes_det(inWin) - t0;
        shift = rand() * T;                        % ⬅ continuous shift in [0,T)
        shifts_applied(tr) = shift;

        nSpk = numel(rel);
        totalSpikesInWin = totalSpikesInWin + nSpk;
        if shift > eps, movedSpikes = movedSpikes + nSpk; end

        spikeTimes_det(inWin) = mod(rel + shift, T) + t0;
    end

    % ===== 2) Shuffle image labels within each load separately =====
    % Load these once (you already did above, but we reuse local copies)
    % enc1, enc2, enc3 are already in workspace from earlier loads.
    % Define load masks
    maskL1 = (enc1 > 0) & (enc2 == 0) & (enc3 == 0);
    maskL2 = (enc1 > 0) & (enc2 > 0) & (enc3 == 0);
    maskL3 = (enc1 > 0) & (enc2 > 0) & (enc3 > 0);

    % Shuffle enc columns independently within each load
    if any(maskL1)
        tmp = enc1(maskL1); enc1(maskL1) = tmp(randperm(numel(tmp)));
    end
    if any(maskL2)
        tmp = enc1(maskL2); enc1(maskL2) = tmp(randperm(numel(tmp)));
        tmp = enc2(maskL2); enc2(maskL2) = tmp(randperm(numel(tmp)));
    end
    if any(maskL3)
        tmp = enc1(maskL3); enc1(maskL3) = tmp(randperm(numel(tmp)));
        tmp = enc2(maskL3); enc2(maskL3) = tmp(randperm(numel(tmp)));
        tmp = enc3(maskL3); enc3(maskL3) = tmp(randperm(numel(tmp)));
    end

    % ===== 3) Recompute single-load labels for detection =====
    % (singleMask was computed earlier; reuse it)
    imgIDs = enc1(singleMask);   % new load-1 labels after shuffling

    % ===== 4) Diagnostics (optional) =====
    movedFrac = (totalSpikesInWin>0) * (movedSpikes / max(1,totalSpikesInWin));
    fprintf('  NullMode diag: movedFrac≈%.2f, meanShift=%.3fs, L1=%d L2=%d L3=%d\n', ...
            movedFrac, mean(shifts_applied), nnz(maskL1), nnz(maskL2), nnz(maskL3));
end




    uImgs    = unique(imgIDs);
    nImgs    = numel(uImgs);
    
    is_locked    = false(1,nImgs);
    peakBin      = zeros(1,nImgs);
    frAvg_store  = cell(1,nImgs);
    
    % NEW: bookkeeping for validity and p-values
    valid_img       = false(1,nImgs);     % image considered in multiplicity
    pvals_per_img   = inf(1,nImgs);       % store raw permutation p-values


    is_locked    = false(1,nImgs);
    peakBin      = zeros(1,nImgs);
    frAvg_store  = cell(1,nImgs);   % per-image frAvg (z-scored, smoothed)

    % ---- Loop over images ---------------------------------------------
    for k = 1:nImgs
        thisImg     = uImgs(k);
        trialMask   = (imgIDs == thisImg);
        trials_here = tStruct(trialMask);
        nTr         = numel(trials_here);
        if nTr < min_trial_count,  continue;  end   % trial-count filter
    
        % ---- Detection matrices (z for stats, raw for spike checks)
        [frAvg, spkMatZ, spkMatRaw] = binGaussZscore(spikeTimes_det, trials_here, ...
                                        bin_width_analysis, total_bins, gKer); %#ok<ASGLU>
    
        % ---- Exclude image groups with NO spikes in ANY trial
        trialHasAnySpike = sum(spkMatRaw, 2) > 0;   % per-trial total within maintenance
        if ~any(trialHasAnySpike)
            continue                                  % not valid for multiplicity
        end
        valid_img(k) = true;
    
        % ---- Peak & cache for later steps
        [~, pkBin]    = max(frAvg);
        frAvg_store{k} = frAvg;
        peakBin(k)     = pkBin;
    
        % ---- Permutation significance on max(frAvg)
        pval = permTest_timeLocked(frAvg, spikeTimes_det, trials_here, ...
                                   bin_width_analysis, total_bins, ...
                                   num_permutations, gKer);
        pvals_per_img(k) = pval;     % NEW: keep the raw p-value
    
        if pval >= alphaLim
            continue
        end
    
        % ---- Spike-in-peak consistency (raw counts)
        if useSpikeInPeak
            winL = max(1, pkBin - spikeWinHalfWidth);
            winR = min(total_bins, pkBin + spikeWinHalfWidth);
            trialsInside = any(spkMatRaw(:,winL:winR) > 0, 2);
            needTrials = max(prm.MinTrialsRequired, ceil(prm.FracTrialsRequired * nTr));
            if sum(trialsInside) < needTrials
                continue
            end
        end
    
        is_locked(k) = true;
    end


    % ---- Bonferroni over REMAINING valid images
nRemain = sum(valid_img);
if nRemain == 0
    continue  % no evaluable images for this unit
end
pvals_corr = min(1, pvals_per_img(valid_img) * nRemain);  % Bonferroni
% apply corrected threshold: only images with corrected p<alpha can be "locked"
tmp_locked = false(1, nRemain);
tmp_locked(:) = is_locked(valid_img) & (pvals_corr < alphaLim);

% rewrite is_locked to reflect corrected decisions
is_locked(:) = false;
is_locked(valid_img) = tmp_locked;

% ---- Global gate: exactly one significant image (+ optional selectivity)
sigImgs = find(is_locked);
if isscalar(sigImgs)
    prefIdx = sigImgs;

    if prm.UseImageSelectivity
        % Single-bin (100 ms) selectivity at each image's own peak
        pkSig = peakBin(prefIdx);
        if pkSig <= 0 || isempty(frAvg_store{prefIdx}), continue; end
        sigVal = frAvg_store{prefIdx}(pkSig);

        otherIdx  = setdiff(1:nImgs, prefIdx);
        otherVals = nan(1, numel(otherIdx));
        for ii = 1:numel(otherIdx)
            jj  = otherIdx(ii);
            pkB = peakBin(jj);
            if pkB > 0 && ~isempty(frAvg_store{jj})
                otherVals(ii) = frAvg_store{jj}(pkB);
            end
        end
        otherVals = otherVals(~isnan(otherVals));
        if isempty(otherVals), continue; end

        imgSelectivity = sigVal - mean(otherVals);
        if imgSelectivity <= prm.ImageSelectivityMin
            continue
        end
    end

    % Accept unit
    time_cell_info = [time_cell_info; ...
        SU.subject_id, SU.unit_id, uImgs(prefIdx), peakBin(prefIdx)];
end


    % ---- Bonferroni over REMAINING valid images
    nRemain = sum(valid_img);
    if nRemain == 0
        continue  % no evaluable images for this unit
    end
    pvals_corr = min(1, pvals_per_img(valid_img) * nRemain);  % Bonferroni
    % apply corrected threshold: only images with corrected p<alpha can be "locked"
    tmp_locked = false(1, nRemain);
    tmp_locked(:) = is_locked(valid_img) & (pvals_corr < alphaLim);
    
    % rewrite is_locked to reflect corrected decisions
    is_locked(:) = false;
    is_locked(valid_img) = tmp_locked;
    
    % ---- Global gate: exactly one significant image (+ optional selectivity)
    sigImgs = find(is_locked);
    if isscalar(sigImgs)
        prefIdx = sigImgs;
    
        if prm.UseImageSelectivity
            % Single-bin (100 ms) selectivity at each image's own peak
            pkSig = peakBin(prefIdx);
            if pkSig <= 0 || isempty(frAvg_store{prefIdx}), continue; end
            sigVal = frAvg_store{prefIdx}(pkSig);
    
            otherIdx  = setdiff(1:nImgs, prefIdx);
            otherVals = nan(1, numel(otherIdx));
            for ii = 1:numel(otherIdx)
                jj  = otherIdx(ii);
                pkB = peakBin(jj);
                if pkB > 0 && ~isempty(frAvg_store{jj})
                    otherVals(ii) = frAvg_store{jj}(pkB);
                end
            end
            otherVals = otherVals(~isnan(otherVals));
            if isempty(otherVals), continue; end
    
            imgSelectivity = sigVal - mean(otherVals);
            if imgSelectivity <= prm.ImageSelectivityMin
                continue
            end
        end
    
        % Accept unit
        time_cell_info = [time_cell_info; ...
            SU.subject_id, SU.unit_id, uImgs(prefIdx), peakBin(prefIdx)];
    end

end

fprintf('Detected %d load-1 image-specific time cells.\n', size(time_cell_info,1));

% -------------------- Build neural_data struct --------------------------
neural_data = struct('patient_id',{},'unit_id',{},'preferred_image',{}, ...
                     'time_field',{},'firing_rates',{},'trial_correctness',{}, ...
                     'brain_region',{},'trial_imageIDs',{},'trial_load',{}, ...
                     'trial_RT',{},'trial_probe_in_out',{});

for r = 1:size(time_cell_info,1)
    pid   = time_cell_info(r,1);
    uid   = time_cell_info(r,2);
    pref  = time_cell_info(r,3);
    tfbin = time_cell_info(r,4);

    SUidx = find([all_units.subject_id]==pid & [all_units.unit_id]==uid, 1);
    if isempty(SUidx)
        warning('Unit %d/%d missing after filtering.', pid, uid);
        continue
    end
    SU = all_units(SUidx);

    % ---- Pull trial-level info ---------------------------------------
    sess = nwbAll{SU.session_count};
    tsM  = sess.intervals_trials.vectordata.get('timestamps_Maintenance').data.load();
    resp = sess.intervals_trials.vectordata.get('response_accuracy').data.load();
    ID1  = sess.intervals_trials.vectordata.get('loadsEnc1_PicIDs').data.load();
    ID2  = sess.intervals_trials.vectordata.get('loadsEnc2_PicIDs').data.load();
    ID3  = sess.intervals_trials.vectordata.get('loadsEnc3_PicIDs').data.load();

    correctVec = double(resp == 1);
    trialImgs  = [ID1(:), ID2(:), ID3(:)];
    trialLoad  = sum(trialImgs ~= 0, 2);

    % ---- Bin spikes for output (session-wide) -------------------------
    nT  = numel(tsM);
    spkM = zeros(nT, total_bins);
    for tr = 1:nT
        for b = 1:total_bins
            t0 = tsM(tr) + (b-1)*bin_width_analysis;
            t1 = t0 + bin_width_analysis;
            spkM(tr,b) = sum(SU.spike_times >= t0 & SU.spike_times < t1);
        end
    end

    % ---- Optional preprocessing for output (smoothed, no z-score) ----
    if preprocess
        for tr = 1:nT
            spkM(tr,:) = conv(spkM(tr,:), gKer, 'same');
        end
    end

    % ---- Brain region (if available) ---------------------------------
    try
        regStr = sess.general_extracellular_ephys_electrodes ...
                     .vectordata.get('location').data.load(SU.electrodes);
        brainRegion = regStr{:};
    catch
        brainRegion = 'unknown';
    end

    % ---- Reaction time & probe in/out --------------------------------
    tsProbe = sess.intervals_trials.vectordata.get('timestamps_Probe').data.load();
    tsResp  = sess.intervals_trials.vectordata.get('timestamps_Response').data.load();
    trialRT = tsResp - tsProbe;

    vd = sess.intervals_trials.vectordata;
    try
        probe_in_out = vd.get('probe_in_out').data.load();
    catch
        probe_in_out = nan(size(tsM));
    end

    % ---- Assemble struct ---------------------------------------------
    nd.patient_id         = pid;
    nd.unit_id            = uid;
    nd.preferred_image    = pref;
    nd.time_field         = tfbin;
    nd.firing_rates       = spkM;
    nd.trial_correctness  = correctVec(:);
    nd.brain_region       = brainRegion;
    nd.trial_imageIDs     = trialImgs;
    nd.trial_load         = trialLoad;
    nd.trial_RT           = trialRT(:);
    nd.trial_probe_in_out = probe_in_out(:);

    neural_data(end+1) = nd; %#ok<AGROW>
end

fprintf('Created neural_data for %d time cells (preprocess=%d).\n', ...
        numel(neural_data), preprocess);

end  % --------------------------- END MAIN -------------------------------


%% ===================== Helper sub-functions ============================

% Bin raw counts + make z-scored smoothed matrices (detection only)
function [frAvg, spkMatZ, spkMatRaw] = binGaussZscore(spikeTimes, trials, binW, totalBins, gKer)
    nT = numel(trials);
    spkMatRaw = zeros(nT,totalBins);
    for k = 1:nT
        t0 = trials(k).tsMaint;
        for b = 1:totalBins
            bs = t0 + (b-1)*binW; be = bs + binW;
            spkMatRaw(k,b) = sum(spikeTimes >= bs & spikeTimes < be);
        end
    end
    spkMatZ = zeros(size(spkMatRaw));
    for k = 1:nT
        spkMatZ(k,:) = zscore_safe(conv(spkMatRaw(k,:), gKer, 'same'));
    end
    frAvg = mean(spkMatZ,1);
end

% Safe per-row z-score (avoid NaNs if std=0)
function Z = zscore_safe(X)
    mu = mean(X,2);
    sd = std(X,0,2);
    sd(sd==0) = 1;           % avoid NaNs; row becomes zeros after centering
    Z = (X - mu) ./ sd;
end

% Permutation test on max(frAvg)
function p = permTest_timeLocked(frAvg, spikeTimes, trials, binW, totalBins, nPerm, gKer)
    obs  = max(frAvg);
    null = zeros(nPerm,1);

    for pIdx = 1:nPerm
        permSpk = [];
        for k = 1:numel(trials)
           tRef = trials(k).tsMaint;
            T    = totalBins * binW;
            trS  = spikeTimes(spikeTimes >= tRef & spikeTimes < tRef+T) - tRef;
            shift = (randi(totalBins) - 1) * binW;
            permTr = mod(trS + shift, T);
            permSpk = [permSpk; permTr + tRef]; %#ok<AGROW>
        end

        permMat = zeros(numel(trials), totalBins);
        for k = 1:numel(trials)
            tRef = trials(k).tsMaint;
            for b = 1:totalBins
                bs = tRef + (b-1)*binW; be = bs + binW;
                permMat(k,b) = sum(permSpk >= bs & permSpk < be);
            end
        end
        for k = 1:size(permMat,1)
            permMat(k,:) = zscore_safe(conv(permMat(k,:), gKer, 'same'));
        end
        null(pIdx) = max(mean(permMat,1));
    end
    p = mean(null >= obs);
end
