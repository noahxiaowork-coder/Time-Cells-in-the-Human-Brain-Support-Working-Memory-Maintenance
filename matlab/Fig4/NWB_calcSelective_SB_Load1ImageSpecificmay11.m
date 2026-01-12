function [neural_data, time_cell_info] = NWB_calcSelective_SB_Load1ImageSpecificmay11( ...
        nwbAll, all_units, bin_width_analysis, preprocess, useSpikeInPeak)
%BUILDNEURALDATA_LOAD1IMAGESPECIFIC  Detect load‑1 image‑specific time cells
% and return their trial‑level firing‑rate matrices in a single call.
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
%   neural_data   – struct array, one element per time cell
%   time_cell_info– numeric matrix [patient, unit, preferredImg, peakBin]
% 
% The detection logic matches the May‑10‑2025 Gaussian‑smoothed pipeline:
%   • 100 ms bins over a 2.5 s maintenance period  (25 bins)
%   • Gaussian kernel σ = 2.5 bins, truncated at ±2 σ
%   • Within‑image permutation test (α = 0.05, 1000 perms)
%   • Spike‑in‑peak consistency (≥ max(4, 70 % trials) with spikes ±2 σ)
%   • Global peak‑dominance criterion
%
% -------------------------------------------------------------------------
% Author:  <your‑name>   Date: 11‑May‑2025
% -------------------------------------------------------------------------

%% ---------------------------- Parameters --------------------------------
rng(42)                              % deterministic permutations

% Detection thresholds
alphaLim         = 0.05;
num_permutations = 1000;
total_bins       = round(2.5 / bin_width_analysis);   % 25 for 100‑ms bins
min_trial_count  = 7;

gKer = GaussianKernal(3, 1.5);

% Spike‑in‑peak rule
% spikeWinHalfWidth = round(2 * gaussSigmaBins);   % ±2 σ window


spikeWinHalfWidth = 2;


%% ------------------------- Optional rate filter -------------------------
% if ~isempty(rateFilter)
%     keep = arrayfun(@(u)  numel(u.spike_times)/(max(u.spike_times)-min(u.spike_times)) ...
%                                >= rateFilter, all_units);
%     all_units = all_units(keep);
% end
num_units = numel(all_units);

%% -------------------------- Time‑cell search ----------------------------
time_cell_info = [];

for iU = 1:num_units
    SU = all_units(iU);
    subject_id = SU.subject_id;
    cellID     = SU.unit_id;

    fprintf('Processing unit %d/%d  [Subject %d  Unit %d]\n', ...
            iU, num_units, subject_id, cellID);

    % ---- Fetch trial metadata -----------------------------------------
    enc1 = nwbAll{SU.session_count}.intervals_trials.vectordata ...
              .get('loadsEnc1_PicIDs').data.load();
    enc2 = nwbAll{SU.session_count}.intervals_trials.vectordata ...
              .get('loadsEnc2_PicIDs').data.load();
    enc3 = nwbAll{SU.session_count}.intervals_trials.vectordata ...
              .get('loadsEnc3_PicIDs').data.load();
    tsM  = nwbAll{SU.session_count}.intervals_trials.vectordata ...
              .get('timestamps_Maintenance').data.load();

    % Keep single‑load trials (enc1 only)
    singleMask = (enc1 > 0) & (enc2 == 0) & (enc3 == 0);
    if ~any(singleMask),  continue;  end

    imgIDs   = enc1(singleMask);
    tStruct  = struct('tsMaint', num2cell(tsM(singleMask)));
    uImgs    = unique(imgIDs);
    nImgs    = numel(uImgs);

    is_locked  = false(1,nImgs);
    peakAmp    = -inf(1,nImgs);
    peakBin    = zeros(1,nImgs);

    % ---- Loop over images ---------------------------------------------
    for k = 1:nImgs
        thisImg     = uImgs(k);
        trialMask   = (imgIDs == thisImg);
        trials_here = tStruct(trialMask);
        nTr         = numel(trials_here);

        if nTr < min_trial_count,  continue;  end

        % ---- Bin, smooth, z‑score per trial ---------------------------
        [frAvg, spkMat] = binGaussZscore(SU.spike_times, trials_here, ...
                                         bin_width_analysis, total_bins, gKer);

        % Store peak for global dominance test
        [pk, pkBin] = max(frAvg);
        peakAmp(k)  = pk;
        peakBin(k)  = pkBin;

        trialsWithSpike = sum(spkMat, 2) > 0;   % logical (#Trials × 1)
        if sum(trialsWithSpike) < 3
            continue
        end




        % ---- Permutation significance ---------------------------------
        p = permTest_timeLocked(frAvg, SU.spike_times, trials_here, ...
                                bin_width_analysis, total_bins, ...
                                num_permutations, gKer);

        if p >= alphaLim,  continue;  end

                 % ---- Spike‑in‑peak consistency (optional) ---------------------
        if useSpikeInPeak
            win = max(1,pkBin-spikeWinHalfWidth) : ...
                  min(total_bins,pkBin+spikeWinHalfWidth);
            trialsWithSpike = any(spkMat(:,win) > 0, 2);
            if sum(trialsWithSpike) < max(3, ceil(0.5*nTr))
                continue
            end
        end

     

        % ---- Image passes all within‑image gates ----------------------
        is_locked(k) = true;
    end

    % ---- Global peak‑dominance gate -----------------------------------
    [~, prefIdx] = max(peakAmp);
    if is_locked(prefIdx) && ...
       all(peakAmp(prefIdx) > peakAmp(setdiff(1:nImgs,prefIdx)))

        time_cell_info = [time_cell_info; ...
                          SU.subject_id, SU.unit_id, ...
                          uImgs(prefIdx), peakBin(prefIdx)]; %#ok<AGROW>
    end


    % %New strict criterion: must be significant for exactly one image
    % sigImgs = find(is_locked);
    % if isscalar(sigImgs)
    %     prefIdx = sigImgs;   % This is the only significant image
    %     time_cell_info = [time_cell_info; ...
    %         SU.subject_id, SU.unit_id, ...
    %         uImgs(prefIdx), peakBin(prefIdx)];
    % end

    
end

fprintf('Detected %d load‑1 image‑specific time cells.\n', size(time_cell_info,1));

%% -------------------- Build neural_data struct -------------------------
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
    if isempty(SUidx),  warning('Unit %d/%d missing after rate filter.',pid,uid); continue; end
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

    % ---- Bin spikes ---------------------------------------------------
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
            spkM(tr,:) = sm;
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

    % ── Reaction time ───────────────────────────────────────────────────────
    tsProbe = sess.intervals_trials.vectordata.get('timestamps_Probe')   ...
                       .data.load();           % (# trials × 1)
    tsResp  = sess.intervals_trials.vectordata.get('timestamps_Response')...
                       .data.load();           % (# trials × 1)
    trialRT = tsResp - tsProbe;                % NaN where a stamp is missing
    
    % ── Probe in/out flag (may be absent in older NWBs) ─────────────────────

    vd = nwbAll{SU.session_count}.intervals_trials.vectordata;
    probe_in_out = vd.get('probe_in_out').data.load();

    % ---- Assemble struct ---------------------------------------------
    nd.patient_id        = pid;
    nd.unit_id           = uid;
    nd.preferred_image   = pref;
    nd.time_field        = tfbin;
    nd.firing_rates      = spkM;
    nd.trial_correctness = correctVec(:);
    nd.brain_region      = brainRegion;
    nd.trial_imageIDs    = trialImgs;
    nd.trial_load        = trialLoad;
    nd.trial_RT          = trialRT(:);
    nd.trial_probe_in_out= probe_in_out(:);

    neural_data(end+1) = nd; %#ok<AGROW>
end

fprintf('Created neural_data for %d time cells (preprocess=%d).\n', ...
        numel(neural_data), preprocess);

end  % --------------------------- END MAIN -------------------------------


%% ===================== Helper sub‑functions ============================

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
