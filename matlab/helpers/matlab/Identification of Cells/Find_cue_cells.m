function [neural_data, time_cell_info, unit_stats] = Find_cue_cells( ...
    nwbAll, all_units, bin_width_analysis, NullMode)

if nargin < 5 || isempty(NullMode),    NullMode = false; end

rng(42);

Twin = 2.5;
assert(abs(Twin/bin_width_analysis - round(Twin/bin_width_analysis)) < 1e-9, ...
    'bin_width_analysis must divide 2.5 s exactly.');
total_bins = round(Twin / bin_width_analysis);

gKer = GaussianKernal(3, 1.5);
gKer = gKer(:)'; gKer = gKer / sum(gKer);

num_units = numel(all_units);
time_cell_info = [];

neural_data = struct('patient_id',{},'unit_id',{},'preferred_image',{}, ...
    'time_field',{},'firing_rates',{},'trial_correctness',{}, ...
    'brain_region',{},'trial_imageIDs',{},'trial_load',{}, ...
    'trial_RT',{},'trial_probe_in_out',{});

unit_stats = struct('patient_id',{},'unit_id',{},'p_unit',{},'method',{},'nCand',{},'note',{});

for iU = 1:num_units
    SU = all_units(iU);

    sess = nwbAll{SU.session_count};
    enc1 = sess.intervals_trials.vectordata.get('loadsEnc1_PicIDs').data.load();
    enc2 = sess.intervals_trials.vectordata.get('loadsEnc2_PicIDs').data.load();
    enc3 = sess.intervals_trials.vectordata.get('loadsEnc3_PicIDs').data.load();
    tsM  = sess.intervals_trials.vectordata.get('timestamps_Maintenance').data.load();

    singleMask = (enc1 > 0) & (enc2 == 0) & (enc3 == 0);
    if ~any(singleMask)
        unit_stats(end+1) = struct('patient_id',SU.subject_id,'unit_id',SU.unit_id,...
            'p_unit',1,'method','maxt','nCand',0,'note','no single trials'); %#ok<AGROW>
        continue
    end

    imgIDs_single = enc1(singleMask);
    tStarts = tsM(singleMask);
    nTrials = numel(tStarts);

    rawMat_all = zeros(nTrials, total_bins, 'double');
    T = total_bins * bin_width_analysis;
    edgeOffsets = (0:total_bins) .* bin_width_analysis;

    for tr = 1:nTrials
        t0 = tStarts(tr);
        edges = t0 + edgeOffsets;
        trSpk = SU.spike_times(SU.spike_times >= t0 & SU.spike_times < t0 + T);
        if ~isempty(trSpk)
            rawMat_all(tr,:) = histcounts(trSpk, edges);
        end
    end

    imgIDs_detect = imgIDs_single;
    rawMat_detect = rawMat_all;
    if NullMode
        rawMat_detect = apply_within_trial_null(rawMat_detect, 'shift', 1);
        imgIDs_detect = imgIDs_detect(randperm(nTrials));
    end

    uImgs = unique(imgIDs_detect);
    nImgs = numel(uImgs);
    nTr_img = arrayfun(@(id) sum(imgIDs_detect == id), uImgs);

    cand_mask_trials = (nTr_img >= 7);
    candIdx = find(cand_mask_trials);

    if isempty(candIdx)
        unit_stats(end+1) = struct('patient_id',SU.subject_id,'unit_id',SU.unit_id,...
            'p_unit',1,'method','maxt','nCand',0,'note','< 7'); %#ok<AGROW>
        continue
    end

    rawSub_cell   = cell(1,nImgs);
    frAvg_obs     = cell(1,nImgs);
    peakAmp_obs   = -inf(1,nImgs);

    for kImg = 1:nImgs
        trialMask = (imgIDs_detect == uImgs(kImg));
        if sum(trialMask) < 7, continue; end

        rawSub = rawMat_detect(trialMask,:);
        rawSub_cell{kImg} = rawSub;

        [~, frAvg] = smooth_and_avg(rawSub, gKer, false);
        frAvg_obs{kImg} = frAvg;
        peakAmp_obs(kImg) = max(frAvg);
    end

    score_obs = peakAmp_obs;

    p_unit = 1;
    appended_any = false;

    % MaxT multiple comparison correction
    T_obs = max(score_obs(candIdx));

    T_null = -Inf(1000,1);
    for b = 1:1000
        Xb = apply_within_trial_null(rawMat_all, 'shift', 1);
        labels_b = imgIDs_single(randperm(nTrials));

        score_b = -inf(1,nImgs);
        for kImg = 1:nImgs
            trialMask_b = (labels_b == uImgs(kImg));
            if sum(trialMask_b) < 7, continue; end
            [~, frb] = smooth_and_avg(Xb(trialMask_b,:), gKer, false);
            score_b(kImg) = max(frb);
        end

        cand_b = find(score_b > -inf);
        if isempty(cand_b)
            T_null(b) = -Inf;
        else
            T_null(b) = max(score_b(cand_b));
        end
    end

    p_unit = (1 + sum(T_null >= T_obs)) / (1000 + 1);

    if p_unit < 0.05
        [~,loc] = max(score_obs(candIdx));
        kSel = candIdx(loc);
        pkBin = argmaxbin(frAvg_obs{kSel});
        if posthoc_keep('posthoc', rawSub_cell{kSel}, frAvg_obs{kSel}, 2, 1, 4, 0.5)
            time_cell_info = [time_cell_info; SU.subject_id, SU.unit_id, uImgs(kSel), pkBin]; %#ok<AGROW>
            appended_any = true;
        end
    end

    if appended_any
        tsM_all = sess.intervals_trials.vectordata.get('timestamps_Maintenance').data.load();
        resp = sess.intervals_trials.vectordata.get('response_accuracy').data.load();
        ID1 = sess.intervals_trials.vectordata.get('loadsEnc1_PicIDs').data.load();
        ID2 = sess.intervals_trials.vectordata.get('loadsEnc2_PicIDs').data.load();
        ID3 = sess.intervals_trials.vectordata.get('loadsEnc3_PicIDs').data.load();

        nT = numel(tsM_all);
        spkM = zeros(nT, total_bins);

        for tr = 1:nT
            t0 = tsM_all(tr);
            edges = t0 + edgeOffsets;
            trSpk = SU.spike_times(SU.spike_times >= t0 & SU.spike_times < t0 + T);
            if ~isempty(trSpk)
                spkM(tr,:) = histcounts(trSpk, edges);
            end
        end


        spkM = zscore(conv2(spkM, gKer, 'same'), 0, 2);
        spkM(~isfinite(spkM)) = 0;


        try
            brainRegion = fetch_brain_region_legacy(sess, SU);
        catch
            brainRegion = 'unknown';
        end

        tsProbe = sess.intervals_trials.vectordata.get('timestamps_Probe').data.load();
        tsResp  = sess.intervals_trials.vectordata.get('timestamps_Response').data.load();
        trialRT = tsResp - tsProbe;

        vd = sess.intervals_trials.vectordata;
        probe_in_out = vd.get('probe_in_out').data.load();

        rows_this_unit = find(time_cell_info(:,1)==SU.subject_id & time_cell_info(:,2)==SU.unit_id);
        for rr = rows_this_unit(:).'
            nd.patient_id = SU.subject_id;
            nd.unit_id = SU.unit_id;
            nd.preferred_image = time_cell_info(rr,3);
            nd.time_field = time_cell_info(rr,4);
            nd.firing_rates = spkM;
            nd.trial_correctness = double(resp == 1);
            nd.brain_region = brainRegion;
            nd.trial_imageIDs = [ID1(:),ID2(:),ID3(:)];
            nd.trial_load = sum(nd.trial_imageIDs ~= 0, 2);
            nd.trial_RT = trialRT(:);
            nd.trial_probe_in_out = probe_in_out(:);
            neural_data(end+1) = nd; %#ok<AGROW>
        end
    end

    unit_stats(end+1) = struct('patient_id',SU.subject_id,'unit_id',SU.unit_id, ...
        'p_unit',p_unit,'method','maxt','nCand',numel(candIdx), ...
        'note',''); %#ok<AGROW>
end

fprintf('Detected %d load-1 image-specific time cells.\n', size(time_cell_info,1));
fprintf('Created neural_data for %d time cells.\n', numel(neural_data));

end

function Xnull = apply_within_trial_null(X, nullKind, jitterHalf)
[nTr,total_bins] = size(X);
Xnull = zeros(size(X),'like',X);

switch lower(nullKind)
    case 'permutebins'
        for tr = 1:nTr
            row = X(tr,:);
            if any(row), Xnull(tr,:) = row(randperm(total_bins));
            else, Xnull(tr,:) = 0; end
        end

    case 'shift'
        shifts = randi(total_bins, nTr, 1) - 1;
        for tr = 1:nTr
            row = X(tr,:);
            if any(row), Xnull(tr,:) = row( mod((0:total_bins-1) - shifts(tr), total_bins) + 1 );
            else, Xnull(tr,:)=0; end
        end

    case 'jitter'
        for tr = 1:nTr
            row = X(tr,:);
            if any(row)
                jit = randi(2*jitterHalf+1,1) - (jitterHalf+1);
                Xnull(tr,:) = row( mod((0:total_bins-1) + jit, total_bins) + 1 );
            else
                Xnull(tr,:) = 0;
            end
        end

    case 'continuous'
        shifts = randi(total_bins, nTr, 1) - 1;
        for tr = 1:nTr
            row = X(tr,:);
            if any(row), Xnull(tr,:) = row( mod((0:total_bins-1) - shifts(tr), total_bins) + 1 );
            else, Xnull(tr,:)=0; end
        end

    otherwise
        error('Unknown nullKind: %s', nullKind);
end
end

function [Z, frAvg] = smooth_and_avg(rawCounts, gKer, useZ)
Z = conv2(rawCounts, gKer, 'same');
if useZ, Z = zscore_rows_safe(Z); end
frAvg = mean(Z, 1);
end

function Z = zscore_rows_safe(X)
Z = zscore(X, 0, 2);
Z(~isfinite(Z)) = 0;
end

function keep = posthoc_keep(spikeGateMode, rawSub, frAvg, ...
    spikeWinHalfWidth, minSpkPerTrialInWin, minTrialsWithSpike, fracTrialsWithSpike)

if ~strcmpi(spikeGateMode,'posthoc'), keep = true; return; end
pkBin = argmaxbin(frAvg);
keep = spike_gate_pass(rawSub, pkBin, spikeWinHalfWidth, minSpkPerTrialInWin, minTrialsWithSpike, fracTrialsWithSpike);
end

function tf = spike_gate_pass(rawSub, pkBin, spikeWinHalfWidth, ...
    minSpkPerTrialInWin, minTrialsWithSpike, fracTrialsWithSpike)

[nTr,total_bins] = size(rawSub);
win = max(1, pkBin - spikeWinHalfWidth) : min(total_bins, pkBin + spikeWinHalfWidth);
spikesInWin = sum(rawSub(:,win), 2);
req = max(minTrialsWithSpike, ceil(fracTrialsWithSpike * nTr));
tf = (sum(spikesInWin >= minSpkPerTrialInWin) >= req);
end

function [p_i, pkBinObs] = permTest_image_trueBonf(rawSub, gKer, nPerm, useZ, ...
    nullKind, jitterHalf, ...
    spikeGateMode, spikeWinHalfWidth, ...
    minSpkPerTrialInWin, minTrialsWithSpike, fracTrialsWithSpike)

[Zobs, frObs] = smooth_and_avg(rawSub, gKer, useZ);
[pkObs, pkBinObs] = max(frObs);

if strcmpi(spikeGateMode,'nullsym')
    if ~spike_gate_pass(rawSub, pkBinObs, spikeWinHalfWidth, minSpkPerTrialInWin, minTrialsWithSpike, fracTrialsWithSpike)
        p_i = 1; return
    end
end

nullScores = -Inf(nPerm,1);

for b = 1:nPerm
    tmp = apply_within_trial_null(rawSub, nullKind, jitterHalf);
    [~, frP] = smooth_and_avg(tmp, gKer, useZ);
    [pkP, pkBinP] = max(frP);

    if strcmpi(spikeGateMode,'nullsym')
        if spike_gate_pass(tmp, pkBinP, spikeWinHalfWidth, minSpkPerTrialInWin, minTrialsWithSpike, fracTrialsWithSpike)
            nullScores(b) = pkP;
        else
            nullScores(b) = -Inf;
        end
    else
        nullScores(b) = pkP;
    end
end

p_i = (1 + sum(nullScores >= pkObs)) / (nPerm + 1);
end

function bin = argmaxbin(v)
[~,bin] = max(v);
end

function out = fetch_brain_region_legacy(sess, SU)
tryFields = {'electrodes','electrode','electrode_index','channel','channels'};
eIdx = [];
for f = 1:numel(tryFields)
    if isfield(SU, tryFields{f}) && ~isempty(SU.(tryFields{f}))
        eIdx = SU.(tryFields{f});
        break
    end
end

tablePaths = { ...
    'general_extracellular_ephys_electrodes', ...
    'ecephys_electrodes', ...
    'general_extracellular_ephys_electrode', ...
    'electrodes' ...
    };

colNames = {'location','location_label','brain_area','region','structure'};
locVals = [];
for t = 1:numel(tablePaths)
    tbl = [];
    try, tbl = getfield(sess, tablePaths{t}); catch, end %#ok<GFLD>
    if isempty(tbl), continue; end

    for c = 1:numel(colNames)
        vec = [];
        try, vec = tbl.vectordata.get(colNames{c}); catch, end
        if isempty(vec), continue; end

        try
            if ~isempty(eIdx)
                locVals = vec.data.load(eIdx(1));
            else
                locVals = vec.data.load();
            end
        catch
            try
                allVals = vec.data.load();
                if ~isempty(eIdx)
                    locVals = allVals(eIdx(1));
                else
                    locVals = allVals;
                end
            catch
                locVals = [];
            end
        end

        if ~isempty(locVals), break; end
    end
    if ~isempty(locVals), break; end
end

if isempty(locVals), out = 'unknown'; return; end

if iscell(locVals), s = locVals{1};
elseif isstring(locVals), s = char(locVals(1));
else, s = locVals; end

s = char(s);
s = strtrim(s);
if isempty(s), s = 'unknown'; end

s = regexprep(s, '_(left|right)$', '', 'ignorecase');
s = regexprep(s, '\s*(\(|\[)?(left|right|lh|rh|L|R)(\)|\])?\s*$', '', 'ignorecase');
out = s;
end
