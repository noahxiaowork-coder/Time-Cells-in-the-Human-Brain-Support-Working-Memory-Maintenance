function out = plot_concept_ratecurve(nwbAll, all_units, neural_data_file, bin_width)
% plot_PSTH_L1_neuronAvg
% Neuron-first PSTH for load-1:
% 1) For each neuron, collect all load-1 trials, split into Pref vs Non-Pref.
% 2) For each trial: bin spikes for Enc-1 (0–1.0s) and Maint (0–2.5s),
% concatenate (0–3.5s), then smooth AFTER concatenation (cross-boundary bleed),
% convert to spikes/s.
% 3) Average trials WITHIN neuron → per-neuron PSTH (Pref / Non-Pref).
% 4) Mean ± SEM ACROSS neurons at each time bin.
%
% Strict colors: Pref = BLUE, Non-Pref = GREY.
%
% Inputs:
% nwbAll, all_units – your NWB/session + unit structs
% neural_data_file – .mat containing 'neural_data'
% bin_width – bin size in seconds (e.g., 0.1)
%
% Output:
% out.t – 1×n time vector (s)
% out.mean_pref/non – 1×n mean PSTH across neurons
% out.sem_pref/non – 1×n SEM across neurons
% out.PSTH_pref/non – (#neurons × nBins) per-neuron PSTHs (NaN if missing)
% out.Nneur_pref/non – 1×n number of neurons contributing per bin
% out.encOffset, out.maintOffset, out.bin_width

% ---------------- Params ----------------
binSz = bin_width;
encOffset = 1.0;
maintOffset = 2.5;
nE = round(encOffset / binSz);
nM = round(maintOffset / binSz);
nTot = nE + nM;
t = (0:nTot-1) * binSz;

% Gaussian smoothing kernel in *bins*; σ ≈ 0.3 s; smooth AFTER concat
gaussKern = GaussianKernal(0.5/binSz, 2);             % row
if any(gaussKern)
    gaussKern = gaussKern / sum(gaussKern);
end

% Colors (strict)
BLUE = [0 0 1];
ORANGE= [1 0.5 0];

% ---------------- Load data ----------------
S = load(neural_data_file, 'neural_data');
neural_data = S.neural_data;
num_neurons = numel(neural_data);

% Per-neuron PSTHs
PSTH_pref_neuron = nan(num_neurons, nTot);
PSTH_non_neuron = nan(num_neurons, nTot);

% ---------------- Main loop (per neuron) ----------------
for ndx = 1:num_neurons
    nd = neural_data(ndx);
    pid = nd.patient_id;
    uid = nd.unit_id;
    
    % match unit
    sIdx = find([all_units.subject_id]==pid & [all_units.unit_id]==uid, 1);
    if isempty(sIdx), continue; end
    SU = all_units(sIdx);
    sess = nwbAll{SU.session_count};
    
    % timestamps
    tsEnc = get_ts(sess,'timestamps_Encoding1');
    tsMai = get_ts(sess,'timestamps_Maintenance');
    if isempty(tsEnc) || isempty(tsMai), continue; end
    
    % trial table
    T = nd.trial_imageIDs;
    if iscell(T), T = cell2mat(T); end
    
    % load-1 trials; preferred defined at column 1
    [trialsL1, ~] = select_trials_by_load(T, 1);
    if isempty(trialsL1), continue; end
    
    isPref = (T(trialsL1,1) == nd.preferred_image);
    prefTrials = trialsL1(isPref);
    nonprefTrials = trialsL1(~isPref);
    
    % per-trial rates (smoothed AFTER concatenation)
    Rpref = [];
    for k = 1:numel(prefTrials)
        idx = prefTrials(k);
        if idx > numel(tsEnc) || idx > numel(tsMai), continue; end
        r = concat_rate_one_trial(SU.spike_times, tsEnc(idx), tsMai(idx), ...
            binSz, encOffset, maintOffset, gaussKern);
        Rpref = [Rpref; r]; %#ok<AGROW>
    end
    
    Rnon = [];
    for k = 1:numel(nonprefTrials)
        idx = nonprefTrials(k);
        if idx > numel(tsEnc) || idx > numel(tsMai), continue; end
        r = concat_rate_one_trial(SU.spike_times, tsEnc(idx), tsMai(idx), ...
            binSz, encOffset, maintOffset, gaussKern);
        Rnon = [Rnon; r]; %#ok<AGROW>
    end
    
    % ---------------- Global z-scoring across ALL trials (Pref + Non) ----------------
    if ~isempty(Rpref) || ~isempty(Rnon)
        allTrials = [Rpref; Rnon]; % combine
        mu_all = mean(allTrials(:), 'omitnan');
        sd_all = std(allTrials(:), 0, 'omitnan');
        if sd_all > 0
            allTrials = (allTrials - mu_all) ./ sd_all;
        else
            allTrials = zeros(size(allTrials)); % avoid NaN if flat
        end
        % split back
        Rpref = allTrials(1:size(Rpref,1), :);
        Rnon  = allTrials(size(Rpref,1)+1:end, :);
    end
    
    % per-neuron trial-average PSTH
    if ~isempty(Rpref), PSTH_pref_neuron(ndx,:) = mean(Rpref, 1, 'omitnan'); end
    if ~isempty(Rnon), PSTH_non_neuron(ndx,:) = mean(Rnon, 1, 'omitnan'); end

end

% ---------------- Population mean ± SEM across neurons ----------------
[mean_pref, sem_pref, Nneur_pref] = mean_sem_across_neurons(PSTH_pref_neuron);
[mean_non, sem_non, Nneur_non ] = mean_sem_across_neurons(PSTH_non_neuron);

% ---------------- Plot ----------------
figure('Name','PSTH (Neuron-first) – Load-1, Enc1+Maint, mean ± SEM');
hold on;

% shaded SEM bands
fillBand(t, mean_non, sem_non, ORANGE, 0.25);
fillBand(t, mean_pref, sem_pref, BLUE, 0.25);

% mean lines
pNon = plot(t, mean_non, 'Color', ORANGE, 'LineWidth', 2);
pPref = plot(t, mean_pref, 'Color', BLUE, 'LineWidth', 2);

xline(encOffset, '--', 'Color', [0 0 0], 'LineWidth', 1.2);
xlabel('Time (s)');
ylabel('Z-scored Firing rate');
title('Load-1 PSTH (Neuron-first): Non-Pref (orange) vs Pref (blue)');
legend([pNon pPref], {'Non-Pref','Pref'}, 'Location','best');
grid on; box off; hold off;

% ---------------- Output ----------------
out = struct( ...
    't', t, ...
    'mean_pref', mean_pref, 'sem_pref', sem_pref, 'Nneur_pref', Nneur_pref, ...
    'mean_non', mean_non, 'sem_non', sem_non, 'Nneur_non', Nneur_non, ...
    'PSTH_pref', PSTH_pref_neuron, 'PSTH_non', PSTH_non_neuron, ...
    'bin_width', binSz, 'encOffset', encOffset, 'maintOffset', maintOffset);
end

% ==================== Helpers ====================
function r = concat_rate_one_trial(spike_times, tE, tM, binSz, offE, offM, ker)
% Build concatenated counts for Enc (0–offE) and Maint (0–offM),
% then smooth AFTER concatenation (allows cross-boundary bleed), → rate.

    % --- histograms for Enc and Maint ---
    edgesE = tE : binSz : (tE + offE);
    edgesM = tM : binSz : (tM + offM);

    cE = histcounts(spike_times, edgesE);
    cM = histcounts(spike_times, edgesM);
    c  = [cE cM];               % 1 × nTot counts

    % --- ensure 1D row & normalized kernel ---
    c   = c(:)';                
    k   = ker(:)';              
    if any(k)
        k = k / sum(k);
    end

    nTot      = numel(c);
    halfWidth = floor((numel(k) - 1) / 2);
    padBins   = min(halfWidth, floor((nTot-1)/2));   % guard against tiny vectors

    if padBins > 0
        % mirror-pad at both ends (time axis)
        c_pad = [ fliplr(c(1:padBins)), ...
                  c, ...
                  fliplr(c(end-padBins+1:end)) ];

        r_pad = conv(c_pad, k, 'same');
        r     = r_pad(padBins+1 : padBins+nTot) ./ binSz;   % crop back to 0–3.5 s
    else
        % fallback: no room to pad
        r = conv(c, k, 'same') ./ binSz;
    end
end

function [mu, se, Nneur] = mean_sem_across_neurons(P)
% P is (#neurons × #timebins) of per-neuron PSTHs (already trial-averaged)
mu = mean(P, 1, 'omitnan'); % mean across neurons per bin
sd = std(P, 0, 1, 'omitnan'); % std across neurons per bin
Nneur = sum(~isnan(P), 1); % neurons contributing per bin
se = sd ./ sqrt(max(Nneur,1)); % SEM = sd / sqrt(N)
se(Nneur <= 1) = NaN; % undefined SEM with <2 neurons
end

function g = makeGaussKernel(sigmaBins, halfWidthSigmas)
% sigmaBins is σ measured in *bins*
if nargin < 2, halfWidthSigmas = 2.5; end
k = max(1, round(halfWidthSigmas * sigmaBins));
x = -k:k;
g = exp(-(x.^2) / (2*sigmaBins^2));
g = g / sum(g);
end

function ts = get_ts(sess,key)
if isKey(sess.intervals_trials.vectordata,key)
    ts = sess.intervals_trials.vectordata.get(key).data.load();
else
    ts = [];
end
end

function [trials, posCol] = select_trials_by_load(trial_imageIDs, load_level)
% Same convention as your other code.
switch load_level
    case 1
        mask = (trial_imageIDs(:,1) ~= 0) & ...
               (trial_imageIDs(:,2) == 0) & ...
               (trial_imageIDs(:,3) == 0);
        posCol = 1;
    case 2
        mask = (trial_imageIDs(:,1) ~= 0) & ...
               (trial_imageIDs(:,2) ~= 0) & ...
               (trial_imageIDs(:,3) == 0);
        posCol = 2;
    case 3
        mask = (trial_imageIDs(:,1) ~= 0) & ...
               (trial_imageIDs(:,2) ~= 0) & ...
               (trial_imageIDs(:,3) ~= 0);
        posCol = 3;
    otherwise
        error('load_level must be 1, 2, or 3.');
end
trials = find(mask);
end

function h = fillBand(t, mu, se, colorRGB, alphaVal)
upper = mu + se;
lower = mu - se;
xx = [t, fliplr(t)];
yy = [upper, fliplr(lower)];
h = fill(xx, yy, colorRGB, 'EdgeColor', 'none'); %#ok<NASGU>
set(h, 'FaceAlpha', alphaVal);
end
