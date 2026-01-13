function out = plot_concept_ratecurve(nwbAll, all_units, neural_data_file, bin_width)

binSz = bin_width;
encOffset = 1.0;
maintOffset = 2.5;
nE = round(encOffset / binSz);
nM = round(maintOffset / binSz);
nTot = nE + nM;
t = (0:nTot-1) * binSz;

gaussKern = GaussianKernal(0.5/binSz, 2);
if any(gaussKern)
    gaussKern = gaussKern / sum(gaussKern);
end

BLUE = [0 0 1];
ORANGE= [1 0.5 0];

S = load(neural_data_file, 'neural_data');
neural_data = S.neural_data;
num_neurons = numel(neural_data);

PSTH_pref_neuron = nan(num_neurons, nTot);
PSTH_non_neuron = nan(num_neurons, nTot);

for ndx = 1:num_neurons
    nd = neural_data(ndx);
    pid = nd.patient_id;
    uid = nd.unit_id;
    
    sIdx = find([all_units.subject_id]==pid & [all_units.unit_id]==uid, 1);
    if isempty(sIdx), continue; end
    SU = all_units(sIdx);
    sess = nwbAll{SU.session_count};
    
    tsEnc = get_ts(sess,'timestamps_Encoding1');
    tsMai = get_ts(sess,'timestamps_Maintenance');
    if isempty(tsEnc) || isempty(tsMai), continue; end
    
    T = nd.trial_imageIDs;
    if iscell(T), T = cell2mat(T); end
    
    [trialsL1, ~] = select_trials_by_load(T, 1);
    if isempty(trialsL1), continue; end
    
    isPref = (T(trialsL1,1) == nd.preferred_image);
    prefTrials = trialsL1(isPref);
    nonprefTrials = trialsL1(~isPref);
    
    Rpref = [];
    for k = 1:numel(prefTrials)
        idx = prefTrials(k);
        if idx > numel(tsEnc) || idx > numel(tsMai), continue; end
        r = concat_rate_one_trial(SU.spike_times, tsEnc(idx), tsMai(idx), ...
            binSz, encOffset, maintOffset, gaussKern);
        Rpref = [Rpref; r];
    end
    
    Rnon = [];
    for k = 1:numel(nonprefTrials)
        idx = nonprefTrials(k);
        if idx > numel(tsEnc) || idx > numel(tsMai), continue; end
        r = concat_rate_one_trial(SU.spike_times, tsEnc(idx), tsMai(idx), ...
            binSz, encOffset, maintOffset, gaussKern);
        Rnon = [Rnon; r];
    end
    
    if ~isempty(Rpref) || ~isempty(Rnon)
        allTrials = [Rpref; Rnon];
        mu_all = mean(allTrials(:), 'omitnan');
        sd_all = std(allTrials(:), 0, 'omitnan');
        if sd_all > 0
            allTrials = (allTrials - mu_all) ./ sd_all;
        else
            allTrials = zeros(size(allTrials));
        end
        Rpref = allTrials(1:size(Rpref,1), :);
        Rnon  = allTrials(size(Rpref,1)+1:end, :);
    end
    
    if ~isempty(Rpref), PSTH_pref_neuron(ndx,:) = mean(Rpref, 1, 'omitnan'); end
    if ~isempty(Rnon), PSTH_non_neuron(ndx,:) = mean(Rnon, 1, 'omitnan'); end
end

[mean_pref, sem_pref, Nneur_pref] = mean_sem_across_neurons(PSTH_pref_neuron);
[mean_non, sem_non, Nneur_non ] = mean_sem_across_neurons(PSTH_non_neuron);

figure('Name','PSTH (Neuron-first) – Load-1, Enc1+Maint, mean ± SEM');
hold on;
fillBand(t, mean_non, sem_non, ORANGE, 0.25);
fillBand(t, mean_pref, sem_pref, BLUE, 0.25);
pNon = plot(t, mean_non, 'Color', ORANGE, 'LineWidth', 2);
pPref = plot(t, mean_pref, 'Color', BLUE, 'LineWidth', 2);
xline(encOffset, '--', 'Color', [0 0 0], 'LineWidth', 1.2);
xlabel('Time (s)');
ylabel('Z-scored Firing rate');
title('Load-1 PSTH (Neuron-first): Non-Pref (orange) vs Pref (blue)');
legend([pNon pPref], {'Non-Pref','Pref'}, 'Location','best');
grid on; box off; hold off;

out = struct( ...
    't', t, ...
    'mean_pref', mean_pref, 'sem_pref', sem_pref, 'Nneur_pref', Nneur_pref, ...
    'mean_non', mean_non, 'sem_non', sem_non, 'Nneur_non', Nneur_non, ...
    'PSTH_pref', PSTH_pref_neuron, 'PSTH_non', PSTH_non_neuron, ...
    'bin_width', binSz, 'encOffset', encOffset, 'maintOffset', maintOffset);
end

function r = concat_rate_one_trial(spike_times, tE, tM, binSz, offE, offM, ker)

edgesE = tE : binSz : (tE + offE);
edgesM = tM : binSz : (tM + offM);

cE = histcounts(spike_times, edgesE);
cM = histcounts(spike_times, edgesM);
c  = [cE cM];

c   = c(:)';                
k   = ker(:)';              
if any(k)
    k = k / sum(k);
end

nTot      = numel(c);
halfWidth = floor((numel(k) - 1) / 2);
padBins   = min(halfWidth, floor((nTot-1)/2));

if padBins > 0
    c_pad = [ fliplr(c(1:padBins)), c, fliplr(c(end-padBins+1:end)) ];
    r_pad = conv(c_pad, k, 'same');
    r     = r_pad(padBins+1 : padBins+nTot) ./ binSz;
else
    r = conv(c, k, 'same') ./ binSz;
end
end

function [mu, se, Nneur] = mean_sem_across_neurons(P)
mu = mean(P, 1, 'omitnan');
sd = std(P, 0, 1, 'omitnan');
Nneur = sum(~isnan(P), 1);
se = sd ./ sqrt(max(Nneur,1));
se(Nneur <= 1) = NaN;
end

function ts = get_ts(sess,key)
if isKey(sess.intervals_trials.vectordata,key)
    ts = sess.intervals_trials.vectordata.get(key).data.load();
else
    ts = [];
end
end

function [trials, posCol] = select_trials_by_load(trial_imageIDs, load_level)
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
h = fill(xx, yy, colorRGB, 'EdgeColor', 'none');
set(h, 'FaceAlpha', alphaVal);
end
