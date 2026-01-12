function plot_dot_mean_pref_vs_nonpref_Enc1(nwbAll, all_units, neural_data_file, bin_width, load_level, use_zscore)
% PLOT_DOT_MEAN_PREF_VS_NONPREF_ENC1
% Dot-with-mean-line comparison of Encoding-1 activity (Pref vs NonPref) across neurons.
%
% Inputs
%   nwbAll             : cell array of NWB sessions
%   all_units          : struct array with fields subject_id, unit_id, session_count, spike_times
%   neural_data_file   : MAT file containing 'neural_data' with fields:
%                        patient_id, unit_id, preferred_image, trial_imageIDs
%   bin_width          : bin size in seconds (e.g., 0.1)
%   load_level         : 1/2/3 to select trials (see select_trials_by_load)
%   use_zscore         : if true, z-score each condition’s trials×bins matrix
%                        across all entries (10-bin vectors across all trials)
%
% Notes
%   - Epoch: Encoding-1 only (0–1 s relative to encoding onset).
%   - FR is computed as spikes/sec per bin. Each trial’s 10-bin vector
%     is optionally smoothed (GaussianKernal(0.3/bin, 1.5), conv 'same').
%   - Per neuron per condition scalar = mean across trials and bins
%     (after optional smoothing + z-scoring).
%   - Balanced trials are NOT enforced here; uses all available trials
%     for each condition at the specified load.

if nargin < 6 || isempty(use_zscore), use_zscore = false; end

encOffset = 1.0;                     % 1 s (Encoding-1)
nBinsEnc  = round(encOffset / bin_width);

% Smoothing kernel (same convention as your other functions)
gaussKern = GaussianKernal(0.3 / bin_width, 1.5);
if ~isempty(gaussKern) && sum(gaussKern) ~= 0
    gaussKern = gaussKern(:)' / sum(gaussKern);
else
    gaussKern = 1;  % no smoothing fallback
end

% Load neural_data
S = load(neural_data_file, 'neural_data');
neural_data = S.neural_data;
num_neurons = numel(neural_data);

pref_vals    = nan(num_neurons,1);
nonpref_vals = nan(num_neurons,1);

infoList = nan(num_neurons,2);   % [patient_id, unit_id] (optional bookkeeping)

for ndx = 1:num_neurons
    nd = neural_data(ndx);
    pid = nd.patient_id;  uid = nd.unit_id;  pImg = nd.preferred_image;

    trial_imageIDs = nd.trial_imageIDs;
    if iscell(trial_imageIDs), trial_imageIDs = cell2mat(trial_imageIDs); end

    % Locate unit & session
    sIdx = find([all_units.subject_id]==pid & [all_units.unit_id]==uid, 1);
    if isempty(sIdx), continue; end
    SU   = all_units(sIdx);
    sess = nwbAll{SU.session_count};

    % Encoding timestamps
    tsEnc = get_ts(sess, 'timestamps_Encoding1');
    if isempty(tsEnc), continue; end

    % Select trials for the specified load
    [trials_this_load, posCol] = select_trials_by_load(trial_imageIDs, load_level);
    if isempty(trials_this_load), continue; end

    % Partition into Pref vs NonPref at the load-defining position
    isPrefAtPos   = (trial_imageIDs(trials_this_load, posCol) == pImg);
    prefTrials    = trials_this_load(isPrefAtPos);
    nonprefTrials = trials_this_load(~isPrefAtPos);

    if isempty(prefTrials) && isempty(nonprefTrials), continue; end

    % Build trials×10 FR matrices (smoothed)
    M_pref    = build_FR_matrix(prefTrials,    tsEnc, SU.spike_times, bin_width, encOffset, gaussKern);
    M_nonpref = build_FR_matrix(nonprefTrials, tsEnc, SU.spike_times, bin_width, encOffset, gaussKern);

    % Optional z-scoring across all entries (across the 10-bin vectors across all trials)
    if use_zscore
        if ~isempty(M_pref)
            muP = mean(M_pref(:), 'omitnan'); sdP = std(M_pref(:), 0, 'omitnan');
            if isfinite(sdP) && sdP>0, M_pref = (M_pref - muP) ./ sdP; else, M_pref = zeros(size(M_pref)); end
        end
        if ~isempty(M_nonpref)
            muN = mean(M_nonpref(:), 'omitnan'); sdN = std(M_nonpref(:), 0, 'omitnan');
            if isfinite(sdN) && sdN>0, M_nonpref = (M_nonpref - muN) ./ sdN; else, M_nonpref = zeros(size(M_nonpref)); end
        end
    end

    % Per-neuron scalar = mean across trials and bins
    if ~isempty(M_pref),    pref_vals(ndx)    = mean(M_pref(:),    'omitnan'); end
    if ~isempty(M_nonpref), nonpref_vals(ndx) = mean(M_nonpref(:), 'omitnan'); end

    infoList(ndx,:) = [pid, uid];
end

% Keep only neurons with both conditions
valid = ~isnan(pref_vals) & ~isnan(nonpref_vals);
pref_vals    = pref_vals(valid);
nonpref_vals = nonpref_vals(valid);

% ---------- Plot: paired dots with mean ± SEM and significance ----------
figure('Name','Encoding-1: Pref vs NonPref (dot + mean line)');
hold on;

x1 = ones(size(nonpref_vals));
x2 = 2*ones(size(pref_vals));

% Paired lines (one line per neuron)
for i = 1:numel(pref_vals)
    plot([1 2], [nonpref_vals(i) pref_vals(i)], '-', 'Color',[0.6 0.6 0.6], 'LineWidth', 1.0);
end

% Dots
plot(x1, nonpref_vals, 'o', 'MarkerFaceColor',[1 0.5 0], 'MarkerEdgeColor','none'); % NonPref (orange)
plot(x2, pref_vals,    'o', 'MarkerFaceColor',[0 0.45 1], 'MarkerEdgeColor','none'); % Pref (blue)

% Means and SEM
mn  = [mean(nonpref_vals), mean(pref_vals)];
sem = [std(nonpref_vals)/sqrt(numel(nonpref_vals)), std(pref_vals)/sqrt(numel(pref_vals))];

plot([0.85 1.15], [mn(1) mn(1)], 'k-', 'LineWidth', 2);
plot([1.85 2.15], [mn(2) mn(2)], 'k-', 'LineWidth', 2);
errorbar([1 2], mn, sem, '.k', 'LineWidth', 1.5);

xlim([0.5 2.5]);
set(gca,'XTick',[1 2],'XTickLabel',{'Non-Pref','Pref'},'FontSize',14);
ylabel(use_zscore_if(use_zscore, 'Z-scored FR (Enc1)', 'Firing Rate (Hz, Enc1)'));
title(sprintf('Encoding-1 (bin = %.1f s) — %d neurons', bin_width, numel(pref_vals)));

% Paired t-test (two-sided); annotate significance
if numel(pref_vals) >= 2
    [p,~,~,stats] = ttest(pref_vals, nonpref_vals); %#ok<ASGLU>
    add_sig_bar(1, 2, mn, sem, p);
end

hold off;

end % main

% ============================ Helpers ===================================

function M = build_FR_matrix(trials, tsEnc, spike_times, binSz, encOffset, ker)
% Returns a trials × nBins matrix (Encoding-1), smoothed row-wise.
    nBins = round(encOffset / binSz);
    if isempty(trials) || isempty(tsEnc)
        M = []; return;
    end
    M = nan(numel(trials), nBins);
    for k = 1:numel(trials)
        idx = trials(k);
        if idx > numel(tsEnc), continue; end
        t0    = tsEnc(idx);
        edges = t0 : binSz : (t0 + encOffset);
        fr    = histcounts(spike_times, edges) ./ binSz;  % 1×nBins, spikes/s
        if numel(ker) > 1
            fr = conv(fr, ker, 'same');
        end
        M(k,:) = fr;
    end
end

function ts = get_ts(sess, key)
    if isKey(sess.intervals_trials.vectordata, key)
        ts = sess.intervals_trials.vectordata.get(key).data.load();
    else
        ts = [];
    end
end

function [trials, posCol] = select_trials_by_load(trial_imageIDs, load_level)
% Same conventions as your LIS helper.
    switch load_level
        case 1
            mask  = (trial_imageIDs(:,1) ~= 0) & (trial_imageIDs(:,2) == 0) & (trial_imageIDs(:,3) == 0);
            posCol = 1;
        case 2
            mask  = (trial_imageIDs(:,1) ~= 0) & (trial_imageIDs(:,2) ~= 0) & (trial_imageIDs(:,3) == 0);
            posCol = 2;
        case 3
            mask  = (trial_imageIDs(:,1) ~= 0) & (trial_imageIDs(:,2) ~= 0) & (trial_imageIDs(:,3) ~= 0);
            posCol = 3;
        otherwise
            error('load_level must be 1, 2, or 3.');
    end
    trials = find(mask);
end

function ylab = use_zscore_if(flag, s1, s2)
    if flag, ylab = s1; else, ylab = s2; end
end

function add_sig_bar(x1, x2, mn, sem, p)
% Draw simple significance bar with stars if p < 0.05
    y = max(mn + sem) * 1.08;
    line([x1 x2], [y y], 'Color', 'k', 'LineWidth', 1.5);
    tl = 0.03 * y;
    line([x1 x1], [y y - tl], 'Color', 'k', 'LineWidth', 1.5);
    line([x2 x2], [y y - tl], 'Color', 'k', 'LineWidth', 1.5);

    if p < 0.001
        stars = '***';
    elseif p < 0.01
        stars = '**';
    elseif p < 0.05
        stars = '*';
    else
        stars = 'n.s.';
    end
    text(mean([x1 x2]), y + 0.01*y, stars, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 14);
end
