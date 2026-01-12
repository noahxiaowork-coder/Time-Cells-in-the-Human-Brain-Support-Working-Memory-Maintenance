function out = showPreferredTF_CorrVsIncorr_OriginalViz_Loads(nwbAll, all_units, neural_data_file, bin_size, useZscore, include_loads)
% SHOWPREFERREDTF_CORRVSINCORR_ORIGINALVIZ_LOADS
%   Preferred trials are those whose last NON-ZERO entry in trial_imageIDs equals preferred_image.
%   Compares within-field (Original TF = 0.1 s) firing in Preferred-Correct vs Preferred-Incorrect.
%   Keeps original bar+pairwise visualization, and lets you specify which LOADS to include.
%
% Inputs:
%   nwbAll, all_units, neural_data_file : as in your original
%   bin_size     : PSTH bin size (e.g., 0.05)
%   useZscore    : logical, z-score within neuron across all preferred trials (default: false)
%   include_loads: vector of loads to include (e.g., 1:3, [1 3]). Default: 1:3
%
% Output:
%   out struct with fields:
%     .per_neuron [N x 2] -> [Preferred-Correct, Preferred-Incorrect]
%     .means, .sems, .pval_right, .n_neurons_used, .loads_used, .figure

if nargin < 5 || isempty(useZscore), useZscore = false; end
if nargin < 6 || isempty(include_loads), include_loads = 1:3; end
include_loads = unique(include_loads(:))';

%% Load data
S = load(neural_data_file, 'neural_data');
neural_data = S.neural_data;

%% PSTH parameters
duration  = 2.5;
psth_bins = 0:bin_size:duration;

%% Gaussian kernel (self-contained)
gaussian_kernel = makeGaussianKernel(0.3, bin_size, 1.5); % sigma=0.3s, ±1.5 SD

%% Allocate
N = numel(neural_data);
per_neuron = nan(N,2);  % [Pref-Correct, Pref-Incorrect]

%% Loop over neurons
for ndx = 1:N
    nd = neural_data(ndx);
    patient_id        = nd.patient_id;
    unit_id           = nd.unit_id;
    time_field        = nd.time_field;           % 0.1 s bins, 1-based
    preferred_image   = nd.preferred_image;      % scalar
    trial_imageIDs    = nd.trial_imageIDs;       % [numTrials x 3]
    trial_correctness = nd.trial_correctness;    % vec (1=correct, 0=incorrect)
    trial_load        = nd.trial_load;           % vec of loads (1/2/3)

    % Match unit
    unit_match = ([all_units.subject_id] == patient_id) & ([all_units.unit_id] == unit_id);
    if ~any(unit_match)
        warning('Unit (patient_id=%d, unit_id=%d) not found. Skipping...', patient_id, unit_id);
        continue;
    end
    SU = all_units(unit_match);
    spike_times = SU.spike_times(:)';
    tsMaint     = nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Maintenance').data.load();

    % Basic checks
    numTrials = size(trial_imageIDs,1);
    if numTrials ~= numel(trial_correctness) || numTrials ~= numel(tsMaint) || numTrials ~= numel(trial_load)
        warning('Trial count mismatch for unit (patient_id=%d, unit_id=%d). Skipping...', patient_id, unit_id);
        continue;
    end

    % ----- Preferred trials: last NON-ZERO element equals preferred_image -----
    lastVals = nan(numTrials,1);
    for t = 1:numTrials
        nz = find(trial_imageIDs(t,:) ~= 0, 1, 'last');
        if ~isempty(nz), lastVals(t) = trial_imageIDs(t,nz); end
    end
    isPreferred = (lastVals == preferred_image);

    % ----- Load filter -----
    keepByLoad = ismember(trial_load, include_loads);

    % Combine: preferred AND in requested loads
    keepMask = isPreferred & keepByLoad;

    idxPrefCorrect   = find(keepMask & (trial_correctness == 1));
    idxPrefIncorrect = find(keepMask & (trial_correctness == 0));

    if isempty(idxPrefCorrect) && isempty(idxPrefIncorrect)
        % nothing to compute for this neuron
        continue;
    end

    % ----- TF window bins -----
    tf_start = (time_field - 1)*0.1;
    tf_end   = time_field*0.1;
    bin_tf_start = find(psth_bins >= tf_start, 1, 'first');
    bin_tf_end   = find(psth_bins >  tf_end,   1, 'first') - 1;
    if isempty(bin_tf_start) || isempty(bin_tf_end) || bin_tf_end < bin_tf_start
        warning('Invalid TF window for unit (patient_id=%d, unit_id=%d). Skipping...', patient_id, unit_id);
        continue;
    end

    % ----- Smoothed PSTHs for the kept preferred trials -----
    psth_pref_correct   = computePSTH(spike_times, tsMaint, idxPrefCorrect, duration, psth_bins, gaussian_kernel, bin_size);
    psth_pref_incorrect = computePSTH(spike_times, tsMaint, idxPrefIncorrect, duration, psth_bins, gaussian_kernel, bin_size);

    % Optional z-score across ALL kept preferred trials (both groups) within-neuron
    if useZscore
        allP = [psth_pref_correct; psth_pref_incorrect];
        if ~isempty(allP)
            muC = mean(allP(:), 'omitnan');
            sdC = std(allP(:), 0, 'omitnan'); if sdC == 0 || isnan(sdC), sdC = 1; end
            if ~isempty(psth_pref_correct),   psth_pref_correct   = (psth_pref_correct   - muC) ./ sdC; end
            if ~isempty(psth_pref_incorrect), psth_pref_incorrect = (psth_pref_incorrect - muC) ./ sdC; end
        end
    end

    % ----- Average within TF window (per trial, then across trials) -----
    if ~isempty(psth_pref_correct)
        valsC = mean(psth_pref_correct(:, bin_tf_start:bin_tf_end), 2, 'omitnan');
        per_neuron(ndx,1) = mean(valsC, 'omitnan');
    end
    if ~isempty(psth_pref_incorrect)
        valsI = mean(psth_pref_incorrect(:, bin_tf_start:bin_tf_end), 2, 'omitnan');
        per_neuron(ndx,2) = mean(valsI, 'omitnan');
    end
end

%% Stats
good = ~any(isnan(per_neuron),2);
vals = per_neuron(good,:);
nGood = size(vals,1);

if nGood >= 2
    [~, p_right] = ttest(vals(:,1), vals(:,2), 'Tail','right', 'Alpha',0.05); % one-sided (Correct > Incorrect)
else
    p_right = NaN;
    warning('Not enough neurons with both conditions. n=%d', nGood);
end

%% Plot (original style, two bars + pairwise annotation; robust y-lim incl. z-score case)
ttl = sprintf('Preferred Trials: TF (0.1 s) | Loads: %s', mat2str(include_loads));
fig = plotPairedSwarmWithCenterLines_OneSided(vals, useZscore, ...
    ttl, ...
    'Average Firing Rate in Time Field (Hz)', ...
    'Z-score Rate in Time Field');

%% Output
out = struct();
out.per_neuron     = per_neuron;
out.means          = mean(vals,1,'omitnan');
out.sems           = std(vals,0,1,'omitnan') ./ sqrt(max(1,nGood));
out.pval_right     = p_right;
out.n_neurons_used = nGood;
out.loads_used     = include_loads;
out.figure         = fig;

end % main

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function figH = plotPairedSwarmWithCenterLines_OneSided(dataMatrix, useZscore, figTitle, yLabelHz, yLabelZ)
% dataMatrix: [N x 2] -> [Pref-Correct, Pref-Incorrect]
% Aesthetic choices inspired by showTimeCellAverages:
% - swarm dots (or jitter fallback)
% - light-gray routed pairing lines via center x
% - mean ± SEM as short horizontal ticks
% - bracket + stars for paired t-test
% - robust y-lims; zero-line for z-score plots

figH = figure('Name', figTitle, 'Position', [100,100,520,540]); hold on;
if isempty(dataMatrix)
    text(0.5,0.5,'No valid data','Units','normalized','FontSize',16, ...
        'HorizontalAlignment','center'); hold off; return;
end

% keep rows with both values
good = ~any(isnan(dataMatrix),2);
X = dataMatrix(good,:);
n = size(X,1);
if n == 0
    text(0.5,0.5,'No paired data','Units','normalized','FontSize',16, ...
        'HorizontalAlignment','center'); hold off; return;
end

% colors (same code as before)
colC = [0 0 1];  % blue  : Pref Correct
colI = [1 0 0];  % red   : Pref Incorrect
pairCol = 0.75*[1 1 1];  % light gray

xL = 1; xR = 2; xC = mean([xL xR]);
A = X(:,1); B = X(:,2);

% stats (one-sided Correct > Incorrect, as in your original)
mA = mean(A,'omitnan'); mB = mean(B,'omitnan');
sA = std(A,0,'omitnan'); sB = std(B,0,'omitnan');
nA = sum(~isnan(A));     nB = sum(~isnan(B));
semA = sA / sqrt(max(1,nA)); semB = sB / sqrt(max(1,nB));
pVal = NaN; if n >= 2, [~,pVal] = ttest(A, B, 'Tail','right', 'Alpha',0.05); end
starStr = getStarString(pVal);

% axes & labels
set(gca,'FontSize',14, 'XTick',[xL xR], 'XTickLabel',{'Pref Correct','Pref Incorrect'});
ylabel(ifelse(useZscore, yLabelZ, yLabelHz), 'FontSize',14);
title(figTitle, 'FontSize',15); box on;

% pairing lines (under points), routed via center
ymid = (A + B) / 2;
for i = 1:n
    plot([xL xC xR], [A(i) ymid(i) B(i)], '-', 'Color', pairCol, 'LineWidth', 0.6, ...
        'HandleVisibility','off');
end

% swarm dots (fallback to jitter if needed)
msz = 24;
haveSwarm = exist('swarmchart','file')==2;
if haveSwarm
    sc1 = swarmchart(repmat(xL,n,1), A, msz, 'filled'); hold on;
    sc1.MarkerFaceColor = colC; sc1.MarkerEdgeColor = 'none'; sc1.MarkerFaceAlpha = 0.9;
    sc1.XJitter = 'density'; sc1.XJitterWidth = 0.2;

    sc2 = swarmchart(repmat(xR,n,1), B, msz, 'filled');
    sc2.MarkerFaceColor = colI; sc2.MarkerEdgeColor = 'none'; sc2.MarkerFaceAlpha = 0.9;
    sc2.XJitter = 'density'; sc2.XJitterWidth = 0.2;
else
    jit = 0.18;
    scatter(xL + (rand(n,1)-0.5)*jit, A, msz, colC, 'filled', ...
        'MarkerFaceAlpha',0.9, 'MarkerEdgeColor','none');
    scatter(xR + (rand(n,1)-0.5)*jit, B, msz, colI, 'filled', ...
        'MarkerFaceAlpha',0.9, 'MarkerEdgeColor','none');
end

% mean ± SEM ticks (clean three-bar style)
tickHalf = 0.2; lwMean = 2.2; lwSem = 1.2;
drawMeanTicks_local(xL, mA, semA, tickHalf, lwMean, lwSem);
drawMeanTicks_local(xR, mB, semB, tickHalf, lwMean, lwSem);

% significance bracket + star text
yMax = max([A;B], [], 'omitnan'); yMin = min([A;B], [], 'omitnan');
pad  = 0.06 * max(eps, yMax - yMin);
ySig = max([mA+semA, mB+semB, yMax]) + pad;
plot([xL xR], [ySig ySig], 'k-', 'LineWidth', 1.4, 'HandleVisibility','off');
txt = sprintf('%s  (p=%.3g)', starStr, pVal);
text(xC, ySig, txt, 'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize',14);

% limits & helpers
xlim([xL-0.55, xR+0.55]);
if useZscore
    % keep zero visible and headroom above bracket
    yAbs = max(abs([A;B]), [], 'omitnan');
    yTop = max([yAbs, ySig + pad]);
    ylim([min(-0.1, yMin - pad), yTop + pad]);
    yline(0,'--','Color',[0.5 0.5 0.5], 'HandleVisibility','off');
else
    ylim([yMin - pad, ySig + pad]);
end

legend({' '},'Location','best','Box','off'); % keeps layout tidy without extra entries
hold off;

    function drawMeanTicks_local(xc, m, sem, halfw, lwM, lwS)
        % center mean
        plot([xc-halfw, xc+halfw], [m m], 'k-', 'LineWidth', lwM, 'HandleVisibility','off');
        % sem whiskers as two short bars
        % plot([xc-halfw/2, xc+halfw/2], [m+sem, m+sem], 'k-', 'LineWidth', lwS, 'HandleVisibility','off');
        % plot([xc-halfw/2, xc+halfw/2], [m-sem, m-sem], 'k-', 'LineWidth', lwS, 'HandleVisibility','off');
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function psth = computePSTH(spike_times, ts, trial_idx, duration, edges, gk, bin_size)
if isempty(trial_idx)
    psth = [];
    return;
end
nT = numel(trial_idx);
nB = numel(edges)-1;
psth = zeros(nT, nB);
for iT = 1:nT
    tStart = ts(trial_idx(iT)); tEnd = tStart + duration;
    spk = spike_times(spike_times >= tStart & spike_times < tEnd) - tStart;
    counts = histcounts(spk, edges);
    psth(iT,:) = conv(counts, gk, 'same') / bin_size; % Hz
end
end

function gk = makeGaussianKernel(sigma_s, bin_size, width_sd)
sigma_bins = sigma_s / bin_size;
halfW = max(1, ceil(width_sd * sigma_bins));
x = -halfW:halfW;
gk = exp(-0.5*(x./sigma_bins).^2);
gk = gk / sum(gk);
end

function starStr = getStarString(pVal)
if isnan(pVal)
    starStr = 'n.s.';
elseif pVal < 1e-3
    starStr = '***';
elseif pVal < 1e-2
    starStr = '**';
elseif pVal < 0.05
    starStr = '*';
else
    starStr = 'n.s.';
end
end

function out = ifelse(cond, valTrue, valFalse)
if cond, out = valTrue; else, out = valFalse; end
end
