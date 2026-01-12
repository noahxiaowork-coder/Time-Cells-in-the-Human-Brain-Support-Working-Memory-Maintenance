function out = showPreferredTF_CorrVsIncorr_ByRegionIntegrated_Loads( ...
            nwbAll, all_units, neural_data_file, bin_size, ...
            useZscore, include_loads, varargin)
% SHOWPREFERREDTF_CORRVSINCORR_BYREGIONINTEGRATED_LOADS
%
%   Regional version of SHOWPREFERREDTF_CORRVSINCORR_ORIGINALVIZ_LOADS.
%
%   For each neuron:
%     - Preferred trials are those whose last NON-ZERO entry in trial_imageIDs
%       equals preferred_image.
%     - Within preferred trials (restricted to include_loads), computes average
%       firing in the Time Field window (0.1 s around time_field bin):
%           Preferred-Correct vs Preferred-Incorrect.
%
%   Then groups neurons by brain region (laterality stripped) and plots
%   Correct vs Incorrect as paired points per region:
%     - Blue  = Preferred-Correct
%     - Red   = Preferred-Incorrect
%     - Grey lines connect paired neurons
%     - One-sided paired t-test (Correct > Incorrect) per region with stars+p.
%
% Inputs:
%   nwbAll, all_units, neural_data_file : as in your other functions
%   bin_size      : PSTH bin size (e.g., 0.05)
%   useZscore     : logical, z-score within neuron across all preferred trials (default: false)
%   include_loads : vector of loads to include (e.g., 1:3, [1 3]). Default: 1:3
%
% Name-value options:
%   'ExcludeVentral'  (default=true)   : drop regions starting with 'ventral'
%   'MinUnits'        (default=0)      : min number of neurons per region to keep
%   'FigWidth'        (default=round(648*1.618))
%   'FigHeight'       (default=648)
%   'FontSize'        (default=20)
%   'MarkerSize'      (default=24)
%   'Jitter'          (default=0.08)
%
% Output:
%   out struct with fields:
%     .perNeuron         [N x 2] -> [Pref-Correct, Pref-Incorrect]
%     .regions           cellstr of region names (after stripping laterality)
%     .xlabels           x-axis labels used (acronyms)
%     .perRegion         struct array with fields .region, .Correct, .Incorrect
%     .figure            handle to regional figure
%     .means_global      [Correct, Incorrect] across all good neurons
%     .sems_global       SEMs across all good neurons
%     .pval_global_right one-sided paired t-test (Correct > Incorrect) across all neurons
%     .n_neurons_used    number of neurons with both conditions
%     .loads_used        include_loads
%     .figSize           [width height]
%     .useZscore         useZscore

if nargin < 5 || isempty(useZscore),    useZscore    = false; end
if nargin < 6 || isempty(include_loads), include_loads = 1:3;  end
include_loads = unique(include_loads(:))';

% ---- options ----
p = inputParser;
p.addParameter('ExcludeVentral', true,  @(b)islogical(b)&&isscalar(b));
p.addParameter('MinUnits',       0,     @(x)isnumeric(x)&&isscalar(x)&&x>=0);
p.addParameter('FigWidth',  round(648*1.618), @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('FigHeight',      648,  @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('FontSize',        20,  @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('MarkerSize',      24,  @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('Jitter',        0.08,  @(x)isnumeric(x)&&isscalar(x)&&x>=0);
p.parse(varargin{:});
opt = p.Results;

% ---- acronym map for x-axis ----
acronym = containers.Map( ...
    {'hippocampus','amygdala','dorsal_anterior_cingulate_cortex', ...
     'pre_supplementary_motor_area','ventral_medial_prefrontal_cortex'}, ...
    {'HPC','AMY','DaCC','PSMA','vmPFC'});

% helper: strip trailing _left/_right and lowercase
stripLat = @(s) regexprep(lower(string(s)), '_(left|right)$', '');

%% Load data
S = load(neural_data_file, 'neural_data');
neural_data = S.neural_data;

%% PSTH parameters
duration  = 2.5;
psth_bins = 0:bin_size:duration;

%% Gaussian kernel (same shape as your original)
gaussian_kernel = makeGaussianKernel_local(0.3, bin_size, 1.5); % sigma=0.3s, ±1.5 SD

%% Allocate per-neuron matrix + regional container
N = numel(neural_data);
per_neuron = nan(N,2);  % [Pref-Correct, Pref-Incorrect]
valsRegional = struct('region',{},'Correct',{},'Incorrect',{});

%% Loop over neurons
for ndx = 1:N
    nd = neural_data(ndx);

    % region (with laterality stripped, ventral exclusion)
    reg = stripLat(nd.brain_region);
    if opt.ExcludeVentral && startsWith(reg,"ventral",'IgnoreCase',true)
        continue;
    end

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
    tsMaint     = nwbAll{SU.session_count}.intervals_trials. ...
                      vectordata.get('timestamps_Maintenance').data.load();

    % Basic checks
    numTrials = size(trial_imageIDs,1);
    if numTrials ~= numel(trial_correctness) || ...
       numTrials ~= numel(tsMaint) || ...
       numTrials ~= numel(trial_load)
        warning('Trial count mismatch for unit (patient_id=%d, unit_id=%d). Skipping...', ...
                patient_id, unit_id);
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
        warning('Invalid TF window for unit (patient_id=%d, unit_id=%d). Skipping...', ...
                patient_id, unit_id);
        continue;
    end

    % ----- Smoothed PSTHs for preferred trials -----
    psth_pref_correct   = computePSTH_local(spike_times, tsMaint, idxPrefCorrect, ...
                                            duration, psth_bins, gaussian_kernel, bin_size);
    psth_pref_incorrect = computePSTH_local(spike_times, tsMaint, idxPrefIncorrect, ...
                                            duration, psth_bins, gaussian_kernel, bin_size);

    % Optional z-score across ALL kept preferred trials (both groups) within-neuron
    if useZscore
        allP = [psth_pref_correct; psth_pref_incorrect];
        if ~isempty(allP)
            muC = mean(allP(:), 'omitnan');
            sdC = std(allP(:), 0, 'omitnan');
            if sdC == 0 || isnan(sdC), sdC = 1; end
            if ~isempty(psth_pref_correct)
                psth_pref_correct = (psth_pref_correct - muC) ./ sdC;
            end
            if ~isempty(psth_pref_incorrect)
                psth_pref_incorrect = (psth_pref_incorrect - muC) ./ sdC;
            end
        end
    end

    % ----- Average within TF window (per trial, then across trials) -----
    fC = NaN; fI = NaN;
    if ~isempty(psth_pref_correct)
        valsC = mean(psth_pref_correct(:, bin_tf_start:bin_tf_end), 2, 'omitnan');
        fC = mean(valsC, 'omitnan');
        per_neuron(ndx,1) = fC;
    end
    if ~isempty(psth_pref_incorrect)
        valsI = mean(psth_pref_incorrect(:, bin_tf_start:bin_tf_end), 2, 'omitnan');
        fI = mean(valsI, 'omitnan');
        per_neuron(ndx,2) = fI;
    end

    % store into regional struct if both finite
    if isfinite(fC) && isfinite(fI)
        k = numel(valsRegional) + 1;
        valsRegional(k).region    = char(reg);
        valsRegional(k).Correct   = fC;
        valsRegional(k).Incorrect = fI;
    end
end

%% Global neuron-level stats (paired) just like original
good = ~any(isnan(per_neuron),2);
vals_global = per_neuron(good,:);
nGood = size(vals_global,1);
if nGood >= 2
    [~, p_global_right] = ttest(vals_global(:,1), vals_global(:,2), ...
                                'Tail','right', 'Alpha',0.05);
else
    p_global_right = NaN;
    warning('Not enough neurons with both conditions. n=%d', nGood);
end

%% Regional plot (Correct vs Incorrect)
ttl = sprintf('Preferred TF (0.1 s): Correct vs Incorrect by Region | Loads: %s', ...
              mat2str(include_loads));
[uniqRegs, perReg, xLabs, hFig] = ...
    regionalPlot_CorrIncorr_local(valsRegional, acronym, useZscore, opt, ...
        ttl, ...
        'Average Firing Rate in Time Field (Hz)', ...
        'Z-score Rate in Time Field');

%% Output
out = struct();
out.perNeuron          = per_neuron;
out.regions            = cellstr(uniqRegs);
out.xlabels            = xLabs;
out.perRegion          = perReg;
out.figure             = hFig;
out.means_global       = mean(vals_global,1,'omitnan');
out.sems_global        = std(vals_global,0,1,'omitnan') ./ sqrt(max(1,nGood));
out.pval_global_right  = p_global_right;
out.n_neurons_used     = nGood;
out.loads_used         = include_loads;
out.figSize            = [opt.FigWidth opt.FigHeight];
out.useZscore          = useZscore;

end % main function


% ======================================================================
% Local: regional plot for Correct vs Incorrect
% ======================================================================
function [uniqRegs, perReg, xLabs, hFig] = regionalPlot_CorrIncorr_local(vals, acronym, useZscore, opt, ...
                                                                          titleStr, yLabelHz, yLabelZ)

if isempty(vals)
    warning('No data to plot for "%s".', titleStr);
    uniqRegs = string.empty(1,0);
    perReg   = struct('region',{},'Correct',{},'Incorrect',{});
    xLabs    = {};
    hFig     = [];
    return;
end

allRegs = string({vals.region});
uniqRegs = unique(allRegs, 'stable');

% drop small regions
counts = arrayfun(@(r) sum(allRegs==r), uniqRegs);
keep   = counts >= opt.MinUnits;
uniqRegs = uniqRegs(keep);
counts   = counts(keep);

if isempty(uniqRegs)
    warning('All regions filtered out by MinUnits for "%s".', titleStr);
    perReg = struct('region',{},'Correct',{},'Incorrect',{});
    xLabs  = {};
    hFig   = [];
    return;
end

R = numel(uniqRegs);
perReg = struct('region',[],'Correct',[],'Incorrect',[]);
for r = 1:R
    mask = (allRegs==uniqRegs(r));
    perReg(r).region    = char(uniqRegs(r));
    perReg(r).Correct   = [vals(mask).Correct].';
    perReg(r).Incorrect = [vals(mask).Incorrect].';
end

% plotting
colCorr = [0 0 1];   % blue   = Correct
colInc  = [1 0 0];   % red    = Incorrect
grey    = [0.75 0.75 0.75];

hFig = figure('Color','w','Units','pixels', ...
              'Position',[100 100 opt.FigWidth opt.FigHeight]);
hold on;

dx   = 0.18;
jitW = opt.Jitter;
xc   = 1:R;
allY = [];

for r = 1:R
    C = perReg(r).Correct;
    I = perReg(r).Incorrect;

    % good pairs for plotting/p-values
    good = isfinite(C) & isfinite(I);
    Cg   = C(good);
    Ig   = I(good);
    nPairs = numel(Cg);

    % draw paired connectors using only good pairs
    if nPairs > 0
        j  = (rand(nPairs,1)-0.5)*jitW;
        xL = (xc(r)-dx)+j;
        xR = (xc(r)+dx)+j;
        for i = 1:nPairs
            plot([xL(i) xR(i)], [Cg(i) Ig(i)], '-', ...
                 'Color', grey, 'LineWidth', 0.7);
        end
        scatter(xL, Cg, opt.MarkerSize, colCorr, 'filled', 'MarkerFaceAlpha',0.85);
        scatter(xR, Ig, opt.MarkerSize, colInc, 'filled', 'MarkerFaceAlpha',0.85);
    end

    % any remaining mismatched values (if any)
    if numel(C) > nPairs
        restC = C(~good);
        if ~isempty(restC)
            j = (rand(numel(restC),1)-0.5)*jitW;
            scatter((xc(r)-dx)+j, restC, opt.MarkerSize, colCorr, ...
                    'filled','MarkerFaceAlpha',0.85);
        end
    end
    if numel(I) > nPairs
        restI = I(~good);
        if ~isempty(restI)
            j = (rand(numel(restI),1)-0.5)*jitW;
            scatter((xc(r)+dx)+j, restI, opt.MarkerSize, colInc, ...
                    'filled','MarkerFaceAlpha',0.85);
        end
    end

    % means (colored)
    muC = mean(C,'omitnan');
    muI = mean(I,'omitnan');
    plot([xc(r)-dx-0.16, xc(r)-dx+0.16], [muC muC], '-', ...
         'Color', colCorr, 'LineWidth', 2.2);
    plot([xc(r)+dx-0.16, xc(r)+dx+0.16], [muI muI], '-', ...
         'Color', colInc, 'LineWidth', 2.2);

    % one-sided paired t-test (Correct > Incorrect) on good pairs
    p = NaN;
    if nPairs >= 2
        try
            [~, p] = ttest(Cg, Ig, 'Tail','right');
        catch
            p = NaN;
        end
    end

    % print & annotate
    fprintf('Region %-20s | nPairs=%-3d | one-sided (Correct>Incorrect) p=%g\n', ...
            perReg(r).region, nPairs, p);

    star = getStarString_local(p);
    yTop = max([C;I], [], 'omitnan');
    if isempty(yTop)
        yTop = max([muC muI]);
    end
    if isempty(yTop) || ~isfinite(yTop), yTop = 0; end
    pad  = 0.10 * max(eps, range([C;I]));
    yStar = yTop + pad;

    % stars + p above region
    text(xc(r), yStar, sprintf('%s\np=%.3g', star, p), ...
         'HorizontalAlignment','center', ...
         'VerticalAlignment','bottom', ...
         'FontSize', opt.FontSize-2);

    allY = [allY; C; I]; %#ok<AGROW>
end

% x labels (acronyms)
xLabs = cell(1,R);
for r = 1:R
    key = char(uniqRegs(r));
    if isKey(acronym, key)
        xLabs{r} = acronym(key);
    else
        xLabs{r} = upper(key);
    end
end
set(gca,'XTick',xc,'XTickLabel',xLabs,'FontSize',opt.FontSize,'Box','off');
xlabel('Brain Region', 'FontSize', opt.FontSize);

if useZscore
    ylabel(yLabelZ, 'FontSize', opt.FontSize);
    yline(0,'--','Color',[0.4 0.4 0.4], 'LineWidth',0.8);
else
    ylabel(yLabelHz, 'FontSize', opt.FontSize);
end

xlim([0.5 R+0.5]);
if isempty(allY) || ~any(isfinite(allY))
    ylim([-1 1]);
else
    yPad = 0.08 * max(eps, max(allY)-min(allY));
    ylim([min(allY)-yPad, max(allY)+2*yPad]);
end
title(titleStr, 'FontSize', opt.FontSize);
grid on;
hold off;

end % regionalPlot_CorrIncorr_local


% ======================================================================
% local helpers
% ======================================================================
function psth = computePSTH_local(spike_times, ts, trial_idx, duration, edges, gk, bin_size)
if isempty(trial_idx)
    psth = [];
    return;
end
nT = numel(trial_idx);
nB = numel(edges)-1;
psth = zeros(nT, nB);
for iT = 1:nT
    tStart = ts(trial_idx(iT)); 
    tEnd   = tStart + duration;
    spk = spike_times(spike_times >= tStart & spike_times < tEnd) - tStart;
    counts = histcounts(spk, edges);
    psth(iT,:) = conv(counts, gk, 'same') / bin_size; % Hz
end
end

function gk = makeGaussianKernel_local(sigma_s, bin_size, width_sd)
sigma_bins = sigma_s / bin_size;
halfW = max(1, ceil(width_sd * sigma_bins));
x = -halfW:halfW;
gk = exp(-0.5*(x./sigma_bins).^2);
gk = gk / sum(gk);
end

function starStr = getStarString_local(pVal)
if ~isfinite(pVal)
    starStr = 'n.s.';
    return;
end
if pVal < 1e-3
    starStr = '***';
elseif pVal < 1e-2
    starStr = '**';
elseif pVal < 0.05
    starStr = '*';
else
    starStr = 'n.s.';
end
end
