function out = showPreferredTF_CorrVsIncorr_ByRegion( ...
            nwbAll, all_units, neural_data_file, bin_size, varargin)

% Enforced behavior:
%   - Z-scoring is always ON
%   - Only Load == 1 trials are included

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

acronym = containers.Map( ...
    {'hippocampus','amygdala','dorsal_anterior_cingulate_cortex', ...
     'pre_supplementary_motor_area','ventral_medial_prefrontal_cortex'}, ...
    {'HPC','AMY','DaCC','PSMA','vmPFC'});

stripLat = @(s) regexprep(lower(string(s)), '_(left|right)$', '');

S = load(neural_data_file, 'neural_data');
neural_data = S.neural_data;

duration  = 2.5;
psth_bins = 0:bin_size:duration;

gaussian_kernel = makeGaussianKernel_local(0.3, bin_size, 1.5);

N = numel(neural_data);
per_neuron = nan(N,2);
valsRegional = struct('region',{},'Correct',{},'Incorrect',{});

for ndx = 1:N
    nd = neural_data(ndx);

    reg = stripLat(nd.brain_region);

    patient_id        = nd.patient_id;
    unit_id           = nd.unit_id;
    time_field        = nd.time_field;
    preferred_image   = nd.preferred_image;
    trial_imageIDs    = nd.trial_imageIDs;
    trial_correctness = nd.trial_correctness;
    trial_load        = nd.trial_load;

    unit_match = ([all_units.subject_id] == patient_id) & ...
                 ([all_units.unit_id]    == unit_id);
    if ~any(unit_match)
        continue;
    end
    SU = all_units(unit_match);
    spike_times = SU.spike_times(:)';
    tsMaint = nwbAll{SU.session_count}.intervals_trials.vectordata ...
              .get('timestamps_Maintenance').data.load();

    numTrials = size(trial_imageIDs,1);
    if numTrials ~= numel(trial_correctness) || ...
       numTrials ~= numel(tsMaint) || ...
       numTrials ~= numel(trial_load)
        continue;
    end

    lastVals = nan(numTrials,1);
    for t = 1:numTrials
        nz = find(trial_imageIDs(t,:) ~= 0, 1, 'last');
        if ~isempty(nz)
            lastVals(t) = trial_imageIDs(t,nz);
        end
    end
    isPreferred = (lastVals == preferred_image);

    % -------------------------------
    % Enforced: only Load == 1
    % -------------------------------
    keepByLoad = (trial_load == 1);
    keepMask  = isPreferred & keepByLoad;

    idxPrefCorrect   = find(keepMask & (trial_correctness == 1));
    idxPrefIncorrect = find(keepMask & (trial_correctness == 0));

    if isempty(idxPrefCorrect) && isempty(idxPrefIncorrect)
        continue;
    end

    tf_start = (time_field - 1)*0.1;
    tf_end   = time_field*0.1;
    bin_tf_start = find(psth_bins >= tf_start, 1, 'first');
    bin_tf_end   = find(psth_bins >  tf_end,   1, 'first') - 1;
    if isempty(bin_tf_start) || isempty(bin_tf_end) || bin_tf_end < bin_tf_start
        continue;
    end

    psth_pref_correct = computePSTH_local(spike_times, tsMaint, idxPrefCorrect, ...
                                          duration, psth_bins, gaussian_kernel, bin_size);
    psth_pref_incorrect = computePSTH_local(spike_times, tsMaint, idxPrefIncorrect, ...
                                            duration, psth_bins, gaussian_kernel, bin_size);

    % -------------------------------
    % Enforced: always Z-score
    % -------------------------------
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

    fC = NaN; 
    fI = NaN;
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

    if isfinite(fC) && isfinite(fI)
        k = numel(valsRegional) + 1;
        valsRegional(k).region    = char(reg);
        valsRegional(k).Correct   = fC;
        valsRegional(k).Incorrect = fI;
    end
end

good = ~any(isnan(per_neuron),2);
vals_global = per_neuron(good,:);
nGood = size(vals_global,1);
if nGood >= 2
    [~, p_global_right] = ttest(vals_global(:,1), vals_global(:,2), ...
                                'Tail','right', 'Alpha',0.05);
else
    p_global_right = NaN;
end

ttl = 'Preferred TF (0.1 s): Correct vs Incorrect by Region | Load = 1 (Z)';

[uniqRegs, perReg, xLabs, hFig] = ...
    regionalPlot_CorrIncorr_local(valsRegional, acronym, true, opt, ...
        ttl, ...
        'Average Firing Rate in Time Field (Hz)', ...
        'Z-score Rate in Time Field');

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
out.loads_used         = 1;
out.figSize            = [opt.FigWidth opt.FigHeight];
out.useZscore          = true;

end


% ========================================================================
% ========================= Helper functions =============================
% ========================================================================

function [uniqRegs, perReg, xLabs, hFig] = regionalPlot_CorrIncorr_local(vals, acronym, useZscore, opt, ...
                                                                          titleStr, yLabelHz, yLabelZ)

if isempty(vals)
    uniqRegs = string.empty(1,0);
    perReg   = struct('region',{},'Correct',{},'Incorrect',{});
    xLabs    = {};
    hFig     = [];
    return;
end

allRegs = string({vals.region});
uniqRegs = unique(allRegs, 'stable');

counts = arrayfun(@(r) sum(allRegs==r), uniqRegs);
keep   = counts >= opt.MinUnits;
uniqRegs = uniqRegs(keep);
counts   = counts(keep);

if isempty(uniqRegs)
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

colCorr = [0 0 1];
colInc  = [1 0 0];
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

    good = isfinite(C) & isfinite(I);
    Cg   = C(good);
    Ig   = I(good);
    nPairs = numel(Cg);

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

    muC = mean(C,'omitnan');
    muI = mean(I,'omitnan');
    plot([xc(r)-dx-0.16, xc(r)-dx+0.16], [muC muC], '-', ...
         'Color', colCorr, 'LineWidth', 2.2);
    plot([xc(r)+dx-0.16, xc(r)+dx+0.16], [muI muI], '-', ...
         'Color', colInc, 'LineWidth', 2.2);

    p = NaN;
    if nPairs >= 2
        try
            [~, p] = ttest(Cg, Ig, 'Tail','right');
        catch
            p = NaN;
        end
    end

    star = getStarString_local(p);
    yTop = max([C;I], [], 'omitnan');
    if isempty(yTop) || ~isfinite(yTop)
        yTop = max([muC muI]);
    end
    pad  = 0.10 * max(eps, range([C;I]));
    yStar = yTop + pad;

    text(xc(r), yStar, sprintf('%s\np=%.3g', star, p), ...
         'HorizontalAlignment','center', ...
         'VerticalAlignment','bottom', ...
         'FontSize', opt.FontSize-2);

    allY = [allY; C; I];
end

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

end


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
    psth(iT,:) = conv(counts, gk, 'same') / bin_size;
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
