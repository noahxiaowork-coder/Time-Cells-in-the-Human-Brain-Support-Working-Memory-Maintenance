function out = showTimeCellAverages_ByRegionIntegrated( ...
    nwbAll, all_units, neural_data_file, bin_size, loadBalanced, varargin)

p = inputParser;
p.addParameter('Measure','TimeField', @(s)ischar(s)||isstring(s));
p.addParameter('FigWidth', round(648*1.618), @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('FigHeight', 648, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('FontSize', 20, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('MarkerSize', 20, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('Jitter', 0.10, @(x)isnumeric(x)&&isscalar(x)&&x>=0);
p.addParameter('ShowSEM', false, @(b)islogical(b)&&isscalar(b));
p.addParameter('ExcludeVentral', true, @(b)islogical(b)&&isscalar(b));
p.addParameter('MinUnits', 0, @(x)isnumeric(x)&&isscalar(x)&&x>=0);
p.addParameter('NumResamples', 1000, @(x)isnumeric(x)&&isscalar(x)&&x>=1);
p.parse(varargin{:});
opt = p.Results;

measure = lower(string(opt.Measure));
nResamp = opt.NumResamples;

acronym = containers.Map( ...
    {'hippocampus','amygdala','dorsal_anterior_cingulate_cortex', ...
     'pre_supplementary_motor_area','ventral_medial_prefrontal_cortex'}, ...
    {'HPC','AMY','DaCC','PSMA','vmPFC'});

S = load(neural_data_file, 'neural_data');
neural_data = S.neural_data;

duration = 2.5;
psth_bins = 0:bin_size:duration;
nBins = numel(psth_bins)-1;
gaussian_kernel = GaussianKernal(0.3 / bin_size, 1.5);

stripLat = @(s) regexprep(lower(string(s)), '_(left|right)$', '');

vals = struct('region',{},'yC',{},'yI',{});

for ndx = 1:numel(neural_data)
    reg = stripLat(neural_data(ndx).brain_region);
    if opt.ExcludeVentral && startsWith(reg,"ventral",'IgnoreCase',true), continue; end

    pid = neural_data(ndx).patient_id;
    uid = neural_data(ndx).unit_id;
    corr = neural_data(ndx).trial_correctness;
    tf   = neural_data(ndx).time_field;
    trialLoad = neural_data(ndx).trial_load;

    m = ([all_units.subject_id]==pid) & ([all_units.unit_id]==uid);
    if ~any(m), continue; end
    SU = all_units(m);

    tsMaint = nwbAll{SU.session_count}.intervals_trials.vectordata ...
        .get('timestamps_Maintenance').data.load();
    spk = SU.spike_times;

    corr      = corr(:);
    trialLoad = trialLoad(:);

    if numel(corr) ~= numel(trialLoad) || numel(corr) ~= numel(tsMaint)
        warning('Trial vectors (correctness/load/timestamps) mismatch for unit pid=%d uid=%d. Skipping.', pid, uid);
        continue;
    end

    idxC_all = find(corr==1);
    idxI     = find(corr==0);
    if isempty(idxC_all) || isempty(idxI)
        continue;
    end

    nC_all = numel(idxC_all);
    nI     = numel(idxI);

    firing_correct_all = zeros(nC_all, nBins);
    firing_incorrect   = zeros(nI,     nBins);

    for iC = 1:nC_all
        t  = idxC_all(iC);
        tr = spk(spk >= tsMaint(t) & spk < tsMaint(t)+duration) - tsMaint(t);
        firing_correct_all(iC,:) = conv(histcounts(tr, psth_bins), gaussian_kernel, 'same') / bin_size;
    end

    for iI = 1:nI
        t  = idxI(iI);
        tr = spk(spk >= tsMaint(t) & spk < tsMaint(t)+duration) - tsMaint(t);
        firing_incorrect(iI,:) = conv(histcounts(tr, psth_bins), gaussian_kernel, 'same') / bin_size;
    end

    combined = [firing_correct_all; firing_incorrect];
    muC = mean(combined(:));
    sdC = std(combined(:));
    if sdC==0 || ~isfinite(sdC), sdC = 1; end
    firing_correct_all = (firing_correct_all - muC) / sdC;
    firing_incorrect   = (firing_incorrect   - muC) / sdC;

    cols = 1:nBins;

    switch measure
        case "extendedtf"
            tf_start  = (tf-1)*0.1;
            tf_end    = tf*0.1;
            ext_start = tf_start - 0.1;
            ext_end   = tf_end   + 0.1;

            if ext_start < 0 || ext_end > 2.5, continue; end

            b0 = find(psth_bins >= ext_start, 1, 'first');
            b1 = find(psth_bins >  ext_end,  1, 'first') - 1;
            if isempty(b0) || isempty(b1) || b1<b0, continue; end
            cols = b0:b1;

        case "timefield"
            tf_start = (tf-1)*0.1;
            tf_end   = tf*0.1;
            b0 = find(psth_bins >= tf_start, 1, 'first');
            b1 = find(psth_bins >  tf_end,   1, 'first') - 1;
            if isempty(b0) || isempty(b1) || b1<b0, continue; end
            cols = b0:b1;

        otherwise
    end

    Yc_trials = mean(firing_correct_all(:,cols), 2, 'omitnan');
    Yi_trials = mean(firing_incorrect(:,cols),   2, 'omitnan');

    if loadBalanced
        [yC, ok] = balancedCorrectMean_bootstrap(Yc_trials, idxC_all, idxI, trialLoad, nResamp);
        if ~ok
            continue;
        end
        yI = mean(Yi_trials, 'omitnan');
    else
        yC = mean(Yc_trials, 'omitnan');
        yI = mean(Yi_trials, 'omitnan');
    end

    if isfinite(yC) && isfinite(yI)
        vals(end+1).region = char(reg);
        vals(end).yC = yC;
        vals(end).yI = yI;
    end
end

if isempty(vals)
    warning('No data to plot after filtering.');
    out = struct();
    return;
end

allRegs = string({vals.region});
uniqRegs = unique(allRegs, 'stable');

counts = arrayfun(@(r) sum(allRegs==r), uniqRegs);
keep = counts >= opt.MinUnits;
uniqRegs = uniqRegs(keep);
counts = counts(keep);

R = numel(uniqRegs);
perReg = struct('region',[],'Yc',[],'Yi',[]);

for r = 1:R
    mask = (allRegs==uniqRegs(r));
    perReg(r).region = char(uniqRegs(r));
    perReg(r).Yc = [vals(mask).yC].';
    perReg(r).Yi = [vals(mask).yI].';
end

blue = [0.20 0.45 0.95];
red  = [0.95 0.25 0.25];
grey = [0.70 0.70 0.70];

hFig = figure('Color','w','Units','pixels', ...
    'Position',[100 100 opt.FigWidth opt.FigHeight]);
hold on;

dx = 0.18;
jitW = opt.Jitter;
xc = 1:R;
allY = [];
pVals = NaN(R,1);
nPairsV = NaN(R,1);

for r = 1:R
    Y1 = perReg(r).Yc;
    Y2 = perReg(r).Yi;

    nPairs = min(numel(Y1), numel(Y2));
    if nPairs>0
        j = (rand(nPairs,1)-0.5)*jitW;
        xL = (xc(r)-dx)+j;
        xR = (xc(r)+dx)+j;

        for i=1:nPairs
            plot([xL(i) xR(i)], [Y1(i) Y2(i)], '-', 'Color', grey, 'LineWidth', 0.8);
        end

        scatter(xL, Y1(1:nPairs), opt.MarkerSize, blue, 'filled', 'MarkerFaceAlpha',0.9);
        scatter(xR, Y2(1:nPairs), opt.MarkerSize, red, 'filled', 'MarkerFaceAlpha',0.9);
    end

    if numel(Y1)>nPairs
        j = (rand(numel(Y1)-nPairs,1)-0.5)*jitW;
        scatter((xc(r)-dx)+j, Y1(nPairs+1:end), opt.MarkerSize, blue, 'filled', 'MarkerFaceAlpha',0.9);
    end

    if numel(Y2)>nPairs
        j = (rand(numel(Y2)-nPairs,1)-0.5)*jitW;
        scatter((xc(r)+dx)+j, Y2(nPairs+1:end), opt.MarkerSize, red, 'filled', 'MarkerFaceAlpha',0.9);
    end

    mu1 = mean(Y1,'omitnan');
    mu2 = mean(Y2,'omitnan');
    line([xc(r)-dx-0.15, xc(r)-dx+0.15], [mu1 mu1], 'Color','k', 'LineWidth', 3);
    line([xc(r)+dx-0.15, xc(r)+dx+0.15], [mu2 mu2], 'Color','k', 'LineWidth', 3);

    if opt.ShowSEM
        se1 = std(Y1,'omitnan') / max(1,sqrt(sum(isfinite(Y1))));
        se2 = std(Y2,'omitnan') / max(1,sqrt(sum(isfinite(Y2))));
        line([xc(r)-dx-0.10, xc(r)-dx+0.10], [mu1+se1 mu1+se1], 'Color','k', 'LineWidth', 1.6);
        line([xc(r)-dx-0.10, xc(r)-dx+0.10], [mu1-se1 mu1-se1], 'Color','k', 'LineWidth', 1.6);
        line([xc(r)+dx-0.10, xc(r)+dx+0.10], [mu2+se2 mu2+se2], 'Color','k', 'LineWidth', 1.6);
        line([xc(r)+dx-0.10, xc(r)+dx+0.10], [mu2-se2 mu2-se2], 'Color','k', 'LineWidth', 1.6);
    end

    p = NaN;
    if numel(Y1)==numel(Y2) && numel(Y1)>=2
        try
            [~,p] = ttest(Y1,Y2);
        catch
        end
    end
    pVals(r) = p;
    nPairsV(r) = nPairs;
    fprintf('Region %-20s | nPairs=%-3d | p=%g\n', perReg(r).region, nPairs, p);

    star = getStarString_local(p);
    yTop = max([Y1;Y2], [], 'omitnan');
    if isempty(yTop), yTop = max([mu1 mu2]); end
    pad = 0.06 * max(eps, range([Y1;Y2]));
    text(xc(r), yTop+pad, star, 'HorizontalAlignment','center', ...
        'VerticalAlignment','bottom', 'FontSize', opt.FontSize+2);

    allY = [allY; Y1; Y2];
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
ylabel('Z-scored rate', 'FontSize', opt.FontSize);
yline(0,'--','Color',[0.6 0.6 0.6], 'LineWidth',0.8);

xlim([0.5 R+0.5]);
yPad = 0.08 * max(eps, max(allY)-min(allY));
ylim([min(allY)-yPad, max(allY)+2*yPad]);

title(sprintf('Correct vs Incorrect by Region — %s', measureTitle(measure)), 'FontSize', opt.FontSize);
hold off;

fprintf('Correct < 0 overall: %d\n', sum(allY(1:2:end) < 0));

out = struct('regions',{uniqRegs}, 'xlabels',{xLabs}, 'perRegion',perReg, ...
             'figure',hFig, 'measure',measure, 'figSize',[opt.FigWidth opt.FigHeight]);

end


function starStr = getStarString_local(pVal)
if ~isfinite(pVal), starStr = 'n.s.'; return; end
if pVal < 1e-3,      starStr = '***';
elseif pVal < 1e-2,  starStr = '**';
elseif pVal < 0.05,  starStr = '*';
else,                starStr = 'n.s.';
end
end

function t = measureTitle(m)
switch string(m)
    case "extendedtf", t='Time Field (±0.1 s)';
    case "timefield",  t='Time Field (0.1 s)';
    otherwise,         t='Maintenance (0–2.5 s)';
end
end

function [yC, ok] = balancedCorrectMean_bootstrap(Yc_trials, idxC_all, idxI, trialLoad, nResamp)

ok = false;
yC = NaN;

if isempty(idxC_all) || isempty(idxI)
    return;
end

loadC = trialLoad(idxC_all);
loadI = trialLoad(idxI);

loadsHere = unique(trialLoad);

yBoot = NaN(nResamp,1);

for b = 1:nResamp
    rowsSel = [];
    for L = loadsHere(:).'
        rowsC_L = find(loadC == L);
        rowsI_L = find(loadI == L);

        nC_L = numel(rowsC_L);
        nI_L = numel(rowsI_L);

        if nC_L==0 || nI_L==0
            continue;
        end

        nUse = min(nC_L, nI_L);

        perm = randperm(nC_L, nUse);
        rowsSel = [rowsSel; rowsC_L(perm)];
    end

    if isempty(rowsSel)
        continue;
    end

    yBoot(b) = mean(Yc_trials(rowsSel), 'omitnan');
end

valid = isfinite(yBoot);
if ~any(valid)
    return;
end

yC = mean(yBoot(valid));
ok = true;
end
