function out = showTimeCellAverages_RT_ByRegionIntegrated( ...
            nwbAll, all_units, neural_data_file, bin_size, ...
            useZscore, use_correct, threshold, varargin)

if nargin < 6 || isempty(use_correct), use_correct = true; end
if nargin < 7 || isempty(threshold),   threshold   = 30;   end

p = inputParser;
p.addParameter('ExcludeVentral', true,  @(b)islogical(b)&&isscalar(b));
p.addParameter('MinUnits',       0,     @(x)isnumeric(x)&&isscalar(x)&&x>=0);
p.addParameter('FigWidth',  round(648*1.618), @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('FigHeight',      648,  @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('FontSize',        20,  @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('MarkerSize',      24,  @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('Jitter',        0.08,  @(x)isnumeric(x)&&isscalar(x)&&x>=0);
p.addParameter('PerLoadPoints', false, @(b)islogical(b)&&isscalar(b));
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
nBins     = numel(psth_bins) - 1;

gauss_kernel = GaussianKernal(0.3 / bin_size, 1.5);

raw_pid = {neural_data.patient_id};
if all(cellfun(@isnumeric, raw_pid))
    pid_vec = cell2mat(raw_pid);
else
    pid_vec = string(raw_pid);
end
patient_ids = unique(pid_vec);

percentiles = nan(numel(patient_ids), 3, 2);
for pIdx = 1:numel(patient_ids)
    if isnumeric(patient_ids(pIdx))
        idx = find(pid_vec == patient_ids(pIdx), 1, 'first');
    else
        idx = find(strcmp(pid_vec, patient_ids(pIdx)), 1, 'first');
    end
    if isempty(idx), continue; end

    rt_all   = neural_data(idx).trial_RT;
    load_all = neural_data(idx).trial_load;
    corr_all = (neural_data(idx).trial_correctness == 1);

    for L = 1:3
        msk = (load_all == L) & ~isnan(rt_all);
        if use_correct
            msk = msk & corr_all;
        end
        if nnz(msk) < 4, continue; end
        percentiles(pIdx, L, 1) = prctile(rt_all(msk), threshold);
        percentiles(pIdx, L, 2) = prctile(rt_all(msk), 100 - threshold);
    end
end

valsTF    = struct('region',{},'fast',{},'slow',{});
valsMaint = struct('region',{},'fast',{},'slow',{});

for ndx = 1:numel(neural_data)
    reg = stripLat(neural_data(ndx).brain_region);
    if opt.ExcludeVentral && startsWith(reg,"ventral",'IgnoreCase',true)
        continue;
    end

    pid     = neural_data(ndx).patient_id;
    uid     = neural_data(ndx).unit_id;
    rtVec   = neural_data(ndx).trial_RT;
    corr    = (neural_data(ndx).trial_correctness == 1);
    loadVec = neural_data(ndx).trial_load;
    tf_bin  = neural_data(ndx).time_field;

    if isnumeric(pid)
        pIdx = find(patient_ids == pid);
    else
        pIdx = find(strcmp(patient_ids, string(pid)));
    end
    if isempty(pIdx), continue; end

    fast_mask = false(size(rtVec));
    slow_mask = false(size(rtVec));
    for L = 1:3
        q1 = percentiles(pIdx, L, 1);
        q3 = percentiles(pIdx, L, 2);
        if isnan(q1) || isnan(q3), continue; end

        msk = (loadVec == L) & ~isnan(rtVec);
        if use_correct
            msk = msk & corr;
        end

        fast_mask(msk) = fast_mask(msk) | (rtVec(msk) <= q1);
        slow_mask(msk) = slow_mask(msk) | (rtVec(msk) >= q3);
    end

    if ~any(fast_mask) || ~any(slow_mask), continue; end

    m = ([all_units.subject_id] == pid) & ([all_units.unit_id] == uid);
    if ~any(m), continue; end
    SU = all_units(m);

    tsMaint = nwbAll{SU.session_count}.intervals_trials. ...
                     vectordata.get('timestamps_Maintenance').data.load();
    spk = SU.spike_times;

    if ~opt.PerLoadPoints
        psth_fast = zeros(nnz(fast_mask), nBins);
        psth_slow = zeros(nnz(slow_mask), nBins);

        ii = 0;
        for t = find(fast_mask)'
            ii = ii + 1;
            s = spk(spk >= tsMaint(t) & spk < tsMaint(t)+duration) - tsMaint(t);
            psth_fast(ii,:) = conv(histcounts(s, psth_bins), gauss_kernel, 'same') / bin_size;
        end
        jj = 0;
        for t = find(slow_mask)'
            jj = jj + 1;
            s = spk(spk >= tsMaint(t) & spk < tsMaint(t)+duration) - tsMaint(t);
            psth_slow(jj,:) = conv(histcounts(s, psth_bins), gauss_kernel, 'same') / bin_size;
        end

        if useZscore
            comb = [psth_fast; psth_slow];
            mu   = mean(comb(:));
            sd   = std(comb(:));
            if sd == 0 || ~isfinite(sd), sd = 2; end
            psth_fast = (psth_fast - mu) / sd;
            psth_slow = (psth_slow - mu) / sd;
        end

        tf_start = (tf_bin-1)*0.1;
        tf_end   = tf_bin*0.1;
        bin_tf_s = find(psth_bins >= tf_start, 1, 'first');
        bin_tf_e = find(psth_bins >  tf_end,   1, 'first') - 1;
        if ~isempty(bin_tf_s) && ~isempty(bin_tf_e) && bin_tf_e >= bin_tf_s
            fTF = mean(mean(psth_fast(:, bin_tf_s:bin_tf_e), 2), 'omitnan');
            sTF = mean(mean(psth_slow(:, bin_tf_s:bin_tf_e), 2), 'omitnan');
            if isfinite(fTF) && isfinite(sTF)
                k = numel(valsTF) + 1;
                valsTF(k).region = char(reg);
                valsTF(k).fast   = fTF;
                valsTF(k).slow   = sTF;
            end
        end

        fM = mean(psth_fast(:), 'omitnan');
        sM = mean(psth_slow(:), 'omitnan');
        if isfinite(fM) && isfinite(sM)
            k = numel(valsMaint) + 1;
            valsMaint(k).region = char(reg);
            valsMaint(k).fast   = fM;
            valsMaint(k).slow   = sM;
        end

    else
        for L = 1:3
            local_fast = fast_mask & (loadVec == L);
            local_slow = slow_mask & (loadVec == L);

            if ~any(local_fast) || ~any(local_slow)
                continue;
            end

            idxF = find(local_fast);
            idxS = find(local_slow);

            psth_fast = zeros(numel(idxF), nBins);
            psth_slow = zeros(numel(idxS), nBins);

            ii = 0;
            for t = idxF'
                ii = ii + 1;
                s = spk(spk >= tsMaint(t) & spk < tsMaint(t)+duration) - tsMaint(t);
                psth_fast(ii,:) = conv(histcounts(s, psth_bins), gauss_kernel, 'same') / bin_size;
            end
            jj = 0;
            for t = idxS'
                jj = jj + 1;
                s = spk(spk >= tsMaint(t) & spk < tsMaint(t)+duration) - tsMaint(t);
                psth_slow(jj,:) = conv(histcounts(s, psth_bins), gauss_kernel, 'same') / bin_size;
            end

            if useZscore
                comb = [psth_fast; psth_slow];
                mu   = mean(comb(:));
                sd   = std(comb(:));
                if sd == 0 || ~isfinite(sd), sd = 2; end
                psth_fast = (psth_fast - mu) / sd;
                psth_slow = (psth_slow - mu) / sd;
            end

            regionLabel = sprintf('%s_L%d', char(reg), L);

            tf_start = (tf_bin-1)*0.1;
            tf_end   = tf_bin*0.1;
            bin_tf_s = find(psth_bins >= tf_start, 1, 'first');
            bin_tf_e = find(psth_bins >  tf_end,   1, 'first') - 1;
            if ~isempty(bin_tf_s) && ~isempty(bin_tf_e) && bin_tf_e >= bin_tf_s
                fTF = mean(mean(psth_fast(:, bin_tf_s:bin_tf_e), 2), 'omitnan');
                sTF = mean(mean(psth_slow(:, bin_tf_s:bin_tf_e), 2), 'omitnan');
                if isfinite(fTF) && isfinite(sTF)
                    k = numel(valsTF) + 1;
                    valsTF(k).region = regionLabel;
                    valsTF(k).fast   = fTF;
                    valsTF(k).slow   = sTF;
                end
            end

            fM = mean(psth_fast(:), 'omitnan');
            sM = mean(psth_slow(:), 'omitnan');
            if isfinite(fM) && isfinite(sM)
                k = numel(valsMaint) + 1;
                valsMaint(k).region = regionLabel;
                valsMaint(k).fast   = fM;
                valsMaint(k).slow   = sM;
            end
        end
    end
end

[uniqRegsTF, perRegTF, xLabsTF, hFigTF] = ...
    regionalPlot_RT(valsTF, acronym, useZscore, opt, ...
    'Fast vs Slow RT by Region — Time Field (0.1 s)', ...
    'Average Firing Rate in Time Field (Hz)', ...
    'Z-score Rate in Time Field');

[uniqRegsMaint, perRegMaint, xLabsMaint, hFigMaint] = ...
    regionalPlot_RT(valsMaint, acronym, useZscore, opt, ...
    'Fast vs Slow RT by Region — Maintenance (0–2.5 s)', ...
    'Average Firing Rate (Hz)', ...
    'Z-score Rate');

out = struct( ...
    'regionsTF',     {uniqRegsTF}, ...
    'xlabelsTF',     {xLabsTF}, ...
    'perRegionTF',    perRegTF, ...
    'figureTF',       hFigTF, ...
    'regionsMaint',  {uniqRegsMaint}, ...
    'xlabelsMaint',  {xLabsMaint}, ...
    'perRegionMaint', perRegMaint, ...
    'figureMaint',    hFigMaint, ...
    'figSize',        [opt.FigWidth opt.FigHeight], ...
    'useZscore',      useZscore, ...
    'use_correct',    use_correct, ...
    'threshold',      threshold, ...
    'PerLoadPoints',  opt.PerLoadPoints);

end


function [uniqRegs, perReg, xLabs, hFig] = regionalPlot_RT(vals, acronym, useZscore, opt, ...
                                                           titleStr, yLabelHz, yLabelZ)

if isempty(vals)
    uniqRegs = string.empty(1,0);
    perReg   = struct('region',{},'Fast',{},'Slow',{});
    xLabs    = {};
    hFig     = [];
    return;
end

allRegs = string({vals.region});
uniqRegs = unique(allRegs, 'stable');

counts = arrayfun(@(r) sum(allRegs==r), uniqRegs);
keep   = counts >= opt.MinUnits;
uniqRegs = uniqRegs(keep);

if isempty(uniqRegs)
    perReg = struct('region',{},'Fast',{},'Slow',{});
    xLabs  = {};
    hFig   = [];
    return;
end

R = numel(uniqRegs);
perReg = struct('region',[],'Fast',[],'Slow',[]);
for r = 1:R
    mask = (allRegs==uniqRegs(r));
    perReg(r).region = char(uniqRegs(r));
    perReg(r).Fast   = [vals(mask).fast].';
    perReg(r).Slow   = [vals(mask).slow].';
end

colFast = [0 0 1];
colSlow = [1 0 0];
grey    = [0.75 0.75 0.75];

hFig = figure('Color','w','Units','pixels', ...
              'Position',[100 100 opt.FigWidth opt.FigHeight]);
hold on;

dx   = 0.18;
jitW = opt.Jitter;
xc   = 1:R;
allY = [];

for r = 1:R
    F = perReg(r).Fast;
    S = perReg(r).Slow;

    good = isfinite(F) & isfinite(S);
    Fg   = F(good);
    Sg   = S(good);
    nPairs = numel(Fg);

    if nPairs > 0
        j  = (rand(nPairs,1)-0.5)*jitW;
        xL = (xc(r)-dx)+j;
        xR = (xc(r)+dx)+j;
        for i = 1:nPairs
            plot([xL(i) xR(i)], [Fg(i) Sg(i)], '-', ...
                 'Color', grey, 'LineWidth', 0.7);
        end
        scatter(xL, Fg, opt.MarkerSize, colFast, 'filled', 'MarkerFaceAlpha',0.85);
        scatter(xR, Sg, opt.MarkerSize, colSlow, 'filled', 'MarkerFaceAlpha',0.85);
    end

    muF = mean(F,'omitnan'); 
    muS = mean(S,'omitnan');
    plot([xc(r)-dx-0.16, xc(r)-dx+0.16], [muF muF], '-', ...
         'Color', colFast, 'LineWidth', 2.2);
    plot([xc(r)+dx-0.16, xc(r)+dx+0.16], [muS muS], '-', ...
         'Color', colSlow, 'LineWidth', 2.2);

    p = NaN;
    if nPairs >= 2
        try
            [~, p] = ttest(Fg, Sg, 'Tail','right');
        catch
            p = NaN;
        end
    end

    star = getStarString_local(p);
    yTop = max([F;S], [], 'omitnan');
    if isempty(yTop)
        yTop = max([muF muS]);
    end
    if isempty(yTop) || ~isfinite(yTop), yTop = 0; end
    pad  = 0.10 * max(eps, range([F;S]));
    yStar = yTop + pad;

    text(xc(r), yStar, sprintf('%s\np=%.3g', star, p), ...
         'HorizontalAlignment','center', ...
         'VerticalAlignment','bottom', ...
         'FontSize', opt.FontSize-2);

    allY = [allY; F; S]; %#ok<AGROW>
end

xLabs = cell(1,R);
for r = 1:R
    key = char(uniqRegs(r));
    tokens = regexp(key, '^(.*)_L(\d+)$', 'tokens', 'once');
    if ~isempty(tokens)
        base    = tokens{1};
        loadStr = tokens{2};
        if isKey(acronym, base)
            xLabs{r} = sprintf('%s L%s', acronym(base), loadStr);
        else
            xLabs{r} = upper(sprintf('%s L%s', base, loadStr));
        end
    else
        if isKey(acronym, key)
            xLabs{r} = acronym(key);
        else
            xLabs{r} = upper(key);
        end
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
