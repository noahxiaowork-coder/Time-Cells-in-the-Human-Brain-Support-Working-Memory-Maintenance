function out = showTimeCellAverages_ByRegionLoadIntegrated( ...
        nwbAll, all_units, neural_data_file, bin_size, varargin)

p = inputParser;
p.addParameter('Measure','TimeField', @(s)ischar(s)||isstring(s));
p.addParameter('FigWidth',  round(648*1.618), @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('FigHeight', 648, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('FontSize',   20, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('MarkerSize', 20, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('Jitter',    0.10, @(x)isnumeric(x)&&isscalar(x)&&x>=0);
p.addParameter('ShowSEM',  false, @(b)islogical(b)&&isscalar(b));
p.addParameter('ExcludeVentral', true, @(b)islogical(b)&&isscalar(b));
p.addParameter('MinUnits',  0, @(x)isnumeric(x)&&isscalar(x)&&x>=0);
p.addParameter('UseOnlyCorrect', false, @(b)islogical(b)&&isscalar(b));
p.parse(varargin{:});
opt = p.Results;
measure = lower(string(opt.Measure));

acronym = containers.Map( ...
    {'hippocampus','amygdala','dorsal_anterior_cingulate_cortex', ...
     'pre_supplementary_motor_area','ventral_medial_prefrontal_cortex'}, ...
    {'HPC','AMY','DaCC','PSMA','vmPFC'});

S = load(neural_data_file, 'neural_data');
neural_data = S.neural_data;

hasCorrectField = isfield(neural_data, 'trial_correctness');

duration  = 2.5;
psth_bins = 0:bin_size:duration;
nBins     = numel(psth_bins)-1;
gaussian_kernel = GaussianKernal(0.3 / bin_size, 1.5);

stripLat = @(s) regexprep(lower(string(s)), '_(left|right)$', '');

vals = struct('region',{},'y1',{},'y2',{},'y3',{});
for ndx = 1:numel(neural_data)
    reg = stripLat(neural_data(ndx).brain_region);
    if opt.ExcludeVentral && startsWith(reg,"ventral",'IgnoreCase',true), continue; end

    pid = neural_data(ndx).patient_id;
    uid = neural_data(ndx).unit_id;
    loadVec = neural_data(ndx).trial_load;

    if hasCorrectField
        correctVec = neural_data(ndx).trial_correctness;
    else
        correctVec = [];
    end

    nTrials = numel(loadVec);
    if ~isempty(correctVec) && numel(correctVec) ~= nTrials
        correctVec = [];
    end

    if opt.UseOnlyCorrect && ~isempty(correctVec)
        trialMask = (correctVec == 1);
    else
        trialMask = true(size(loadVec));
    end

    tf = neural_data(ndx).time_field;

    m = ([all_units.subject_id]==pid) & ([all_units.unit_id]==uid);
    if ~any(m), continue; end
    SU = all_units(m);

    tsMaint = nwbAll{SU.session_count}.intervals_trials.vectordata ...
              .get('timestamps_Maintenance').data.load();
    spk = SU.spike_times;

    idxL1 = find(loadVec==1 & trialMask);
    idxL2 = find(loadVec==2 & trialMask);
    idxL3 = find(loadVec==3 & trialMask);

    firingL1 = zeros(numel(idxL1), nBins);
    firingL2 = zeros(numel(idxL2), nBins);
    firingL3 = zeros(numel(idxL3), nBins);

    for iT = 1:numel(idxL1)
        t  = idxL1(iT);
        if t > numel(tsMaint), continue; end
        tr = spk(spk >= tsMaint(t) & spk < tsMaint(t)+duration) - tsMaint(t);
        firingL1(iT,:) = conv(histcounts(tr, psth_bins), gaussian_kernel, 'same') / bin_size;
    end
    for iT = 1:numel(idxL2)
        t  = idxL2(iT);
        if t > numel(tsMaint), continue; end
        tr = spk(spk >= tsMaint(t) & spk < tsMaint(t)+duration) - tsMaint(t);
        firingL2(iT,:) = conv(histcounts(tr, psth_bins), gaussian_kernel, 'same') / bin_size;
    end
    for iT = 1:numel(idxL3)
        t  = idxL3(iT);
        if t > numel(tsMaint), continue; end
        tr = spk(spk >= tsMaint(t) & spk < tsMaint(t)+duration) - tsMaint(t);
        firingL3(iT,:) = conv(histcounts(tr, psth_bins), gaussian_kernel, 'same') / bin_size;
    end

    combined = [];
    if ~isempty(firingL1), combined = [combined; firingL1]; end
    if ~isempty(firingL2), combined = [combined; firingL2]; end
    if ~isempty(firingL3), combined = [combined; firingL3]; end
    if ~isempty(combined)
        muC = mean(combined(:));
        sdC = std(combined(:));
        if sdC==0 || ~isfinite(sdC), sdC = 1; end
        if ~isempty(firingL1), firingL1 = (firingL1 - muC) / sdC; end
        if ~isempty(firingL2), firingL2 = (firingL2 - muC) / sdC; end
        if ~isempty(firingL3), firingL3 = (firingL3 - muC) / sdC; end
    end

    y1 = NaN; y2 = NaN; y3 = NaN;
    switch measure
        case "extendedtf"
            tf_start = (tf-1)*0.1; tf_end = tf*0.1;
            ext_start = tf_start - 0.1; ext_end = tf_end + 0.1;
            if ext_start < 0 || ext_end > 2.5, continue; end
            b0 = find(psth_bins >= ext_start, 1, 'first');
            b1 = find(psth_bins >  ext_end,   1, 'first') - 1;
            if isempty(b0) || isempty(b1) || b1<b0, continue; end
            if ~isempty(firingL1), y1 = mean(mean(firingL1(:,b0:b1),2)); end
            if ~isempty(firingL2), y2 = mean(mean(firingL2(:,b0:b1),2)); end
            if ~isempty(firingL3), y3 = mean(mean(firingL3(:,b0:b1),2)); end

        case "timefield"
            tf_start = (tf-1)*0.1; tf_end = tf*0.1;
            b0 = find(psth_bins >= tf_start, 1, 'first');
            b1 = find(psth_bins >  tf_end,   1, 'first') - 1;
            if isempty(b0) || isempty(b1) || b1<b0, continue; end
            if ~isempty(firingL1), y1 = mean(mean(firingL1(:,b0:b1),2)); end
            if ~isempty(firingL2), y2 = mean(mean(firingL2(:,b0:b1),2)); end
            if ~isempty(firingL3), y3 = mean(mean(firingL3(:,b0:b1),2)); end

        otherwise
            if ~isempty(firingL1), y1 = mean(mean(firingL1,2)); end
            if ~isempty(firingL2), y2 = mean(mean(firingL2,2)); end
            if ~isempty(firingL3), y3 = mean(mean(firingL3,2)); end
    end

    if any(isfinite([y1 y2 y3]))
        vals(end+1).region = char(reg);
        vals(end).y1 = y1;
        vals(end).y2 = y2;
        vals(end).y3 = y3;
    end
end

if isempty(vals)
    out = struct();
    return;
end

allRegs = string({vals.region});
uniqRegs = unique(allRegs, 'stable');

counts = arrayfun(@(r) sum(allRegs==r), uniqRegs);
keep   = counts >= opt.MinUnits;
uniqRegs = uniqRegs(keep);
counts   = counts(keep);

R = numel(uniqRegs);
perReg = struct('region',[],'Y1',[],'Y2',[],'Y3',[]);
for r = 1:R
    mask = (allRegs==uniqRegs(r));
    perReg(r).region = char(uniqRegs(r));
    perReg(r).Y1 = [vals(mask).y1].';
    perReg(r).Y2 = [vals(mask).y2].';
    perReg(r).Y3 = [vals(mask).y3].';
end

col1 = [0.20 0.45 0.95];
col2 = [0.30 0.75 0.30];
col3 = [0.95 0.25 0.25];
grey = [0.70 0.70 0.70];

figure('Color','w','Units','pixels', ...
       'Position',[100 100 opt.FigWidth opt.FigHeight]);
hold on;

dx = 0.23;
offsets = [-dx, 0, dx];
jitW = opt.Jitter;
xc   = 1:R;
allY = [];

for r = 1:R
    Y1 = padToLength(perReg(r).Y1(:), []);
    Y2 = padToLength(perReg(r).Y2(:), []);
    Y3 = padToLength(perReg(r).Y3(:), []);
    maxN = max([numel(Y1), numel(Y2), numel(Y3)]);
    Y1 = padToLength(Y1, maxN);
    Y2 = padToLength(Y2, maxN);
    Y3 = padToLength(Y3, maxN);

    j1 = (rand(maxN,1)-0.5)*jitW;
    j2 = (rand(maxN,1)-0.5)*jitW;
    j3 = (rand(maxN,1)-0.5)*jitW;

    x1 = xc(r) + offsets(1) + j1;
    x2 = xc(r) + offsets(2) + j2;
    x3 = xc(r) + offsets(3) + j3;

    for k = 1:maxN
        yk = [Y1(k) Y2(k) Y3(k)];
        m  = isfinite(yk);
        if sum(m) >= 2
            idx = find(m);
            xs = [x1(k) x2(k) x3(k)];
            for jj = 1:numel(idx)-1
                plot(xs(idx(jj:jj+1)), ...
                     yk(idx(jj:jj+1)), '-', ...
                     'Color', grey, 'LineWidth', 0.8, 'HandleVisibility','off');
            end
        end
    end

    scatter(x1, Y1, opt.MarkerSize, col1, 'filled', 'MarkerFaceAlpha',0.9);
    scatter(x2, Y2, opt.MarkerSize, col2, 'filled', 'MarkerFaceAlpha',0.9);
    scatter(x3, Y3, opt.MarkerSize, col3, 'filled', 'MarkerFaceAlpha',0.9);

    mu1 = mean(Y1,'omitnan');
    mu2 = mean(Y2,'omitnan');
    mu3 = mean(Y3,'omitnan');

    line([xc(r)+offsets(1)-0.12, xc(r)+offsets(1)+0.12], [mu1 mu1], 'Color','k', 'LineWidth', 3);
    line([xc(r)+offsets(2)-0.12, xc(r)+offsets(2)+0.12], [mu2 mu2], 'Color','k', 'LineWidth', 3);
    line([xc(r)+offsets(3)-0.12, xc(r)+offsets(3)+0.12], [mu3 mu3], 'Color','k', 'LineWidth', 3);

    allY = [allY; Y1; Y2; Y3];
end

hL1 = scatter(NaN,NaN,opt.MarkerSize,col1,'filled');
hL2 = scatter(NaN,NaN,opt.MarkerSize,col2,'filled');
hL3 = scatter(NaN,NaN,opt.MarkerSize,col3,'filled');
legend([hL1 hL2 hL3], {'Load 1','Load 2','Load 3'}, 'Location','best','Box','off');

xLabs = cell(1,R);
for r = 1:R
    key = char(uniqRegs(r));
    if isKey(acronym, key), xLabs{r} = acronym(key);
    else, xLabs{r} = upper(key);
    end
end
set(gca,'XTick',xc,'XTickLabel',xLabs,'FontSize',opt.FontSize,'Box','off');
xlabel('Brain Region', 'FontSize', opt.FontSize);
ylabel('Z-scored rate', 'FontSize', opt.FontSize);
yline(0,'--','Color',[0.6 0.6 0.6], 'LineWidth',0.8);
xlim([0.5 R+0.5]);
yPad = 0.08 * max(eps, max(allY)-min(allY));
ylim([min(allY)-yPad, max(allY)+2*yPad]);
hold off;

out = struct('regions',{uniqRegs}, 'xlabels',{xLabs}, 'perRegion',perReg, ...
             'figure',gcf, 'measure',measure, 'figSize',[opt.FigWidth opt.FigHeight]);
end


function v = padToLength(v, L)
v = v(:);
if isempty(L)
    return;
end
if numel(v) < L
    v(end+1:L,1) = NaN;
end
end
