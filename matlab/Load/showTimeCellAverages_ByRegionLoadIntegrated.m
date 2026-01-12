function out = showTimeCellAverages_ByRegionLoadIntegrated( ...
        nwbAll, all_units, neural_data_file, bin_size, useZscore, varargin)
% Same pipeline as showTimeCellAverages_ByRegionIntegrated, but grouping
% trials by trial_load (1,2,3) instead of trial_correctness (0/1).
%
% Added option:
%   'UseOnlyCorrect' (default: false)
%       If true, restrict to trials with neural_data(ndx).trial_correctness == 1
%
% For each neuron, compute a scalar per-load value (Load 1/2/3) using the
% same Measure definition ("TimeField", "ExtendedTF", or "Maintenance"),
% then aggregate by region and plot Load 1 vs Load 2 vs Load 3.

% ---- options ----
p = inputParser;
p.addParameter('Measure','TimeField', @(s)ischar(s)||isstring(s));  % 'ExtendedTF'|'TimeField'|'Maintenance'
p.addParameter('FigWidth',  round(648*1.618), @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('FigHeight', 648, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('FontSize',   20, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('MarkerSize', 20, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('Jitter',    0.10, @(x)isnumeric(x)&&isscalar(x)&&x>=0);
p.addParameter('ShowSEM',  false, @(b)islogical(b)&&isscalar(b));
p.addParameter('ExcludeVentral', true, @(b)islogical(b)&&isscalar(b));
p.addParameter('MinUnits',  0, @(x)isnumeric(x)&&isscalar(x)&&x>=0);

% NEW toggle: use only correct trials (trial_correctness == 1)
p.addParameter('UseOnlyCorrect', false, @(b)islogical(b)&&isscalar(b));

p.parse(varargin{:});
opt = p.Results;
measure = lower(string(opt.Measure));

% ---- acronym map for x-axis ----
acronym = containers.Map( ...
    {'hippocampus','amygdala','dorsal_anterior_cingulate_cortex', ...
     'pre_supplementary_motor_area','ventral_medial_prefrontal_cortex'}, ...
    {'HPC','AMY','DaCC','PSMA','vmPFC'});

% ---- load neural_data ----
S = load(neural_data_file, 'neural_data');
neural_data = S.neural_data;

hasCorrectField = isfield(neural_data, 'trial_correctness');

if opt.UseOnlyCorrect && ~hasCorrectField
    warning('UseOnlyCorrect=true but neural_data has no field trial_correctness. Using all trials instead.');
end

% ---- constants: EXACTLY as in your original function ----
duration  = 2.5;
psth_bins = 0:bin_size:duration;
nBins     = numel(psth_bins)-1;
gaussian_kernel = GaussianKernal(0.3 / bin_size, 1.5);

% helper: strip only a trailing laterality suffix
stripLat = @(s) regexprep(lower(string(s)), '_(left|right)$', '');

% ---- compute per-neuron yL1/yL2/yL3 ONCE ----
vals = struct('region',{},'y1',{},'y2',{},'y3',{});
for ndx = 1:numel(neural_data)
    reg = stripLat(neural_data(ndx).brain_region);
    if opt.ExcludeVentral && startsWith(reg,"ventral",'IgnoreCase',true), continue; end

    pid = neural_data(ndx).patient_id;
    uid = neural_data(ndx).unit_id;
    loadVec = neural_data(ndx).trial_load;   % 1,2,3

    % NEW: correctness vector (1 = correct, 0 = incorrect)
    if hasCorrectField
        correctVec = neural_data(ndx).trial_correctness;
    else
        correctVec = [];
    end

    % basic sanity for lengths
    nTrials = numel(loadVec);
    if ~isempty(correctVec) && numel(correctVec) ~= nTrials
        warning('Unit (patient %d, unit %d): trial_correctness length mismatch with trial_load. Ignoring correctness for this unit.', ...
            pid, uid);
        correctVec = [];
    end

    % build trial mask
    if opt.UseOnlyCorrect && ~isempty(correctVec)
        trialMask = (correctVec == 1);  % keep only correct trials
    else
        trialMask = true(size(loadVec));
    end

    tf      = neural_data(ndx).time_field;   % 1-based (0.1 s)

    % match unit
    m = ([all_units.subject_id]==pid) & ([all_units.unit_id]==uid);
    if ~any(m), continue; end
    SU = all_units(m);

    tsMaint = nwbAll{SU.session_count}.intervals_trials.vectordata ...
              .get('timestamps_Maintenance').data.load();
    spk = SU.spike_times;

    % indices for each load, AFTER dropping incorrect trials
    idxL1 = find(loadVec==1 & trialMask);
    idxL2 = find(loadVec==2 & trialMask);
    idxL3 = find(loadVec==3 & trialMask);

    firingL1 = zeros(numel(idxL1), nBins);
    firingL2 = zeros(numel(idxL2), nBins);
    firingL3 = zeros(numel(idxL3), nBins);

    % build trials×bins firing (Hz) for each load, SAME smoothing
    for iT = 1:numel(idxL1)
        t  = idxL1(iT);
        if t > numel(tsMaint), continue; end  % safety
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

    % ORIGINAL global z across all trials×bins for THIS neuron (now across loads)
    if useZscore
        combined = [];
        if ~isempty(firingL1), combined = [combined; firingL1]; end %#ok<AGROW>
        if ~isempty(firingL2), combined = [combined; firingL2]; end %#ok<AGROW>
        if ~isempty(firingL3), combined = [combined; firingL3]; end %#ok<AGROW>
        if ~isempty(combined)
            muC = mean(combined(:));
            sdC = std(combined(:));
            if sdC==0 || ~isfinite(sdC), sdC = 1; end
            if ~isempty(firingL1), firingL1 = (firingL1 - muC) / sdC; end
            if ~isempty(firingL2), firingL2 = (firingL2 - muC) / sdC; end
            if ~isempty(firingL3), firingL3 = (firingL3 - muC) / sdC; end
        end
    end

    % scalar per-neuron values – EXACT SAME definitions as before
    y1 = NaN; y2 = NaN; y3 = NaN;
    switch measure
        case "extendedtf"   % +/- 0.1 s (skip edges)
            tf_start = (tf-1)*0.1; tf_end = tf*0.1;
            ext_start = tf_start - 0.1; ext_end = tf_end + 0.1;
            if ext_start < 0 || ext_end > 2.5, continue; end
            b0 = find(psth_bins >= ext_start, 1, 'first');
            b1 = find(psth_bins >  ext_end,   1, 'first') - 1;
            if isempty(b0) || isempty(b1) || b1<b0, continue; end
            if ~isempty(firingL1), y1 = mean(mean(firingL1(:,b0:b1),2)); end
            if ~isempty(firingL2), y2 = mean(mean(firingL2(:,b0:b1),2)); end
            if ~isempty(firingL3), y3 = mean(mean(firingL3(:,b0:b1),2)); end

        case "timefield"    % exact 0.1 s window
            tf_start = (tf-1)*0.1; tf_end = tf*0.1;
            b0 = find(psth_bins >= tf_start, 1, 'first');
            b1 = find(psth_bins >  tf_end,   1, 'first') - 1;
            if isempty(b0) || isempty(b1) || b1<b0, continue; end
            if ~isempty(firingL1), y1 = mean(mean(firingL1(:,b0:b1),2)); end
            if ~isempty(firingL2), y2 = mean(mean(firingL2(:,b0:b1),2)); end
            if ~isempty(firingL3), y3 = mean(mean(firingL3(:,b0:b1),2)); end

        otherwise           % "maintenance": mean over 0–2.5 s
            if ~isempty(firingL1), y1 = mean(mean(firingL1,2)); end
            if ~isempty(firingL2), y2 = mean(mean(firingL2,2)); end
            if ~isempty(firingL3), y3 = mean(mean(firingL3,2)); end
    end

    if any(isfinite([y1 y2 y3]))
        vals(end+1).region = char(reg); %#ok<AGROW>
        vals(end).y1 = y1;
        vals(end).y2 = y2;
        vals(end).y3 = y3;
    end
end

if isempty(vals)
    warning('No data to plot after filtering.');
    out = struct();
    return;
end

% ---- group by region (post-compute) ----
allRegs = string({vals.region});
uniqRegs = unique(allRegs, 'stable');

% drop tiny regions if requested
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

% ---- plot (single axis; Load 1/2/3 per region) ----
col1 = [0.20 0.45 0.95];  % Load 1: blue
col2 = [0.30 0.75 0.30];  % Load 2: green
col3 = [0.95 0.25 0.25];  % Load 3: red
grey = [0.70 0.70 0.70];

hFig = figure('Color','w','Units','pixels', ...
              'Position',[100 100 opt.FigWidth opt.FigHeight]);
hold on;

dx = 0.23;                                % offsets for 3 loads
offsets = [-dx, 0, dx];
jitW = opt.Jitter;
xc   = 1:R;
allY = [];

pVals = NaN(R,1);

for r = 1:R
    Y1 = perReg(r).Y1;
    Y2 = perReg(r).Y2;
    Y3 = perReg(r).Y3;

    % all three are same length (one value per neuron); keep as column
    Y1 = Y1(:); Y2 = Y2(:); Y3 = Y3(:);
    maxN = max([numel(Y1), numel(Y2), numel(Y3)]);
    if isempty(maxN) || maxN==0
        continue;
    end

    % OPTIONAL: if you kept padToLength, you can still call it here
    Y1 = padToLength(Y1, maxN);
    Y2 = padToLength(Y2, maxN);
    Y3 = padToLength(Y3, maxN);

    % same jitter used for both lines and dots
    j1 = (rand(maxN,1)-0.5)*jitW;
    j2 = (rand(maxN,1)-0.5)*jitW;
    j3 = (rand(maxN,1)-0.5)*jitW;

    x1 = xc(r) + offsets(1) + j1;
    x2 = xc(r) + offsets(2) + j2;
    x3 = xc(r) + offsets(3) + j3;

    % ---------- grey lines connecting each neuron across loads ----------
    for k = 1:maxN
        xk = [x1(k) x2(k) x3(k)];
        yk = [Y1(k) Y2(k) Y3(k)];
        m  = isfinite(yk);           % only connect where we have data

        if sum(m) >= 2
            idx = find(m);           % e.g. [1 2 3], or [1 3], or [2 3]
            for jj = 1:numel(idx)-1
                plot(xk(idx(jj:jj+1)), yk(idx(jj:jj+1)), '-', ...
                     'Color', grey, 'LineWidth', 0.8, 'HandleVisibility', 'off');
            end
        end
    end
    % ------------------------------------------------------------------------

    % scatter for each load
    scatter(x1, Y1, opt.MarkerSize, col1, 'filled', 'MarkerFaceAlpha',0.9);
    scatter(x2, Y2, opt.MarkerSize, col2, 'filled', 'MarkerFaceAlpha',0.9);
    scatter(x3, Y3, opt.MarkerSize, col3, 'filled', 'MarkerFaceAlpha',0.9);

    % means (bold black); optional SEM
    mu1 = mean(Y1,'omitnan');
    mu2 = mean(Y2,'omitnan');
    mu3 = mean(Y3,'omitnan');

    line([xc(r)+offsets(1)-0.12, xc(r)+offsets(1)+0.12], [mu1 mu1], 'Color','k', 'LineWidth', 3);
    line([xc(r)+offsets(2)-0.12, xc(r)+offsets(2)+0.12], [mu2 mu2], 'Color','k', 'LineWidth', 3);
    line([xc(r)+offsets(3)-0.12, xc(r)+offsets(3)+0.12], [mu3 mu3], 'Color','k', 'LineWidth', 3);

    if opt.ShowSEM
        se1 = std(Y1,'omitnan') / max(1,sqrt(sum(isfinite(Y1))));
        se2 = std(Y2,'omitnan') / max(1,sqrt(sum(isfinite(Y2))));
        se3 = std(Y3,'omitnan') / max(1,sqrt(sum(isfinite(Y3))));
        line([xc(r)+offsets(1)-0.08, xc(r)+offsets(1)+0.08], [mu1+se1 mu1+se1], 'Color','k', 'LineWidth', 1.6);
        line([xc(r)+offsets(1)-0.08, xc(r)+offsets(1)+0.08], [mu1-se1 mu1-se1], 'Color','k', 'LineWidth', 1.6);
        line([xc(r)+offsets(2)-0.08, xc(r)+offsets(2)+0.08], [mu2+se2 mu2+se2], 'Color','k', 'LineWidth', 1.6);
        line([xc(r)+offsets(2)-0.08, xc(r)+offsets(2)+0.08], [mu2-se2 mu2-se2], 'Color','k', 'LineWidth', 1.6);
        line([xc(r)+offsets(3)-0.08, xc(r)+offsets(3)+0.08], [mu3+se3 mu3+se3], 'Color','k', 'LineWidth', 1.6);
        line([xc(r)+offsets(3)-0.08, xc(r)+offsets(3)+0.08], [mu3-se3 mu3-se3], 'Color','k', 'LineWidth', 1.6);
    end

    % ---------- paired 2-tailed t-tests between loads ----------
    Ymat = [Y1 Y2 Y3];

    p12 = NaN; p23 = NaN; p13 = NaN;

    m12 = isfinite(Y1) & isfinite(Y2);
    if nnz(m12) >= 2
        try
            [~,p12] = ttest(Y1(m12), Y2(m12));  % Load 1 vs Load 2
        catch
            p12 = NaN;
        end
    end

    m23 = isfinite(Y2) & isfinite(Y3);
    if nnz(m23) >= 2
        try
            [~,p23] = ttest(Y2(m23), Y3(m23));  % Load 2 vs Load 3
        catch
            p23 = NaN;
        end
    end

    m13 = isfinite(Y1) & isfinite(Y3);
    if nnz(m13) >= 2
        try
            [~,p13] = ttest(Y1(m13), Y3(m13));  % Load 1 vs Load 3
        catch
            p13 = NaN;
        end
    end

    % store if you want them in the output
    pVals12(r,1) = p12;
    pVals23(r,1) = p23;
    pVals13(r,1) = p13;

    fprintf('Region %-20s | nUnits=%-3d | p12=%g | p23=%g | p13=%g\n', ...
        perReg(r).region, maxN, p12, p23, p13);

    % ---------- draw brackets + stars between each pair ----------
    yRegion = Ymat(isfinite(Ymat));
    if isempty(yRegion)
        yRegion = [mu1 mu2 mu3];
    end
    yBase = max(yRegion);
    yRange = range(yRegion);
    if ~isfinite(yRange) || yRange==0, yRange = 1; end
    step = 0.06 * yRange;

    % x-coordinates (centers of each load group, without jitter)
    xL1c = xc(r) + offsets(1);
    xL2c = xc(r) + offsets(2);
    xL3c = xc(r) + offsets(3);

    % Load1 vs Load2
    drawPairSigLine(xL1c, xL2c, yBase + step, p12, opt.FontSize);

    % Load2 vs Load3
    drawPairSigLine(xL2c, xL3c, yBase + 2*step, p23, opt.FontSize);

    % Load1 vs Load3 (above the other two)
    drawPairSigLine(xL1c, xL3c, yBase + 3*step, p13, opt.FontSize);

    allY = [allY; Y1; Y2; Y3]; %#ok<AGROW>
end

% legend (using dummy handles so we don't overwrite)
hL1 = scatter(NaN,NaN,opt.MarkerSize,col1,'filled');
hL2 = scatter(NaN,NaN,opt.MarkerSize,col2,'filled');
hL3 = scatter(NaN,NaN,opt.MarkerSize,col3,'filled');

leg = legend([hL1 hL2 hL3], {'Load 1','Load 2','Load 3'}, 'Location','best');
set(leg, 'Box','off', 'AutoUpdate','off');

% x labels (acronyms)
xLabs = cell(1,R);
for r = 1:R
    key = char(uniqRegs(r));
    if isKey(acronym, key), xLabs{r} = acronym(key);
    else, xLabs{r} = upper(key);
    end
end
set(gca,'XTick',xc,'XTickLabel',xLabs,'FontSize',opt.FontSize,'Box','off');
xlabel('Brain Region', 'FontSize', opt.FontSize);
if useZscore
    ylabel('Z-scored rate', 'FontSize', opt.FontSize);
    yline(0,'--','Color',[0.6 0.6 0.6], 'LineWidth',0.8);
else
    ylabel('Rate (Hz)', 'FontSize', opt.FontSize);
end
xlim([0.5 R+0.5]);
yPad = 0.08 * max(eps, max(allY)-min(allY));
ylim([min(allY)-yPad, max(allY)+2*yPad]);
hold off;

out = struct('regions',{uniqRegs}, 'xlabels',{xLabs}, 'perRegion',perReg, ...
             'figure',gcf, 'measure',measure, 'figSize',[opt.FigWidth opt.FigHeight], ...
             'pVals',pVals);
end

% ---- local helpers ----
function v = padToLength(v, L)
v = v(:);
if numel(v) < L
    v(end+1:L,1) = NaN;
end
end

function starStr = getStarString_local(pVal)
if ~isfinite(pVal), starStr = 'n.s.'; return; end
if pVal < 1e-3, starStr = '***';
elseif pVal < 1e-2, starStr = '**';
elseif pVal < 0.05, starStr = '*';
else, starStr = 'n.s.';
end
end

function t = measureTitle(m)
switch string(m)
    case "extendedtf", t='Time Field (±0.1 s)';
    case "timefield",  t='Time Field (0.1 s)';
    otherwise,         t='Maintenance (0–2.5 s)';
end
end

function drawPairSigLine(x1, x2, y, pVal, fontSize)
% Draws a little bracket + significance star between x1 and x2 at height y
if ~isfinite(pVal)
    return;
end
star = getStarString_local(pVal);
if strcmp(star,'n.s.')
    % If you prefer to only show significant, you can return here instead:
    % return;
end

% horizontal bracket
line([x1 x1 x2 x2], [y-0.01*y y y y-0.01*y], ...
     'Color','k', 'LineWidth', 1, 'HandleVisibility','off');

% star text
text(mean([x1 x2]), y + 0.01*y, star, ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','bottom', ...
    'FontSize', fontSize);
end
