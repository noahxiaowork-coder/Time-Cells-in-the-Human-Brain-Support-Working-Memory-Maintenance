function out = showTimeCellAverages_ByRegion( ...
        nwbAll, all_units, neural_data_file, bin_size, useZscore, varargin)

phi = (1+sqrt(5))/2;
p = inputParser;
p.addParameter('Measure','TimeField', @(s)ischar(s)||isstring(s));
p.addParameter('FigHeight', 648, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('FigWidth',  [],  @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('FontSize',   20, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('MarkerSize', 20, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('Regions',   {}, @(c)iscellstr(c) || isstring(c));
p.addParameter('Jitter',   0.10, @(x)isnumeric(x)&&isscalar(x)&&x>=0);
p.addParameter('MinUnits',    3,  @(x)isnumeric(x)&&isscalar(x)&&x>=0);
p.addParameter('AcronymMap', defaultAcronymMap(), @(m)isAcronymMap(m));
p.parse(varargin{:});
opt = p.Results;
if isempty(opt.FigWidth), opt.FigWidth = round(opt.FigHeight*phi); end
measure = lower(string(opt.Measure));

S = load(neural_data_file, 'neural_data');
neural_data = S.neural_data;

duration  = 2.5;
edges     = 0:bin_size:duration;
nBins     = numel(edges)-1;
gaussian_kernel = GaussianKernal(0.3 / bin_size, 1.5);

normRegion = @(s) regexprep(lower(string(s)), '_(left|right)$', '');
allRegions = arrayfun(@(x) normRegion(x.brain_region), neural_data, 'uni', true);

keepIdx = ~startsWith(allRegions,"ventral",'IgnoreCase',true);
neural_data = neural_data(keepIdx);
allRegions  = allRegions(keepIdx);

uniqRegions = unique(allRegions,'stable');

if ~isempty(opt.Regions)
    want = string(lower(strtrim(opt.Regions)));
    uniqRegions = uniqRegions(ismember(uniqRegions, want));
    if isempty(uniqRegions)
        error('None of the requested regions found after laterality stripping and ventral exclusion.');
    end
end

R = numel(uniqRegions);
perReg = struct('region',[],'Yc',[],'Yi',[],'nUnits',0);
for r = 1:R
    perReg(r).region = char(uniqRegions(r));
end

for ndx = 1:numel(neural_data)
    reg = normRegion(neural_data(ndx).brain_region);
    ridx = find(uniqRegions == reg, 1);
    if isempty(ridx), continue; end

    pid = neural_data(ndx).patient_id;
    uid = neural_data(ndx).unit_id;
    corr = neural_data(ndx).trial_correctness;
    tf_bin = neural_data(ndx).time_field;

    m = ([all_units.subject_id]==pid) & ([all_units.unit_id]==uid);
    if ~any(m), continue; end
    SU = all_units(m);

    tsMaint = nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Maintenance').data.load();
    nT = numel(tsMaint); if nT==0, continue; end

    X = zeros(nT, nBins);
    for k = 1:nT
        s0 = tsMaint(k); s1 = s0 + duration;
        tr = SU.spike_times(SU.spike_times >= s0 & SU.spike_times < s1) - s0;
        counts = histcounts(tr, edges);
        X(k,:) = conv(counts, gaussian_kernel, 'same') / bin_size;
    end

    if useZscore
        muC = mean(X(:)); sdC = std(X(:));
        if sdC==0 || ~isfinite(sdC), sdC=1; end
        X = (X - muC) / sdC;
    end

    idxC = find(corr==1); idxI = find(corr==0);
    XC = X(idxC,:); XI = X(idxI,:);

    switch measure
        case "extendedtf"
            if tf_bin>1 && tf_bin<nBins
                yC = mean(mean(XC(:, tf_bin-1:tf_bin+1),2), 'omitnan');
                yI = mean(mean(XI(:, tf_bin-1:tf_bin+1),2), 'omitnan');
            else, continue;
            end
        case "timefield"
            if tf_bin>=1 && tf_bin<=nBins
                yC = mean(XC(:, tf_bin), 'omitnan');
                yI = mean(XI(:, tf_bin), 'omitnan');
            else, continue;
            end
        otherwise
            yC = mean(mean(XC,2), 'omitnan');
            yI = mean(mean(XI,2), 'omitnan');
    end

    if isfinite(yC) && isfinite(yI)
        perReg(ridx).Yc(end+1,1) = yC;
        perReg(ridx).Yi(end+1,1) = yI;
        perReg(ridx).nUnits = perReg(ridx).nUnits + 1;
    end
end

keepR = arrayfun(@(s) s.nUnits >= opt.MinUnits, perReg);
perReg = perReg(keepR);
uniqRegions = uniqRegions(keepR);
R = numel(uniqRegions);

blue = [0.20 0.45 0.95];
red  = [0.95 0.25 0.25];
grey = [0.70 0.70 0.70];

hFig = figure('Color','w','Units','pixels', ...
              'Position',[100 100 opt.FigWidth opt.FigHeight]);
hold on;

dx   = 0.18;
jitW = opt.Jitter;
xCenters = 1:R;
allY = [];

for r = 1:R
    Y1 = perReg(r).Yc;
    Y2 = perReg(r).Yi;

    nPairs = min(numel(Y1), numel(Y2));
    if nPairs > 0
        j = (rand(nPairs,1)-0.5)*jitW;
        xL = (xCenters(r) - dx) + j;
        xR = (xCenters(r) + dx) + j;
        ymid = (Y1(1:nPairs) + Y2(1:nPairs))/2;
        for i=1:nPairs
            plot([xL(i) xCenters(r) xR(i)], [Y1(i) ymid(i) Y2(i)], '-', ...
                 'Color', grey, 'LineWidth', 0.8, 'HandleVisibility','off');
        end
        scatter(xL, Y1(1:nPairs), opt.MarkerSize, blue, 'filled', 'MarkerFaceAlpha',0.9);
        scatter(xR, Y2(1:nPairs), opt.MarkerSize, red,  'filled', 'MarkerFaceAlpha',0.9);
    end

    if numel(Y1) > nPairs
        j = (rand(numel(Y1)-nPairs,1)-0.5)*jitW;
        scatter((xCenters(r)-dx)+j, Y1(nPairs+1:end), opt.MarkerSize, blue, 'filled', 'MarkerFaceAlpha',0.9);
    end
    if numel(Y2) > nPairs
        j = (rand(numel(Y2)-nPairs,1)-0.5)*jitW;
        scatter((xCenters(r)+dx)+j, Y2(nPairs+1:end), opt.MarkerSize, red,  'filled', 'MarkerFaceAlpha',0.9);
    end

    m1 = mean(Y1,'omitnan'); m2 = mean(Y2,'omitnan');
    drawMeanTick(xCenters(r)-dx, m1);
    drawMeanTick(xCenters(r)+dx, m2);

    p = NaN;
    if numel(Y1)==numel(Y2) && numel(Y1)>=2
        try, [~,p]=ttest(Y1, Y2); catch, p=NaN; end
    end
    star = p2star(p);
    yTop = max([Y1; Y2], [], 'omitnan');
    if isempty(yTop), yTop = max([m1 m2]); end
    if ~isempty(yTop) && isfinite(yTop)
        pad = 0.06 * max(eps, range([Y1;Y2]));
        text(xCenters(r), max([m1 m2 yTop])+pad, star, 'HorizontalAlignment','center', ...
             'VerticalAlignment','bottom', 'FontSize', opt.FontSize-2);
    end

    allY = [allY; Y1; Y2];
end

xlim([0.5 R+0.5]);
if isempty(allY), ylim([0 1]); else
    yPad = 0.08 * max(eps, max(allY)-min(allY));
    ylim([min(allY)-yPad, max(allY)+2*yPad]);
end

acLabs = mapAcronyms(uniqRegions, opt.AcronymMap);
set(gca, 'XTick', xCenters, 'XTickLabel', acLabs, ...
         'FontSize', opt.FontSize, 'Box','off');
xtickangle(0);
xlabel('Brain Region', 'FontSize', opt.FontSize);

if useZscore
    ylabel('Z-scored rate', 'FontSize', opt.FontSize);
    yline(0,'--','Color',[0.6 0.6 0.6], 'LineWidth',0.8, 'HandleVisibility','off');
else
    ylabel('Rate (Hz)', 'FontSize', opt.FontSize);
end

title(sprintf('Correct vs Incorrect by Region — %s', measureTitle(measure)), 'FontSize', opt.FontSize);
hold off;

out = struct('regions',{uniqRegions}, 'perRegion',perReg, 'figure',hFig, 'measure',measure);

end

function drawMeanTick(x, m)
    dx = 0.15; line([x-dx, x+dx],[m m],'Color','k','LineWidth',3);
end
function s = p2star(p)
    if ~isfinite(p), s='n.s.'; return; end
    if p < 1e-3, s='***';
    elseif p < 1e-2, s='**';
    elseif p < 5e-2, s='*';
    else, s='n.s.';
    end
end

function ok = isAcronymMap(m)
    ok = isa(m,'containers.Map') || isstruct(m) || istable(m);
end
function M = defaultAcronymMap()
    keys = {'dorsal_anterior_cingulate_cortex','pre_supplementary_motor_area', ...
            'hippocampus','amygdala','ventral_medial_prefrontal_cortex'};
    vals = {'dACC','pre-SMA','HPC','AMY','vmPFC'};
    M = containers.Map(keys, vals);
end
function labs = mapAcronyms(uniqRegions, mapObj)
    regs = string(uniqRegions);
    labs = cell(size(regs));
    if isa(mapObj,'containers.Map')
        M = containers.Map(lower(string(mapObj.keys)), values(mapObj));
    elseif isstruct(mapObj)
        f = fieldnames(mapObj);
        M = containers.Map(lower(string(f)), struct2cell(mapObj));
    elseif istable(mapObj) && all(ismember({'region','acronym'}, string(mapObj.Properties.VariableNames)))
        M = containers.Map(lower(string(mapObj.region)), cellstr(string(mapObj.acronym)));
    else
        M = defaultAcronymMap();
    end
    for i = 1:numel(regs)
        key = char(lower(regs(i)));
        if isKey(M, key)
            labs{i} = string(M(key));
        else
            t = regexprep(regs(i),'(^|_)([a-z])','${upper($2)}');
            labs{i} = strrep(t,'_',' ');
        end
    end
end

function t = measureTitle(m)
    switch string(m)
        case "extendedtf", t='Extended TF (±0.1 s)';
        case "timefield",  t='Time Field (0.1 s)';
        otherwise,         t='Maintenance (0–2.5 s)';
    end
end
