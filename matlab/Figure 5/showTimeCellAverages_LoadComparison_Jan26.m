function showTimeCellAverages_LoadComparison(nwbAll, all_units, neural_data_file, bin_size, ~, ~, excludeAllNegZ)

if nargin < 7 || isempty(excludeAllNegZ)
    excludeAllNegZ = false;
end

load(neural_data_file, 'neural_data');

duration  = 2.5;
psth_bins = 0:bin_size:duration;
nBins     = numel(psth_bins) - 1;

gaussian_kernel = GaussianKernal(0.3 / bin_size, 1.5);

loadColors = [0 0 1;
              0 0.7 0;
              1 0 0];

for useZscore = [false true]
    for use_correct = [false true]

        nUnitsAll = length(neural_data);
        all_tf_load_raw = nan(nUnitsAll, 3);
        all_tf_load_z   = nan(nUnitsAll, 3);

        for ndx = 1:length(neural_data)
            patient_id        = neural_data(ndx).patient_id;
            unit_id           = neural_data(ndx).unit_id;
            trial_load        = neural_data(ndx).trial_load;
            trial_correctness = neural_data(ndx).trial_correctness;
            time_field        = neural_data(ndx).time_field;

            unit_match = ([all_units.subject_id] == patient_id) & ([all_units.unit_id] == unit_id);
            if ~any(unit_match)
                continue;
            end
            SU = all_units(unit_match);

            tsMaint     = nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Maintenance').data.load();
            spike_times = SU.spike_times;

            firing_by_load_raw = cell(1,3);
            for iL = 1:3
                if use_correct
                    these_trials = find(trial_load == iL & trial_correctness == 1);
                else
                    these_trials = find(trial_load == iL);
                end
                if isempty(these_trials), continue; end

                tmp_firing = zeros(length(these_trials), nBins);
                for iT = 1:length(these_trials)
                    tIdx   = these_trials(iT);
                    tStart = tsMaint(tIdx);
                    tr_sp  = spike_times(spike_times >= tStart & spike_times < (tStart + duration)) - tStart;
                    spike_counts = histcounts(tr_sp, psth_bins);
                    tmp_firing(iT,:) = conv(spike_counts, gaussian_kernel, 'same') / bin_size;
                end
                firing_by_load_raw{iL} = tmp_firing;
            end

            firing_by_load_z = firing_by_load_raw;
            combined_all = vertcat(firing_by_load_raw{:});
            if ~isempty(combined_all)
                muC = mean(combined_all(:));
                sdC = std(combined_all(:));
                if sdC == 0 || ~isfinite(sdC), sdC = 1; end
                for iL = 1:3
                    if ~isempty(firing_by_load_z{iL})
                        firing_by_load_z{iL} = (firing_by_load_z{iL} - muC) / sdC;
                    end
                end
            end

            tf_start = (time_field - 1) * 0.1;
            tf_end   =  time_field * 0.1;
            bin_tf_start = find(psth_bins >= tf_start, 1, 'first');
            bin_tf_end   = find(psth_bins > tf_end,  1, 'first') - 1;

            if ~isempty(bin_tf_start) && ~isempty(bin_tf_end) && bin_tf_end >= bin_tf_start
                for iL = 1:3
                    if ~isempty(firing_by_load_raw{iL})
                        trial_vals_raw = mean(firing_by_load_raw{iL}(:, bin_tf_start:bin_tf_end), 2, 'omitnan');
                        all_tf_load_raw(ndx, iL) = mean(trial_vals_raw, 'omitnan');
                    end
                    if ~isempty(firing_by_load_z{iL})
                        trial_vals_z = mean(firing_by_load_z{iL}(:, bin_tf_start:bin_tf_end), 2, 'omitnan');
                        all_tf_load_z(ndx, iL) = mean(trial_vals_z, 'omitnan');
                    end
                end
            end
        end

        if excludeAllNegZ
            haveAll3 = all(isfinite(all_tf_load_z), 2);
            allNeg   = all(all_tf_load_z < 0,  2);
            toDrop   = haveAll3 & allNeg;
            if any(toDrop)
                all_tf_load_raw(toDrop, :) = NaN;
                all_tf_load_z(toDrop, :)   = NaN;
            end
        end

        figTitle = sprintf('Figure 1 | useZscore=%d, use_correct=%d | exclAllNegZ=%d', ...
                           useZscore, use_correct, excludeAllNegZ);

        yLabelHz = 'Average Firing Rate in Time Field (Hz)';
        yLabelZ  = 'Z-score Rate in Time Field';

        dataToPlot = ifelse(useZscore, all_tf_load_z, all_tf_load_raw);

        dotPlotWithPairwise_OneSided_Conn(dataToPlot, useZscore, figTitle, yLabelHz, yLabelZ, loadColors);

    end
end
end


function dotPlotWithPairwise_OneSided_Conn(dataMatrix, useZscore, figTitle, yLabelHz, yLabelZ, loadColors, figPosition)

if ~exist('figPosition','var') || isempty(figPosition)
    figPosition = [100, 100, 560, 460];
end

figure('Name', figTitle, 'Position', figPosition);
hold on;

goodIdx = ~all(isnan(dataMatrix), 2);
vals    = dataMatrix(goodIdx, :);

set(gca,'XTick',1:3,'XTickLabel',{'Load 1','Load 2','Load 3'}, 'FontSize',16);
ylabel(ifelse(useZscore, yLabelZ, yLabelHz), 'FontSize',16);
title(figTitle, 'FontSize',16);
box on;

if isempty(vals)
    hold off; 
    return;
end

nUnits = size(vals,1);
for i = 1:nUnits
    x = [1 2 3];
    y = vals(i,:);
    mask = ~isnan(y);
    x = x(mask); y = y(mask);
    if numel(x) >= 2
        plot(x, y, '-', 'Color', [0.75 0.75 0.75], 'LineWidth', 0.6, 'HandleVisibility','off');
    end
end

haveSwarm = exist('swarmchart','file')==2;
msz = 28;
jitterWidth = 0.10;
for iL = 1:3
    col = loadColors(iL,:);
    y   = vals(:, iL);
    y   = y(~isnan(y));
    if isempty(y), continue; end

    if haveSwarm
        sc = swarmchart(ones(numel(y),1)*iL, y, msz, 'filled');
        sc.MarkerFaceColor = col;
        sc.MarkerEdgeColor = 'none';
        try, sc.MarkerFaceAlpha = 0.85; catch, end
        sc.XJitter = 'density';
        sc.XJitterWidth = jitterWidth;
    else
        x = iL + (rand(numel(y),1)-0.5)*2*jitterWidth;
        s = scatter(x, y, msz, 'filled');
        s.MarkerFaceColor = col;
        s.MarkerEdgeColor = 'none';
        try, s.MarkerFaceAlpha = 0.85; catch, end
    end
end

for iL = 1:3
    col = loadColors(iL,:);
    m = mean(vals(:, iL), 'omitnan');
    if ~isnan(m)
        plot([iL-0.18, iL+0.18], [m, m], '-', 'LineWidth', 2.2, 'Color', col);
    end
end

grid on;

pairs = { [1,2], 1.5; [2,3], 2.5; [1,3], 2.0; };
yMax  = max(vals, [], 'all', 'omitnan');
yMin  = min(vals, [], 'all', 'omitnan');
rangeY = max(1e-6, yMax - yMin);
offset = 0.08 * rangeY;

for iP = 1:size(pairs,1)
    loadA = pairs{iP,1}(1);
    loadB = pairs{iP,1}(2);
    xMid  = pairs{iP,2};
    A     = vals(:, loadA);
    B     = vals(:, loadB);

    if sum(~isnan(A) & ~isnan(B)) < 2, continue; end

    [~, pVal] = ttest(A, B, 'Alpha', 0.05, 'Tail', 'right');

    localMax = max([A; B], [], 'omitnan');
    if isempty(localMax) || ~isfinite(localMax), localMax = yMax; end
    yLevel = localMax + (1.0 + iP) * offset;

    plot([loadA, loadB], [yLevel, yLevel], 'k-', 'LineWidth', 1.5);
    text(xMid, yLevel, sprintf('%s\np=%.3g', getStarString(pVal), pVal), ...
        'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',14);
end

ylim([yMin - 0.06*rangeY, yMax + (size(pairs,1)+2)*offset]);

if useZscore
    yline(0,'--','Color',[0.4 0.4 0.4], 'HandleVisibility','off');
end

hold off;
end


function starStr = getStarString(pVal)
if pVal < 0.001
    starStr = '***';
elseif pVal < 0.01
    starStr = '**';
elseif pVal < 0.05
    starStr = '*';
else
    starStr = 'n.s.';
end
end


function out = ifelse(cond, valTrue, valFalse)
if cond
    out = valTrue;
else
    out = valFalse;
end
end
