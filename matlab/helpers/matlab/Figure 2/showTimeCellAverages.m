function showTimeCellAverages(nwbAll, all_units, neural_data_file, bin_size)

load(neural_data_file, 'neural_data');

duration  = 2.5;
psth_bins = 0:bin_size:duration;
nBins     = length(psth_bins) - 1;

gaussian_kernel = GaussianKernal(0.3 / bin_size, 1.5);

all_timefieldFR_correct   = zeros(length(neural_data), 1);
all_timefieldFR_incorrect = zeros(length(neural_data), 1);
valid_timefield_count     = 0;

for ndx = 1 : length(neural_data)
    patient_id        = neural_data(ndx).patient_id;
    unit_id           = neural_data(ndx).unit_id;
    trial_correctness = neural_data(ndx).trial_correctness;
    time_field        = neural_data(ndx).time_field;

    unit_match = ([all_units.subject_id] == patient_id) & ([all_units.unit_id] == unit_id);
    if ~any(unit_match)
        continue;
    end
    SU = all_units(unit_match);

    tsMaint     = nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Maintenance').data.load();
    spike_times = SU.spike_times;

    correct_idx   = find(trial_correctness == 1);
    incorrect_idx = find(trial_correctness == 0);

    firing_correct   = zeros(length(correct_idx), nBins);
    firing_incorrect = zeros(length(incorrect_idx), nBins);

    for iC = 1:length(correct_idx)
        tC = correct_idx(iC);
        trial_spikes = spike_times(spike_times >= tsMaint(tC) & spike_times < tsMaint(tC)+duration) - tsMaint(tC);
        spike_counts = histcounts(trial_spikes, psth_bins);
        firing_correct(iC,:) = conv(spike_counts, gaussian_kernel, 'same') / bin_size;
    end

    for iI = 1:length(incorrect_idx)
        tI = incorrect_idx(iI);
        trial_spikes = spike_times(spike_times >= tsMaint(tI) & spike_times < tsMaint(tI)+duration) - tsMaint(tI);
        spike_counts = histcounts(trial_spikes, psth_bins);
        firing_incorrect(iI,:) = conv(spike_counts, gaussian_kernel, 'same') / bin_size;
    end

    combined = [firing_correct; firing_incorrect];
    muC = mean(combined(:));
    sdC = std(combined(:));
    if sdC==0 || ~isfinite(sdC), sdC = 1; end
    firing_correct   = (firing_correct   - muC) / sdC;
    firing_incorrect = (firing_incorrect - muC) / sdC;

    tf_start  = (time_field - 1) * 0.1;
    tf_end    = time_field * 0.1;

    bin_tf_start = find(psth_bins >= tf_start, 1, 'first');
    bin_tf_end   = find(psth_bins >  tf_end,   1, 'first') - 1;

    if ~isempty(bin_tf_start) && ~isempty(bin_tf_end) && bin_tf_end >= bin_tf_start
        valid_timefield_count = valid_timefield_count + 1;
        c_vals = firing_correct(:, bin_tf_start:bin_tf_end);
        i_vals = firing_incorrect(:, bin_tf_start:bin_tf_end);
        all_timefieldFR_correct(valid_timefield_count)   = mean(mean(c_vals,2));
        all_timefieldFR_incorrect(valid_timefield_count) = mean(mean(i_vals,2));
    end
end

all_timefieldFR_correct   = all_timefieldFR_correct(1:valid_timefield_count);
all_timefieldFR_incorrect = all_timefieldFR_incorrect(1:valid_timefield_count);

fig = figure('Name','Time-Field Firing','Position',[140,120,500,500]);
pairedSwarmWithCenterLines([1 2], all_timefieldFR_correct, all_timefieldFR_incorrect, ...
                   'Time Field (0.1 s) — Grand Average', ...
                   'Z-scored Rate in Time Field', true);

outname = replace(neural_data_file, '.mat', '_TimeField_dotplot.pdf');
exportgraphics(fig, outname, 'ContentType','vector');

end


function pairedSwarmWithCenterLines(xcats, Y1, Y2, ttl, ylab, useZscore)

assert(isvector(Y1) && isvector(Y2) && numel(Y1)==numel(Y2));
Y1 = Y1(:); 
Y2 = Y2(:); 
n  = numel(Y1);

xL = xcats(1); 
xR = xcats(2); 
xC = mean(xcats);

m1 = mean(Y1,'omitnan'); 
m2 = mean(Y2,'omitnan');
s1 = std(Y1,'omitnan');  
s2 = std(Y2,'omitnan');
n1 = sum(~isnan(Y1));    
n2 = sum(~isnan(Y2));
sem1 = s1/sqrt(max(1,n1)); 
sem2 = s2/sqrt(max(1,n2));
[~, p] = ttest(Y1, Y2, 'Alpha',0.05,'Tail','both');
starStr = getStarString(p);

hold on;
set(gca,'FontSize',12,'XTick',xcats,'XTickLabel',{'Correct','Incorrect'});
xlabel('Condition'); 
ylabel(ylab); 
title(ttl);
box on;

ymid = (Y1 + Y2)/2;
for i = 1:n
    plot([xL xC xR], [Y1(i) ymid(i) Y2(i)], '-', ...
        'Color', [0.75 0.75 0.75], 'LineWidth', 0.5, ...
        'HandleVisibility','off');
end

haveSwarm = exist('swarmchart','file')==2;
msz = 20;
if haveSwarm
    sc1 = swarmchart(repmat(xL,n,1), Y1, msz, 'filled');
    sc1.MarkerFaceColor = [0.20 0.45 0.95]; 
    sc1.MarkerEdgeColor = 'none';
    sc1.MarkerFaceAlpha = 0.85; 
    sc1.XJitterWidth = 0.18; 
    sc1.XJitter = 'density';

    sc2 = swarmchart(repmat(xR,n,1), Y2, msz, 'filled');
    sc2.MarkerFaceColor = [0.95 0.25 0.25]; 
    sc2.MarkerEdgeColor = 'none';
    sc2.MarkerFaceAlpha = 0.85; 
    sc2.XJitterWidth = 0.18; 
    sc2.XJitter = 'density';
else
    jit = 0.12;
    scatter(xL + (rand(n,1)-0.5)*jit, Y1, msz, [0.20 0.45 0.95], 'filled', ...
            'MarkerFaceAlpha',0.85, 'MarkerEdgeColor','none');
    scatter(xR + (rand(n,1)-0.5)*jit, Y2, msz, [0.95 0.25 0.25], 'filled', ...
            'MarkerFaceAlpha',0.85, 'MarkerEdgeColor','none');
end

tickHalf = 0.18; 
lwMean = 2.5; 
lwSem  = 1.2;
drawMeanTicks(xL, m1, sem1, tickHalf, lwMean, lwSem);
drawMeanTicks(xR, m2, sem2, tickHalf, lwMean, lwSem);

yMax = max([Y1; Y2], [], 'omitnan');
yMin = min([Y1; Y2], [], 'omitnan');
pad  = 0.06 * max(eps, yMax - yMin);
ySig = max([m1+sem1, m2+sem2, yMax]) + pad;
plot([xL xR], [ySig ySig], 'k-', 'LineWidth', 1.5);
text(xC, ySig, starStr, 'HorizontalAlignment','center', ...
     'VerticalAlignment','bottom', 'FontSize', 14);

xlim([xL-0.6, xR+0.6]);
ylim([yMin - pad, ySig + pad]);

yline(0,'--','Color',[0.4 0.4 0.4], 'HandleVisibility','off');

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


function drawMeanTicks(x, m, sem, dx, lwMean, lwSem)
line([x-dx, x+dx],[m m],'Color','k','LineWidth',lwMean);
end
