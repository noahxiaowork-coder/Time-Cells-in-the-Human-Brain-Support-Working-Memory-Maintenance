function showTimeCellAverages_loadMatched(nwbAll, all_units, neural_data_file, bin_size)

load(neural_data_file, 'neural_data');

duration  = 2.5;
psth_bins = 0:bin_size:duration;
nBins     = numel(psth_bins) - 1;

gaussian_kernel = GaussianKernal(0.3 / bin_size, 1.5);

N = numel(neural_data);

all_timefieldFR_correct   = zeros(N,1);
all_timefieldFR_incorrect = zeros(N,1);
valid_timefield_count     = 0;

rng(20250710);

for ndx = 1:N
    nd = neural_data(ndx);

    unit_match = ([all_units.subject_id] == nd.patient_id) & ...
                 ([all_units.unit_id]    == nd.unit_id);
    if ~any(unit_match)
        continue;
    end
    SU = all_units(unit_match);

    sess     = nwbAll{SU.session_count};
    tsMaint  = sess.intervals_trials.vectordata.get('timestamps_Maintenance').data.load();
    spike_ts = SU.spike_times;

    corr_vals = nd.trial_correctness(:);
    if ~isfield(nd,'trial_load')
        error('neural_data(%d) has no field trial_load.', ndx);
    end
    load_vals = nd.trial_load(:);

    nTrials = numel(tsMaint);
    if numel(corr_vals) ~= nTrials || numel(load_vals) ~= nTrials
        continue;
    end

    inc_idx  = (corr_vals == 0);
    corr_idx = (corr_vals == 1);

    if ~any(inc_idx)
        sel_corr = [];
    else
        sel_corr = [];
        for L = [1 2 3]
            inc_L  = find( inc_idx & (load_vals == L));
            if isempty(inc_L), continue; end
            nInc_L = numel(inc_L);

            corr_L = find(corr_idx & (load_vals == L));
            if isempty(corr_L)
                globC = find(corr_idx);
                if isempty(globC)
                    sel_corr = [];
                    break;
                end
                pickL = randsample(globC, nInc_L, true);
            elseif numel(corr_L) >= nInc_L
                pickL = randsample(corr_L, nInc_L, false);
            else
                need  = nInc_L - numel(corr_L);
                topup = randsample(corr_L, need, true);
                pickL = [corr_L; topup];
            end
            sel_corr = [sel_corr; pickL(:)];
        end
        sel_corr = sel_corr(:);
    end

    inc_ids = find(inc_idx);
    firing_incorrect = zeros(numel(inc_ids), nBins);
    for k = 1:numel(inc_ids)
        t0 = tsMaint(inc_ids(k));
        spikes = spike_ts(spike_ts >= t0 & spike_ts < t0 + duration) - t0;
        counts = histcounts(spikes, psth_bins);
        firing_incorrect(k,:) = conv(counts, gaussian_kernel, 'same') / bin_size;
    end

    if isempty(sel_corr)
        firing_correct_ds = nan(0, nBins);
    else
        firing_correct_ds = zeros(numel(sel_corr), nBins);
        for k = 1:numel(sel_corr)
            t0 = tsMaint(sel_corr(k));
            spikes = spike_ts(spike_ts >= t0 & spike_ts < t0 + duration) - t0;
            counts = histcounts(spikes, psth_bins);
            firing_correct_ds(k,:) = conv(counts, gaussian_kernel, 'same') / bin_size;
        end
    end

    combined = [firing_correct_ds; firing_incorrect];
    muC = mean(combined(:));
    sdC = std(combined(:));
    if sdC==0 || ~isfinite(sdC), sdC = 1; end
    firing_correct_ds = (firing_correct_ds - muC) / sdC;
    firing_incorrect  = (firing_incorrect  - muC) / sdC;

    tf_idx   = nd.time_field;
    tf_start = (tf_idx - 1) * 0.1;
    tf_end   = tf_idx * 0.1;

    bin_tf_start = find(psth_bins >= tf_start, 1, 'first');
    bin_tf_end   = find(psth_bins >  tf_end,   1, 'first') - 1;

    if ~isempty(bin_tf_start) && ~isempty(bin_tf_end) && bin_tf_end >= bin_tf_start
        valid_timefield_count = valid_timefield_count + 1;
        c_vals = firing_correct_ds(:, bin_tf_start:bin_tf_end);
        i_vals = firing_incorrect(:,    bin_tf_start:bin_tf_end);
        all_timefieldFR_correct(valid_timefield_count)   = mean(mean(c_vals,2), 'omitnan');
        all_timefieldFR_incorrect(valid_timefield_count) = mean(mean(i_vals,2), 'omitnan');
    end
end

all_timefieldFR_correct   = all_timefieldFR_correct(1:valid_timefield_count);
all_timefieldFR_incorrect = all_timefieldFR_incorrect(1:valid_timefield_count);

figure('Name','Time-Field Firing','Position',[140,120,500,500]);
pairedSwarmWithCenterLines([1 2], all_timefieldFR_correct, all_timefieldFR_incorrect, ...
    'Time Field (0.1 s) — Load-matched Correct', ...
    'Z-scored Rate in Time Field', true);

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


function drawMeanTicks(x, m, sem, dx, lwMean, lwSem)
line([x-dx, x+dx],[m m],'Color','k','LineWidth',lwMean);
end
