function showTimeCellAverages_loadMatched(nwbAll, all_units, neural_data_file, bin_size, useZscore)

load(neural_data_file, 'neural_data');

duration  = 2.5;
psth_bins = 0:bin_size:duration;
nBins     = numel(psth_bins) - 1;

gaussian_kernel = GaussianKernal(0.3 / bin_size, 1.5);

N = numel(neural_data);
all_extTF_correct   = zeros(N,1);
all_extTF_incorrect = zeros(N,1);
valid_extTF_count   = 0;

all_timefieldFR_correct   = zeros(N,1);
all_timefieldFR_incorrect = zeros(N,1);
valid_timefield_count     = 0;

all_maintFR_correct   = zeros(N,1);
all_maintFR_incorrect = zeros(N,1);
valid_maint_count     = 0;

rng(20250710);

for ndx = 1:N
    nd = neural_data(ndx);

    unit_match = ([all_units.subject_id] == nd.patient_id) & ...
                 ([all_units.unit_id]    == nd.unit_id);
    if ~any(unit_match)
        warning('Unit (patient_id=%d, unit_id=%d) not found. Skipping...', nd.patient_id, nd.unit_id);
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
        warning('Trial count mismatch for neuron %d. Skipping...', ndx);
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
        t  = inc_ids(k);
        t0 = tsMaint(t);
        spikes = spike_ts(spike_ts >= t0 & spike_ts < t0 + duration) - t0;
        counts = histcounts(spikes, psth_bins);
        firing_incorrect(k,:) = conv(counts, gaussian_kernel, 'same') / bin_size;
    end

    if isempty(sel_corr)
        firing_correct_ds = nan(0, nBins);
    else
        firing_correct_ds = zeros(numel(sel_corr), nBins);
        for k = 1:numel(sel_corr)
            t  = sel_corr(k);
            t0 = tsMaint(t);
            spikes = spike_ts(spike_ts >= t0 & spike_ts < t0 + duration) - t0;
            counts = histcounts(spikes, psth_bins);
            firing_correct_ds(k,:) = conv(counts, gaussian_kernel, 'same') / bin_size;
        end
    end

    if useZscore
        if ~isempty(firing_correct_ds)
            firing_correct_ds = zscore(firing_correct_ds, 0, 2);
        end
        if ~isempty(firing_incorrect)
            firing_incorrect   = zscore(firing_incorrect, 0, 2);
        end
    end

    tf_idx   = nd.time_field;
    tf_start = (tf_idx - 1) * 0.1;
    tf_end   = tf_idx * 0.1;
    ext_start = tf_start - 0.1;
    ext_end   = tf_end   + 0.1;

    if ext_start >= 0 && ext_end <= duration
        valid_extTF_count = valid_extTF_count + 1;

        bin_start = find(psth_bins >= ext_start, 1, 'first');
        bin_end   = find(psth_bins >  ext_end,   1, 'first') - 1;

        if ~isempty(bin_start) && ~isempty(bin_end) && bin_end >= bin_start
            c_vals = firing_correct_ds(:, bin_start:bin_end);
            i_vals = firing_incorrect(:,    bin_start:bin_end);

            all_extTF_correct(valid_extTF_count)   = mean(mean(c_vals,2), 'omitnan');
            all_extTF_incorrect(valid_extTF_count) = mean(mean(i_vals,2), 'omitnan');
        else
            valid_extTF_count = valid_extTF_count - 1;
        end
    end

    bin_tf_start = find(psth_bins >= tf_start, 1, 'first');
    bin_tf_end   = find(psth_bins >  tf_end,   1, 'first') - 1;

    if ~isempty(bin_tf_start) && ~isempty(bin_tf_end) && bin_tf_end >= bin_tf_start
        valid_timefield_count = valid_timefield_count + 1;
        c_vals = firing_correct_ds(:, bin_tf_start:bin_tf_end);
        i_vals = firing_incorrect(:,    bin_tf_start:bin_tf_end);
        all_timefieldFR_correct(valid_timefield_count)   = mean(mean(c_vals,2), 'omitnan');
        all_timefieldFR_incorrect(valid_timefield_count) = mean(mean(i_vals,2), 'omitnan');
    end

    valid_maint_count = valid_maint_count + 1;
    all_maintFR_correct(valid_maint_count)   = mean(mean(firing_correct_ds,2), 'omitnan');
    all_maintFR_incorrect(valid_maint_count) = mean(mean(firing_incorrect, 2), 'omitnan');
end

all_extTF_correct   = all_extTF_correct(1:valid_extTF_count);
all_extTF_incorrect = all_extTF_incorrect(1:valid_extTF_count);

all_timefieldFR_correct   = all_timefieldFR_correct(1:valid_timefield_count);
all_timefieldFR_incorrect = all_timefieldFR_incorrect(1:valid_timefield_count);

all_maintFR_correct   = all_maintFR_correct(1:valid_maint_count);
all_maintFR_incorrect = all_maintFR_incorrect(1:valid_maint_count);

figure('Name','Extended Time Field (+/-0.1s)','Position',[100,100,500,500]);
ylab = ternary(~useZscore, 'Extended TF Rate (Hz)', 'Extended TF (Z-score units)');
pairedSwarmWithCenterLines([1 2], all_extTF_correct, all_extTF_incorrect, ...
    'Time Field \pm 0.1 s (0.3 s window) — Load-matched Correct', ylab, useZscore);

fig2 = figure('Name','Time-Field Firing','Position',[140,120,500,500]);
ylab = ternary(~useZscore, 'Avg. Rate in Time Field (Hz)', 'Z-scored Rate in Time Field');
pairedSwarmWithCenterLines([1 2], all_timefieldFR_correct, all_timefieldFR_incorrect, ...
    'Time Field (0.1 s) — Load-matched Correct', ylab, useZscore);

outname = replace(neural_data_file, '.mat', '_TimeField_dotplot_loadMatched.pdf');
exportgraphics(fig2, outname, 'ContentType','vector');
fprintf(' Saved Time Field dot plot (load-matched): %s\n', outname);

figure('Name','Maintenance Firing','Position',[180,140,500,500]);
ylab = ternary(~useZscore, 'Avg. Rate (Hz)', 'Z-scored Rate');
pairedSwarmWithCenterLines([1 2], all_maintFR_correct, all_maintFR_incorrect, ...
    'Maintenance (0–2.5 s) — Load-matched Correct', ylab, useZscore);

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

assert(isvector(Y1) && isvector(Y2) && numel(Y1)==numel(Y2), ...
       'Y1 and Y2 must be vectors of equal length.');
Y1 = Y1(:); Y2 = Y2(:); n = numel(Y1);
xL = xcats(1); xR = xcats(2); xC = mean(xcats);

m1 = mean(Y1,'omitnan'); m2 = mean(Y2,'omitnan');
s1 = std(Y1,'omitnan');  s2 = std(Y2,'omitnan');
n1 = sum(~isnan(Y1));    n2 = sum(~isnan(Y2));
sem1 = s1/sqrt(max(1,n1)); sem2 = s2/sqrt(max(1,n2));
[~, p] = ttest(Y1, Y2, 'Alpha',0.05,'Tail','both');
starStr = getStarString(p);
disp(p)

hold on;
set(gca,'FontSize',12,'XTick',xcats,'XTickLabel',{'Correct','Incorrect'});
xlabel('Condition'); ylabel(ylab); title(ttl);
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
    sc1 = swarmchart(repmat(xL,n,1), Y1, msz, 'filled'); hold on;
    sc1.MarkerFaceColor = [0.20 0.45 0.95]; sc1.MarkerEdgeColor = 'none';
    sc1.MarkerFaceAlpha = 0.85; sc1.XJitterWidth = 0.18; sc1.XJitter = 'density';

    sc2 = swarmchart(repmat(xR,n,1), Y2, msz, 'filled');
    sc2.MarkerFaceColor = [0.95 0.25 0.25]; sc2.MarkerEdgeColor = 'none';
    sc2.MarkerFaceAlpha = 0.85; sc2.XJitterWidth = 0.18; sc2.XJitter = 'density';
else
    jit = 0.12;
    scatter(xL + (rand(n,1)-0.5)*jit, Y1, msz, [0.20 0.45 0.95], 'filled', ...
            'MarkerFaceAlpha',0.85, 'MarkerEdgeColor','none');
    scatter(xR + (rand(n,1)-0.5)*jit, Y2, msz, [0.95 0.25 0.25], 'filled', ...
            'MarkerFaceAlpha',0.85, 'MarkerEdgeColor','none');
end

tickHalf = 0.18; lwMean = 2.5; lwSem = 1.2;
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

if useZscore
    yline(0,'--','Color',[0.4 0.4 0.4], 'HandleVisibility','off');
end
end


function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end

function drawMeanTicks(x, m, sem, dx, lwMean, lwSem)
line([x-dx, x+dx],[m m],'Color','k','LineWidth',lwMean);
end
