function showTimeCellAverages(nwbAll, all_units, neural_data_file, bin_size, useZscore)
% SHOWTIMECELLAVERAGES_3FIGS_WINDOW
%   This function produces three separate figures:
%     1) Extended Time Field (+/- 0.1s) bar chart (correct vs. incorrect).
%        - Skips neurons whose time field is <0.1s from start or end of session.
%        - Optionally z-scores single-trial PSTHs before averaging.
%        - A bar chart with significance for correct vs. incorrect.
%     2) The existing Time Field (0.1s window) bar chart (correct vs. incorrect)
%        with improved star visibility.
%     3) Maintenance (0--2.5s) bar chart (correct vs. incorrect).
%
%   Also includes a significance bar in each bar chart.
%
%   INPUTS:
%     nwbAll           - Cell array of NWB objects or data referencing each session
%     all_units        - Struct array of single-unit data (fields: subject_id, unit_id, spike_times, etc.)
%     params           - (Unused, but can store extra analysis parameters)
%     neural_data_file - Path to .mat file containing 'neural_data'
%     field_width      - (Unused here, but might define the time field duration if needed)
%     useZscore        - Boolean; if true, z-score at the single-trial level
%
%   OUTPUT: Three separate figures:
%     (1) Extended time-field (+/-0.1s) bar chart
%     (2) Time-field bar chart
%     (3) Maintenance bar chart

%% Load data
load(neural_data_file, 'neural_data');

%% PSTH parameters   
duration  = 2.5;      % analyze 0--2.5 s
psth_bins = 0:bin_size:duration;
nBins     = length(psth_bins) - 1;

% %% Gaussian kernel (for smoothing)
% gaussian_sigma = 2;  % in bin units
% kernel_size    = round(5 * gaussian_sigma);
% x_kernel       = -kernel_size : kernel_size;
% gaussian_kernel = exp(-(x_kernel.^2)/(2*gaussian_sigma^2));
% gaussian_kernel = gaussian_kernel / sum(gaussian_kernel);

gaussian_kernel = GaussianKernal(0.3 / bin_size, 1.5);



% 
% gaussian_kernel = gaussian_kernel(:)';           % row vector
% if any(gaussian_kernel)
%     gaussian_kernel = gaussian_kernel / sum(gaussian_kernel);
% end
% 
% sigma_bins = 0.3 / bin_size;                     % σ in bins (0.3 s / bin_size)
% pad_bins   = ceil(3 * sigma_bins);               % ~3σ padding
% 
% 




%% Allocate arrays
% 1) Extended time-field (+/- 0.1s)
all_extTF_correct   = zeros(length(neural_data), 1);
all_extTF_incorrect = zeros(length(neural_data), 1);
valid_extTF_count   = 0;  % for how many neurons is the extended TF feasible?

% 2) The original time-field measure
all_timefieldFR_correct   = zeros(length(neural_data), 1);
all_timefieldFR_incorrect = zeros(length(neural_data), 1);

% 3) Maintenance measure
all_maintFR_correct   = zeros(length(neural_data), 1);
all_maintFR_incorrect = zeros(length(neural_data), 1);

% We'll also track how many neurons total we process for the latter two measures
valid_timefield_count = 0;
valid_maint_count     = 0;

%% Loop over each neuron
for ndx = 1 : length(neural_data)
    patient_id       = neural_data(ndx).patient_id;
    unit_id          = neural_data(ndx).unit_id;
    trial_correctness= neural_data(ndx).trial_correctness;
    time_field       = neural_data(ndx).time_field;  % 1-based index => 0.1s window

    % match single unit
    unit_match = ([all_units.subject_id] == patient_id) & ([all_units.unit_id] == unit_id);
    if ~any(unit_match)
        warning('Unit (patient_id=%d, unit_id=%d) not found. Skipping...', patient_id, unit_id);
        continue;
    end
    SU = all_units(unit_match);

    tsMaint     = nwbAll{SU.session_count}.intervals_trials.vectordata.get('timestamps_Maintenance').data.load();
    spike_times = SU.spike_times;

    correct_idx   = find(trial_correctness == 1);
    incorrect_idx = find(trial_correctness == 0);

    %% Bin & Smooth single trials for correct vs. incorrect
    firing_correct   = zeros(length(correct_idx), nBins);
    firing_incorrect = zeros(length(incorrect_idx), nBins);

    for iC = 1:length(correct_idx)
        tC = correct_idx(iC);
        trial_spikes = spike_times(spike_times >= tsMaint(tC) & spike_times < tsMaint(tC)+duration);
        trial_spikes = trial_spikes - tsMaint(tC);
        spike_counts = histcounts(trial_spikes, psth_bins);
        smoothed     = conv(spike_counts, gaussian_kernel, 'same') / bin_size;
        firing_correct(iC,:) = smoothed;
    end
    for iI = 1:length(incorrect_idx)
        tI = incorrect_idx(iI);
        trial_spikes = spike_times(spike_times >= tsMaint(tI) & spike_times < tsMaint(tI)+duration);
        trial_spikes = trial_spikes - tsMaint(tI);
        spike_counts = histcounts(trial_spikes, psth_bins);
        smoothed     = conv(spike_counts, gaussian_kernel, 'same') / bin_size;
        firing_incorrect(iI,:) = smoothed;
    end
    % 
    % for iC = 1:length(correct_idx)
    % tC = correct_idx(iC);
    % trial_spikes = spike_times(spike_times >= tsMaint(tC) & spike_times < tsMaint(tC)+duration);
    % trial_spikes = trial_spikes - tsMaint(tC);
    % spike_counts = histcounts(trial_spikes, psth_bins);
    % sc           = spike_counts(:)';          % row
    % 
    % if nBins > 2*pad_bins
    %     % mirror-pad in time
    %     sc_pad = [ fliplr(sc(1:pad_bins)), ...
    %                sc, ...
    %                fliplr(sc(end-pad_bins+1:end)) ];
    %     sm_pad   = conv(sc_pad, gaussian_kernel, 'same');
    %     smoothed = sm_pad(pad_bins+1 : pad_bins+nBins) / bin_size;
    % else
    %     % fallback if window is too short
    %     smoothed = conv(sc, gaussian_kernel, 'same') / bin_size;
    % end
    % 
    % firing_correct(iC,:) = smoothed;
    % end
    % 
    % 
    % for iI = 1:length(incorrect_idx)
    % tI = incorrect_idx(iI);
    % trial_spikes = spike_times(spike_times >= tsMaint(tI) & spike_times < tsMaint(tI)+duration);
    % trial_spikes = trial_spikes - tsMaint(tI);
    % spike_counts = histcounts(trial_spikes, psth_bins);
    % sc           = spike_counts(:)';
    % 
    % if nBins > 2*pad_bins
    %     sc_pad = [ fliplr(sc(1:pad_bins)), ...
    %                sc, ...
    %                fliplr(sc(end-pad_bins+1:end)) ];
    %     sm_pad   = conv(sc_pad, gaussian_kernel, 'same');
    %     smoothed = sm_pad(pad_bins+1 : pad_bins+nBins) / bin_size;
    % else
    %     smoothed = conv(sc, gaussian_kernel, 'same') / bin_size;
    % end
    % 
    % firing_incorrect(iI,:) = smoothed;
    % end




    %% Optional Z-score (across all trials for this neuron)
    if useZscore
        combined = [firing_correct; firing_incorrect];
        muC      = mean(combined(:));
        sdC      = std(combined(:));
        firing_correct   = (firing_correct   - muC) / sdC;
        firing_incorrect = (firing_incorrect - muC) / sdC;
    end

    % if useZscore
    %     firing_correct   = zscore(firing_correct, 0, 2);   % row-wise (trial-wise)
    %     firing_incorrect = zscore(firing_incorrect, 0, 2);
    % end


    % -----------
    % (1) Extended Time Field (+/- 0.1s)
    %     time_field: from tf_start to tf_end, each 0.1s wide.
    %     extended window => [tf_start - 0.1, tf_end + 0.1], total 0.3s.
    %     skip if tf_start < 0.1 or tf_end > 2.4 (since we can't do +/- 0.1s)
    tf_start = (time_field - 1) * 0.1;
    tf_end   = time_field * 0.1;
    ext_start = tf_start - 0.1;
    ext_end   = tf_end   + 0.1;

    if (ext_start < 0) || (ext_end > 2.5)
        % We skip this neuron for the extended TF measure
    else
        % Good neuron for extended window
        valid_extTF_count = valid_extTF_count + 1;

        % Find bin indices that lie in [ext_start, ext_end]
        bin_start = find(psth_bins >= ext_start, 1, 'first');
        bin_end   = find(psth_bins >  ext_end,   1, 'first') - 1;

        if isempty(bin_start) || isempty(bin_end) || bin_end < bin_start
            % weird boundary condition => skip
        else
            % average across those bins, then average across trials
            c_vals = firing_correct(:, bin_start:bin_end);     % (#correctTrials x #binsInWindow)
            i_vals = firing_incorrect(:, bin_start:bin_end);
            all_extTF_correct(valid_extTF_count)   = mean(mean(c_vals,2));
            all_extTF_incorrect(valid_extTF_count) = mean(mean(i_vals,2));
        end
    end

    % -----------
    % (2) Original Time Field measure (0.1s window)
    % We'll store it for *all* neurons that we can find bins for.
    bin_tf_start = find(psth_bins >= tf_start, 1, 'first');
    bin_tf_end   = find(psth_bins > tf_end,   1, 'first') - 1;


    if ~isempty(bin_tf_start) && ~isempty(bin_tf_end) && bin_tf_end >= bin_tf_start
        valid_timefield_count = valid_timefield_count + 1;
        c_vals = firing_correct(:, bin_tf_start:bin_tf_end);
        i_vals = firing_incorrect(:, bin_tf_start:bin_tf_end);
        all_timefieldFR_correct(valid_timefield_count)   = mean(mean(c_vals,2));
        all_timefieldFR_incorrect(valid_timefield_count) = mean(mean(i_vals,2));
    end

    % -----------
    % (3) Maintenance measure (0--2.5s)
    valid_maint_count = valid_maint_count + 1;
    c_vals_maint = mean(firing_correct, 2);   % average across all bins for each trial
    i_vals_maint = mean(firing_incorrect, 2);
    all_maintFR_correct(valid_maint_count)   = mean(c_vals_maint);
    all_maintFR_incorrect(valid_maint_count) = mean(i_vals_maint);

end % end for each neuron

%% Now truncate arrays to the valid_count for each measure
all_extTF_correct   = all_extTF_correct(1:valid_extTF_count);
all_extTF_incorrect = all_extTF_incorrect(1:valid_extTF_count);

all_timefieldFR_correct   = all_timefieldFR_correct(1:valid_timefield_count);
all_timefieldFR_incorrect = all_timefieldFR_incorrect(1:valid_timefield_count);

all_maintFR_correct   = all_maintFR_correct(1:valid_maint_count);
all_maintFR_incorrect = all_maintFR_incorrect(1:valid_maint_count);


%% ===================== FIGURE 1: Extended Time Field (+/- 0.1s) =====================
fig1 = figure('Name','Extended Time Field (+/-0.1s)','Position',[100,100,500,500]);

ylab = ternary(~useZscore, 'Extended TF Rate (Hz)', 'Extended TF (Z-score units)');
pairedSwarmWithCenterLines([1 2], all_extTF_correct, all_extTF_incorrect, ...
                   'Time Field \pm 0.1 s (0.3 s window)', ylab, useZscore);

%% ===================== FIGURE 2: Original Time Field (0.1s) =====================
fig2 = figure('Name','Time-Field Firing','Position',[140,120,500,500]);

ylab = ternary(~useZscore, 'Avg. Rate in Time Field (Hz)', 'Z-scored Rate in Time Field');
pairedSwarmWithCenterLines([1 2], all_timefieldFR_correct, all_timefieldFR_incorrect, ...
                   'Time Field (0.1 s) — Grand Average', ylab, useZscore);

% Save as PDF
outname = replace(neural_data_file, '.mat', '_TimeField_dotplot.pdf');
exportgraphics(fig2, outname, 'ContentType','vector');
fprintf(' Saved Time Field dot plot: %s\n', outname);

%% ===================== FIGURE 3: Maintenance (0--2.5 s) =====================
fig3 = figure('Name','Maintenance Firing','Position',[180,140,500,500]);

ylab = ternary(~useZscore, 'Avg. Rate (Hz)', 'Z-scored Rate');
pairedSwarmWithCenterLines([1 2], all_maintFR_correct, all_maintFR_incorrect, ...
                   'Maintenance (0–2.5 s) — Grand Average', ylab, useZscore);


end % function


%% Helper function: convert p-value to star string
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
% Pretty paired dot plot:
% - Dots as swarms at xcats(1) and xcats(2)
% - Paired lines routed through center x (xc = mean(xcats))
% - Mean ± SEM shown as three short horizontal ticks
% - Bracket + stars for paired t-test

assert(isvector(Y1) && isvector(Y2) && numel(Y1)==numel(Y2), ...
       'Y1 and Y2 must be vectors of equal length.');
Y1 = Y1(:); Y2 = Y2(:); n = numel(Y1);
xL = xcats(1); xR = xcats(2); xC = mean(xcats);

% ---- compute stats
m1 = mean(Y1,'omitnan'); m2 = mean(Y2,'omitnan');
s1 = std(Y1,'omitnan');  s2 = std(Y2,'omitnan');
n1 = sum(~isnan(Y1));    n2 = sum(~isnan(Y2));
sem1 = s1/sqrt(max(1,n1)); sem2 = s2/sqrt(max(1,n2));
[~, p] = ttest(Y1, Y2, 'Alpha',0.05,'Tail','both');
disp(p)
starStr = getStarString(p);


% ---- axes & look
hold on;
set(gca,'FontSize',12,'XTick',xcats,'XTickLabel',{'Correct','Incorrect'});
xlabel('Condition'); ylabel(ylab); title(ttl);
box on;

% ---- route paired lines via center, draw UNDER the points
% y-midpoint per pair for a gentle bend
ymid = (Y1 + Y2)/2;
for i = 1:n
    plot([xL xC xR], [Y1(i) ymid(i) Y2(i)], '-', ...
        'Color', [0.75 0.75 0.75], 'LineWidth', 0.5, ...
        'HandleVisibility','off');
end

% ---- dots as swarms (fallback to jitter if swarmchart not available)
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
    % fallback: light jitter
    jit = 0.12;
    scatter(xL + (rand(n,1)-0.5)*jit, Y1, msz, [0.20 0.45 0.95], 'filled', ...
            'MarkerFaceAlpha',0.85, 'MarkerEdgeColor','none');
    scatter(xR + (rand(n,1)-0.5)*jit, Y2, msz, [0.95 0.25 0.25], 'filled', ...
            'MarkerFaceAlpha',0.85, 'MarkerEdgeColor','none');
end

% ---- mean ± SEM ticks (three short horizontal bars)
tickHalf = 0.18; lwMean = 2.5; lwSem = 1.2;
drawMeanTicks(xL, m1, sem1, tickHalf, lwMean, lwSem);
drawMeanTicks(xR, m2, sem2, tickHalf, lwMean, lwSem);

% ---- bracket + stars
yMax = max([Y1; Y2], [], 'omitnan');
yMin = min([Y1; Y2], [], 'omitnan');
pad  = 0.06 * max(eps, yMax - yMin);
ySig = max([m1+sem1, m2+sem2, yMax]) + pad;
plot([xL xR], [ySig ySig], 'k-', 'LineWidth', 1.5);
text(xC, ySig, starStr, 'HorizontalAlignment','center', ...
     'VerticalAlignment','bottom', 'FontSize', 14);

% ---- tidy limits
xlim([xL-0.6, xR+0.6]);
ylim([yMin - pad, ySig + pad]);

% zero-line helps for z-scored plots
if useZscore
    yline(0,'--','Color',[0.4 0.4 0.4], 'HandleVisibility','off');
end
end


function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end
