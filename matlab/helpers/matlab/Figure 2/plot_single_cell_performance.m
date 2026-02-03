function [p_value, is_valid] = plot_single_cell_performance(nwbAll, all_units, neural_data, ...
                                      bin_size, subject_id, unit_id, use_zscore)

% PLOT_SINGLE_CELL_PERFORMANCE
% Raster + dual PSTHs (correct vs. incorrect) for ONE unit.
% Pipeline:
%  1) per-trial rate histograms
%  2) smooth each trial (conv2, kernel sums to 1)
%  3) (optional) z-score each trial across time
%  4) mean across trials + SEM from the smoothed (+ z-scored) trials

if nargin < 7, use_zscore = false; end
p_value = NaN;
is_valid = false;
doplot = false;


% -------------------------------------------------------------------------
% FIND THIS UNIT 
ndx = find([neural_data.patient_id]==subject_id & ...
           [neural_data.unit_id]  ==unit_id, 1, 'first');
if isempty(ndx)
    warning('Unit %d/%d not found in neural_data.',subject_id,unit_id);
    return
end

m = ([all_units.subject_id]==subject_id & [all_units.unit_id]==unit_id);
if ~any(m)
    warning('Unit %d/%d not found in all_units.',subject_id,unit_id);
    return
end
SU     = all_units(m);
tf_bin = double(neural_data(ndx).time_field);
corr_v = neural_data(ndx).trial_correctness;

% -------------------------------------------------------------------------
% GET SPIKES & TRIALS
bins    = 0:bin_size:2.5;    
n_bins  = numel(bins)-1;
tsMaint = nwbAll{SU.session_count}.intervals_trials ...
                           .vectordata.get('timestamps_Maintenance') ...
                           .data.load();
spk     = SU.spike_times;
n_trials= numel(tsMaint);

% Gaussian kernel for smoothing (normalize to sum=1)
gauss_kernel = GaussianKernal(5, 2.0);

gauss_kernel = gauss_kernel(:)';          % row vector
if any(gauss_kernel)
    gauss_kernel = gauss_kernel / sum(gauss_kernel);
end

sigma_bins = 5;                           % this is your sigma in "bins"
pad_bins   = ceil(3 * sigma_bins);        % 3σ padding for edge safety


% -------------------------------------------------------------------------
% Time-field extent from pooled raw rates (unchanged)
fr_all = zeros(n_trials,n_bins);
aligned_spikes = cell(n_trials,1);  % also cache for raster
for t = 1:n_trials
    s = spk(spk>=tsMaint(t) & spk<tsMaint(t)+2.5) - tsMaint(t);
    aligned_spikes{t} = s;
    fr_all(t,:) = histcounts(s,bins) / bin_size;
end
mean_fr_all = mean(fr_all,1);
bl  = mean(mean_fr_all);
thr = bl + 0.5*std(mean_fr_all);

centre_t   = (tf_bin-0.5)*0.1;
[~,centre] = min(abs(bins(1:end-1)-centre_t));
left = centre;  while left>1       && mean_fr_all(left)>=thr,  left  = left-1; end
right= centre;  while right<n_bins && mean_fr_all(right)>=thr, right = right+1; end

field_start = (tf_bin-1)*0.1;
field_end   =  tf_bin   *0.1;
t_start = min(bins(left), field_start);
t_end   = max(bins(right), field_end);
if t_end <= t_start
    t_start = field_start;
    t_end   = field_end;
end

% -------------------------------------------------------------------------
% SPLIT correct vs incorrect
corr_idx = find(corr_v==1);
inc_idx  = find(corr_v==0);

% Per-trial histograms for each group
fr_corr = zeros(numel(corr_idx), n_bins);
fr_inc  = zeros(numel(inc_idx ), n_bins);
for ii = 1:numel(corr_idx)
    tr = corr_idx(ii);
    s  = aligned_spikes{tr};
    fr_corr(ii,:) = histcounts(s,bins) / bin_size;
end
for jj = 1:numel(inc_idx)
    tr = inc_idx(jj);
    s  = aligned_spikes{tr};
    fr_inc(jj,:) = histcounts(s,bins) / bin_size;
end

% -------------------------------------------------------------------------
% NEW PIPELINE STEPS
% 2) Smooth each trial across time (row-wise)
% 2) Smooth each trial across time (row-wise) WITH TEMPORAL PADDING
if ~isempty(fr_corr)
    fr_corr_pad = [ fliplr(fr_corr(:,1:pad_bins)) , ...
                    fr_corr , ...
                    fliplr(fr_corr(:,end-pad_bins+1:end)) ];
    sm_corr_pad  = conv2(fr_corr_pad, gauss_kernel, 'same');
    sm_corr_trials = sm_corr_pad(:, pad_bins+1 : pad_bins+n_bins);
else
    sm_corr_trials = fr_corr;   % empty
end

if ~isempty(fr_inc)
    fr_inc_pad = [ fliplr(fr_inc(:,1:pad_bins)) , ...
                   fr_inc , ...
                   fliplr(fr_inc(:,end-pad_bins+1:end)) ];
    sm_inc_pad  = conv2(fr_inc_pad, gauss_kernel, 'same');
    sm_inc_trials = sm_inc_pad(:, pad_bins+1 : pad_bins+n_bins);
else
    sm_inc_trials = fr_inc;     % empty
end


% 3) Optional trial-level z-score AFTER smoothing
if use_zscore
    sm_corr_trials = zscore_trials(sm_corr_trials);
    sm_inc_trials  = zscore_trials(sm_inc_trials);
    y_label = 'Z-score';
else
    y_label = 'Firing Rate (Hz)';
end

% 4) Mean & SEM from (smoothed [+ z-scored]) trials
if isempty(sm_corr_trials)
    m_corr = nan(1,n_bins); sem_corr = nan(1,n_bins);
else
    m_corr   = mean(sm_corr_trials, 1, 'omitnan');
    sem_corr = std (sm_corr_trials, 0, 1, 'omitnan') / sqrt(size(sm_corr_trials,1));
end
if isempty(sm_inc_trials)
    m_inc = nan(1,n_bins); sem_inc = nan(1,n_bins);
else
    m_inc    = mean(sm_inc_trials , 1, 'omitnan');
    sem_inc  = std (sm_inc_trials , 0, 1, 'omitnan') / sqrt(size(sm_inc_trials ,1));
end


% -------------------------------------------------------------------------
% Empirical one-sided test in the time-field bin
t = bins(1:end-1);

% time-field bin center is defined in your code as (tf_bin-0.5)*0.1 seconds
tf_center_t = (tf_bin - 0.5) * 0.1;
[~, tf_bin_idx] = min(abs(t - tf_center_t));

nResamp = 1000;
p_value = nan;

if ~isempty(sm_inc_trials) && ~isempty(sm_corr_trials)
    nInc = size(sm_inc_trials, 1);
    obs_inc = mean(sm_inc_trials(:, tf_bin_idx), 'omitnan');

    nCorr = size(sm_corr_trials, 1);

    resamp_means = nan(nResamp,1);
    for k = 1:nResamp
        pick = randi(nCorr, [nInc, 1]);  % with replacement
        resamp_means(k) = mean(sm_corr_trials(pick, tf_bin_idx), 'omitnan');
    end


    p_value = mean(resamp_means <= obs_inc); % or >= depending on hypothesis
    is_valid = true;

end



if doplot
% -------------------------------------------------------------------------
% ============== PLOT ==============
figure('Units','pixels','Position',[100 100 400 648]);

% -------- Raster --------
subplot(2,1,1); hold on
set(gca,'FontSize',16);
fill([t_start t_start t_end t_end], ...
     [0 n_trials+1 n_trials+1 0], ...
     [0.6 0.8 1],'EdgeColor','none','FaceAlpha',0.4, ...
     'HandleVisibility','off');  % not in legend

% Correct trials in blue, Incorrect in red
for ii = 1:numel(corr_idx)
    s = aligned_spikes{corr_idx(ii)};
    if ~isempty(s), scatter(s, ii*ones(size(s)), 9,'b','filled','HandleVisibility','off'); end
end
for jj = 1:numel(inc_idx)
    s = aligned_spikes{inc_idx(jj)};
    if ~isempty(s), scatter(s, (numel(corr_idx)+jj)*ones(size(s)), 9,'r','filled','HandleVisibility','off'); end
end

xlim([0 2.5]); ylim([0 n_trials+1]);
xlabel('Time (s)','FontSize',16); ylabel('Trial','FontSize',16);
title(sprintf('Subject %d | Unit %d', subject_id, unit_id), ...
      'Interpreter','none','FontSize',16);

% -------- PSTH --------
subplot(2,1,2); hold on
set(gca,'FontSize',16);

% Robust y-limits (works for z-scores or rates)
t = bins(1:end-1);
upper_env = nanmax([m_corr + sem_corr; m_inc + sem_inc], [], 1);
lower_env = nanmin([m_corr - sem_corr; m_inc - sem_inc], [], 1);
ymax = max(upper_env, [], 2); if isempty(ymax) || isnan(ymax), ymax = 1; end
ymin = min(lower_env, [], 2); if isempty(ymin) || isnan(ymin), ymin = 0; end
if ~use_zscore, ymin = max(0, ymin); end      % rates can't be negative
pad  = 0.05 * max(1, ymax - ymin);
yl   = [ymin - pad, ymax + pad];

% Highlight window (no legend)
patch([t_start t_start t_end t_end],[yl(1) yl(2) yl(2) yl(1)], ...
      [0.6 0.8 1],'EdgeColor','none','FaceAlpha',0.2, ...
      'HandleVisibility','off');

% Shaded SEM first (hidden), then lines (for legend)
fill([t fliplr(t)], [m_corr+sem_corr fliplr(m_corr-sem_corr)], ...
     'b','EdgeColor','none','FaceAlpha',0.2,'HandleVisibility','off');
hC = plot(t, m_corr,'b-','LineWidth',2,'DisplayName','Correct');

fill([t fliplr(t)], [m_inc+sem_inc fliplr(m_inc-sem_inc)], ...
     'r','EdgeColor','none','FaceAlpha',0.2,'HandleVisibility','off');
hI = plot(t, m_inc,'r-','LineWidth',2,'DisplayName','Incorrect');


% Mark the time-field bin and print p-value

if ~isnan(p_value)
    txt = sprintf('Empirical p = %.3g (N_{inc}=%d)', ...
                  p_value, size(sm_inc_trials,1));
else
    txt = 'Empirical p = NaN (need both correct + incorrect trials)';
end
text(0.02, 0.98, txt, 'Units','normalized', ...
     'VerticalAlignment','top', 'FontSize', 12);

legend([hC hI],'Location','best','FontSize',16,'Box','on');
xlim([0 2.5]); ylim(yl);
xlabel('Time (s)','FontSize',16); ylabel(y_label,'FontSize',16);

end
end

% -------------------------------------------------------------------------
function zmat = zscore_trials(mat)
% Z-score each row (trial) across time bins
mu  = mean(mat,2);
sig = std(mat,0,2);
sig(sig==0) = 1;                 % avoid divide-by-zero
zmat = (mat - mu) ./ sig;
end

