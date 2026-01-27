function plot_single_cell_RTspeed(nwbAll, all_units, neural_data, ...
                                  bin_size, subject_id, unit_id, ...
                                  use_correct, use_zscore)

if nargin < 7 || isempty(use_correct), use_correct = true;  end
if nargin < 8 || isempty(use_zscore),  use_zscore  = false; end
pctThresh = 30;

ndx = find([neural_data.patient_id]==subject_id & ...
           [neural_data.unit_id]==unit_id, 1,'first');
if isempty(ndx)
    warning('Unit %d/%d not found in neural_data.',subject_id,unit_id);
    return
end
m = ([all_units.subject_id]==subject_id & [all_units.unit_id]==unit_id);
if ~any(m)
    warning('Unit %d/%d not found in all_units.',subject_id,unit_id);
    return
end
SU = all_units(m);

tf_bin   = double(neural_data(ndx).time_field);
rt_vec   = neural_data(ndx).trial_RT;
corr_vec = neural_data(ndx).trial_correctness==1;
load_v   = neural_data(ndx).trial_load;
is_valid = ~isnan(rt_vec);

fast_mask = false(size(rt_vec));
slow_mask = false(size(rt_vec));

for L = 1:3
    if use_correct
        mask_L = (load_v == L) & is_valid & corr_vec;
    else
        mask_L = (load_v == L) & is_valid;
    end

    if nnz(mask_L) < 5
        continue;
    end

    p_low  = prctile(rt_vec(mask_L),        pctThresh);
    p_high = prctile(rt_vec(mask_L), 100 - pctThresh);

    f_mask = (rt_vec <= p_low)  & mask_L;
    s_mask = (rt_vec >= p_high) & mask_L;

    fast_mask = fast_mask | f_mask;
    slow_mask = slow_mask | s_mask;
end

fast_idx = find(fast_mask);
slow_idx = find(slow_mask);
if isempty(fast_idx) || isempty(slow_idx)
    warning('Unit %d/%d skipped (no trials in one RT bracket).',subject_id,unit_id);
    return
end

tsMaint = nwbAll{SU.session_count}.intervals_trials ...
                   .vectordata.get('timestamps_Maintenance') ...
                   .data.load();
spk     = SU.spike_times;
n_trials = numel(tsMaint);

bins   = 0:bin_size:2.5;
n_bins = numel(bins)-1;

gauss_sigma_b = 0.2 / bin_size;
k_size        = round(5*gauss_sigma_b);
xkern         = -k_size:k_size;
gauss_kernel  = exp(-(xkern.^2)/(2*gauss_sigma_b^2));
gauss_kernel  = gauss_kernel / sum(gauss_kernel);

fr_all = zeros(n_trials,n_bins);
for t = 1:n_trials
    s = spk(spk>=tsMaint(t) & spk<tsMaint(t)+2.5) - tsMaint(t);
    fr_all(t,:) = histcounts(s,bins)/bin_size;
end
mean_fr_all = mean(fr_all,1);
baseline = mean(mean_fr_all);
thr      = baseline + 0.5*std(mean_fr_all);

centre_t   = (tf_bin-0.5)*0.1;
[~,centre] = min(abs(bins(1:end-1)-centre_t));
left  = centre; while left>1       && mean_fr_all(left) >= thr, left  = left-1; end
right = centre; while right<n_bins && mean_fr_all(right)>= thr, right = right+1; end
t_start = bins(left);  t_end = bins(right);

fr_fast = buildFR(spk, tsMaint, fast_idx, bins, bin_size, use_zscore);
fr_slow = buildFR(spk, tsMaint, slow_idx, bins, bin_size, use_zscore);

fr_fast_sm = smoothTrials(fr_fast, gauss_kernel);
fr_slow_sm = smoothTrials(fr_slow, gauss_kernel);

m_fast   = mean(fr_fast_sm, 1);
m_slow   = mean(fr_slow_sm, 1);
sem_fast = std(fr_fast_sm, 0, 1) / sqrt(size(fr_fast_sm, 1));
sem_slow = std(fr_slow_sm, 0, 1) / sqrt(size(fr_slow_sm, 1));

sm_fast  = m_fast;
sm_slow  = m_slow;
sm_sem_f = sem_fast;
sm_sem_s = sem_slow;

figure('Units','pixels','Position',[100 100 400 648],'Color','w');

subplot(2,1,1); hold on; set(gca,'FontSize',16)
n_plot = numel(fast_idx)+numel(slow_idx);

fill([t_start t_start t_end t_end],[0 n_plot+1 n_plot+1 0], ...
     [0.6 0.8 1],'EdgeColor','none','FaceAlpha',0.4,'HandleVisibility','off');

row = 1;
for tr = fast_idx(:)'
    s = spk(spk>=tsMaint(tr) & spk<tsMaint(tr)+2.5) - tsMaint(tr);
    scatter(s, row*ones(size(s)), 8,'b','filled','HandleVisibility','off');
    row = row+1;
end
for tr = slow_idx(:)'
    s = spk(spk>=tsMaint(tr) & spk<tsMaint(tr)+2.5) - tsMaint(tr);
    scatter(s, row*ones(size(s)), 8,'r','filled','HandleVisibility','off');
    row = row+1;
end
xlim([0 2.5]); ylim([0 n_plot+1]);
xlabel('Time (s)','FontSize',16); ylabel('Trial','FontSize',16);
ttl = sprintf('S%d U%d | RT split within loads', subject_id, unit_id);
if use_correct, ttl = [ttl ' | correct-only']; end
title(ttl,'Interpreter','none','FontSize',16);

subplot(2,1,2); hold on; set(gca,'FontSize',16)
xlabel('Time (s)','FontSize',16);
if use_zscore
    ylabel('Z-score','FontSize',16);
else
    ylabel('Firing Rate (Hz)','FontSize',16);
end

t = bins(1:end-1);

fill([t fliplr(t)], [sm_fast+sm_sem_f fliplr(sm_fast-sm_sem_f)], ...
     'b','EdgeColor','none','FaceAlpha',0.25,'HandleVisibility','off');
hFast = plot(t, sm_fast, 'b-', 'LineWidth',2, ...
             'DisplayName', 'Fast');

fill([t fliplr(t)], [sm_slow+sm_sem_s fliplr(sm_slow-sm_sem_s)], ...
     'r','EdgeColor','none','FaceAlpha',0.25,'HandleVisibility','off');
hSlow = plot(t, sm_slow, 'r-', 'LineWidth',2, ...
             'DisplayName', 'Slow');

yl = autoPadY(sm_fast,sm_sem_f,sm_slow,sm_sem_s);
patch([t_start t_start t_end t_end],[yl(1) yl(2) yl(2) yl(1)], ...
      [0.6 0.8 1],'EdgeColor','none','FaceAlpha',0.2,'HandleVisibility','off');

legend([hFast hSlow],'Location','best','FontSize',16,'Box','on');
xlim([0 2.5]); ylim(yl);

end

function FR = buildFR(spk, ts, idx, bins, bin_size, zflag)
nT = numel(idx); nb = numel(bins)-1;
FR = zeros(nT,nb);
for k = 1:nT
    tr = idx(k);
    s  = spk(spk>=ts(tr) & spk<ts(tr)+2.5) - ts(tr);
    r  = histcounts(s,bins)/bin_size;
    if zflag
        mu = mean(r); sd = std(r);
        if sd==0, r = zeros(size(r)); else, r = (r - mu) / sd; end
    end
    FR(k,:) = r;
end
end

function yl = autoPadY(varargin)
allVals = [varargin{:}];
yMin = min(allVals); yMax = max(allVals);
pad  = 0.1*(yMax-yMin+eps);
yl   = [yMin-pad, yMax+pad];
end

function Xsm = smoothTrials(X, kernel)
if isempty(X)
    Xsm = X;
    return;
end

k = kernel(:)';
if any(k)
    k = k / sum(k);
end

[~, nBins] = size(X);
halfWidth  = floor((numel(k) - 1) / 2);
pad_bins   = min(halfWidth, floor((nBins-1)/2));

if pad_bins > 0
    Xpad = [ fliplr(X(:,1:pad_bins)), ...
             X, ...
             fliplr(X(:,end-pad_bins+1:end)) ];

    Xsm_pad = conv2(Xpad, k, 'same');
    Xsm = Xsm_pad(:, pad_bins+1 : pad_bins+nBins);
else
    Xsm = conv2(X, k, 'same');
end
end
