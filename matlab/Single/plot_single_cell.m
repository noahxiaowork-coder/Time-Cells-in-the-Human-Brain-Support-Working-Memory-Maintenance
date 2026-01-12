function plot_single_cell(nwbAll, all_units, neural_data, bin_size, subject_id, unit_id)
% Raster + PSTH for one neuron.

ndx = find([neural_data.patient_id]==subject_id & ...
           [neural_data.unit_id]  ==unit_id, 1, 'first');
if isempty(ndx)
    warning('No entry for subject %d / unit %d in neural_data.',subject_id,unit_id);
    return
end

m  = ([all_units.subject_id]==subject_id & [all_units.unit_id]==unit_id);
if ~any(m)
    warning('Unit %d/%d not found in all_units.',subject_id,unit_id);
    return
end
SU       = all_units(m);
tf_bin10 = double(neural_data(ndx).time_field);   % 10 Hz bins (0.1 s)

sigma_bins   = 0.5 / bin_size;
pad_bins     = ceil(3 * sigma_bins);

bins         = 0:bin_size:2.5;
n_bins       = numel(bins)-1;
gauss_kernel = GaussianKernal(0.5 / bin_size, 2);
gauss_kernel = gauss_kernel(:)';                   
gauss_kernel = gauss_kernel / sum(gauss_kernel);   % normalize

tsMaint  = nwbAll{SU.session_count} ...
           .intervals_trials.vectordata ...
           .get('timestamps_Maintenance').data.load();
spk      = SU.spike_times;
n_trials = numel(tsMaint);

fr = zeros(n_trials,n_bins);
for t = 1:n_trials
    s       = spk(spk>=tsMaint(t) & spk<tsMaint(t)+2.5) - tsMaint(t);
    fr(t,:) = histcounts(s,bins)/bin_size;
end

% Use the raw mean to estimate a data-driven highlight window.
mean_fr_raw = mean(fr,1);
bl          = mean(mean_fr_raw);
thr         = bl + 0.5*std(mean_fr_raw);

centre_t   = (tf_bin10-0.5)*0.1;                 
[~,centre] = min(abs(bins(1:end-1)-centre_t));
left  = centre;  while left>1       && mean_fr_raw(left)  >= thr, left  = left-1; end
right = centre;  while right<n_bins && mean_fr_raw(right) >= thr, right = right+1; end

time_start_dynamic = min(bins(left), (tf_bin10-1)*0.1);   
time_end_dynamic   = max(bins(right), tf_bin10*0.1);

sigma_bins   = 0.5 / bin_size;
pad_bins     = ceil(3 * sigma_bins);

% Mirror-pad in time to avoid edge artifacts when smoothing.
fr_padded = [ fliplr(fr(:,1:pad_bins)) , fr , fliplr(fr(:,end-pad_bins+1:end)) ];
sm_fr_padded = conv2(fr_padded, gauss_kernel, 'same');
sm_fr = sm_fr_padded(:, pad_bins+1 : pad_bins+n_bins);

% Optional: z-score each trial across time after smoothing.
% mu  = mean(sm_fr, 2);
% sig = std (sm_fr, 0, 2);
% sig(sig==0) = 1;
% sm_fr = (sm_fr - mu) ./ sig;

mean_fr = mean(sm_fr, 1);
sem_fr  = std (sm_fr, 0, 1) / sqrt(n_trials);

figure('Units','pixels','Position',[100 100 400 648]);

subplot(2,1,1); hold on
fill([time_start_dynamic time_start_dynamic time_end_dynamic time_end_dynamic], ...
     [0 n_trials+1 n_trials+1 0], [0.6 0.8 1], 'EdgeColor','none','FaceAlpha',0.4);

for tr = 1:n_trials
    s = spk(spk>=tsMaint(tr) & spk<tsMaint(tr)+2.5) - tsMaint(tr);
    scatter(s, tr*ones(size(s)), 9, [0.2 0.2 0.2],'filled');
end
xlim([0 2.5]); ylim([0 n_trials+1]);
title(sprintf('Subject %d – Unit %d',subject_id,unit_id));
xlabel('Time (s)'); ylabel('Trial');

subplot(2,1,2); hold on
t = bins(1:end-1);

plot(t, mean_fr,'b-','LineWidth',2);
fill([t fliplr(t)], [mean_fr+sem_fr fliplr(mean_fr-sem_fr)], ...
     'b','EdgeColor','none','FaceAlpha',0.3);
xlim([0 2.5]);
xlabel('Time (s)'); ylabel('Firing rate (Hz)');  
end
