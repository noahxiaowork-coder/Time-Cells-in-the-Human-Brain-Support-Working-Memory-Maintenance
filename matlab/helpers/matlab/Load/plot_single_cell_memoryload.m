function plot_single_cell_memoryload(nwbAll, all_units, neural_data, ...
                                     bin_size, subject_id, unit_id, ...
                                     use_zscore, use_correct)

if nargin < 7, use_zscore  = false; end
if nargin < 8, use_correct = false; end

ndx = find([neural_data.patient_id]==subject_id & ...
           [neural_data.unit_id]==unit_id, 1, 'first');
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

tfbin  = double(neural_data(ndx).time_field);
load_v = neural_data(ndx).trial_load;
corr_v = neural_data(ndx).trial_correctness;

tsMaint = nwbAll{SU.session_count}.intervals_trials ...
                    .vectordata.get('timestamps_Maintenance') ...
                    .data.load();
spk = SU.spike_times;
nTrials = numel(tsMaint);

bins   = 0:bin_size:2.5;
n_bins = numel(bins)-1;

gauss_kernel = GaussianKernal(0.5 / bin_size, 2);
gauss_kernel = gauss_kernel(:)';   
if any(gauss_kernel)
    gauss_kernel = gauss_kernel / sum(gauss_kernel);
end

sigma_bins = 0.5 / bin_size;
pad_bins   = min( ceil(3 * sigma_bins), floor((n_bins-1)/2) );

FR_all = zeros(nTrials,n_bins);
for t = 1:nTrials
    s = spk(spk>=tsMaint(t) & spk<tsMaint(t)+2.5) - tsMaint(t);
    FR_all(t,:) = histcounts(s,bins)/bin_size;
end
mFR = mean(FR_all,1);
base = mean(mFR);
thr  = base + 0.5*std(mFR);

centre_t   = (tfbin-0.5)*0.1;
[~,centre] = min(abs(bins(1:end-1)-centre_t));

left = centre;  while left>1       && mFR(left)>=thr, left  = left-1; end
right= centre;  while right<n_bins && mFR(right)>=thr, right = right+1; end

t_start = min(bins(left), (tfbin-1)*0.1);
t_end   = max(bins(right), tfbin*0.1);
if t_end <= t_start
    t_start = (tfbin-1)*0.1;
    t_end   = tfbin*0.1;
end

if use_correct
    idx1 = find(load_v==1 & corr_v==1);
    idx2 = find(load_v==2 & corr_v==1);
    idx3 = find(load_v==3 & corr_v==1);
else
    idx1 = find(load_v==1);
    idx2 = find(load_v==2);
    idx3 = find(load_v==3);
end

clustered_idx = [idx1(:); idx2(:); idx3(:)];
nClustered    = numel(clustered_idx);

figure('Units','pixels','Position',[100 100 400 648]);

subplot(2,1,1); hold on
set(gca,'FontSize',16);

fill([t_start t_start t_end t_end], [0 nClustered+1 nClustered+1 0], ...
     [0.6 0.8 1],'EdgeColor','none','FaceAlpha',0.4);

title(sprintf('Subject %d, Unit %d, TFbin %d%s', ...
      subject_id,unit_id,tfbin, ternary(use_correct,' (correct only)','')), ...
      'Interpreter','none','FontSize',16);
xlabel('Time (s)','FontSize',16);
ylabel('Trial','FontSize',16);

trialRow = 1;
for i = 1:nClustered
    tr = clustered_idx(i);
    s  = spk(spk>=tsMaint(tr) & spk<tsMaint(tr)+2.5) - tsMaint(tr);

    switch load_v(tr)
        case 1, baseColor = [0 0 1];
        case 2, baseColor = [0 0.6 0];
        case 3, baseColor = [1 0 0];
        otherwise, baseColor = [0 0 0];
    end
    if corr_v(tr)==1, c = baseColor; else, c = 0.5*baseColor + 0.5; end
    scatter(s, trialRow*ones(size(s)), 8, c, 'filled');
    trialRow = trialRow + 1;
end
ylim([0 nClustered+1]);

subplot(2,1,2); hold on
set(gca,'FontSize',16);
xlabel('Time (s)','FontSize',16);

if use_zscore
    y_label = 'Z-score';
else
    y_label = 'Firing Rate (Hz)';
end
ylabel(y_label, 'FontSize', 16);

[m1,s1] = psth_group(spk, tsMaint, idx1, use_zscore, bins, bin_size);
[m2,s2] = psth_group(spk, tsMaint, idx2, use_zscore, bins, bin_size);
[m3,s3] = psth_group(spk, tsMaint, idx3, use_zscore, bins, bin_size);

m1 = smooth_with_padding(m1, gauss_kernel, pad_bins);
s1 = smooth_with_padding(s1, gauss_kernel, pad_bins);
m2 = smooth_with_padding(m2, gauss_kernel, pad_bins);
s2 = smooth_with_padding(s2, gauss_kernel, pad_bins);
m3 = smooth_with_padding(m3, gauss_kernel, pad_bins);
s3 = smooth_with_padding(s3, gauss_kernel, pad_bins);

t = bins(1:end-1);

h1 = plot_fill(t,m1,s1,[0 0 1]);
h2 = plot_fill(t,m2,s2,[0 0.6 0]);
h3 = plot_fill(t,m3,s3,[1 0 0]);

set(h1,'DisplayName','Load 1');
set(h2,'DisplayName','Load 2');
set(h3,'DisplayName','Load 3');
legend([h1 h2 h3],'Location','best','FontSize',16,'Box','on');

yl = ylim;
patch([t_start t_start t_end t_end],[yl(1) yl(2) yl(2) yl(1)], ...
      [0.6 0.8 1],'EdgeColor','none','FaceAlpha',0.2,'HandleVisibility','off');

xlim([0 2.5]);

allVals = [m1+s1 m1-s1 m2+s2 m2-s2 m3+s3 m3-s3];
yMin = min(allVals); yMax = max(allVals);
pad  = 0.1*(yMax-yMin + eps);
ylim([yMin-pad, yMax+pad]);

end


function [mean_fr, sem_fr] = psth_group(spk, ts, idx, zflag, bins, bin_size)
nT = numel(idx);
nb = numel(bins)-1;
if nT==0, mean_fr=zeros(1,nb); sem_fr=zeros(1,nb); return; end
FR = zeros(nT,nb);
for k = 1:nT
    tr = idx(k);
    s  = spk(spk>=ts(tr)&spk<ts(tr)+2.5) - ts(tr);
    r  = histcounts(s,bins)/bin_size;
    if zflag
        mu = mean(r); sd = std(r);
        if sd==0, r = zeros(size(r)); else, r = (r-mu)/sd; end
    end
    FR(k,:) = r;
end
mean_fr = mean(FR,1);
sem_fr  = std(FR,0,1)/sqrt(nT);
end


function hLine = plot_fill(t,m,sem,col)
fill([t fliplr(t)], [m+sem fliplr(m-sem)], col, ...
     'EdgeColor','none','FaceAlpha',0.2,'HandleVisibility','off');
hLine = plot(t,m,'Color',col,'LineWidth',2);
end


function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end


function sm = smooth_with_padding(vec, k, pad_bins)
vec = vec(:)';
if numel(vec) <= 2*pad_bins
    sm = conv(vec, k, 'same');
    return
end
vpad = [ fliplr(vec(1:pad_bins)) , ...
         vec , ...
         fliplr(vec(end-pad_bins+1:end)) ];
vsm  = conv(vpad, k, 'same');
sm   = vsm(pad_bins+1 : pad_bins+numel(vec));
end
