function plot_single_cell_XvsNonX_heat(nwbAll, all_units, neural_data, ...
                                  bin_size, subject_id, unit_id, ...
                                  use_zscore)
% PLOT_SINGLE_CELL_XVSNONX
% Raster + dual PSTHs (X vs Non-X trials) for ONE unit.
% INPUTS:
%   nwbAll      : cell array of NWB objects
%   all_units   : struct array of single-unit metadata
%   neural_data : struct array already loaded
%   bin_size    : PSTH bin size in seconds (e.g. 0.05)
%   subject_id  : subject ID for this cell
%   unit_id     : unit ID for this cell
%   use_zscore  : true to z-score within each trial

if nargin < 7, use_zscore = false; end

% -------------------- Find unit --------------------
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

% -------------------- Grab metadata --------------------
tf_bin   = double(neural_data(ndx).time_field);
pref_img = neural_data(ndx).preferred_image;
imgIDs   = neural_data(ndx).trial_imageIDs;

tsMaint  = nwbAll{SU.session_count}.intervals_trials ...
                    .vectordata.get('timestamps_Maintenance') ...
                    .data.load();
spk = SU.spike_times;
nTrials = numel(tsMaint);

% -------------------- Trial split --------------------
enc1 = imgIDs(:,1);  enc2 = imgIDs(:,2);  enc3 = imgIDs(:,3);
X_idx    = find(enc1==pref_img & enc2==0 & enc3==0);
NonX_idx = find(enc1~=pref_img & enc2==0 & enc3==0);

% -------------------- PSTH parameters --------------------
bins   = 0:bin_size:2.5;  n_bins = numel(bins)-1;

% Gaussian kernel (2 bins = 100 ms)
gaussian_sigma = 2;
k_size   = round(2.5 * gaussian_sigma);
x_kernel = -k_size:k_size;
gauss_k  = exp(-(x_kernel.^2)/(2*gaussian_sigma^2));
gauss_k  = gauss_k / sum(gauss_k);

% -------------------- Field window: only X trials --------------------
fr_all = zeros(numel(X_idx), n_bins);
for t = 1:numel(X_idx)
    s = spk(spk>=tsMaint(X_idx(t)) & spk<tsMaint(X_idx(t))+2.5) - tsMaint(X_idx(t));
    fr_all(t,:) = histcounts(s,bins)/bin_size;
end
mean_fr = mean(fr_all,1);
baseline = mean(mean_fr);
thr = baseline + 0.5*std(mean_fr);

centre_t   = (tf_bin-0.5)*0.1;
[~,centre] = min(abs(bins(1:end-1)-centre_t));

left  = centre; while left>1      && mean_fr(left)>=thr, left  = left-1; end
right = centre; while right<n_bins&& mean_fr(right)>=thr, right = right+1; end

t_start = min(bins(left), (tf_bin-1)*0.1);
t_end   = max(bins(right), tf_bin*0.1);
if t_end <= t_start
    t_start = (tf_bin-1)*0.1;
    t_end   = tf_bin*0.1;
end

% -------------------- Plot figure --------------------
% ======= HEAT-MAPS INSTEAD OF DOT RASTER ===============================
subplot(2,1,1); cla; hold on; set(gca,'FontSize',12)

% ---------- build trial × bin matrices ----------
matX    = buildHeatMatrix(X_idx   , spk, tsMaint, bins, bin_size);
matNonX = buildHeatMatrix(NonX_idx, spk, tsMaint, bins, bin_size);

% ---------- OPTIONAL: sort rows by peak-bin ----------
[~,ordX]    = sort(max(matX   ,[],2));  matX    = matX   (ordX   ,:);
[~,ordNonX] = sort(max(matNonX,[],2));  matNonX = matNonX(ordNonX,:);

% ---------- stack and plot ----------
matAll  = [matX; matNonX];                   %# rows = nX + nNonX
imagesc(bins(1:end-1), 1:size(matAll,1), matAll);
colormap(gca, parula);   caxis([0 1]);       % 0-1 after min-max scaling

% labels & cosmetics
xlabel('Time (s)');  ylabel('Trial');
title(sprintf('S%d U%d – Heat-map (X on top, Non-X below)', ...
              subject_id, unit_id));
xlim([0 2.5]);  ylim([0.5, size(matAll,1)+0.5]);

% highlight time-field window
patch([t_start t_start t_end t_end], ...
      [0.5 size(matAll,1)+0.5 size(matAll,1)+0.5 0.5], ...
      [0.6 0.8 1],'EdgeColor','none','FaceAlpha',0.20,'HandleVisibility','off');

cb = colorbar('Location','eastoutside');     % create bar, keep handle
ylabel(cb,'Min–max-normalised z-score');     % now add the label


% === PSTH ===
subplot(2,1,2); hold on; set(gca,'FontSize',12);
xlabel('Time (s)','FontSize',12);
if use_zscore
    ylabel('Z‑score','FontSize',12);
else
    ylabel('Rate (spk/s)','FontSize',12);
end

labels = {'X Trials','Non‑X Trials'};
for g = 1:2
    idx = groups{g};
    col = colors{g};

    [mFR, sFR] = computePSTH(spk, tsMaint, idx, bin_size, bins, use_zscore);
    mFR = conv(mFR, gauss_k,'same');
    sFR = conv(sFR, gauss_k,'same');

    t = bins(1:end-1);
    fill([t fliplr(t)], [mFR+sFR fliplr(mFR-sFR)], col, ...
         'EdgeColor','none','FaceAlpha',0.25);
    plot(t, mFR, 'Color',col,'LineWidth',2);
end

yl = ylim;
patch([t_start t_start t_end t_end],[yl(1) yl(2) yl(2) yl(1)], ...
      [0.6 0.8 1],'EdgeColor','none','FaceAlpha',0.2,'HandleVisibility','off');

legend(labels,'Location','best','FontSize',12);
xlim([0 2.5]);

end

% ------------------------------------------------------------------------
function [mean_fr, sem_fr] = computePSTH(spike_times, tsMaint, idx, ...
                                         bin_size, bins, use_z)
% Single‑group PSTH, optional z‑score per trial
nT = numel(idx);
nb = numel(bins)-1;
if nT==0, mean_fr=zeros(1,nb); sem_fr=zeros(1,nb); return; end

FR = zeros(nT,nb);
for k = 1:nT
    tr  = idx(k);
    s   = spike_times(spike_times>=tsMaint(tr) & spike_times< tsMaint(tr)+2.5) - tsMaint(tr);
    cnt = histcounts(s,bins);
    r   = cnt / bin_size;

    if use_z
        mu = mean(r);  sd = std(r);
        if sd==0, r = zeros(size(r));
        else      r = (r - mu) / sd;
        end
    end
    FR(k,:) = r;
end
mean_fr = mean(FR,1);
sem_fr  = std(FR,0,1) / sqrt(nT);
end
function M = buildHeatMatrix(idx, spk, tsMaint, bins, bin_size)
% Returns (#trials × #bins) matrix: per-trial z-score then min–max scale

nT = numel(idx);
nB = numel(bins)-1;
M  = zeros(nT,nB);

for ii = 1:nT
    tr   = idx(ii);
    s    = spk(spk>=tsMaint(tr) & spk<tsMaint(tr)+2.5) - tsMaint(tr);
    r    = histcounts(s,bins)/bin_size;            % rate per bin
    rz   = zscoreRow(r);                           % per-trial z-score
    M(ii,:) = minMaxNorm(rz);                      % 0-1 scaling
end
end

function y = zscoreRow(x)
mu = mean(x);  sd = std(x);
if sd==0,  y = zeros(size(x)); else, y = (x-mu)/sd; end
end

function y = minMaxNorm(x)
rng = max(x)-min(x);
if rng==0, y = zeros(size(x));
else       y = (x-min(x))/rng;
end
end
