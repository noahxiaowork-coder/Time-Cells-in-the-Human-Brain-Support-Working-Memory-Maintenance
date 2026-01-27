function plot_correct_incorrect_heatmaps(nwbAll, all_units, ...
    neural_data_file, bin_width)
%PLOT_CORRECT_INCORRECT_HEATMAPS Four Maintenance-epoch heat-maps
% 1) All trials
% 2) Correct trials
% 3) Incorrect trials
% 4) Correct trials down-sampled to #incorrect
%
% All PSTHs are rebuilt from raw spike times:
% – bins of BIN_WIDTH seconds across the 2.5-s Maintenance epoch
% – Gaussian smoothing, σ = 0.2 s
%
% Figures are fixed to 648 × 400 pixels.
% ------------------------------------------------ parameters ------------
maintDur = 2.5;
nBins = round(maintDur / bin_width);
rng(20250710);

% ---- Gaussian kernel (σ = 0.2 s) ---------------------------------------
% sigmaBins = 0.2 / bin_width;
% kSize = round(2*sigmaBins);
% x = -kSize:kSize;
% gKernel = exp(-(x.^2) ./ (2*sigmaBins^2));
% gKernel = gKernel ./ sum(gKernel);
gKernel = GaussianKernal(0.3 / bin_width, 1.5);
% ------------------------------------------------------------------------

load(neural_data_file,'neural_data');
nNeurons = numel(neural_data);
assert(nNeurons>0,'No neurons in neural_data.');

% pre-allocate
all_psth = nan(nNeurons,nBins);
corr_psth = nan(nNeurons,nBins);
incorr_psth = nan(nNeurons,nBins);
corr_ds_psth= nan(nNeurons,nBins);
peak_bins = nan(nNeurons,1);

% ================================================= main loop ============
for i = 1:nNeurons
    nd = neural_data(i);
    pid = nd.patient_id;
    uid = nd.unit_id;

    % ---- locate raw spikes --------------------------------------------
    idx = find([all_units.subject_id]==pid & ...
               [all_units.unit_id]==uid,1);
    if isempty(idx), continue; end
    SU = all_units(idx);

    sess = nwbAll{SU.session_count};
    tsMai = sess.intervals_trials.vectordata ...
                  .get('timestamps_Maintenance').data.load();
    nTrials = numel(tsMai);
    if nTrials == 0, continue; end

    corr_vals = nd.trial_correctness(:);   % 1 / 0
    if numel(corr_vals) ~= nTrials
        % safety
        warning('Trial count mismatch for unit %d – skipping',i);
        continue;
    end

    % ---- build single-trial, smoothed FR matrix -----------------------
    fr = nan(nTrials,nBins);
    for t = 1:nTrials
        edges = tsMai(t):bin_width:(tsMai(t)+maintDur);
        if numel(edges)~=nBins+1, continue; end
        tmp = histcounts(SU.spike_times,edges) ./ bin_width;
        fr(t,:) = conv(tmp,gKernel,'same');
    end

    % ---- decide trial classes ----------------------------------------
    corr_idx = corr_vals == 1;
    inc_idx  = corr_vals == 0;

    % mean PSTHs
    avg_all  = mean(fr, 1,'omitnan');
    avg_corr = mean(fr(corr_idx,:),1,'omitnan');
    avg_inc  = mean(fr(inc_idx ,:),1,'omitnan');

    % down-sample correct → same n as incorrect
    nCorr = nnz(corr_idx);
    nInc  = nnz(inc_idx);
    if nInc>0 && nCorr>=nInc
        pick = randsample(find(corr_idx), nInc);
        avg_corr_ds = mean(fr(pick,:),1,'omitnan');
    elseif nInc>0
        avg_corr_ds = avg_corr;   % fewer correct than incorrect
    else
        avg_corr_ds = nan(1,nBins);
    end

    all_psth(i,:)     = avg_all;
    corr_psth(i,:)    = avg_corr;
    incorr_psth(i,:)  = avg_inc;
    corr_ds_psth(i,:) = avg_corr_ds;
    [~,peak_bins(i)]  = max(avg_all);
end

% ================================================= post-process =========
% sort by peak latency in All-trials PSTH
[~,ord] = sort(peak_bins,'ascend','MissingPlacement','last');
all_psth     = all_psth(ord,:);
corr_psth    = corr_psth(ord,:);
incorr_psth  = incorr_psth(ord,:);
corr_ds_psth = corr_ds_psth(ord,:);

% row-wise min–max normalise independently where needed
all_psth_n   = rowwise_minmax_normalize(all_psth);
corr_full_n  = rowwise_minmax_normalize(corr_psth);

% paired normalisation for Incorrect vs Down-sampled Correct
[incorr_n, corr_ds_n] = ...
    rowwise_minmax_normalize_together(incorr_psth, corr_ds_psth);

% ------------------------------------------------ plotting --------------
cmap = parula;
tCenters = (0:nBins-1)*bin_width + bin_width/2;
tickStep = 0.5;
tTicks = 0:tickStep:maintDur;
xt = arrayfun(@(t) find_closest_bin(tCenters,t), tTicks);
xLabs = arrayfun(@(t) sprintf('%.1f',t),tTicks,'uni',0);

makeFig(all_psth_n,  'All Trials');
makeFig(corr_full_n,'Correct Trials');
makeFig(incorr_n,   'Incorrect Trials (scaled vs. DS-Correct)');
makeFig(corr_ds_n,  'Correct Trials – Down-sampled');
disp('Done – correct/incorrect heat-maps generated.');

% =================== SAVE DS-Correct and Incorrect heatmaps as PDF ============
fig = figure('Name','Incorrect vs DS-Correct Heatmaps');

% Use tiledlayout with tight spacing
t = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');

% 1) Incorrect Trials
ax1 = nexttile(t,1);
imagesc(ax1, incorr_n, [0 1]);
colormap(ax1, cmap);
title(ax1, 'Incorrect Trials');
xlabel(ax1,'Time (s)');
ylabel(ax1,'Neuron');
set(ax1,'TickDir','out','XTick',xt,'XTickLabel',xLabs);

% 2) Down-sampled Correct Trials
ax2 = nexttile(t,2);
imagesc(ax2, corr_ds_n, [0 1]);
colormap(ax2, cmap);
title(ax2, 'Correct Trials – Down-sampled');
xlabel(ax2,'Time (s)');
ylabel(ax2,'Neuron');
set(ax2,'TickDir','out','XTick',xt,'XTickLabel',xLabs);

% Shared colorbar
cb = colorbar;
cb.Layout.Tile = 'east';

% ✅ Ensure the figure size includes all elements
fig.Units = 'inches';
fig.Position(3:4) = [8 4];   % width x height in inches (adjust as needed)

% ✅ Ensure axes fit tightly in the figure
set([ax1 ax2], 'LooseInset', get(ax1,'TightInset'));

% ✅ Export to vector SVG, complete, no clipping
exportgraphics(t, replace(neural_data_file,'.mat','_incorr_vs_corrds_heatmaps.svg'), ...
               'ContentType','vector');
disp('Saved Incorrect and DS-Correct heatmaps as SVG.');

% ---------------------- nested helper ----------------------------------
    function makeFig(mat,ttl)
        figure('Name',ttl,'Units','pixels','Position',[100 100 400 648]);
        imagesc(mat,[0 1]);
        colormap(cmap);
        colorbar;
        xlabel('Time (s)');
        ylabel('Neuron (sorted by peak)');
        title(['Time Cells: ' ttl]);
        set(gca,'TickDir','out','XTick',xt,'XTickLabel',xLabs);
    end
end
% ---------- end main function

function mat = rowwise_minmax_normalize(mat)
%ROWWISE_MINMAX_NORMALIZE Normalizes each row to [0,1] using min-max across that row only.
for r = 1:size(mat,1)
    row = mat(r,:);
    if all(isnan(row))   % handle all-NaN rows
        continue;
    end
    min_val = min(row);
    max_val = max(row);
    if max_val > min_val
        mat(r,:) = (row - min_val) / (max_val - min_val);
    else
        % If values are constant (or effectively constant), set them to 0.5
        mat(r,:) = 0.5;
    end
end
end

function [mat1_norm, mat2_norm] = rowwise_minmax_normalize_together(mat1, mat2)
%ROWWISE_MINMAX_NORMALIZE_TOGETHER
% For each row (i.e., neuron), combine the values from MAT1(i,:) and MAT2(i,:)
% to get a single min and max. Then normalize both to [0,1].
%
% This ensures that each neuron has the same color scale between
% the two matrices (e.g. comparing incorrect vs. correct-downsampled PSTHs).

mat1_norm = zeros(size(mat1));
mat2_norm = zeros(size(mat2));

for r = 1:size(mat1,1)
    row1 = mat1(r,:);
    row2 = mat2(r,:);
    row_combined = [row1, row2];

    % If everything is NaN, skip
    if all(isnan(row_combined))
        mat1_norm(r,:) = row1;
        mat2_norm(r,:) = row2;
        continue;
    end

    min_val = min(row_combined);
    max_val = max(row_combined);

    % Handle normal row
    if max_val > min_val
        mat1_norm(r,:) = (row1 - min_val) / (max_val - min_val);
        mat2_norm(r,:) = (row2 - min_val) / (max_val - min_val);
    else
        % If the combined row is constant, set everything to 0.5
        mat1_norm(r,:) = 0.5;
        mat2_norm(r,:) = 0.5;
    end
end
end

function idx = find_closest_bin(time_centers, target_time)
%FIND_CLOSEST_BIN Finds the bin index in time_centers closest to target_time.
[~, idx] = min(abs(time_centers - target_time));
end

function avg_xcorr = compute_avg_cross_correlation(heatmap)
% Compute average cross-correlation across rows after peak alignment.

[num_neurons, num_bins] = size(heatmap);
middle_bin = round(num_bins / 2);   % Define the center position for alignment

% Step 1: Find peak bin for each neuron
[~, peak_bins] = max(heatmap, [], 2);   % Find peak bin for each row

% Step 2: Align each row by shifting so that peaks are centered
aligned_heatmap = zeros(size(heatmap));
for i = 1:num_neurons
    shift_amount = middle_bin - peak_bins(i);
    aligned_heatmap(i, :) = circshift(heatmap(i, :), shift_amount, 2);
end

% Step 3: Compute pairwise cross-correlation (Pearson correlation)
corr_matrix = corr(aligned_heatmap', 'Rows', 'pairwise');

% Step 4: Compute average cross-correlation (excluding diagonal)
avg_xcorr = mean(corr_matrix(~eye(num_neurons)), 'omitnan');
end
