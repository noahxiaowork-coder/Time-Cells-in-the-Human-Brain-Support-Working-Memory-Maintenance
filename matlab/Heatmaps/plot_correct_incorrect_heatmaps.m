function plot_correct_incorrect_heatmaps(nwbAll, all_units, ...
    neural_data_file, bin_width)

maintDur = 2.5;
nBins = round(maintDur / bin_width);
rng(20250710);

gKernel = GaussianKernal(0.3 / bin_width, 1.5);

load(neural_data_file,'neural_data');
nNeurons = numel(neural_data);
assert(nNeurons>0,'No neurons in neural_data.');

all_psth = nan(nNeurons,nBins);
corr_psth = nan(nNeurons,nBins);
incorr_psth = nan(nNeurons,nBins);
corr_ds_psth= nan(nNeurons,nBins);
peak_bins = nan(nNeurons,1);

for i = 1:nNeurons
    nd = neural_data(i);
    pid = nd.patient_id;
    uid = nd.unit_id;

    idx = find([all_units.subject_id]==pid & ...
               [all_units.unit_id]==uid,1);
    if isempty(idx), continue; end
    SU = all_units(idx);

    sess = nwbAll{SU.session_count};
    tsMai = sess.intervals_trials.vectordata ...
                  .get('timestamps_Maintenance').data.load();
    nTrials = numel(tsMai);
    if nTrials == 0, continue; end

    corr_vals = nd.trial_correctness(:);
    if numel(corr_vals) ~= nTrials
        warning('Trial count mismatch for unit %d – skipping',i);
        continue;
    end

    fr = nan(nTrials,nBins);
    for t = 1:nTrials
        edges = tsMai(t):bin_width:(tsMai(t)+maintDur);
        if numel(edges)~=nBins+1, continue; end
        tmp = histcounts(SU.spike_times,edges) ./ bin_width;
        fr(t,:) = conv(tmp,gKernel,'same');
    end

    corr_idx = corr_vals == 1;
    inc_idx  = corr_vals == 0;

    avg_all  = mean(fr, 1,'omitnan');
    avg_corr = mean(fr(corr_idx,:),1,'omitnan');
    avg_inc  = mean(fr(inc_idx ,:),1,'omitnan');

    nCorr = nnz(corr_idx);
    nInc  = nnz(inc_idx);
    if nInc>0 && nCorr>=nInc
        pick = randsample(find(corr_idx), nInc);
        avg_corr_ds = mean(fr(pick,:),1,'omitnan');
    elseif nInc>0
        avg_corr_ds = avg_corr;
    else
        avg_corr_ds = nan(1,nBins);
    end

    all_psth(i,:)     = avg_all;
    corr_psth(i,:)    = avg_corr;
    incorr_psth(i,:)  = avg_inc;
    corr_ds_psth(i,:) = avg_corr_ds;
    [~,peak_bins(i)]  = max(avg_all);
end

[~,ord] = sort(peak_bins,'ascend','MissingPlacement','last');
all_psth     = all_psth(ord,:);
corr_psth    = corr_psth(ord,:);
incorr_psth  = incorr_psth(ord,:);
corr_ds_psth = corr_ds_psth(ord,:);

all_psth_n   = rowwise_minmax_normalize(all_psth);
corr_full_n  = rowwise_minmax_normalize(corr_psth);

[incorr_n, corr_ds_n] = ...
    rowwise_minmax_normalize_together(incorr_psth, corr_ds_psth);

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

fig = figure('Name','Incorrect vs DS-Correct Heatmaps');

t = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');

ax1 = nexttile(t,1);
imagesc(ax1, incorr_n, [0 1]);
colormap(ax1, cmap);
title(ax1, 'Incorrect Trials');
xlabel(ax1,'Time (s)');
ylabel(ax1,'Neuron');
set(ax1,'TickDir','out','XTick',xt,'XTickLabel',xLabs);

ax2 = nexttile(t,2);
imagesc(ax2, corr_ds_n, [0 1]);
colormap(ax2, cmap);
title(ax2, 'Correct Trials – Down-sampled');
xlabel(ax2,'Time (s)');
ylabel(ax2,'Neuron');
set(ax2,'TickDir','out','XTick',xt,'XTickLabel',xLabs);

cb = colorbar;
cb.Layout.Tile = 'east';

fig.Units = 'inches';
fig.Position(3:4) = [8 4];

set([ax1 ax2], 'LooseInset', get(ax1,'TightInset'));


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


function mat = rowwise_minmax_normalize(mat)
for r = 1:size(mat,1)
    row = mat(r,:);
    if all(isnan(row))
        continue;
    end
    min_val = min(row);
    max_val = max(row);
    if max_val > min_val
        mat(r,:) = (row - min_val) / (max_val - min_val);
    else
        mat(r,:) = 0.5;
    end
end
end


function [mat1_norm, mat2_norm] = rowwise_minmax_normalize_together(mat1, mat2)

mat1_norm = zeros(size(mat1));
mat2_norm = zeros(size(mat2));

for r = 1:size(mat1,1)
    row1 = mat1(r,:);
    row2 = mat2(r,:);
    row_combined = [row1, row2];

    if all(isnan(row_combined))
        mat1_norm(r,:) = row1;
        mat2_norm(r,:) = row2;
        continue;
    end

    min_val = min(row_combined);
    max_val = max(row_combined);

    if max_val > min_val
        mat1_norm(r,:) = (row1 - min_val) / (max_val - min_val);
        mat2_norm(r,:) = (row2 - min_val) / (max_val - min_val);
    else
        mat1_norm(r,:) = 0.5;
        mat2_norm(r,:) = 0.5;
    end
end
end


function idx = find_closest_bin(time_centers, target_time)
[~, idx] = min(abs(time_centers - target_time));
end