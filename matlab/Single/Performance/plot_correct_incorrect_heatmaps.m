function plot_correct_incorrect_heatmaps(nwbAll, all_units, ...
                                         neural_data_file, bin_width)
% Build and plot Maintenance-epoch heatmaps for all/correct/incorrect and
% a down-sampled correct set matched to incorrect counts.

maintDur   = 2.5;
nBins      = round(maintDur / bin_width);
% rng(20250710);
rng(42)

gKernel = GaussianKernal(0.3 / bin_width, 1.5);

if any(gKernel)
    gKernel = gKernel / sum(gKernel);
end

halfWidth = floor((numel(gKernel) - 1) / 2);
padBins   = min(halfWidth, floor((nBins - 1)/2));

% Precompute edges offsets so each trial has exactly nBins+1 edges.
idx    = 0:nBins;
widths = idx * bin_width;

load(neural_data_file,'neural_data');
nNeurons = numel(neural_data);
assert(nNeurons>0,'No neurons in neural_data.');

all_psth     = nan(nNeurons,nBins);
corr_psth    = nan(nNeurons,nBins);
incorr_psth  = nan(nNeurons,nBins);
corr_ds_psth = nan(nNeurons,nBins);
peak_bins    = nan(nNeurons,1);

for i = 1:nNeurons
    nd  = neural_data(i);
    pid = nd.patient_id;  uid = nd.unit_id;

    idxUnit = find([all_units.subject_id]==pid & ...
                   [all_units.unit_id]==uid, 1);
    if isempty(idxUnit), continue; end
    SU   = all_units(idxUnit);
    sess = nwbAll{SU.session_count};

    tsMai = sess.intervals_trials.vectordata ...
                  .get('timestamps_Maintenance').data.load();
    nTrials = numel(tsMai);
    if nTrials == 0, continue; end

    corr_vals = nd.trial_correctness(:);
    load_vals = nd.trial_load(:);

    if numel(corr_vals) ~= nTrials || numel(load_vals) ~= nTrials
        warning('Trial count mismatch for unit %d – skipping', i);
        continue;
    end

    fr = nan(nTrials,nBins);
    st = SU.spike_times;
    for t = 1:nTrials
        t0    = tsMai(t);
        edges = t0 + widths;
        t1    = edges(end);

        winMask = (st >= t0) & (st <= t1);

        counts = histcounts(st(winMask), edges);
        rate   = counts ./ bin_width;
        rate   = rate(:)';

        if padBins > 0 && nBins > 2*padBins
            % Mirror-pad in time to reduce edge artifacts.
            rate_pad = [ fliplr(rate(1:padBins)), ...
                         rate, ...
                         fliplr(rate(end-padBins+1:end)) ];

            sm_pad   = conv(rate_pad, gKernel, 'same');
            fr(t,:)  = sm_pad(padBins+1 : padBins + nBins);
        else
            fr(t,:) = conv(rate, gKernel, 'same');
        end
    end

    corr_idx =  corr_vals == 1;
    inc_idx  =  corr_vals == 0;

    avg_all  = mean(fr,            1, 'omitnan');
    avg_corr = mean(fr(corr_idx,:),1, 'omitnan');
    avg_inc  = mean(fr(inc_idx ,:),1, 'omitnan');

    % Down-sample correct trials to match incorrect counts within each load.
    loads = [1 2 3];

    corr_idx = (corr_vals == 1);
    inc_idx  = (corr_vals == 0);

    avg_all  = mean(fr,            1, 'omitnan');
    avg_corr = mean(fr(corr_idx,:),1, 'omitnan');
    avg_inc  = mean(fr(inc_idx ,:),1, 'omitnan');

    if ~any(inc_idx)
        avg_corr_ds = nan(1,nBins);
    else
        selected_corr_trials = [];

        for L = loads
            inc_L_mask   = inc_idx  & (load_vals == L);
            corr_L_mask  = corr_idx & (load_vals == L);

            nInc_L  = nnz(inc_L_mask);
            if nInc_L == 0
                continue;
            end

            corr_L_ids = find(corr_L_mask);
            nCorr_L    = numel(corr_L_ids);

            if nCorr_L == 0
                % No correct trials at this load: sample from all correct with replacement.
                base_ids = find(corr_idx);
                if isempty(base_ids)
                    selected_corr_trials = [];
                    break;
                end
                pick_L = randsample(base_ids, nInc_L, true);
            elseif nCorr_L >= nInc_L
                pick_L = randsample(corr_L_ids, nInc_L, false);
            else
                % Not enough correct trials at this load: top up with replacement.
                extra  = randsample(corr_L_ids, nInc_L - nCorr_L, true);
                pick_L = [corr_L_ids; extra];
            end

            selected_corr_trials = [selected_corr_trials; pick_L(:)];
        end

        if isempty(selected_corr_trials)
            avg_corr_ds = avg_corr;
        else
            avg_corr_ds = mean(fr(selected_corr_trials,:), 1, 'omitnan');
        end
    end

    all_psth(i,:)     = avg_all;
    corr_psth(i,:)    = avg_corr;
    incorr_psth(i,:)  = avg_inc;
    corr_ds_psth(i,:) = avg_corr_ds;

    [~, pk] = max(avg_all, [], 'omitnan');
    if isempty(pk) || all(isnan(avg_all)), pk = NaN; end
    peak_bins(i) = pk;
end

% Sort neurons by peak latency in the all-trials PSTH.
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
tCenters  = (0:nBins-1)*bin_width + bin_width/2;
tickStep  = 0.5;
tTicks    = 0:tickStep:maintDur;
xt        = arrayfun(@(t) find_closest_bin(tCenters,t), tTicks);
xLabs     = arrayfun(@(t) sprintf('%.1f',t),tTicks,'uni',0);

makeFig(all_psth_n,   'All Trials');
makeFig(corr_full_n,  'Correct Trials');
makeFig(incorr_n,     'Incorrect Trials (scaled vs. DS-Correct)');
makeFig(corr_ds_n,    'Correct Trials – Down-sampled');

disp('Done – correct/incorrect heat-maps generated.');

fig = figure('Name','Incorrect vs DS-Correct Heatmaps');

t = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');

ax1 = nexttile(t,1);
imagesc(ax1, incorr_n, [0 1]);
colormap(ax1, cmap);
title(ax1, 'Incorrect Trials');
xlabel(ax1,'Time (s)'); ylabel(ax1,'Neuron');
set(ax1,'TickDir','out','XTick',xt,'XTickLabel',xLabs);

ax2 = nexttile(t,2);
imagesc(ax2, corr_ds_n, [0 1]);
colormap(ax2, cmap);
title(ax2, 'Correct Trials – Down-sampled');
xlabel(ax2,'Time (s)'); ylabel(ax2,'Neuron');
set(ax2,'TickDir','out','XTick',xt,'XTickLabel',xLabs);

cb = colorbar;
cb.Layout.Tile = 'east';

fig.Units = 'inches';
fig.Position(3:4) = [8 4];
set([ax1 ax2], 'LooseInset', get(ax1,'TightInset'));
exportgraphics(t, replace(neural_data_file,'.mat','_incorr_vs_corrds_heatmaps.svg'), ...
               'ContentType','vector');

disp('Saved Incorrect and DS-Correct heatmaps as SVG.');

    function makeFig(mat,ttl)
        figure('Name',ttl,'Units','pixels','Position',[100 100 400 648]);
        imagesc(mat,[0 1]); colormap(cmap); colorbar;
        xlabel('Time (s)'); ylabel('Neuron (sorted by peak)');
        title(['Time Cells: ' ttl]);
        set(gca,'TickDir','out','XTick',xt,'XTickLabel',xLabs);
    end
end




function mat = rowwise_minmax_normalize(mat)
% Normalize each row to [0,1] (NaN-safe).

    for r = 1:size(mat,1)
        row = mat(r,:);
        if all(isnan(row)), continue; end
        min_val = min(row, [], 'omitnan');
        max_val = max(row, [], 'omitnan');
        if max_val > min_val
            mask = ~isnan(row);
            rowN = nan(size(row));
            rowN(mask) = (row(mask) - min_val) / (max_val - min_val);
            mat(r,:) = rowN;
        else
            % Constant row (ignoring NaNs): set observed values to 0.5.
            mask = ~isnan(row);
            rowN = nan(size(row));
            rowN(mask) = 0.5;
            mat(r,:) = rowN;
        end
    end
end

function [mat1_norm, mat2_norm] = rowwise_minmax_normalize_together(mat1, mat2)
% Normalize two matrices together row-by-row using a shared min/max.

    mat1_norm = nan(size(mat1));
    mat2_norm = nan(size(mat2));

    for r = 1:size(mat1,1)
        row1 = mat1(r,:);
        row2 = mat2(r,:);
        row_combined = [row1, row2];

        if all(isnan(row_combined))
            mat1_norm(r,:) = row1;
            mat2_norm(r,:) = row2;
            continue;
        end

        min_val = min(row_combined, [], 'omitnan');
        max_val = max(row_combined, [], 'omitnan');

        if max_val > min_val
            m1 = ~isnan(row1); tmp1 = nan(size(row1));
            m2 = ~isnan(row2); tmp2 = nan(size(row2));
            tmp1(m1) = (row1(m1) - min_val) / (max_val - min_val);
            tmp2(m2) = (row2(m2) - min_val) / (max_val - min_val);
            mat1_norm(r,:) = tmp1;
            mat2_norm(r,:) = tmp2;
        else
            % Constant combined row.
            m1 = ~isnan(row1); tmp1 = nan(size(row1)); tmp1(m1) = 0.5;
            m2 = ~isnan(row2); tmp2 = nan(size(row2)); tmp2(m2) = 0.5;
            mat1_norm(r,:) = tmp1;
            mat2_norm(r,:) = tmp2;
        end
    end
end

function idx = find_closest_bin(time_centers, target_time)
% Find the index of the bin center closest to target_time.
    [~, idx] = min(abs(time_centers - target_time));
end
