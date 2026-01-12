function plot_prefEnc1_heatmap(nwbAll, all_units, neural_data, bin_width)
% PLOT_PREFENC1_HEATMAP_VALIDTF
% Plots Encoding-1 heatmap (1s) for neurons with valid time_field.
% Only includes load-1 preferred-image trials for those neurons.

rng(42);  % deterministic sorting

%% Parameters
encOffset   = 1.0;                              % Encoding-1 window: 1 s
nBinsEnc    = round(encOffset / bin_width);    % e.g. 10 bins for 100 ms
gaussKern   = GaussianKernal(0.3 / bin_width, 1.5);

%% Filter for valid time_field
validIdx = find(arrayfun(@(nd) isfield(nd,'time_field') && ...
                                  ~isempty(nd.time_field) && ...
                                  ~isnan(nd.time_field), ...
                         neural_data));

num_neurons = numel(validIdx);
encMat      = nan(num_neurons, nBinsEnc);    % (#neurons × #bins)

%% Main Loop
for i = 1:num_neurons
    nd  = neural_data(validIdx(i));
    pid = nd.patient_id;
    uid = nd.unit_id;
    prefImg = nd.preferred_image;
    imgIDs  = nd.trial_imageIDs;

    if isempty(imgIDs) || isnan(prefImg), continue; end
    if iscell(imgIDs), imgIDs = cell2mat(imgIDs); end

    % Match to all_units for spike times
    uIdx = find([all_units.subject_id]==pid & [all_units.unit_id]==uid, 1);
    if isempty(uIdx), continue; end
    SU   = all_units(uIdx);
    sess = nwbAll{SU.session_count};

    % Get Encoding-1 timestamps
    if ~isKey(sess.intervals_trials.vectordata, 'timestamps_Encoding1')
        continue;
    end
    tsEnc = sess.intervals_trials.vectordata ...
                  .get('timestamps_Encoding1').data.load();

    % Preferred load-1 trials
    enc1 = imgIDs(:,1);  enc2 = imgIDs(:,2);  enc3 = imgIDs(:,3);
    load1Trials = find(enc1 > 0 & enc2 == 0 & enc3 == 0);
    prefTrials  = load1Trials(enc1(load1Trials) == prefImg);

    if numel(prefTrials) < 3, continue; end

    % Extract Encoding-1 firing rate rows
    mat = nan(numel(prefTrials), nBinsEnc);
    for t = 1:numel(prefTrials)
        tr = prefTrials(t);
        if tr > numel(tsEnc), continue; end
        t0 = tsEnc(tr);
        edges = t0 : bin_width : (t0 + encOffset);
        fr = histcounts(SU.spike_times, edges) ./ bin_width;
        mat(t,:) = conv(fr, gaussKern, 'same');
    end

    % Average across trials
    encMat(i,:) = mean(mat, 1, 'omitnan');
end

%% Normalization (min–max per neuron)
encNorm = encMat;
for r = 1:size(encMat,1)
    row = encMat(r,:);
    mn = min(row,[],'omitnan');
    mx = max(row,[],'omitnan');
    if mx <= mn, mn = 0; mx = 1; end
    encNorm(r,:) = (row - mn) ./ (mx - mn);
end

%% Sort neurons by peak bin (ascending latency)
[~, peakBin] = max(encNorm, [], 2);
[~, sortIdx] = sort(peakBin, 'ascend');
encSorted    = encNorm(sortIdx,:);

%% Plot heatmap
figure('Name','Enc1 Heatmap | Valid Time-Field Neurons Only');
imagesc(encSorted);
colormap('parula'); caxis([0 1]); colorbar;

xt = 0:0.2:1.0;
xticks = round(xt / bin_width); xticks(1) = 1;
set(gca, 'XTick', xticks, ...
         'XTickLabel', arrayfun(@(x) sprintf('%.1f',x), xt, 'UniformOutput', false));

xlabel('Time (s)');
ylabel('Neuron (sorted by Enc1 peak)');
title('Encoding-1 | Preferred Trials (Valid Time-Field Only)');

end
