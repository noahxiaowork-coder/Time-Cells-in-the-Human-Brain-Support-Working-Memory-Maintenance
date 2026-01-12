function plot_LIS_integratedHM(nwbAll, all_units, neural_data_file, bin_width, load_level)
% ...
% PLOT_LIS_ENCODING_MAINT – load‑1 trial analysis (Encoding‑1 + Maintenance)
% -----------------------------------------------------------------------
% 1. Extracts spike‑time firing‑rates for BOTH epochs directly from NWB:
%       * Encoding‑1 : 1.0 s window, 100 ms bins  (10 bins)
%       * Maintenance: 2.5 s window, 100 ms bins  (25 bins)
% 2. Builds four matrices (#neurons × #bins) – Pref/NonPref × Enc/Maint.
% 3. Per neuron, min–max normalises all rows using the Maint‑Pref row.
% 4. Sorts neurons late→early by the peak bin of Maint‑Pref.
% 5. Plots three figure windows:
%       (a) Four individual heatmaps (Enc & Maint, Pref & NonPref)
%       (b) Concatenated Enc+Maint heatmaps (Pref and NonPref) over 3.5 s
%       (c) Bar plot comparing Enc‑1 Pref vs NonPref (all 100‑ms bins)
% -----------------------------------------------------------------------
% Noah Kong • April 28 2025 – updated to include concatenated heatmaps

% rng('shuffle'); 
if nargin < 5 || isempty(load_level)
    load_level = 1;     % 1, 2, or 3
end

rng(42)
%% Parameters
binSz        = bin_width;          % 100 ms
encOffset    = 1.0;          % 1 s  (Encoding-1)
maintOffset  = 2.5;          % 2.5 s (Maintenance)

% % ---------- Gaussian smoothing kernel (σ = 2 bins = 0.2 s) -------------
% gaussSigmaBins = 2;                          % σ in *bins*, not seconds
% kSize          = round(2.5 * gaussSigmaBins);  % ±2.5 σ
% 
% % gaussSigmaBins = 1;                          % σ in *bins*, not seconds
% % kSize          = round(3 * gaussSigmaBins);  % ±2.5 σ
% %Less smooth ones
% x              = -kSize : kSize;
% gaussKern      = exp(-(x.^2) / (2 * gaussSigmaBins^2));
% gaussKern      = gaussKern / sum(gaussKern); % unit area

gaussKern = GaussianKernal(0.3 / binSz, 1.5);
% -----------------------------------------------------------------------

nBinsEnc   = round(encOffset   / binSz);     % 10
nBinsMaint = round(maintOffset / binSz);     % 25

load(neural_data_file, 'neural_data');
num_neurons = numel(neural_data);

% Pre‑allocate
enc_pref       = nan(num_neurons, nBinsEnc);
enc_nonpref    = nan(num_neurons, nBinsEnc);
maint_pref     = nan(num_neurons, nBinsMaint);
maint_nonpref  = nan(num_neurons, nBinsMaint);
infoList       = nan(num_neurons,2);

%% Main loop over neurons ------------------------------------------------
for ndx = 1:num_neurons
    nd = neural_data(ndx);
    pid = nd.patient_id;  uid = nd.unit_id;  pImg = nd.preferred_image;
    trial_imageIDs = nd.trial_imageIDs;           % (#trials × 3)

    if iscell(trial_imageIDs)
        trial_imageIDs = cell2mat(trial_imageIDs);   % now a double matrix
    end

    % Match into all_units to get spike‑times & session index
    sIdx = find([all_units.subject_id]==pid & [all_units.unit_id]==uid,1);
    if isempty(sIdx), continue; end
    SU   = all_units(sIdx);
    sess = nwbAll{SU.session_count};

    tsEnc = get_ts(sess,'timestamps_Encoding1');
    tsMai = get_ts(sess,'timestamps_Maintenance');
    if isempty(tsEnc) || isempty(tsMai), continue; end

   % ----------------- Identify load-specific preferred / non-preferred trials
    [trials_this_load, posCol] = select_trials_by_load(trial_imageIDs, load_level);
    
    if isempty(trials_this_load), continue; end
    
    isPrefAtPos = (trial_imageIDs(trials_this_load, posCol) == pImg);
    prefTrials    = trials_this_load(isPrefAtPos);
    nonprefTrials = trials_this_load(~isPrefAtPos);
    
    nMin = min(numel(prefTrials), numel(nonprefTrials));
    if nMin == 0, continue; end
    prefTrials    = datasample(prefTrials,    nMin, 'Replace', false);
    nonprefTrials = datasample(nonprefTrials, nMin, 'Replace', false);


    % ----------------- Average firing‑rate rows
    rowC = avgFR_concat(prefTrials,    tsEnc, tsMai, SU, binSz, ...
                    encOffset, maintOffset, gaussKern);
    enc_pref(ndx,:)      = rowC(1:nBinsEnc);
    maint_pref(ndx,:)    = rowC(nBinsEnc+1:end);
    
    rowC = avgFR_concat(nonprefTrials, tsEnc, tsMai, SU, binSz, ...
                        encOffset, maintOffset, gaussKern);
    enc_nonpref(ndx,:)   = rowC(1:nBinsEnc);
    maint_nonpref(ndx,:) = rowC(nBinsEnc+1:end);

    infoList(ndx,:)    = [pid uid];
end

%% ----------------- Per‑neuron min‑max using Maint‑Pref row
enc_pref_n       = nan(size(enc_pref));
enc_nonpref_n    = nan(size(enc_nonpref));
maint_pref_n     = nan(size(maint_pref));
maint_nonpref_n  = nan(size(maint_nonpref));
for i = 1:num_neurons
    base = maint_pref(i,:);
    if all(isnan(base)), continue; end
    mn = min(base,[],'omitnan');  mx = max(base,[],'omitnan');
    if mx<=mn, mn=0; mx=1; end
    sc = mx-mn;
    enc_pref_n(i,:)      = (enc_pref(i,:)      - mn)./sc;
    enc_nonpref_n(i,:)   = (enc_nonpref(i,:)   - mn)./sc;
    maint_pref_n(i,:)    = (base               - mn)./sc;
    maint_nonpref_n(i,:) = (maint_nonpref(i,:) - mn)./sc;
end

%% ----------------- Sort neurons by Maint‑Pref peak (late → early)
[~,pk]   = max(maint_pref_n,[],2);
[~,ord]  = sort(pk,'ascend');
enc_pref_n      = enc_pref_n(ord,:);
enc_nonpref_n   = enc_nonpref_n(ord,:);
maint_pref_n    = maint_pref_n(ord,:);
maint_nonpref_n = maint_nonpref_n(ord,:);

%% ---------- NEW: per-neuron min–max across Enc-Pref & Enc-NonPref -----
enc_pref_n2    = nan(size(enc_pref));
enc_nonpref_n2 = nan(size(enc_nonpref));

for i = 1:num_neurons
    % gather BOTH Encoding rows for this neuron
    vv = [enc_pref(i,:)  enc_nonpref(i,:)];
    mn = min(vv,[],'omitnan');
    mx = max(vv,[],'omitnan');
    if mx <= mn, mn = 0; mx = 1; end      % guard flat / NaN rows
    scale = mx - mn;

    enc_pref_n2(i,:)    = (enc_pref(i,:)    - mn) ./ scale;
    enc_nonpref_n2(i,:) = (enc_nonpref(i,:) - mn) ./ scale;
end

% keep the same late→early sort order you already derived
enc_pref_n2    = enc_pref_n2(ord,:);
enc_nonpref_n2 = enc_nonpref_n2(ord,:);

%% ---------- NEW PLOT ---------------------------------------------------
plot_enc_heatmaps(enc_nonpref_n2, enc_pref_n2, binSz, encOffset);


%% ----------------- Concatenate Enc + Maint (3.5 s, 35 bins)
cat_pref    = [enc_pref_n,  maint_pref_n ];
cat_nonpref = [enc_nonpref_n, maint_nonpref_n ];

%% --- NEW: split Preferred trials into two random halves --------------
% Re‑build A & B using **all** Preferred trials (no nMin subsampling)
% ----------------------------------------------------------------------
cat_pref_A = nan(num_neurons, size(cat_pref,2));
cat_pref_B = nan(num_neurons, size(cat_pref,2));

for i = 1:num_neurons
    % ---------- book‑keeping for this neuron ---------------------------
    nd   = neural_data(i);
    pid  = nd.patient_id;
    uid  = nd.unit_id;

    sIdx = find([all_units.subject_id]==pid & ...
                [all_units.unit_id]   ==uid , 1);
    if isempty(sIdx), continue; end
    SU   = all_units(sIdx);
    sess = nwbAll{SU.session_count};

    tsEnc = get_ts(sess,'timestamps_Encoding1');
    tsMai = get_ts(sess,'timestamps_Maintenance');
    if isempty(tsEnc) || isempty(tsMai),  continue;  end
    % ---------- ALL preferred trials for the chosen load -------------------
    trial_imageIDs = nd.trial_imageIDs;
    if iscell(trial_imageIDs),  trial_imageIDs = cell2mat(trial_imageIDs); end
    
    [trials_this_load, posCol] = select_trials_by_load(trial_imageIDs, load_level);
    if isempty(trials_this_load), continue; end
    
    allPref = trials_this_load(trial_imageIDs(trials_this_load, posCol) == nd.preferred_image);
    if numel(allPref) < 2
        continue;
    end


    % ---------- random ½‑split, no replacement -------------------------
    ord     = randperm(numel(allPref));
    cut     = floor(numel(allPref)/2);
    idxA    = allPref(ord(1:cut));
    idxB    = allPref(ord(cut+1:end));

    % ---------- build average FR rows (Enc + Maint, 35 bins) ----------
    cat_pref_A(i,:) = avgFR_concat(idxA, tsEnc, tsMai, SU, ...
                                   binSz, encOffset, maintOffset, ...
                                   gaussKern);
    cat_pref_B(i,:) = avgFR_concat(idxB, tsEnc, tsMai, SU, ...
                                   binSz, encOffset, maintOffset, ...
                                   gaussKern);
end

% ----------------- normalise & sort (unchanged) ------------------------
cat_pref_A_n = localMinMax(cat_pref_A, nBinsEnc+1);
cat_pref_B_n = localMinMax(cat_pref_B, nBinsEnc+1);

[~, pkA]  = max(cat_pref_A_n(:, nBinsEnc+1:end), [], 2);
[~, ordA] = sort(pkA, 'ascend');

cat_pref_A_n = cat_pref_A_n(ordA,:);
cat_pref_B_n = cat_pref_B_n(ordA,:);


%% ----------------- Statistics (Enc1 only) -----------------------------
vecP  = enc_pref_n2(:);  vecNP = enc_nonpref_n2(:);
val   = ~isnan(vecP) & ~isnan(vecNP);
[pEnc,~,~,stats] = ttest(vecP(val), vecNP(val),'Tail','right'); %#ok<ASGLU>
barM = [mean(vecNP(val)), mean(vecP(val))];
barS = [std(vecNP(val))/sqrt(sum(val)), std(vecP(val))/sqrt(sum(val))];

%% ----------------- Plotting -------------------------------------------
% ----- (a) Individual epoch heatmaps (unchanged helper) ---------------
plot_individual_heatmaps(enc_nonpref_n, enc_pref_n, ...
                         maint_nonpref_n, maint_pref_n, ...
                         binSz, encOffset, maintOffset);

% ----- (b) Concatenated maps in **separate** windows -------------------
plot_concatenated_single(cat_nonpref, binSz, encOffset+maintOffset, ...
                         'Non‑Preferred (0–3.5 s)');

plot_concatenated_single(cat_pref,    binSz, encOffset+maintOffset, ...
                         'Preferred (0–3.5 s)');

% ----- (c) Half‑split Preferred maps -----------------------------------
plot_concatenated_single(cat_pref_A_n, binSz, encOffset+maintOffset, ...
                         'Pref Half‑A (sort = own Maint peak)');

plot_concatenated_single(cat_pref_B_n, binSz, encOffset+maintOffset, ...
                         'Pref Half‑B (row‑order = Half‑A)');

end

%% =====================================================================
% Helper functions -----------------------------------------------------
function ts = get_ts(sess,key)
if isKey(sess.intervals_trials.vectordata,key)
    ts = sess.intervals_trials.vectordata.get(key).data.load();
else, ts = []; end
end

function row = avgFR(trials, ts, SU, binSz, offset, ker)
% Returns a single [1 × #bins] vector:
%   – spikes are binned per trial,
%   – each trial vector is Gaussian-smoothed (conv 'same'),
%   – finally averaged across trials.
    nBins = round(offset / binSz);
    if isempty(trials) || isempty(ts)
        row = nan(1,nBins);  return;
    end

    mat = nan(numel(trials), nBins);
    for k = 1:numel(trials)
        idx = trials(k);  if idx > numel(ts), continue; end
        t0   = ts(idx);
        edges = t0 : binSz : (t0 + offset);
        FR    = histcounts(SU.spike_times, edges) ./ binSz; % spikes s⁻¹
        FR    = conv(FR, ker, 'same');                      % ⟵ smoothing
        mat(k,:) = FR;
    end
    row = mean(mat,1,'omitnan');
end

function plot_individual_heatmaps(eNP,eP,mNP,mP,binSzE,offE,offM)
figure('Name','Separate Heatmaps');

% ---------- Encoding – Non‑Pref ---------------------------------------
subplot(2,2,1);  imagesc(eNP);  colormap('parula');  caxis([0 1]);  colorbar;
xt      = 0:0.5:offE;                    % 0 → 1 s
xt_idx  = round(xt/binSzE);  xt_idx(1)=1;
set(gca,'XTick',xt_idx,'XTickLabel',cellstr(num2str(xt','%.1f')));
ylabel('Neuron');  title('Enc – Non‑Pref');

% ---------- Encoding – Pref -------------------------------------------
subplot(2,2,2);  imagesc(eP);  colormap('parula');  caxis([0 1]);  colorbar;
set(gca,'XTick',xt_idx,'XTickLabel',cellstr(num2str(xt','%.1f')));
title('Enc – Pref');

% ---------- Maintenance – Non‑Pref ------------------------------------
subplot(2,2,3);  imagesc(mNP);  colormap('parula');  caxis([0 1]);  colorbar;
xt2     = 0:0.5:offM;                     % 0 → 2.5 s
xt2_idx = round(xt2/binSzE);  xt2_idx(1)=1;
set(gca,'XTick',xt2_idx,'XTickLabel',cellstr(num2str(xt2','%.1f')));
ylabel('Neuron');  title('Maint – Non‑Pref');

% ---------- Maintenance – Pref ----------------------------------------
subplot(2,2,4);  imagesc(mP);  colormap('parula');  caxis([0 1]);  colorbar;
set(gca,'XTick',xt2_idx,'XTickLabel',cellstr(num2str(xt2','%.1f')));
title('Maint – Pref');
end

% -------- Concatenated map in its own window + red divider -------------
function plot_concatenated_single(mat, binSz, totalT, ttl)
    figure('Name',ttl, 'Position',[400, 400, 400, 648]);
    imagesc(mat); colormap('parula'); caxis([0 1]); colorbar;

    xt     = 0:0.5:totalT;               % 0 → 3.5 s
    xt_idx = round(xt/binSz);  xt_idx(1)=1;
    set(gca,'XTick',xt_idx,'XTickLabel',cellstr(num2str(xt','%.1f')), 'FontSize', 20);

    ylabel('Img-spec Time Cells', FontSize=20); xlabel('Time (s)', FontSize=20);
    title(ttl);

    hold on
    xline(round(1/binSz),'r--','LineWidth',3);   % 1 s divider
    hold off
end

function plot_bar(mn,sem,p)
figure('Name','Enc Pref vs NonPref');
bar(mn,'FaceColor','flat'); hold on;
errorbar(1:2,mn,sem,'.k','LineWidth',1.5);
set(gca,'XTick',1:2,'XTickLabel',{'Non‑Pref','Pref'});
ylabel('Normalised FR'); 
title('Enc1 (all 100 ms bins)');

if p < 0.05
    % Set Y position for the significance line
    y = max(mn + sem) * 1.05; 
    line([1 2], [y y], 'Color', 'k', 'LineWidth', 1.5); % Horizontal line
    % Add small "brackets" (ticks) going downward
    tick_length = 0.02 * y; % Length of the small brackets
    line([1 1], [y y - tick_length], 'Color', 'k', 'LineWidth', 1.5);
    line([2 2], [y y - tick_length], 'Color', 'k', 'LineWidth', 1.5);
    
    % Determine significance stars
    if p < 0.001
        stars = '***';
    elseif p < 0.01
        stars = '**';
    else
        stars = '*';
    end
    
    % Place the stars slightly above the horizontal line
    text(1.5, y - 0.5*tick_length, stars, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 20)
end
end



%% Tiny helper
function cells = num2cellstr(vec)
% vec → cellstr with one decimal
cells = arrayfun(@(x) sprintf('%.1f',x), vec, 'UniformOutput', false);
end

%% ------------------ HELPER FUNCTION -----------------------
function rowFR = averageEncodingFR(trialInds, tsEnc, SU, binSz, offset)
% For each trial in trialInds, bin spikes from trial onset to onset+offset,
% then average across those trials, returning a row vector [1 x #bins].
    if isempty(trialInds) || isempty(tsEnc)
        rowFR = nan(1, round(offset/binSz));
        return;
    end
    allFR = [];
    for t = 1:length(trialInds)
        idxTrial = trialInds(t);
        if idxTrial > length(tsEnc), continue; end
        tOnset   = tsEnc(idxTrial);

        bin_edges   = tOnset : binSz : (tOnset + offset);
        spikeCounts = histcounts(SU.spike_times, bin_edges);
        FR          = spikeCounts / binSz;  % spikes/sec
        allFR       = [allFR; FR]; %#ok<AGROW>
    end
    rowFR = mean(allFR,1,'omitnan');
end

function row = avgFR_concat(trials, tsEnc, tsMai, SU, binSz, offE, offM, ker)
% Returns a single 1 × 35 vector (10 + 25 bins):
%   – Enc-1 and Maintenance binned per trial
%   – concatenated, Gaussian-smoothed ('same')
%   – finally averaged across trials
    nEnc  = round(offE / binSz);      % 10
    nMaint= round(offM / binSz);      % 25
    nTot  = nEnc + nMaint;           % 35

    if isempty(trials) || isempty(tsEnc) || isempty(tsMai)
        row = nan(1,nTot);  return;
    end

    mat = nan(numel(trials), nTot);
    for k = 1:numel(trials)
        idx = trials(k);
        if idx > numel(tsEnc) || idx > numel(tsMai),  continue;  end

        % -------- Encoding-1 segment (0–1 s)
        tE  = tsEnc(idx);
        edgesE  = tE : binSz : (tE + offE);
        frEnc   = histcounts(SU.spike_times, edgesE) ./ binSz;  % 1 × 10

        % -------- Maintenance segment (0–2.5 s relative to its onset)
        tM  = tsMai(idx);
        edgesM = tM : binSz : (tM + offM);
        frMaint = histcounts(SU.spike_times, edgesM) ./ binSz;  % 1 × 25

        % -------- Concatenate and smooth across the 35 bins
        fr = conv([frEnc frMaint], ker, 'same');                % 1 × 35
        mat(k,:) = fr;
    end
    row = mean(mat,1,'omitnan');
end

function plot_enc_heatmaps(eNP, eP, binSz, offE)
figure('Name','Enc Pref vs Enc NonPref ‒ per‑neuron min–max');

xt     = 0:0.5:offE;
xt_idx = round(xt/binSz);  xt_idx(1)=1;

% Non‑Pref --------------------------------------------------------------
subplot(1,2,1);  imagesc(eNP);  colormap('parula');  caxis([0 1]);  colorbar;
set(gca,'XTick',xt_idx,'XTickLabel',cellstr(num2str(xt','%.1f')));
ylabel('Neuron');  title('Encoding – Non‑Pref');

% Pref ------------------------------------------------------------------
subplot(1,2,2);  imagesc(eP);   colormap('parula');  caxis([0 1]);  colorbar;
set(gca,'XTick',xt_idx,'XTickLabel',cellstr(num2str(xt','%.1f')));
title('Encoding – Pref');
end


function plot_scatter_with_mean(data1, data2, p)
% data1 and data2 are vectors of samples for the two conditions

figure('Name','Enc1 Pref vs NonPref');
hold on;

% Jitter for better visualization (small random x offset)
jitterAmount = 0.15; 

% Plot data points
scatter(ones(size(data1)) + randn(size(data1)) * jitterAmount, data1, 'filled');
scatter(2 * ones(size(data2)) + randn(size(data2)) * jitterAmount, data2, 'filled');

% Plot mean lines
mean1 = mean(data1);
mean2 = mean(data2);
plot([1 - 0.2, 1 + 0.2], [mean1 mean1], 'k-', 'LineWidth', 2);
plot([2 - 0.2, 2 + 0.2], [mean2 mean2], 'k-', 'LineWidth', 2);

% Axes and labels
set(gca, 'XTick', [1 2], 'XTickLabel', {'Non‑Pref','Pref'});
ylabel('Normalised FR');
title('Enc1 (all 100 ms bins)');
xlim([0.5 2.5]);

% Statistical significance marker
if p < 0.05
    y = max([data1(:); data2(:)]) * 1.05;
    line([1 2], [y y], 'Color', 'k', 'LineWidth', 1.5);
    tick_length = 0.02 * y;
    line([1 1], [y y - tick_length], 'Color', 'k', 'LineWidth', 1.5);
    line([2 2], [y y - tick_length], 'Color', 'k', 'LineWidth', 1.5);

    % Determine significance stars
    if p < 0.001
        stars = '***';
    elseif p < 0.01
        stars = '**';
    else
        stars = '*';
    end
    
    text(1.5, y - 0.5 * tick_length, stars, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 20);
end

hold off;
end

% ------------ SCALE EACH ROW by its own Maintenance min‑max ------------
function matN = localMinMax(mat, maintStartCol)
% mat             – (#neurons × #bins) matrix
% maintStartCol   – column index of the first Maintenance bin
%
% Returns the same‑size matrix with every row rescaled to [0,1] using
% that row’s min & max taken **only from its Maintenance segment**.

    matN = nan(size(mat));
    for r = 1:size(mat,1)
        base = mat(r, maintStartCol:end);           % the 25 Maint bins
        if all(isnan(base)), continue; end
        mn = min(base,[],'omitnan');  mx = max(base,[],'omitnan');
        if mx <= mn                    % flat or NaN row guard
            mn = 0; mx = 1;
        end
        matN(r,:) = (mat(r,:) - mn) ./ (mx - mn);
    end
end

function [trials, posCol] = select_trials_by_load(trial_imageIDs, load_level)
% Returns:
%   trials : linear indices of trials that match the specified load
%   posCol : which image position (1/2/3) defines "preferred" at that load
%
% Conventions assumed:
%   load=1 : only 1st image present (col1~=0, col2==0, col3==0)
%   load=2 : first two images present (col1~=0, col2~=0, col3==0)
%   load=3 : all three present (col1~=0, col2~=0, col3~=0)

    switch load_level
        case 1
            mask  = (trial_imageIDs(:,1) ~= 0) & ...
                    (trial_imageIDs(:,2) == 0) & ...
                    (trial_imageIDs(:,3) == 0);
            posCol = 1;

        case 2
            mask  = (trial_imageIDs(:,1) ~= 0) & ...
                    (trial_imageIDs(:,2) ~= 0) & ...
                    (trial_imageIDs(:,3) == 0);
            posCol = 2;

        case 3
            mask  = (trial_imageIDs(:,1) ~= 0) & ...
                    (trial_imageIDs(:,2) ~= 0) & ...
                    (trial_imageIDs(:,3) ~= 0);
            posCol = 3;

        otherwise
            error('load_level must be 1, 2, or 3.');
    end

    trials = find(mask);
end
