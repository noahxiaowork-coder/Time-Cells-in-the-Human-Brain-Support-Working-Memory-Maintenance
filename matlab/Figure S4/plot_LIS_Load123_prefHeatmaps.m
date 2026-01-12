function plot_LIS_Load123_prefHeatmaps(nwbAll, all_units, neural_data_file, ref_load, LocalN)
% plot_LIS_Load123_prefHeatmaps  Plots concatenated preferred-image heatmaps for loads 1-3.
%
% Syntax
%   plot_LIS_Load123_prefHeatmaps(nwbAll, all_units, neural_data_file, ref_load)
%   plot_LIS_Load123_prefHeatmaps(..., LocalN)
%
% Inputs
%   nwbAll            – cell array of NWB files
%   all_units         – struct array with all single-unit metadata
%   neural_data_file  – .mat file that stores "neural_data" variable
%   ref_load          – reference load for sorting / global-ref normalisation (1 | 2 | 3)
%   LocalN (logical)  – when true, each heat-map row is normalised to its own
%                        min-max in the maintenance period (local normalisation).
%                        Sorting is **still** performed with respect to *ref_load*.
%                        Default: false (legacy, ref-based normalisation).
%
% Description
%   The function concatenates firing-rate matrices for three working-memory loads.
%   Rows are sorted by the peak maintenance response in *ref_load*.  Two
%   normalisation modes are available:
%     • LocalN = false  – Min-max of each row is taken from the maintenance
%                         segment of *ref_load* (legacy behaviour).
%     • LocalN = true   – Min-max of each row is taken from its **own**
%                         maintenance segment, but rows are *still* ordered
%                         according to *ref_load*.
%
%   For Load-2, preferred trials are further split into:
%     • pref-X   – preferred image in Encoding-1
%     • X-pref   – preferred image in Encoding-2
%   and visualised as two separate heat-maps.
% -------------------------------------------------------------------------
% 2025-06-04  X.Kong / GPT-4o-3
% 2025-12-04  L2 split into pref-X vs X-pref heatmaps
% -------------------------------------------------------------------------

%% ----------------------------------------------------------------------
% Input handling
% -----------------------------------------------------------------------
if nargin < 5 || isempty(LocalN)
    LocalN = false;            % backward-compatibility
end

if ~isscalar(ref_load) || ~ismember(ref_load, [1 2 3])
    error('ref_load must be 1, 2, or 3');
end

%% ----------------------------------------------------------------------
% Parameters
% -----------------------------------------------------------------------
rng(42)
binSz     = 0.1;      % s
encDur    = 1.0;      % s
maintDur  = 2.5;      % s
encBins   = round(encDur   / binSz);
maintBins = round(maintDur / binSz);
nTot_L1   = encBins       + maintBins;
nTot_L2   = 2*encBins     + maintBins;
nTot_L3   = 3*encBins     + maintBins;

pxPerSecond = 100;      % controls *all* figure widths
barFrac      = 0.06;    % 6 % of fig height  → mini-bar height
fontSzBar    = 10;      % label size inside the bar
colPref      = [1 0.6 0.6];   % light red
colEnc       = [0.65 0.65 0.65];

gaussKern = GaussianKernal(0.3 / binSz, 1.5);


% ensure kernel is a normalized row vector
gaussKern = gaussKern(:)';
if any(gaussKern)
    gaussKern = gaussKern / sum(gaussKern);
end



%% ----------------------------------------------------------------------
% Load neural data
% -----------------------------------------------------------------------
load(neural_data_file, 'neural_data');
num_neurons = numel(neural_data);

cat_L1_P      = nan(num_neurons, nTot_L1);
cat_L1_NP     = nan(num_neurons, nTot_L1);
cat_L2_P_comb = nan(num_neurons, nTot_L2);  % combined pref-X & X-pref (for ref/stats)
cat_L2_P_prefX  = nan(num_neurons, nTot_L2); % NEW: pref-X
cat_L2_P_Xpref  = nan(num_neurons, nTot_L2); % NEW: X-pref
cat_L3_P      = nan(num_neurons, nTot_L3);

%% ----------------------------------------------------------------------
% Build concatenated matrices
% -----------------------------------------------------------------------
for i = 1:num_neurons
    nd  = neural_data(i);
    pid = nd.patient_id;  uid = nd.unit_id;
    sIdx = find([all_units.subject_id]==pid & [all_units.unit_id]==uid,1);
    if isempty(sIdx), continue; end
    SU   = all_units(sIdx);
    sess = nwbAll{SU.session_count};

    % Get trial-aligned timestamps
    tsEnc1 = get_ts(sess, 'timestamps_Encoding1');
    tsEnc2 = get_ts(sess, 'timestamps_Encoding2');
    tsEnc3 = get_ts(sess, 'timestamps_Encoding3');
    tsMai  = get_ts(sess, 'timestamps_Maintenance');
    if isempty(tsEnc1) || isempty(tsMai), continue; end

    trial_imageIDs = nd.trial_imageIDs;
    if iscell(trial_imageIDs), trial_imageIDs = cell2mat(trial_imageIDs); end
    pImg = nd.preferred_image;

    % --- Load-1 trials --------------------------------------------------
    L1 = find(trial_imageIDs(:,1)~=0 & trial_imageIDs(:,2)==0 & trial_imageIDs(:,3)==0);
    pref_L1    = L1(trial_imageIDs(L1,1)==pImg);
    nonpref_L1 = L1(trial_imageIDs(L1,1)~=pImg);

    if ~isempty(pref_L1)
        cat_L1_P(i,:) = buildLoadRow(pref_L1, tsEnc1, [], [], tsMai, 1, SU, gaussKern, binSz);
    end
    if ~isempty(nonpref_L1)
        cat_L1_NP(i,:) = buildLoadRow(nonpref_L1, tsEnc1, [], [], tsMai, 1, SU, gaussKern, binSz);
    end

    % --- Load-2 trials: split into pref-X vs X-pref ---------------------
    % pref-X  : preferred image in Encoding-1, 2nd slot non-zero, no 3rd
    L2_prefX = find(trial_imageIDs(:,1)==pImg & trial_imageIDs(:,2)~=0 & trial_imageIDs(:,3)==0);
    % X-pref  : preferred image in Encoding-2, no 3rd
    L2_Xpref = find(trial_imageIDs(:,2)==pImg & trial_imageIDs(:,3)==0);

    if ~isempty(L2_prefX)
        cat_L2_P_prefX(i,:) = buildLoadRow(L2_prefX, tsEnc1, tsEnc2, [], tsMai, 2, SU, gaussKern, binSz);
    end
    if ~isempty(L2_Xpref)
        cat_L2_P_Xpref(i,:) = buildLoadRow(L2_Xpref, tsEnc1, tsEnc2, [], tsMai, 2, SU, gaussKern, binSz);
    end

    % For reference / stats, use a combined L2 (mean over available variants)
    if ~all(isnan(cat_L2_P_prefX(i,:))) || ~all(isnan(cat_L2_P_Xpref(i,:)))
        tmp = [cat_L2_P_prefX(i,:); cat_L2_P_Xpref(i,:)];
        cat_L2_P_comb(i,:) = mean(tmp,1,'omitnan');
    end

    % --- Load-3 trials --------------------------------------------------
    L3 = find(trial_imageIDs(:,3)==pImg);
    if ~isempty(L3)
        cat_L3_P(i,:) = buildLoadRow(L3, tsEnc1, tsEnc2, tsEnc3, tsMai, 3, SU, gaussKern, binSz);
    end
end

%% ----------------------------------------------------------------------
% Reference matrix + divCol for vertical demarcation
% -----------------------------------------------------------------------
switch ref_load
    case 1
        refMat  = cat_L1_P;          divCol = encBins       + 1;
    case 2
        refMat  = cat_L2_P_Xpref;    divCol = 2*encBins     + 1;  % <-- X-PREF ONLY
    case 3
        refMat  = cat_L3_P;          divCol = 3*encBins     + 1;
end

%% ----------------------------------------------------------------------
% Normalisation
% -----------------------------------------------------------------------
if LocalN
    % ---- Local row-wise min-max ---------------------------------------
    cat_L1_P_n        = localMinMax(cat_L1_P,        divCol);
    cat_L1_NP_n       = localMinMax(cat_L1_NP,       divCol);
    cat_L2_P_comb_n   = localMinMax(cat_L2_P_comb,   divCol);
    cat_L2_P_prefX_n  = localMinMax(cat_L2_P_prefX,  divCol);
    cat_L2_P_Xpref_n  = localMinMax(cat_L2_P_Xpref,  divCol);
    cat_L3_P_n        = localMinMax(cat_L3_P,        divCol);
else
    % ---- Legacy: min-max taken from refMat ----------------------------
    cat_L1_P_n        = localMinMax_fromRef(cat_L1_P,        refMat, divCol);
    cat_L1_NP_n       = localMinMax_fromRef(cat_L1_NP,       refMat, divCol);
    cat_L2_P_comb_n   = localMinMax_fromRef(cat_L2_P_comb,   refMat, divCol);
    cat_L2_P_prefX_n  = localMinMax_fromRef(cat_L2_P_prefX,  refMat, divCol);
    cat_L2_P_Xpref_n  = localMinMax_fromRef(cat_L2_P_Xpref,  refMat, divCol);
    cat_L3_P_n        = localMinMax_fromRef(cat_L3_P,        refMat, divCol);
end

% Matrix used for sorting is **always** the refMat with its own local
% normalisation so that ordering logic is independent of LocalN.
norm_refMat = localMinMax(refMat, divCol);

% Sort by time-to-peak within maintenance period
[~, pk]  = max(norm_refMat(:, divCol:end), [], 2, 'includenan');
[~, ord] = sort(pk, 'ascend');

% Apply order to *all* matrices
cat_L1_P_n        = cat_L1_P_n(ord, :);
cat_L1_NP_n       = cat_L1_NP_n(ord, :);
cat_L2_P_comb_n   = cat_L2_P_comb_n(ord, :);
cat_L2_P_prefX_n  = cat_L2_P_prefX_n(ord, :);
cat_L2_P_Xpref_n  = cat_L2_P_Xpref_n(ord, :);
cat_L3_P_n        = cat_L3_P_n(ord, :);

%% ----------------------------------------------------------------------
% Plotting & saving
% -----------------------------------------------------------------------
save_path = '/Users/xiangxuankong/Desktop/Human TC Support WM/Figures/Figure S5';

PX = 12;            % keep block pixel size constant

% Load-1 (non-pref + pref) –- no colour bar
plot_concatenated_single(cat_L1_NP_n, binSz, encDur+maintDur, ...
    '', encBins+1, save_path, 'L1_NP_refL3', PX, false);
plot_concatenated_single(cat_L1_P_n , binSz, encDur+maintDur, ...
    '', encBins+1, save_path, 'L1_P_refL3' , PX, false);

% Load-2 –- TWO versions: pref-X and X-pref (both no colour bar)
plot_concatenated_single(cat_L2_P_prefX_n , binSz, 2*encDur+maintDur, ...
    '', 2*encBins+1, save_path,'L2_prefX_refL3', PX, false);
plot_concatenated_single(cat_L2_P_Xpref_n , binSz, 2*encDur+maintDur, ...
    '', 2*encBins+1, save_path,'L2_Xpref_refL3', PX, false);

% (Optional) If you still want a single combined L2 map, uncomment:
% plot_concatenated_single(cat_L2_P_comb_n , binSz, 2*encDur+maintDur, ...
%     '', 2*encBins+1, save_path,'L2_P_comb_refL3', PX, false);

% Load-3 –- keep a single colour bar
plot_concatenated_single(cat_L3_P_n , binSz, 3*encDur+maintDur, ...
    '', 3*encBins+1, save_path,'L3_P_refL3', PX, true);

%% ----------------------------------------------------------------------
% Firing Pattern Similarity (Pearson r) – color-coded bars + black sig bars
% -----------------------------------------------------------------------
% Use the combined L2 matrix for load-2 similarity
refMats_sorted = {cat_L1_P_n, cat_L2_P_Xpref_n, cat_L3_P_n};   % <-- X-PREF ONLY
divCols        = [encBins+1, 2*encBins+1, 3*encBins+1];
load_labels    = {'Load1','Load2','Load3'};

load_colors    = {[0 0 1], [0 0.6 0], [1 0 0]}; % 1=blue, 2=green, 3=red

% Reference (normalized, sorted)
ref_sorted_norm = refMats_sorted{ref_load};

% Pearson r vs reference for each load
pearson_mat = nan(size(ref_sorted_norm,1), 3);
for j = 1:3
    test_sorted_norm = refMats_sorted{j};
    for i = 1:size(ref_sorted_norm,1)
        ref_row  = ref_sorted_norm(i, divCols(ref_load):end);
        test_row = test_sorted_norm(i,  divCols(j):end);
        pearson_mat(i,j) = corr(ref_row', test_row', 'Rows','pairwise');
    end
end

% Build bars: two non-ref loads
compLoads = setdiff(1:3, ref_load);
labels    = load_labels(compLoads);
dataCols  = [pearson_mat(:,compLoads(1)), pearson_mat(:,compLoads(2))];
barColors = [load_colors{compLoads(1)}; load_colors{compLoads(2)}];

% Restrict to neurons with all values
idxBoth = all(~isnan(dataCols),2);
X = dataCols(idxBoth,:);

% Fisher z mean ± SEM → r-space
Z   = atanh(X);
zmu = mean(Z,1,'omitnan');
zse = std(Z,0,1,'omitnan') ./ sqrt(sum(~isnan(Z),1));
rmu = tanh(zmu);
rse = (1 - rmu.^2) .* zse;

% Plot
figure('Name','Firing Pattern Similarity');
hb = bar(rmu,'FaceColor','flat');
for b = 1:numel(rmu)
    hb.CData(b,:) = barColors(b,:);
end
hold on
errorbar(1:numel(rmu), rmu, rse, 'k', 'LineStyle','none', 'LineWidth',1.5);

set(gca,'XTick',1:numel(rmu), 'XTickLabel', labels, 'FontSize',20);
ylabel('Pearson r','FontSize',20);
title(sprintf('Firing Pattern Similarity (reference = %s)', load_labels{ref_load}));

% Paired t-tests + black sig bars
pairs = nchoosek(1:numel(rmu),2);
yl = ylim; yBase = yl(2); yStep = 0.06*(yl(2)-yl(1));
for k = 1:size(pairs,1)
    a = pairs(k,1); b = pairs(k,2);
    [~,p,~,st] = ttest(X(:,a), X(:,b), "Tail","right");
    fprintf('Paired t-test %s vs %s (n=%d): t(%d)=%.3f, p=%.4g\n', ...
        labels{a}, labels{b}, size(X,1), st.df, st.tstat, p);
    y = yBase + (k-1)*yStep;
    addSigStar(gca,a,b,y,p); % helper below
end
ylim([yl(1), yBase + size(pairs,1)*yStep + 0.02*(yl(2)-yl(1))]);
hold off

% Save stats
save(fullfile(save_path, sprintf('pearson_similarity_refL%d.mat', ref_load)), ...
     'pearson_mat','compLoads','idxBoth','labels','rmu','rse','pairs');

%% Helper: significance star (black bar + black text)
function addSigStar(ax, x1, x2, y, p)
    if p < 1e-3, star = '***';
    elseif p < 1e-2, star = '**';
    elseif p < 0.05, star = '*';
    else, star = 'n.s.';
    end
    line(ax, [x1 x1 x2 x2], [y-0.01 y y y-0.01], 'Color','k', 'LineWidth',1.5);
    text(ax, mean([x1 x2]), y + 0.01, star, ...
        'HorizontalAlignment','center','FontSize',16, ...
        'FontWeight','bold','Color','k');
end



end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                              HELPERS                                 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function ts = get_ts(sess, key)
    % Convenience accessor for interval timestamps
    if isKey(sess.intervals_trials.vectordata, key)
        ts = sess.intervals_trials.vectordata.get(key).data.load();
    else
        ts = [];
    end
end

function matN = localMinMax(mat, maintStartCol)
    % Row-wise min-max *within the same matrix* using maintenance segment
    matN = nan(size(mat));
    for r = 1:size(mat,1)
        base = mat(r, maintStartCol:end);
        if all(isnan(base)), continue; end
        mn = min(base, [], 'omitnan');
        mx = max(base, [], 'omitnan');
        if mx <= mn, mn = 0; mx = 1; end
        matN(r,:) = (mat(r,:) - mn) / (mx - mn);
    end
end

function matN = localMinMax_fromRef(mat, ref, maintStartCol)
    % Row-wise min-max where min & max are taken from *ref* matrix
    matN = nan(size(mat));
    for r = 1:size(mat,1)
        base = ref(r, maintStartCol:end);
        if all(isnan(base)), continue; end
        mn = min(base, [], 'omitnan');
        mx = max(base, [], 'omitnan');
        if mx <= mn, mn = 0; mx = 1; end
        matN(r,:) = (mat(r,:) - mn) / (mx - mn);
    end
end

function row = buildLoadRow(trials, ts1, ts2, ts3, tsM, loadTag, SU, ker, binSz)
    % Concatenate encoding (×loadTag) + maintenance, smooth with Gaussian,
    % and average across trials.
    %
    % Uses mirror padding to avoid Gaussian edge artifacts.
    % Guards against empty or degenerate kernels.

    % ---- early exit if no trials ----
    if isempty(trials)
        row = nan(1, 10*loadTag + 25);   % 10 bins per 1 s enc, 25 for 2.5 s maint
        return;
    end

    nBins = 10*loadTag + 25;            % total bins for this load
    nT    = numel(trials);
    mat   = nan(nT, nBins);

    % ---- prepare kernel safely ----
    k = ker(:)';                        % ensure row
    if isempty(k) || all(k==0)
        useSmoothing = false;
    else
        k = k / sum(k);                 % normalize again (cheap & safe)
        useSmoothing = true;
    end

    % ---- loop over trials ----
    for it = 1:nT
        t  = trials(it);
        fr = [];

        % Encoding 1
        if ~isempty(ts1)
            edges1 = ts1(t):binSz:(ts1(t)+1.0);
            c1     = histcounts(SU.spike_times, edges1);
            fr     = [fr, c1 ./ binSz];
        end

        % Encoding 2 (for load ≥2)
        if loadTag >= 2 && ~isempty(ts2)
            edges2 = ts2(t):binSz:(ts2(t)+1.0);
            c2     = histcounts(SU.spike_times, edges2);
            fr     = [fr, c2 ./ binSz];
        end

        % Encoding 3 (for load 3)
        if loadTag == 3 && ~isempty(ts3)
            edges3 = ts3(t):binSz:(ts3(t)+1.0);
            c3     = histcounts(SU.spike_times, edges3);
            fr     = [fr, c3 ./ binSz];
        end

        % Maintenance (0–2.5 s)
        if ~isempty(tsM)
            edgesM = tsM(t):binSz:(tsM(t)+2.5);
            cM     = histcounts(SU.spike_times, edgesM);
            fr     = [fr, cM ./ binSz];
        end

        % If something went wrong with timing, skip this trial
        if numel(fr) ~= nBins
            continue;
        end

        fr_row = fr(:)';   % ensure row vector

        if ~useSmoothing
            % No valid kernel: just store unsmoothed rate
            mat(it,:) = fr_row;
        else
            % ---- mirror-pad to avoid edge artifacts ----
            nTot    = nBins;                        % == numel(fr_row)
            kerHalf = floor((numel(k) - 1) / 2);
            padBins = min(kerHalf, floor((nTot - 1)/2));  % cap for safety

            if padBins > 0
                % Mirror-pad left and right
                fr_pad = [ fliplr(fr_row(1:padBins)), ...
                           fr_row, ...
                           fliplr(fr_row(end-padBins+1:end)) ];

                sm_pad    = conv(fr_pad, k, 'same');
                mat(it,:) = sm_pad(padBins+1 : padBins+nTot);
            else
                % Kernel too short or window too small → fallback
                mat(it,:) = conv(fr_row, k, 'same');
            end
        end
    end

    % ---- average across trials for this neuron and load ----
    row = mean(mat, 1, 'omitnan');
end


function plot_concatenated_single(mat, binSz, totalT, ttl, div_col, ...
                                  save_path, tag, pxPerBin, showColorbar)
% plot_concatenated_single  Variable-width heat-map with optional colour bar
%                           and a top segment bar that always matches the
%                           data panel (ticks & divider aligned to bin edges).
%
% Required (first six) arguments are unchanged.
%
% Optional
%   pxPerBin      – pixels per 0.1-s bin   (default 12)
%   showColorbar  – true / false           (default true)
%
% ------------------------------------------------------------ 2025-08-07 %

if nargin < 8 || isempty(pxPerBin),     pxPerBin     = 12;  end
if nargin < 9 || isempty(showColorbar), showColorbar = true; end

%% 1 ▸ MAIN HEAT-MAP
hFig = figure('Name', ttl);
ax   = axes('Parent', hFig);
imagesc(ax, mat);
colormap(ax, 'parula');
caxis(ax, [0 1]);

% Keep axis width stable relative to the colorbar choice
if showColorbar
    colorbar(ax);
else
    set(ax,'Position',[0.13 0.11 0.815 0.815]);  % slightly wider if no colorbar
end

% Explicit data-edge limits (so centers are j and edges are j±0.5)
nBins = size(mat,2);
xlim(ax, [0.5, nBins + 0.5]);

% ----- Ticks on EDGES (align with divider) -----
tstep       = 0.5;                               % seconds per major tick
xt          = 0:tstep:totalT;                    % tick times in seconds
t_boundary  = (div_col - 1) * binSz;             % end-of-encoding time (s)
xt          = [xt, t_boundary];                  % ensure boundary tick exists
xt          = unique(round(xt,10), 'stable');    % de-dupe w/ numeric safety
xt_pos      = xt ./ binSz + 0.5;                 % map time → edge coordinates

% Clip to x-limits (robust to FP noise)
mask        = (xt_pos >= 0.5 - 1e-9) & (xt_pos <= nBins + 0.5 + 1e-9);
xt_pos      = xt_pos(mask);
xt          = xt(mask);

set(ax,'XTick',xt_pos, ...
       'XTickLabel',cellstr(num2str(xt','%.1f')), ...
       'FontSize',14);

ylabel(ax,'Image-specific Time Cells','FontSize',14);
xlabel(ax,'Time (s)','FontSize',14);

hold(ax,'on');
xline(ax, div_col - 0.5, 'r--', 'LineWidth', 3);  % true bin edge
hold(ax,'off');

%% 2 ▸ TOP SEGMENT BAR (uses normalized coords, matches main axis width)
nEnc              = str2double(regexp(tag,'L(\d)','tokens','once')); % 1|2|3
encBinsPerEpoch   = (div_col - 1) / max(nEnc,1);
maintBins         = nBins - nEnc * encBinsPerEpoch;
prefIdx           = whichPrefEnc(tag);
addSegmentBarLargeNorm(ax, encBinsPerEpoch, maintBins, nEnc, prefIdx, nBins);

%% 3 ▸ FIGURE SIZE (pixel-per-bin, independent of colorbar)
width_px  = nBins * pxPerBin;
height_px = 400;
dpi       = 100;
set(hFig,'Units','inches',...
         'Position',[1 1 width_px/dpi height_px/dpi],...
         'PaperUnits','inches',...
         'PaperPosition',[0 0 width_px/dpi height_px/dpi]);

% (Saving code omitted for brevity; re-enable if needed)
end

function addSegmentBarLargeNorm(mainAx,encBins,maintBins,nEnc,...
                                prefEncIdx,nBins)
pos  = mainAx.Position;
barH = 0.06;

% keep within figure
maxBarTop = 0.99;
barH = min(barH, maxBarTop - (pos(2) + pos(4)));
barH = max(barH, 0.02);   % don’t let it collapse completely

barAx = axes('Position',[pos(1) pos(2)+pos(4) pos(3) barH], 'Units','normalized');
axis(barAx,'off'); hold(barAx,'on');
axis(barAx,[0 1 0 1]);    % fix limits (avoids autoscale surprises)
set(barAx,'Clipping','on');

encFrac   = encBins   / nBins;
maintFrac = maintBins / nBins;

xLeft = 0;
for k = 1:nEnc
    xRight = xLeft + encFrac;
    if k==prefEncIdx
        col=[1 0.6 0.6]; label='Pref';
    else
        col=[0.65 0.65 0.65]; label=sprintf('Enc %d',k);
    end
    patch(barAx,[xLeft xRight xRight xLeft],[0 0 1 1],col,'EdgeColor','k');
    text(barAx,(xLeft+xRight)/2,0.5,label,'HorizontalAlignment','center',...
         'VerticalAlignment','middle','FontWeight','bold','FontSize',14);
    xLeft = xRight;
end
xRight = xLeft + maintFrac;
patch(barAx,[xLeft xRight xRight xLeft],[0 0 1 1],[0.5843    0.8157    0.9882],'EdgeColor','k');
text(barAx,(xLeft+xRight)/2,0.5,'Maint','HorizontalAlignment','center',...
     'VerticalAlignment','middle','FontWeight','bold', 'FontSize',14);
end

%=======================================================================%
function idx = whichPrefEnc(tag)
% Decide which encoding epoch is "preferred" for the top bar.
%  L1_*   → Enc-1
%  L2_*   → by default Enc-2, but:
%           tags containing 'prefX'  → Enc-1
%           tags containing 'Xpref'  → Enc-2
%  L3_*   → Enc-3
if contains(tag,'_NP')
    idx = 0;
    return;
end
if contains(tag,'prefX')
    idx = 1;    % pref-X → preferred in Encoding-1
    return;
elseif contains(tag,'Xpref')
    idx = 2;    % X-pref → preferred in Encoding-2
    return;
end
m = regexp(tag,'L(\d)','tokens','once');
idx = str2double(m{1});
end
