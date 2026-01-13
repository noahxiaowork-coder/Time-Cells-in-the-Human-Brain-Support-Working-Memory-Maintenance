function plot_LIS_Load123_prefHeatmaps(nwbAll, all_units, neural_data_file, ref_load, LocalN)

if nargin < 5 || isempty(LocalN)
    LocalN = false;
end

if ~isscalar(ref_load) || ~ismember(ref_load, [1 2 3])
    error('ref_load must be 1, 2, or 3');
end

rng(42)
binSz     = 0.1;
encDur    = 1.0;
maintDur  = 2.5;
encBins   = round(encDur   / binSz);
maintBins = round(maintDur / binSz);
nTot_L1   = encBins       + maintBins;
nTot_L2   = 2*encBins     + maintBins;
nTot_L3   = 3*encBins     + maintBins;

pxPerSecond = 100;
barFrac      = 0.06;
fontSzBar    = 10;
colPref      = [1 0.6 0.6];
colEnc       = [0.65 0.65 0.65];

gaussKern = GaussianKernal(0.3 / binSz, 1.5);
gaussKern = gaussKern(:)';
if any(gaussKern)
    gaussKern = gaussKern / sum(gaussKern);
end

load(neural_data_file, 'neural_data');
num_neurons = numel(neural_data);

cat_L1_P        = nan(num_neurons, nTot_L1);
cat_L1_NP       = nan(num_neurons, nTot_L1);
cat_L2_P_comb   = nan(num_neurons, nTot_L2);
cat_L2_P_prefX  = nan(num_neurons, nTot_L2);
cat_L2_P_Xpref  = nan(num_neurons, nTot_L2);
cat_L3_P        = nan(num_neurons, nTot_L3);

for i = 1:num_neurons
    nd  = neural_data(i);
    pid = nd.patient_id;
    uid = nd.unit_id;
    sIdx = find([all_units.subject_id]==pid & [all_units.unit_id]==uid,1);
    if isempty(sIdx), continue; end
    SU   = all_units(sIdx);
    sess = nwbAll{SU.session_count};

    tsEnc1 = get_ts(sess, 'timestamps_Encoding1');
    tsEnc2 = get_ts(sess, 'timestamps_Encoding2');
    tsEnc3 = get_ts(sess, 'timestamps_Encoding3');
    tsMai  = get_ts(sess, 'timestamps_Maintenance');
    if isempty(tsEnc1) || isempty(tsMai), continue; end

    trial_imageIDs = nd.trial_imageIDs;
    if iscell(trial_imageIDs), trial_imageIDs = cell2mat(trial_imageIDs); end
    pImg = nd.preferred_image;

    L1 = find(trial_imageIDs(:,1)~=0 & trial_imageIDs(:,2)==0 & trial_imageIDs(:,3)==0);
    pref_L1    = L1(trial_imageIDs(L1,1)==pImg);
    nonpref_L1 = L1(trial_imageIDs(L1,1)~=pImg);

    if ~isempty(pref_L1)
        cat_L1_P(i,:) = buildLoadRow(pref_L1, tsEnc1, [], [], tsMai, 1, SU, gaussKern, binSz);
    end
    if ~isempty(nonpref_L1)
        cat_L1_NP(i,:) = buildLoadRow(nonpref_L1, tsEnc1, [], [], tsMai, 1, SU, gaussKern, binSz);
    end

    L2_prefX = find(trial_imageIDs(:,1)==pImg & trial_imageIDs(:,2)~=0 & trial_imageIDs(:,3)==0);
    L2_Xpref = find(trial_imageIDs(:,2)==pImg & trial_imageIDs(:,3)==0);

    if ~isempty(L2_prefX)
        cat_L2_P_prefX(i,:) = buildLoadRow(L2_prefX, tsEnc1, tsEnc2, [], tsMai, 2, SU, gaussKern, binSz);
    end
    if ~isempty(L2_Xpref)
        cat_L2_P_Xpref(i,:) = buildLoadRow(L2_Xpref, tsEnc1, tsEnc2, [], tsMai, 2, SU, gaussKern, binSz);
    end

    if ~all(isnan(cat_L2_P_prefX(i,:))) || ~all(isnan(cat_L2_P_Xpref(i,:)))
        tmp = [cat_L2_P_prefX(i,:); cat_L2_P_Xpref(i,:)];
        cat_L2_P_comb(i,:) = mean(tmp,1,'omitnan');
    end

    L3 = find(trial_imageIDs(:,3)==pImg);
    if ~isempty(L3)
        cat_L3_P(i,:) = buildLoadRow(L3, tsEnc1, tsEnc2, tsEnc3, tsMai, 3, SU, gaussKern, binSz);
    end
end

switch ref_load
    case 1
        refMat  = cat_L1_P;        divCol = encBins + 1;
    case 2
        refMat  = cat_L2_P_Xpref;  divCol = 2*encBins + 1;
    case 3
        refMat  = cat_L3_P;        divCol = 3*encBins + 1;
end

if LocalN
    cat_L1_P_n        = localMinMax(cat_L1_P,        divCol);
    cat_L1_NP_n       = localMinMax(cat_L1_NP,       divCol);
    cat_L2_P_comb_n   = localMinMax(cat_L2_P_comb,   divCol);
    cat_L2_P_prefX_n  = localMinMax(cat_L2_P_prefX,  divCol);
    cat_L2_P_Xpref_n  = localMinMax(cat_L2_P_Xpref,  divCol);
    cat_L3_P_n        = localMinMax(cat_L3_P,        divCol);
else
    cat_L1_P_n        = localMinMax_fromRef(cat_L1_P,        refMat, divCol);
    cat_L1_NP_n       = localMinMax_fromRef(cat_L1_NP,       refMat, divCol);
    cat_L2_P_comb_n   = localMinMax_fromRef(cat_L2_P_comb,   refMat, divCol);
    cat_L2_P_prefX_n  = localMinMax_fromRef(cat_L2_P_prefX,  refMat, divCol);
    cat_L2_P_Xpref_n  = localMinMax_fromRef(cat_L2_P_Xpref,  refMat, divCol);
    cat_L3_P_n        = localMinMax_fromRef(cat_L3_P,        refMat, divCol);
end

norm_refMat = localMinMax(refMat, divCol);
[~, pk]  = max(norm_refMat(:, divCol:end), [], 2, 'includenan');
[~, ord] = sort(pk, 'ascend');

cat_L1_P_n        = cat_L1_P_n(ord, :);
cat_L1_NP_n       = cat_L1_NP_n(ord, :);
cat_L2_P_comb_n   = cat_L2_P_comb_n(ord, :);
cat_L2_P_prefX_n  = cat_L2_P_prefX_n(ord, :);
cat_L2_P_Xpref_n  = cat_L2_P_Xpref_n(ord, :);
cat_L3_P_n        = cat_L3_P_n(ord, :);

save_path = '/Users/xiangxuankong/Desktop/Human TC Support WM/Figures/Figure S5';
PX = 12;

plot_concatenated_single(cat_L1_NP_n, binSz, encDur+maintDur, ...
    '', encBins+1, save_path, 'L1_NP_refL3', PX, false);
plot_concatenated_single(cat_L1_P_n , binSz, encDur+maintDur, ...
    '', encBins+1, save_path, 'L1_P_refL3' , PX, false);

plot_concatenated_single(cat_L2_P_prefX_n , binSz, 2*encDur+maintDur, ...
    '', 2*encBins+1, save_path,'L2_prefX_refL3', PX, false);
plot_concatenated_single(cat_L2_P_Xpref_n , binSz, 2*encDur+maintDur, ...
    '', 2*encBins+1, save_path,'L2_Xpref_refL3', PX, false);

plot_concatenated_single(cat_L3_P_n , binSz, 3*encDur+maintDur, ...
    '', 3*encBins+1, save_path,'L3_P_refL3', PX, true);

end


function ts = get_ts(sess, key)
    if isKey(sess.intervals_trials.vectordata, key)
        ts = sess.intervals_trials.vectordata.get(key).data.load();
    else
        ts = [];
    end
end

function matN = localMinMax(mat, maintStartCol)
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

    if isempty(trials)
        row = nan(1, 10*loadTag + 25);
        return;
    end

    nBins = 10*loadTag + 25;
    nT    = numel(trials);
    mat   = nan(nT, nBins);

    k = ker(:)';
    if isempty(k) || all(k==0)
        useSmoothing = false;
    else
        k = k / sum(k);
        useSmoothing = true;
    end

    for it = 1:nT
        t  = trials(it);
        fr = [];

        if ~isempty(ts1)
            edges1 = ts1(t):binSz:(ts1(t)+1.0);
            c1     = histcounts(SU.spike_times, edges1);
            fr     = [fr, c1 ./ binSz];
        end

        if loadTag >= 2 && ~isempty(ts2)
            edges2 = ts2(t):binSz:(ts2(t)+1.0);
            c2     = histcounts(SU.spike_times, edges2);
            fr     = [fr, c2 ./ binSz];
        end

        if loadTag == 3 && ~isempty(ts3)
            edges3 = ts3(t):binSz:(ts3(t)+1.0);
            c3     = histcounts(SU.spike_times, edges3);
            fr     = [fr, c3 ./ binSz];
        end

        if ~isempty(tsM)
            edgesM = tsM(t):binSz:(tsM(t)+2.5);
            cM     = histcounts(SU.spike_times, edgesM);
            fr     = [fr, cM ./ binSz];
        end

        if numel(fr) ~= nBins
            continue;
        end

        fr_row = fr(:)';

        if ~useSmoothing
            mat(it,:) = fr_row;
        else
            nTot    = nBins;
            kerHalf = floor((numel(k) - 1) / 2);
            padBins = min(kerHalf, floor((nTot - 1)/2));

            if padBins > 0
                fr_pad = [ fliplr(fr_row(1:padBins)), ...
                           fr_row, ...
                           fliplr(fr_row(end-padBins+1:end)) ];

                sm_pad    = conv(fr_pad, k, 'same');
                mat(it,:) = sm_pad(padBins+1 : padBins+nTot);
            else
                mat(it,:) = conv(fr_row, k, 'same');
            end
        end
    end

    row = mean(mat, 1, 'omitnan');
end


function plot_concatenated_single(mat, binSz, totalT, ttl, div_col, ...
                                  save_path, tag, pxPerBin, showColorbar)

if nargin < 8 || isempty(pxPerBin),     pxPerBin     = 12;  end
if nargin < 9 || isempty(showColorbar), showColorbar = true; end

hFig = figure('Name', ttl);
ax   = axes('Parent', hFig);
imagesc(ax, mat);
colormap(ax, 'parula');
caxis(ax, [0 1]);

if showColorbar
    colorbar(ax);
else
    set(ax,'Position',[0.13 0.11 0.815 0.815]);
end

nBins = size(mat,2);
xlim(ax, [0.5, nBins + 0.5]);

tstep       = 0.5;
xt          = 0:tstep:totalT;
t_boundary  = (div_col - 1) * binSz;
xt          = [xt, t_boundary];
xt          = unique(round(xt,10), 'stable');
xt_pos      = xt ./ binSz + 0.5;

mask        = (xt_pos >= 0.5 - 1e-9) & (xt_pos <= nBins + 0.5 + 1e-9);
xt_pos      = xt_pos(mask);
xt          = xt(mask);

set(ax,'XTick',xt_pos, ...
       'XTickLabel',cellstr(num2str(xt','%.1f')), ...
       'FontSize',14);

ylabel(ax,'Image-specific Time Cells','FontSize',14);
xlabel(ax,'Time (s)','FontSize',14);

hold(ax,'on');
xline(ax, div_col - 0.5, 'r--', 'LineWidth', 3);
hold(ax,'off');

nEnc            = str2double(regexp(tag,'L(\d)','tokens','once'));
encBinsPerEpoch = (div_col - 1) / max(nEnc,1);
maintBins       = nBins - nEnc * encBinsPerEpoch;
prefIdx         = whichPrefEnc(tag);
addSegmentBarLargeNorm(ax, encBinsPerEpoch, maintBins, nEnc, prefIdx, nBins);

width_px  = nBins * pxPerBin;
height_px = 400;
dpi       = 100;
set(hFig,'Units','inches',...
         'Position',[1 1 width_px/dpi height_px/dpi],...
         'PaperUnits','inches',...
         'PaperPosition',[0 0 width_px/dpi height_px/dpi]);

end

function addSegmentBarLargeNorm(mainAx,encBins,maintBins,nEnc,...
                                prefEncIdx,nBins)
pos  = mainAx.Position;
barH = 0.06;

maxBarTop = 0.99;
barH = min(barH, maxBarTop - (pos(2) + pos(4)));
barH = max(barH, 0.02);

barAx = axes('Position',[pos(1) pos(2)+pos(4) pos(3) barH], 'Units','normalized');
axis(barAx,'off'); hold(barAx,'on');
axis(barAx,[0 1 0 1]);
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
patch(barAx,[xLeft xRight xRight xLeft],[0 0 1 1],[0.5843 0.8157 0.9882],'EdgeColor','k');
text(barAx,(xLeft+xRight)/2,0.5,'Maint','HorizontalAlignment','center',...
     'VerticalAlignment','middle','FontWeight','bold', 'FontSize',14);
end

function idx = whichPrefEnc(tag)
if contains(tag,'_NP')
    idx = 0;
    return;
end
if contains(tag,'prefX')
    idx = 1;
    return;
elseif contains(tag,'Xpref')
    idx = 2;
    return;
end
m = regexp(tag,'L(\d)','tokens','once');
idx = str2double(m{1});
end
