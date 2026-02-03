function plot_LIS_integratedHM(nwbAll, all_units, neural_data_file, bin_width, load_level)

if nargin < 5 || isempty(load_level)
    load_level = 1;
end

rng(42)

binSz       = bin_width;
encOffset   = 1.0;
maintOffset = 2.5;

gaussKern = GaussianKernal(0.3 / binSz, 1.5);

nBinsEnc   = round(encOffset   / binSz);
nBinsMaint = round(maintOffset / binSz);

load(neural_data_file, 'neural_data');
num_neurons = numel(neural_data);

enc_pref       = nan(num_neurons, nBinsEnc);
enc_nonpref    = nan(num_neurons, nBinsEnc);
maint_pref     = nan(num_neurons, nBinsMaint);
maint_nonpref  = nan(num_neurons, nBinsMaint);
infoList       = nan(num_neurons,2);

for ndx = 1:num_neurons
    nd = neural_data(ndx);
    pid = nd.patient_id;  
    uid = nd.unit_id;  
    pImg = nd.preferred_image;
    trial_imageIDs = nd.trial_imageIDs;

    if iscell(trial_imageIDs)
        trial_imageIDs = cell2mat(trial_imageIDs);
    end

    sIdx = find([all_units.subject_id]==pid & [all_units.unit_id]==uid,1);
    if isempty(sIdx), continue; end
    SU   = all_units(sIdx);
    sess = nwbAll{SU.session_count};

    tsEnc = get_ts(sess,'timestamps_Encoding1');
    tsMai = get_ts(sess,'timestamps_Maintenance');
    if isempty(tsEnc) || isempty(tsMai), continue; end

    [trials_this_load, posCol] = select_trials_by_load(trial_imageIDs, load_level);
    if isempty(trials_this_load), continue; end
    
    isPrefAtPos = (trial_imageIDs(trials_this_load, posCol) == pImg);
    prefTrials    = trials_this_load(isPrefAtPos);
    nonprefTrials = trials_this_load(~isPrefAtPos);
    
    nMin = min(numel(prefTrials), numel(nonprefTrials));
    if nMin == 0, continue; end
    prefTrials    = datasample(prefTrials,    nMin, 'Replace', false);
    nonprefTrials = datasample(nonprefTrials, nMin, 'Replace', false);

    rowC = avgFR_concat(prefTrials, tsEnc, tsMai, SU, binSz, encOffset, maintOffset, gaussKern);
    enc_pref(ndx,:)      = rowC(1:nBinsEnc);
    maint_pref(ndx,:)    = rowC(nBinsEnc+1:end);
    
    rowC = avgFR_concat(nonprefTrials, tsEnc, tsMai, SU, binSz, encOffset, maintOffset, gaussKern);
    enc_nonpref(ndx,:)   = rowC(1:nBinsEnc);
    maint_nonpref(ndx,:) = rowC(nBinsEnc+1:end);

    infoList(ndx,:) = [pid uid];
end

enc_pref_n       = nan(size(enc_pref));
enc_nonpref_n    = nan(size(enc_nonpref));
maint_pref_n     = nan(size(maint_pref));
maint_nonpref_n  = nan(size(maint_nonpref));

for i = 1:num_neurons
    base = maint_pref(i,:);
    if all(isnan(base)), continue; end
    mn = min(base,[],'omitnan');  
    mx = max(base,[],'omitnan');
    if mx<=mn, mn=0; mx=1; end
    sc = mx-mn;
    enc_pref_n(i,:)      = (enc_pref(i,:)      - mn)./sc;
    enc_nonpref_n(i,:)   = (enc_nonpref(i,:)   - mn)./sc;
    maint_pref_n(i,:)    = (base               - mn)./sc;
    maint_nonpref_n(i,:) = (maint_nonpref(i,:) - mn)./sc;
end

[~,pk]   = max(maint_pref_n,[],2);
[~,ord]  = sort(pk,'ascend');
enc_pref_n      = enc_pref_n(ord,:);
enc_nonpref_n   = enc_nonpref_n(ord,:);
maint_pref_n    = maint_pref_n(ord,:);
maint_nonpref_n = maint_nonpref_n(ord,:);

enc_pref_n2    = nan(size(enc_pref));
enc_nonpref_n2 = nan(size(enc_nonpref));

for i = 1:num_neurons
    vv = [enc_pref(i,:)  enc_nonpref(i,:)];
    mn = min(vv,[],'omitnan');
    mx = max(vv,[],'omitnan');
    if mx <= mn, mn = 0; mx = 1; end
    scale = mx - mn;
    enc_pref_n2(i,:)    = (enc_pref(i,:)    - mn) ./ scale;
    enc_nonpref_n2(i,:) = (enc_nonpref(i,:) - mn) ./ scale;
end

enc_pref_n2    = enc_pref_n2(ord,:);
enc_nonpref_n2 = enc_nonpref_n2(ord,:);

plot_enc_heatmaps(enc_nonpref_n2, enc_pref_n2, binSz, encOffset);

cat_pref    = [enc_pref_n,  maint_pref_n ];
cat_nonpref = [enc_nonpref_n, maint_nonpref_n ];

cat_pref_A = nan(num_neurons, size(cat_pref,2));
cat_pref_B = nan(num_neurons, size(cat_pref,2));

for i = 1:num_neurons
    nd   = neural_data(i);
    pid  = nd.patient_id;
    uid  = nd.unit_id;

    sIdx = find([all_units.subject_id]==pid & [all_units.unit_id]==uid , 1);
    if isempty(sIdx), continue; end
    SU   = all_units(sIdx);
    sess = nwbAll{SU.session_count};

    tsEnc = get_ts(sess,'timestamps_Encoding1');
    tsMai = get_ts(sess,'timestamps_Maintenance');
    if isempty(tsEnc) || isempty(tsMai),  continue;  end
    
    trial_imageIDs = nd.trial_imageIDs;
    if iscell(trial_imageIDs),  trial_imageIDs = cell2mat(trial_imageIDs); end
    
    [trials_this_load, posCol] = select_trials_by_load(trial_imageIDs, load_level);
    if isempty(trials_this_load), continue; end
    
    allPref = trials_this_load(trial_imageIDs(trials_this_load, posCol) == nd.preferred_image);
    if numel(allPref) < 2, continue; end

    ordp  = randperm(numel(allPref));
    cut   = floor(numel(allPref)/2);
    idxA  = allPref(ordp(1:cut));
    idxB  = allPref(ordp(cut+1:end));

    cat_pref_A(i,:) = avgFR_concat(idxA, tsEnc, tsMai, SU, binSz, encOffset, maintOffset, gaussKern);
    cat_pref_B(i,:) = avgFR_concat(idxB, tsEnc, tsMai, SU, binSz, encOffset, maintOffset, gaussKern);
end

cat_pref_A_n = localMinMax(cat_pref_A, nBinsEnc+1);
cat_pref_B_n = localMinMax(cat_pref_B, nBinsEnc+1);

[~, pkA]  = max(cat_pref_A_n(:, nBinsEnc+1:end), [], 2);
[~, ordA] = sort(pkA, 'ascend');

cat_pref_A_n = cat_pref_A_n(ordA,:);
cat_pref_B_n = cat_pref_B_n(ordA,:);

plot_individual_heatmaps(enc_nonpref_n, enc_pref_n, maint_nonpref_n, maint_pref_n, binSz, encOffset, maintOffset);

plot_concatenated_single(cat_nonpref, binSz, encOffset+maintOffset, 'Non-Preferred (0–3.5 s)');
plot_concatenated_single(cat_pref,    binSz, encOffset+maintOffset, 'Preferred (0–3.5 s)');
plot_concatenated_single(cat_pref_A_n, binSz, encOffset+maintOffset, 'Pref Half-A');
plot_concatenated_single(cat_pref_B_n, binSz, encOffset+maintOffset, 'Pref Half-B');

end

function ts = get_ts(sess,key)
if isKey(sess.intervals_trials.vectordata,key)
    ts = sess.intervals_trials.vectordata.get(key).data.load();
else
    ts = [];
end
end

function row = avgFR_concat(trials, tsEnc, tsMai, SU, binSz, offE, offM, ker)
nEnc   = round(offE / binSz);
nMaint = round(offM / binSz);
nTot   = nEnc + nMaint;

if isempty(trials) || isempty(tsEnc) || isempty(tsMai)
    row = nan(1,nTot);  
    return;
end

mat = nan(numel(trials), nTot);
for k = 1:numel(trials)
    idx = trials(k);
    if idx > numel(tsEnc) || idx > numel(tsMai), continue; end

    tE = tsEnc(idx);
    edgesE = tE : binSz : (tE + offE);
    frEnc = histcounts(SU.spike_times, edgesE) ./ binSz;

    tM = tsMai(idx);
    edgesM = tM : binSz : (tM + offM);
    frMaint = histcounts(SU.spike_times, edgesM) ./ binSz;

    fr = conv([frEnc frMaint], ker, 'same');
    mat(k,:) = fr;
end
row = mean(mat,1,'omitnan');
end

function plot_enc_heatmaps(eNP, eP, binSz, offE)
figure('Name','Enc Pref vs Enc NonPref');
xt     = 0:0.5:offE;
xt_idx = round(xt/binSz);  
xt_idx(1)=1;

subplot(1,2,1);  
imagesc(eNP);  
colormap('parula');  
caxis([0 1]);  
colorbar;
set(gca,'XTick',xt_idx,'XTickLabel',cellstr(num2str(xt','%.1f')));
ylabel('Neuron');  
title('Encoding – Non-Pref');

subplot(1,2,2);  
imagesc(eP);   
colormap('parula');  
caxis([0 1]);  
colorbar;
set(gca,'XTick',xt_idx,'XTickLabel',cellstr(num2str(xt','%.1f')));
title('Encoding – Pref');
end

function plot_individual_heatmaps(eNP,eP,mNP,mP,binSzE,offE,offM)
figure('Name','Separate Heatmaps');

subplot(2,2,1);  
imagesc(eNP);  
colormap('parula');  
caxis([0 1]);  
colorbar;
xt = 0:0.5:offE;
xt_idx = round(xt/binSzE);  
xt_idx(1)=1;
set(gca,'XTick',xt_idx,'XTickLabel',cellstr(num2str(xt','%.1f')));
ylabel('Neuron');  
title('Enc – Non-Pref');

subplot(2,2,2);  
imagesc(eP);  
colormap('parula');  
caxis([0 1]);  
colorbar;
set(gca,'XTick',xt_idx,'XTickLabel',cellstr(num2str(xt','%.1f')));
title('Enc – Pref');

subplot(2,2,3);  
imagesc(mNP);  
colormap('parula');  
caxis([0 1]);  
colorbar;
xt2 = 0:0.5:offM;
xt2_idx = round(xt2/binSzE);  
xt2_idx(1)=1;
set(gca,'XTick',xt2_idx,'XTickLabel',cellstr(num2str(xt2','%.1f')));
ylabel('Neuron');  
title('Maint – Non-Pref');

subplot(2,2,4);  
imagesc(mP);  
colormap('parula');  
caxis([0 1]);  
colorbar;
set(gca,'XTick',xt2_idx,'XTickLabel',cellstr(num2str(xt2','%.1f')));
title('Maint – Pref');
end

function plot_concatenated_single(mat, binSz, totalT, ttl)
figure('Name',ttl, 'Position',[400, 400, 400, 648]);
imagesc(mat);  
colormap('parula');  
caxis([0 1]);  
colorbar;

xt     = 0:0.5:totalT;
xt_idx = round(xt/binSz);  
xt_idx(1)=1;
set(gca,'XTick',xt_idx,'XTickLabel',cellstr(num2str(xt','%.1f')), 'FontSize', 20);

ylabel('Img-spec Time Cells');  
xlabel('Time (s)');
title(ttl);

hold on
xline(round(1/binSz),'r--','LineWidth',3);
hold off
end

function matN = localMinMax(mat, maintStartCol)
matN = nan(size(mat));
for r = 1:size(mat,1)
    base = mat(r, maintStartCol:end);
    if all(isnan(base)), continue; end
    mn = min(base,[],'omitnan');  
    mx = max(base,[],'omitnan');
    if mx <= mn
        mn = 0; mx = 1;
    end
    matN(r,:) = (mat(r,:) - mn) ./ (mx - mn);
end
end

function [trials, posCol] = select_trials_by_load(trial_imageIDs, load_level)
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
