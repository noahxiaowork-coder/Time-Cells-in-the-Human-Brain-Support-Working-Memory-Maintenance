function plot_single_cell_XvsNonX(nwbAll, all_units, neural_data, ...
                                  bin_size, subject_id, unit_id, ...
                                  use_zscore, show_multi_nonpref)
% PLOT_SINGLE_CELL_XVSNONX
% Raster + PSTHs (X vs Non-X trials) for ONE unit.
% Now includes Encoding-1 (0–1 s) concatenated with Maintenance (1–3.5 s).
%
% Z-scoring (if enabled) is performed per trial across the full 35-bin
% concatenated vector AFTER smoothing, then averaged across trials.
%
% show_multi_nonpref:
%   true  -> PSTH: one curve per image (multi-colored)
%   false -> PSTH: preferred (blue) vs ALL non-preferred (orange, as before)

if nargin < 7, use_zscore = false; end
if nargin < 8, show_multi_nonpref = false; end   % <-- default: single orange curve

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
tf_bin   = double(neural_data(ndx).time_field);  % maintenance time-field (in 100 ms bins)
pref_img = neural_data(ndx).preferred_image;
imgIDs   = neural_data(ndx).trial_imageIDs;
if iscell(imgIDs), imgIDs = cell2mat(imgIDs); end

sess      = nwbAll{SU.session_count};
% trial timestamps
tsEnc  = get_ts(sess,'timestamps_Encoding1');
tsMaint= get_ts(sess,'timestamps_Maintenance');

if isempty(tsMaint)
    warning('Maintenance timestamps missing for S%d U%d.',subject_id,unit_id);
    return
end
if isempty(tsEnc)
    warning('Encoding timestamps missing for S%d U%d; plotting maintenance only.',subject_id,unit_id);
end

spk     = SU.spike_times;
nTrials = max([numel(tsEnc), numel(tsMaint)]);

% -------------------- Trial split (load-1 convention) --------------------
% X trials: enc1 == preferred, enc2==0, enc3==0
% NonX:     enc1 ~= preferred, enc2==0, enc3==0
enc1 = imgIDs(:,1);  enc2 = imgIDs(:,2);  enc3 = imgIDs(:,3);
X_idx    = find(enc1==pref_img & enc2==0 & enc3==0);
NonX_idx = find(enc1~=pref_img & enc2==0 & enc3==0); %#ok<NASGU>

% === Per-image groups for plotting (raster & PSTH) =======================
valid_idx = find(enc2==0 & enc3==0); % all load-1 trials used for plotting
if isempty(valid_idx)
    warning('No load-1 trials (enc2==0 & enc3==0) for S%d U%d.',subject_id,unit_id);
    return
end

img_valid   = enc1(valid_idx);
img_unique  = unique(img_valid);

% Put preferred image first if it appears among valid trials
if any(img_unique == pref_img)
    img_unique(img_unique == pref_img) = [];
    img_order = [pref_img; img_unique];
else
    img_order = img_unique;
end

nGroups = numel(img_order);
groups  = cell(1,nGroups);
labels  = cell(1,nGroups);

for g = 1:nGroups
    thisImg   = img_order(g);
    tmp       = valid_idx(img_valid == thisImg);
    groups{g} = tmp(:)';  % enforce 1×N row vector

    if thisImg == pref_img
        labels{g} = sprintf('Preferred (img %d)', thisImg);
    else
        labels{g} = sprintf('Img %d', thisImg);
    end
end


% Colors for RASTER: one color per image (preferred forced to blue).
% These colors are used ONLY for the raster (no orange here).
base_cmap = lines(max(nGroups,1));
base_cmap = base_cmap(1:nGroups, :);
if img_order(1) == pref_img
    base_cmap(1,:) = [0 0 1]; % preferred image in blue
end
colors_raster = mat2cell(base_cmap, ones(1,nGroups), 3);

% -------------------- PSTH groups depending on toggle --------------------
if show_multi_nonpref
    % One PSTH per image (multi-colored)
    psth_groups = groups;
    psth_labels = labels;
    psth_colors = colors_raster;   % reuse same colors
else
    % Collapse all non-preferred into a single orange curve, keep preferred blue
    if any(img_order == pref_img)
        gPref = find(img_order == pref_img, 1, 'first');
        idx_pref    = groups{gPref};
        idx_nonpref = cell2mat(groups(setdiff(1:nGroups, gPref)));

        psth_groups = {idx_pref, idx_nonpref};
        psth_labels = {labels{gPref}, 'Non-Preferred'};
        psth_colors = {[0 0 1], [1 0.5 0]};  % blue, orange (as before)
    else
        % No preferred trials in this session: all trials are "non-preferred"
        idx_all     = cell2mat(groups);
        psth_groups = {idx_all};
        psth_labels = {'All Non-Preferred'};
        psth_colors = {[1 0.5 0]};           % single orange curve
    end
end
nPSTHgroups = numel(psth_groups);

% -------------------- Epoch parameters -----------------------------------
encOffset   = 1.0;   % seconds (Encoding-1)
maintOffset = 2.5;   % seconds (Maintenance)
totalT      = encOffset + maintOffset;  % 3.5 s
tVec        = 0:bin_size:(totalT-bin_size);   % left-edge time stamps (35 bins if bin=0.1)

% Gaussian kernel (normalize to preserve scale)
% gauss_k = GaussianKernal(0.3/bin_size, 1.5);
% Gaussian kernel (normalize to preserve scale)
gauss_k = GaussianKernal(0.5/bin_size, 2);
gauss_k = gauss_k(:)';                       
if sum(gauss_k) ~= 0
    gauss_k = gauss_k / sum(gauss_k);
end

sigma_bins = 0.5 / bin_size;                 % σ in bins (0.5 s / bin_size)
pad_bins   = ceil(3 * sigma_bins);           % ~3σ padding on each side

% -------------------- Field window (Maintenance part only) ---------------
% Estimate maintenance response window using ONLY X trials, raw (no smoothing)
binsMaint   = 0:bin_size:maintOffset;  n_binsM = numel(binsMaint)-1;
fr_allM = zeros(numel(X_idx), n_binsM);
for t = 1:numel(X_idx)
    tr = X_idx(t);
    if tr > numel(tsMaint), continue; end
    s  = spk(spk>=tsMaint(tr) & spk<tsMaint(tr)+maintOffset) - tsMaint(tr);
    fr_allM(t,:) = histcounts(s,binsMaint)/bin_size;
end
mean_frM = mean(fr_allM,1,'omitnan');
baseline = mean(mean_frM,'omitnan');
thr      = baseline + 0.5*std(mean_frM,[],'omitnan');

centre_t   = (tf_bin-0.5)*0.1;   % tf_bin in 100 ms bins, convert to s (bin center)
[~,centre] = min(abs(binsMaint(1:end-1)-centre_t));
left  = centre; while left>1              && mean_frM(left)>=thr,  left  = left-1; end
right = centre; while right<n_binsM && mean_frM(right)>=thr, right = right+1; end

t_start_M = min(binsMaint(left),  (tf_bin-1)*0.1);
t_end_M   = max(binsMaint(right), tf_bin*0.1);
if t_end_M <= t_start_M, t_start_M = (tf_bin-1)*0.1; t_end_M = tf_bin*0.1; end

% In concatenated 0–3.5 s axis, maintenance segment is shifted by +encOffset
t_start_cat = encOffset + t_start_M;
t_end_cat   = encOffset + t_end_M;

% -------------------- Figure --------------------------------------------
figure('Name',sprintf('S%d U%d pref=%d',subject_id,unit_id,pref_img), ...
       'Units','pixels','Position',[100 100 400 648]);

% === RASTER over concatenated timeline (0–1 s Enc, 1–3.5 s Maint) ===
subplot(2,1,1); hold on; set(gca,'FontSize',16);
% maintenance field shading (on concatenated axis)
fill([t_start_cat t_start_cat t_end_cat t_end_cat],[0 nTrials+1 nTrials+1 0], ...
     [0.6 0.8 1],'EdgeColor','none','FaceAlpha',0.4,'HandleVisibility','off');

xlabel('Time (s)','FontSize',16); ylabel('Trial','FontSize',16);
title(sprintf('Subject %d, Unit %d, Pref Img %d',subject_id,unit_id,pref_img), ...
      'Interpreter','none','FontSize',16);

offset = 0;
for g = 1:nGroups
    idx = groups{g};
    col = colors_raster{g};
    for i = 1:numel(idx)
        tr = idx(i);

        % Encoding spikes mapped to 0..1 s
        if ~isempty(tsEnc) && tr <= numel(tsEnc)
            sE = spk(spk>=tsEnc(tr) & spk<tsEnc(tr)+encOffset) - tsEnc(tr);
            if ~isempty(sE)
                scatter(sE, offset + i * ones(size(sE)), 9, col, ...
                        'filled','HandleVisibility','off');
            end
        end

        % Maintenance spikes mapped to 1..3.5 s (shift by +encOffset)
        if tr <= numel(tsMaint)
            sM = spk(spk>=tsMaint(tr) & spk<tsMaint(tr)+maintOffset) - tsMaint(tr) + encOffset;
            if ~isempty(sM)
                scatter(sM, offset + i * ones(size(sM)), 9, col, ...
                        'filled','HandleVisibility','off');
            end
        end
    end
    offset = offset + numel(idx); % stack rows by group/color
end

xline(encOffset,'k--','LineWidth',1.5,'HandleVisibility','off'); % Enc | Maint divider
xlim([0 totalT]); ylim([0 offset+1]);

% === PSTH over concatenated 3.5 s ======================================
subplot(2,1,2); hold on; set(gca,'FontSize',16);
xlabel('Time (s)','FontSize',16);

if use_zscore, ylab = "Z-score"; else, ylab = "Firing Rate (Hz)"; end
ylabel(ylab,'FontSize',16);

hLines = gobjects(1,nPSTHgroups);

for g = 1:nPSTHgroups
    idx = psth_groups{g};
    col = psth_colors{g};

    % Mean & SEM from SMOOTHED (+ optional per-trial z-scored) 35-bin trials
    [mFR35, sFR35] = computePSTH35(spk, tsEnc, tsMaint, idx, bin_size, ...
                               encOffset, maintOffset, use_zscore, ...
                               gauss_k, pad_bins);



    % No extra smoothing here!
    fill([tVec fliplr(tVec)], [mFR35+sFR35 fliplr(mFR35-sFR35)], col, ...
         'EdgeColor','none','FaceAlpha',0.25,'HandleVisibility','off');
    hLines(g) = plot(tVec, mFR35, 'Color',col,'LineWidth',2,'DisplayName',psth_labels{g});
end

yl = ylim;
patch([t_start_cat t_start_cat t_end_cat t_end_cat],[yl(1) yl(2) yl(2) yl(1)], ...
      [0.6 0.8 1],'EdgeColor','none','FaceAlpha',0.2,'HandleVisibility','off');

xline(encOffset,'k--','LineWidth',1.5,'HandleVisibility','off'); % Enc | Maint divider
legend(hLines, psth_labels,'Location','best','FontSize',16,'Box','on');
xlim([0 totalT]);

end % === main function end ===


% ------------------------------------------------------------------------
function [mean_fr35, sem_fr35] = computePSTH35(spike_times, tsEnc, tsMaint, idx, ...
                                               bin_size, encOffset, maintOffset, ...
                                               use_z, gauss_k, pad_bins)


nT   = numel(idx);
nEnc = round(encOffset   / bin_size);
nMai = round(maintOffset / bin_size);
nTot = nEnc + nMai;

if nT==0
    mean_fr35=zeros(1,nTot);
    sem_fr35 =zeros(1,nTot);
    return;
end

FR = nan(nT, nTot);
for k = 1:nT
    tr = idx(k);

    % Encoding fragment
    frE = nan(1,nEnc);
    if ~isempty(tsEnc) && tr <= numel(tsEnc)
        edgesE = tsEnc(tr) : bin_size : (tsEnc(tr) + encOffset);
        frE    = histcounts(spike_times, edgesE) ./ bin_size;
    end

    % Maintenance fragment
    frM = nan(1,nMai);
    if tr <= numel(tsMaint)
        edgesM = tsMaint(tr) : bin_size : (tsMaint(tr) + maintOffset);
        frM    = histcounts(spike_times, edgesM) ./ bin_size;
    end

    fr = [frE frM];                      % 1 × nTot
    fr = fr(:)';                         % row
    
    if nTot > 2*pad_bins                 % enough room to pad safely
        vpad = [ fliplr(fr(1:pad_bins)) , ...
                 fr , ...
                 fliplr(fr(end-pad_bins+1:end)) ];
        vsm  = conv(vpad, gauss_k, 'same');
        fr   = vsm(pad_bins+1 : pad_bins+nTot);    % crop back to 3.5 s
    else
        % fall back if window is too short for padding (shouldn't happen with 3.5 s)
        fr = conv(fr, gauss_k, 'same');
    end


    if use_z
        mu = mean(fr,'omitnan'); sd = std(fr,0,'omitnan');
        if isfinite(sd) && sd>0
            fr = (fr - mu) ./ sd;            % z-score across 35 bins
        else
            fr = zeros(size(fr));            % guard flat rows
        end
    end

    FR(k,:) = fr;
end

mean_fr35 = mean(FR, 1, 'omitnan');
sem_fr35  = std (FR, 0, 1, 'omitnan') / sqrt(sum(all(~isnan(FR),2)));

end

% ------------------------- small helper -------------------------------
function ts = get_ts(sess,key)
if isKey(sess.intervals_trials.vectordata,key)
    ts = sess.intervals_trials.vectordata.get(key).data.load();
else
    ts = [];
end
end
