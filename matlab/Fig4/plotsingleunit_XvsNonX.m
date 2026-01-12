function plotsingleunit_XvsNonX(nwbAll, all_units, ...
                                neural_data_file, bin_size, ...
                                use_zscore)
% PLOTSINGLEUNIT_XVSNONX  Raster + dual PSTHs (X vs. Non‑X trials)
% Adjusted:
% * Fixed axis font size to 12.
% * Figure size set to 400x648 pixels.
% * Field window estimation uses only preferred image (X) trials.

if nargin < 6, use_zscore = false; end

load(neural_data_file,'neural_data');

bins       = 0:bin_size:2.5;
n_bins     = numel(bins)-1;

gauss_k = GaussianKernal(0.3 / bin_size, 1.5);

for ndx = 20: 100 %numel(neural_data)
    pid      = neural_data(ndx).patient_id;
    uid      = neural_data(ndx).unit_id;
    tf_bin   = double(neural_data(ndx).time_field);
    pref_img = neural_data(ndx).preferred_image;
    imgIDs   = neural_data(ndx).trial_imageIDs;

    m = ([all_units.subject_id]==pid & [all_units.unit_id]==uid);
    if ~any(m)
        warning('Unit %d/%d not found',pid,uid); continue;
    end
    SU = all_units(find(m,1));

    tsMaint = nwbAll{SU.session_count}.intervals_trials ...
                        .vectordata.get('timestamps_Maintenance') ...
                        .data.load();
    spk     = SU.spike_times;
    nTrials = numel(tsMaint);

    if nTrials ~= size(imgIDs,1)
        warning('Trial count mismatch for patient %d, unit %d',pid,uid);
    end

    enc1 = imgIDs(:,1); enc2 = imgIDs(:,2); enc3 = imgIDs(:,3);

    X_idx    = find(enc1==pref_img & enc2==0 & enc3==0);
    NonX_idx = find(enc1~=pref_img & enc2==0 & enc3==0);

    % --- Use only X trials for field estimation ---
    fr_all = zeros(numel(X_idx),n_bins);
    for t = 1:numel(X_idx)
        s = spk(spk>=tsMaint(X_idx(t)) & spk<tsMaint(X_idx(t))+2.5) - tsMaint(X_idx(t));
        fr_all(t,:) = histcounts(s,bins)/bin_size;
    end
    mean_fr = mean(fr_all,1);
    baseline = mean(mean_fr);
    thr      = baseline + 0.5*std(mean_fr);

    centre_t   = (tf_bin-0.5)*0.1;
    [~,centre] = min(abs(bins(1:end-1)-centre_t));

    left  = centre;
    while left>1      && mean_fr(left)  >= thr, left  = left-1; end
    right = centre;
    while right<n_bins && mean_fr(right) >= thr, right = right+1; end

    t_start = bins(left);
    t_end   = bins(right);

    field_start = (tf_bin-1)*0.1;
    field_end   =  tf_bin   *0.1;
    t_start = min(t_start, field_start);
    t_end   = max(t_end,   field_end);
    if t_end <= t_start
        t_start = field_start;  t_end = field_end;
    end

    % --- Figure ---
    figure('Name',sprintf('P%d U%d pref=%d',pid,uid,pref_img), ...
           'Units','pixels','Position',[100 100 400 648]);

    % === RASTER ===
    subplot(2,1,1); hold on
    fill([t_start t_start t_end t_end],[0 nTrials+1 nTrials+1 0], ...
         [0.6 0.8 1],'EdgeColor','none','FaceAlpha',0.4);
    xlabel('Time (s)','FontSize',12); ylabel('Trial','FontSize',12);
    title(sprintf('Patient %d, Unit %d, Pref Img %d',pid,uid,pref_img), ...
          'Interpreter','none','FontSize',12);

    groups = {X_idx, NonX_idx};
    colors = {'b', [1 0.5 0]};
    offset = 0;
    for g = 1:2
        idx = groups{g};
        col = colors{g};
        for i = 1:numel(idx)
            tr = idx(i);
            s  = spk(spk>=tsMaint(tr) & spk<tsMaint(tr)+2.5) - tsMaint(tr);
            scatter(s, offset+i*ones(size(s)), 9, col,'filled');
        end
        offset = offset + numel(idx);
    end
    xlim([0 2.5]); ylim([0 offset+1]);
    set(gca,'FontSize',12);

    % === PSTH ===
    subplot(2,1,2); hold on
    if use_zscore
        ylabel('Z‑score','FontSize',12);
    else
        ylabel('Rate (spk/s)','FontSize',12);
    end
    xlabel('Time (s)','FontSize',12);
    
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
        set(gca,'FontSize',12);
    
    end
    end


% ========================================================================
function [mean_fr, sem_fr] = computePSTH(spike_times, tsMaint, idx, ...
                                         bin_size, bins, use_z)
% Single‑group PSTH, optional z‑score across time bins per trial
    nT = numel(idx);
    nb = numel(bins)-1;
    if nT==0, mean_fr=zeros(1,nb); sem_fr=zeros(1,nb); return; end

    FR = zeros(nT,nb);
    for k = 1:nT
        tr  = idx(k);
        s   = spike_times(spike_times>=tsMaint(tr) & ...
                          spike_times< tsMaint(tr)+2.5) - tsMaint(tr);
        cnt = histcounts(s,bins);
        r   = cnt/bin_size;

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