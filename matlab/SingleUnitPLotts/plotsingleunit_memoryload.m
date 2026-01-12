function plotsingleunit_memoryload(nwbAll, all_units, ...
                                   neural_data_file, field_width, ...
                                   use_zscore, use_correct)

if nargin < 6, use_zscore  = false; end
if nargin < 7, use_correct = false; end

load(neural_data_file,'neural_data');

bin_size   = 0.05;
bins       = 0:bin_size:2.5;
n_bins     = numel(bins)-1;

gaussian_sigma  = 2;
kernel_size     = round(5*gaussian_sigma);
x_kernel        = -kernel_size:kernel_size;
gauss_kernel    = exp(-(x_kernel.^2)/(2*gaussian_sigma^2));
gauss_kernel    = gauss_kernel / sum(gauss_kernel);

for ndx = 111:150
    pid   = neural_data(ndx).patient_id;
    uid   = neural_data(ndx).unit_id;
    tfbin = double(neural_data(ndx).time_field);
    load_v = neural_data(ndx).trial_load;
    corr_v = neural_data(ndx).trial_correctness;

    m = ([all_units.subject_id]==pid & [all_units.unit_id]==uid);
    if ~any(m), warning('Unit %d/%d not found',pid,uid); continue; end
    SU = all_units(find(m,1));

    tsMaint = nwbAll{SU.session_count}.intervals_trials ...
                        .vectordata.get('timestamps_Maintenance') ...
                        .data.load();
    spk     = SU.spike_times;
    nTrials = numel(tsMaint);

    FR_all = zeros(nTrials,n_bins);
    for t = 1:nTrials
        s = spk(spk>=tsMaint(t)&spk<tsMaint(t)+2.5) - tsMaint(t);
        FR_all(t,:) = histcounts(s,bins)/bin_size;
    end
    mFR = mean(FR_all,1);
    base = mean(mFR);
    thr  = base + 0.5*std(mFR);

    centre_t   = (tfbin-0.5)*0.1;
    [~,centre] = min(abs(bins(1:end-1)-centre_t));

    left = centre;
    while left>1 && mFR(left)>=thr,  left  = left-1; end
    right = centre;
    while right<n_bins && mFR(right)>=thr, right = right+1; end

    t_start = bins(left);
    t_end   = bins(right);

    f_start = (tfbin-1)*0.1;
    f_end   =  tfbin*0.1;
    t_start = min(t_start, f_start);
    t_end   = max(t_end,   f_end);
    if t_end<=t_start, t_start=f_start; t_end=f_end; end

    if use_correct
        idx1 = find(load_v==1 & corr_v==1);
        idx2 = find(load_v==2 & corr_v==1);
        idx3 = find(load_v==3 & corr_v==1);
    else
        idx1 = find(load_v==1);
        idx2 = find(load_v==2);
        idx3 = find(load_v==3);
    end

    figure('Units','pixels','Position',[100 100 400 648]);

    subplot(2,1,1); hold on
    set(gca,'FontSize',12);
    fill([t_start t_start t_end t_end], [0 nTrials+1 nTrials+1 0], ...
         [0.6 0.8 1],'EdgeColor','none','FaceAlpha',0.4);
    
    title(sprintf('Patient %d, Unit %d, Time Field %d',pid,uid,tfbin), ...
          'Interpreter','none','FontSize',12);
    xlabel('Time (s)','FontSize',12);
    ylabel('Trial','FontSize',12);
    
    subplot(2,1,2); hold on
    set(gca,'FontSize',12);
    xlabel('Time (s)','FontSize',12);
    if use_zscore
        ylabel('Z-score','FontSize',12);
    else
        ylabel('Rate (spk/s)','FontSize',12);
    end

    [m1,s1] = psth_group(spk,tsMaint,idx1,use_zscore,bins,bin_size);
    [m2,s2] = psth_group(spk,tsMaint,idx2,use_zscore,bins,bin_size);
    [m3,s3] = psth_group(spk,tsMaint,idx3,use_zscore,bins,bin_size);

    m1 = conv(m1,gauss_kernel,'same'); s1 = conv(s1,gauss_kernel,'same');
    m2 = conv(m2,gauss_kernel,'same'); s2 = conv(s2,gauss_kernel,'same');
    m3 = conv(m3,gauss_kernel,'same'); s3 = conv(s3,gauss_kernel,'same');

    t = bins(1:end-1);
    plot_fill(t,m1,s1,'b');
    plot_fill(t,m2,s2,'g');
    plot_fill(t,m3,s3,'r');

    yl = ylim;
    patch([t_start t_start t_end t_end],[yl(1) yl(2) yl(2) yl(1)], ...
          [0.6 0.8 1],'EdgeColor','none','FaceAlpha',0.2,'HandleVisibility','off');

    legend({'Load 1','Load 2','Load 3'},'Location','best');
    xlim([0 2.5]);

    allVals = [m1+s1 m1-s1 m2+s2 m2-s2 m3+s3 m3-s3];
    yMin = min(allVals); yMax = max(allVals);
    pad  = 0.1*(yMax-yMin + eps);
    ylim([yMin-pad, yMax+pad]);
end
end

function [mean_fr, sem_fr] = psth_group(spk, ts, idx, zflag, bins, bin_size)
    nT = numel(idx);
    nb = numel(bins)-1;
    if nT==0, mean_fr=zeros(1,nb); sem_fr=zeros(1,nb); return; end
    FR = zeros(nT,nb);
    for k = 1:nT
        tr = idx(k);
        s  = spk(spk>=ts(tr)&spk<ts(tr)+2.5) - ts(tr);
        r  = histcounts(s,bins)/bin_size;
        if zflag
            mu = mean(r); sd = std(r);
            if sd==0, r = zeros(size(r)); else, r = (r-mu)/sd; end
        end
        FR(k,:) = r;
    end
    mean_fr = mean(FR,1);
    sem_fr  = std(FR,0,1)/sqrt(nT);
end

function plot_fill(t,m,sem,col)
    fill([t fliplr(t)], [m+sem fliplr(m-sem)], col, ...
         'EdgeColor','none','FaceAlpha',0.2);
    plot(t,m,'Color',col,'LineWidth',2);
end
