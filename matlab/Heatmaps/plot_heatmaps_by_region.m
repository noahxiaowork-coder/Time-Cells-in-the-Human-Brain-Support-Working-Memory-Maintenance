function plot_heatmaps_by_region(nwbAll, all_units, neural_data_file, ...
                                 bin_width, use_correct)
%PLOT_HEATMAPS_BY_REGION  Region-wise Maintenance heat-maps rebuilt from raw spikes
%
% INPUTS
%   nwbAll            – {1×S} cell of NWB session objects
%   all_units         – struct array with fields: subject_id, unit_id,
%                       session_count, spike_times
%   neural_data_file  – MAT file holding `neural_data` (for metadata only)
%   bin_width         – size of PSTH bin in seconds (e.g. 0.100)
%   use_correct       – logical (default = false).  If true, keep only
%                       trials where neural_data.trial_correctness == 1
%
% OUTPUT
%   • One figure per unified brain region (fixed 648 × 400 px window)
%   • A horizontal bar chart: % time-cells per region
%
%   PSTH construction:
%       – 2.5-s Maintenance epoch
%       – histogram → bins of BIN_WIDTH
%       – Gaussian smoothing, σ = 0.2 s (kernel truncated ±2 σ)

% -------------------------------------------------- parameters ----------
if nargin < 5, use_correct = false; end
maintDur = 2.5;                         % Maintenance duration (s)
nBins    = round(maintDur/bin_width);   % #bins per PSTH

% ----- Gaussian kernel --------------------------------------------------
sigmaBins = 0.2 / bin_width;
kSize     = round(2*sigmaBins);
x         = -kSize:kSize;
gKernel   = exp(-(x.^2) ./ (2*sigmaBins^2));
gKernel   = gKernel ./ sum(gKernel);

% ------------------------------------------------------------------------
load(neural_data_file,'neural_data');
nNeurons = numel(neural_data);
assert(nNeurons>0,'No neurons in neural_data.');

% ======= STEP 1: unify region labels (strip _left/_right) ===============
region = cell(nNeurons,1);
for i = 1:nNeurons
    nm = neural_data(i).brain_region;
    if endsWith(nm,'_left')
        nm = extractBefore(nm, strlength(nm)-strlength('_left')+1);
    elseif endsWith(nm,'_right')
        nm = extractBefore(nm, strlength(nm)-strlength('_right')+1);
    end
    region{i} = nm;
end
uniqReg = unique(region);

% containers for summary stats
reg_time_cells  = zeros(numel(uniqReg),1);
reg_total_cells = zeros(numel(uniqReg),1);

% ======= X-axis tick helpers ============================================
tCenters   = (0:nBins-1)*bin_width + bin_width/2;
baseTicks  = 0:0.5:maintDur;
xtick_pos  = arrayfun(@(t) find_closest_bin(tCenters,t), baseTicks);
xtick_lab  = arrayfun(@(t) sprintf('%.1f',t),baseTicks,'uni',0);
cmap       = parula;

% ========================================================================
%                      REGION-WISE LOOP
% ========================================================================
for r = 1:numel(uniqReg)
    regName = uniqReg{r};
    idxReg  = strcmp(region, regName);
    reg_total_cells(r) = nnz(idxReg);
    if reg_total_cells(r)==0, continue; end

    fr_all   = nan(reg_total_cells(r), nBins);   % neuron × time
    peak_bin = nan(reg_total_cells(r), 1);
    tCellCnt = 0;
    row      = 0;

    for i = find(idxReg)'
        row = row + 1;
        nd  = neural_data(i);
        pid = nd.patient_id;  uid = nd.unit_id;

        % -- locate spike train -----------------------------------------
        j = find([all_units.subject_id]==pid & ...
                 [all_units.unit_id]==uid, 1);
        if isempty(j), continue; end
        SU   = all_units(j);
        sess = nwbAll{SU.session_count};

        tsMai = sess.intervals_trials.vectordata ...
                       .get('timestamps_Maintenance').data.load();
        nTr   = numel(tsMai);
        if nTr==0, continue; end

        corr_vals = nd.trial_correctness(:);
        if numel(corr_vals)~=nTr, continue; end

        if use_correct
            keepOK = (corr_vals == 1);
        else
            keepOK = true(size(corr_vals));
        end


        % -- build smoothed FR matrix -----------------------------------
        FR = nan(nTr,nBins);
        for t = 1:nTr
            if ~keepOK(t), continue; end
            edges = tsMai(t):bin_width:(tsMai(t)+maintDur);
            if numel(edges)~=nBins+1, continue; end
            tmp   = histcounts(SU.spike_times,edges)./bin_width;
            FR(t,:) = conv(tmp,gKernel,'same');
        end

        avgFR = mean(FR,1,'omitnan');
        fr_all(row,:) = avgFR;

        [~,pbin] = max(avgFR);  peak_bin(row) = pbin;

        % ----- crude time-cell criterion (keep your own if needed) -----
        % if max(avgFR,[],'omitnan') > 1.5
        %     tCellCnt = tCellCnt + 1;
        % end
    end

    reg_time_cells(r) = tCellCnt;

    % -- sort + normalise ----------------------------------------------
    [~,ord] = sort(peak_bin,'ascend','MissingPlacement','last');
    H = fr_all(ord,:);
    H = rowwise_minmax_normalize(H);

    % -- plot -----------------------------------------------------------
    figure('Name',regName,'Units','pixels','Position',[100 100 648 400]);
    imagesc(H,[0 1]); colormap(cmap); colorbar;
    xlabel('Time (s)'); ylabel('Neuron');
    title(sprintf('Time Cells: %s (All Trials)',regName));
    set(gca,'TickDir','out','XTick',xtick_pos,'XTickLabel',xtick_lab);
end

% ======= summary bar plot ==============================================
pct = (reg_time_cells ./ reg_total_cells) * 100;
figure('Name','Percent Time Cells by Region', ...
       'Units','pixels','Position',[100 100 700 400]);
barh(pct,'FaceColor',[0.2 0.6 0.8],'EdgeColor','none');
xlabel('% Time Cells'); ylabel('Brain Region');
title('% Time Cells per Region (Maintenance)');
set(gca,'YTick',1:numel(uniqReg), 'YTickLabel',uniqReg); grid on;

end  % =================== end main function =============================

% -----------------------------------------------------------------------
function mat = rowwise_minmax_normalize(mat)
for r = 1:size(mat,1)
    row = mat(r,:);
    if all(isnan(row)), continue; end
    mn = min(row,[],'omitnan'); mx = max(row,[],'omitnan');
    if mx>mn
        mat(r,:) = (row-mn)/(mx-mn);
    else
        mat(r,:) = 0.5;
    end
end
end
% -----------------------------------------------------------------------
function idx = find_closest_bin(tc,target)
[~,idx] = min(abs(tc - target));
end
