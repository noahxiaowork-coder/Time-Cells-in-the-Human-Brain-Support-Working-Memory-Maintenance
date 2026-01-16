 function plot_time_cell_heatmaps(nwbAll, all_units, ...
                                          neural_data_file, bin_width)
% PLOT_TIME_CELL_HEATMAPS_FROM_RAW
%   Rebuilds odd-, even- and all-trial PSTHs for every single unit
%   directly from raw spike times in NWB, then plots three heat-maps:
%       1) All trials       – sorted by peak of ALL-trial PSTH
%       2) Odd trials       \
%       3) Even trials       > both sorted by peak of ODD-trial PSTH
%
% INPUTS
%   nwbAll   – {cell} of NWB objects, one per recording session
%   all_units– struct array with fields .subject_id, .session_count, .unit_id
%   neural_data_file – MAT-file created by your earlier pipeline
%                      (only used for meta info: patient_id, unit_id, trial‐
%                       aligned timestamps, etc.)
%   bin_width – PSTH bin size in seconds (e.g. 0.05)
%
% OUTPUT
%   Saves a single PDF with the three heat-maps into the same folder that
%   contained neural_data_file.

%% =========== parameters (edit if you want) =============================
maint_dur = 2.5;                           % maintenance interval (s)
n_bins    = round(maint_dur / bin_width);  % bins along x-axis


gkernel = GaussianKernal(0.3 / bin_width, 1.5);
% make kernel a normalized row vector

if any(gkernel)
    gkernel = gkernel / sum(gkernel);
end

% kernel width in bins (σ = 0.3 s)
sigma_bins = 0.3 / bin_width;
halfWidth  = floor((numel(gkernel) - 1)/2);
pad_bins   = min(halfWidth, floor((n_bins-1)/2));   % safety guard


%% =========== load meta-data only (trial times, labels) =================
load(neural_data_file,'neural_data');      % will *not* use .firing_rates
n_neurons = numel(neural_data);
assert(n_neurons>0,'No neurons inside neural_data.');

% Pre-allocate big arrays (NaN for safety)
all_psth  = nan(n_neurons,n_bins);
odd_psth  = nan(n_neurons,n_bins);
even_psth = nan(n_neurons,n_bins);
peak_all  = nan(n_neurons,1);
peak_odd  = nan(n_neurons,1);

%% =========== MAIN LOOP over neurons ===================================
for u = 1:n_neurons
    nd   = neural_data(u);
    pid  = nd.patient_id;
    uid  = nd.unit_id;

    % ---- locate the same single unit inside all_units / nwbAll ---------
    j = find([all_units.subject_id]==pid & ...
             [all_units.unit_id]==uid, 1);
    if isempty(j), warning('Unit not found in all_units, skip'); continue; end

    SU     = all_units(j);
    nwb    = nwbAll{SU.session_count};

    % -------- get per-trial reference timestamp for Maintenance --------
    ts_maint = nwb.intervals_trials ...
                  .vectordata.get('timestamps_Maintenance').data.load();
    n_trials = numel(ts_maint);
    if n_trials==0, continue; end

    % -------- build single-trial firing-rate matrix --------------------
    fr = nan(n_trials, n_bins);
    spike_t = SU.spike_times;          % raw spike vector (sec, vis-à-vis file)
    for t = 1:n_trials
        edges = ts_maint(t):bin_width:(ts_maint(t)+maint_dur);
        if numel(edges) ~= n_bins+1, continue; end
        % cnt = histcounts(spike_t, edges);
        % fr(t,:) = conv(cnt ./ bin_width, gkernel, 'same');


        cnt = histcounts(spike_t, edges);        % 1 × n_bins
        rate = cnt ./ bin_width;                 % Hz
        rate = rate(:)';                         % row

        if pad_bins > 0 && n_bins > 2*pad_bins
            % mirror-pad along time axis
            rate_pad = [ fliplr(rate(1:pad_bins)), ...
                         rate, ...
                         fliplr(rate(end-pad_bins+1:end)) ];

            sm_pad   = conv(rate_pad, gkernel, 'same');
            fr(t,:)  = sm_pad(pad_bins+1 : pad_bins+n_bins);
        else
            % fallback if window is too short
            fr(t,:) = conv(rate, gkernel, 'same');
        end


        %fr(t,:) = zscore(fr(t,:));
    end

    % Odd/even index vectors
    odd_idx  = mod(1:n_trials,2)==1;
    even_idx = ~odd_idx;

    % Average PSTHs
    avg_all  = mean(fr,            1,'omitnan');
    avg_odd  = mean(fr(odd_idx ,:),1,'omitnan');
    avg_even = mean(fr(even_idx,:),1,'omitnan');

    % Store
    all_psth(u,:)  = avg_all;
    odd_psth(u,:)  = avg_odd;
    even_psth(u,:) = avg_even;

    [~,peak_all(u)] = max(avg_all);
    [~,peak_odd(u)] = max(avg_odd);
end

%% =========== sort, normalise, plot ====================================
% Sort indices
[~,ord_all] = sort(peak_all,'ascend','MissingPlacement','last');
[~,ord_odd] = sort(peak_odd,'ascend','MissingPlacement','last');

% Apply sorting
S = @(M,ord) M(ord,:);
all_sorted  = S(all_psth , ord_all);
odd_sorted  = S(odd_psth , ord_odd);
even_sorted = S(even_psth, ord_odd);

% Row-wise min–max (vectorised)
all_sorted  = rowwise_minmax_normalize(all_sorted);
odd_sorted  = rowwise_minmax_normalize(odd_sorted);
even_sorted = rowwise_minmax_normalize(even_sorted);

% Time axis (left edges + half-bin)
t_centers = ((0:n_bins-1)+0.5)*bin_width;
tick_step = 0.5;
tick_labs = 0:tick_step:maint_dur;
xt        = arrayfun(@(t) find_closest_bin(t_centers,t), tick_labs);

% Plot three panels in one tiledlayout
cmap = parula;
fig  = figure('Name','Time-Cell Heat-maps');
t    = tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');

for k = 1:3
    ax = nexttile(t);
    switch k
        case 1, mat = all_sorted;  ttl = 'All Trials';
        case 2, mat = odd_sorted;  ttl = 'Odd Trials';
        case 3, mat = even_sorted; ttl = 'Even Trials';
    end
    imagesc(ax,mat,[0 1]);
    title(ax,ttl); xlabel(ax,'Time (s)'); ylabel(ax,'Neuron');
    set(ax,'TickDir','out','XTick',xt,'XTickLabel',tick_labs);
end

colormap(fig,cmap);   % set palette for the whole figure

cb = colorbar;        % create colour-bar on the current axes
cb.Layout.Tile = 'east';  % dock it to the right of the 1×3 grid
                    % works fine with tiledlayout

exportgraphics(t, replace(neural_data_file,'.mat','_raw_heatmaps.pdf'), ...
               'ContentType','vector');
disp('Heat-maps generated from raw spikes.');

end % ---------- main function end --------------------------------------


%% =========== helper functions ==========================================
function mat = rowwise_minmax_normalize(mat)
    mins   = min(mat,[],2,'omitnan');
    maxs   = max(mat,[],2,'omitnan');
    span   = maxs - mins;
    const  = span==0 | isnan(span);
    mat    = (mat - mins) ./ span;
    %mat    = mat ./ maxs;
    mat(const,:) = NaN;               % transparent if desired
end

function idx = find_closest_bin(t_centers, tgt)
    [~,idx] = min(abs(t_centers - tgt));
end

function mat = peak_normalize(mat)
    maxs   = max(mat,[],2,'omitnan');
    mat    = mat ./ maxs;
    % mat(const,:) = NaN;               % transparent if desired
end
