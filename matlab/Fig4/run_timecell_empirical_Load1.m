function region_stats = run_timecell_empirical_Load1(nwbAll, all_units, bin_width_analysis, useSpikeInPeak, num_iterations, print_every)
% RUN_TIMECELL_EMPIRICAL_LOAD1
% Empirical p-values per region (observed vs null) for load-1 image-specific time cells,
% using the *fast* detector that returns only significance flags + regions.
%
% Inputs
%   nwbAll, all_units           : dataset and units
%   bin_width_analysis          : e.g., 0.1
%   useSpikeInPeak (logical)    : pass-through to detector
%   num_iterations (default 500): # null iterations
%   print_every (default 1)     : print progress every k iterations (1 = every iter)
%
% Output
%   region_stats : table {Region, ObservedCount, NullMean, EmpiricalP}

    if nargin < 5 || isempty(num_iterations), num_iterations = 500; end
    if nargin < 6 || isempty(print_every),    print_every    = 1;   end

    rng(42);  % reproducible unless you change this

    % ----------------------- Observed pass -----------------------
    fprintf('Running observed detection (FAST)...\n');
    [sig_obs, areas_obs, ~] = NWB_calcSelective_SB_Load1ImageSpecificmay11_fast( ...
        nwbAll, all_units, bin_width_analysis, useSpikeInPeak);

    regions_obs   = cellfun(@canon_region, areas_obs, 'uni', false);
    obs_counts    = counts_by_region_sig(regions_obs, sig_obs.time_cells);

    % Initialize running tallies over union of regions
    exceed_map = containers.Map('KeyType','char','ValueType','double');
    sum_map    = containers.Map('KeyType','char','ValueType','double');

    obs_regs = sort(obs_counts.keys());
    for i = 1:numel(obs_regs)
        exceed_map(obs_regs{i}) = 0;
        sum_map(obs_regs{i})    = 0;
    end

    % ------------------ Null iteration scaffolding ----------------
    total_bins = round(2.5 / bin_width_analysis);
    L          = total_bins * bin_width_analysis;

    % Cache maintenance timestamps per session
    ts_cache = containers.Map('KeyType','double','ValueType','any');

    fprintf('Running %d null iterations...\n', num_iterations);
    for it = 1:num_iterations
        t_it = tic;

        % --- Build shuffled units: per-trial circular shift in [0,L) ---
        all_units_shuf = all_units;  % shallow copy: only spike_times replaced
        for iU = 1:numel(all_units)
            SU  = all_units(iU);
            sid = SU.session_count;

            if ~isKey(ts_cache, sid)
                ts = nwbAll{sid}.intervals_trials.vectordata ...
                         .get('timestamps_Maintenance').data.load();
                ts_cache(sid) = ts(:);
            end
            tsM = ts_cache(sid);

            st = SU.spike_times(:);
            for k = 1:numel(tsM)
                t0 = tsM(k); t1 = t0 + L;
                in = (st >= t0 & st < t1);
                if any(in)
                    shift_amt = rand() * L;
                    rel = st(in) - t0;
                    rel = mod(rel + shift_amt, L);
                    st(in) = rel + t0;
                end
            end
            all_units_shuf(iU).spike_times = st;
        end

        % --- FAST detection on shuffled dataset ---
        [sig_null, areas_null, ~] = NWB_calcSelective_SB_Load1ImageSpecificmay11_fast( ...
            nwbAll, all_units_shuf, bin_width_analysis, useSpikeInPeak);

        % --- Tally per region for this iteration ---
        regions_null    = cellfun(@canon_region, areas_null, 'uni', false);
        null_counts_map = counts_by_region_sig(regions_null, sig_null.time_cells);

        % Ensure union of regions is represented
        null_regs = null_counts_map.keys();
        for j = 1:numel(null_regs)
            r = null_regs{j};
            if ~isKey(exceed_map, r), exceed_map(r) = 0; end
            if ~isKey(sum_map, r),    sum_map(r)    = 0; end
            if ~isKey(obs_counts, r), obs_counts(r) = 0; end
        end

        % Update tallies
        all_regs = sort(unique([exceed_map.keys(), null_counts_map.keys()])); %#ok<CCAT>
        for j = 1:numel(all_regs)
            r = all_regs{j};
            obs_ct  = 0; if isKey(obs_counts, r),  obs_ct  = obs_counts(r);  end
            null_ct = 0; if isKey(null_counts_map, r), null_ct = null_counts_map(r); end

            sum_map(r) = sum_map(r) + null_ct;
            if null_ct >= obs_ct
                exceed_map(r) = exceed_map(r) + 1;
            end
        end

        % ----------- PROGRESS LOGGING (your ask) --------------
        if mod(it, print_every)==0 || it==1 || it==num_iterations
            total_sig = sum(sig_null.time_cells);
            fprintf('Null iter %d/%d: %d cells detected (%.2f sec)\n', ...
                    it, num_iterations, total_sig, toc(t_it));

            % OPTIONAL: print the top 3 regions for this iteration
            % comment out these 8 lines if you want quieter logs
            if ~isempty(null_regs)
                % build a small array for sorting
                pairs = cell(numel(null_regs), 2);
                for a = 1:numel(null_regs)
                    reg = null_regs{a};
                    pairs{a,1} = reg;
                    pairs{a,2} = null_counts_map(reg);
                end
                [~, ord] = sort(cell2mat(pairs(:,2)), 'descend');
                topk = min(3, numel(ord));
                msg = join(string(pairs(ord(1:topk),1)) + ":" + string(cell2mat(pairs(ord(1:topk),2))).', ", ");
                fprintf('    Top regions: %s\n', msg{1});
            end
        end
    end

    % ----------------------- Finalize table ----------------------
    regs = sort(exceed_map.keys());
    R    = numel(regs);
    ObservedCount = zeros(R,1);
    NullMean      = zeros(R,1);
    EmpiricalP    = zeros(R,1);

    for i = 1:R
        r = regs{i};
        if isKey(obs_counts, r), ObservedCount(i) = obs_counts(r); end
        NullMean(i)   = sum_map(r) / max(1, num_iterations);
        EmpiricalP(i) = (exceed_map(r) + 1) / (num_iterations + 1);
    end

    region_stats = table(regs(:), ObservedCount, NullMean, EmpiricalP, ...
        'VariableNames', {'Region','ObservedCount','NullMean','EmpiricalP'});

    fprintf('\nEmpirical region stats (I=%d):\n', num_iterations);
    disp(region_stats);
end

% -------------------------- Helpers --------------------------------
function canon = canon_region(str_in)
    s = lower(strtrim(str_in));
    canon = regexprep(s, '_(left|right)$', '');
end

function cmap = counts_by_region_sig(region_cell, is_sig)
    % Map: region -> # of significant units
    cmap = containers.Map('KeyType','char','ValueType','double');
    for i = 1:numel(region_cell)
        if is_sig(i)
            r = region_cell{i};
            if ~isKey(cmap, r), cmap(r) = 1; else, cmap(r) = cmap(r) + 1; end
        end
    end
end
