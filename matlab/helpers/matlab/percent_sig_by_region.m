function stats = percent_sig_by_region(nwbAll, all_units, neural_data, bin_size, alpha, use_zscore)

if nargin < 5 || isempty(alpha), alpha = 0.05; end
if nargin < 6 || isempty(use_zscore), use_zscore = false; end

% normalize region labels
n = numel(neural_data);
region = strings(n,1);

for i = 1:n
    br = neural_data(i).brain_region;

    if iscell(br)
        if isempty(br), continue; end
        br = br{1};
    end
    if isempty(br) || ~ischar(br), continue; end

    br_norm = lower(strtrim(br));
    br_norm = regexprep(br_norm, '_(left|right)$', '');
    region(i) = string(br_norm);
end

valid_region = region ~= "";
regions = unique(region(valid_region));

% storage
pvals   = nan(n,1);
isvalid = false(n,1);

for i = 1:n
    if region(i) == "", continue; end

    subj = neural_data(i).patient_id;
    unit = neural_data(i).unit_id;

    % IMPORTANT: requires plot_single_cell_performance to return p_value
    [pvals(i), isvalid(i)] = plot_single_cell_performance( ...
        nwbAll, all_units, neural_data, bin_size, subj, unit, use_zscore);
end

% build stats table
nR = numel(regions);
Region      = strings(nR,1);
N_total     = zeros(nR,1);
N_valid     = zeros(nR,1);
N_sig       = zeros(nR,1);
Pct_sig     = nan(nR,1);

for r = 1:nR
    Region(r)  = regions(r);
    idx        = (region == regions(r));
    N_total(r) = sum(idx);

    idxv       = idx & isvalid & ~isnan(pvals);
    N_valid(r) = sum(idxv);

    N_sig(r)   = sum(idxv & (pvals < alpha));

    if N_valid(r) > 0
        Pct_sig(r) = 100 * N_sig(r) / N_valid(r);
    end
end

stats = table(Region, N_total, N_valid, N_sig, Pct_sig);
stats = sortrows(stats, "Pct_sig", "descend");

disp(stats);
end
