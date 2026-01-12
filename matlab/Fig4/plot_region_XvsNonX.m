function plot_region_XvsNonX(nwbAll, all_units, neural_data, ...
                             bin_size, region_name, ...
                             use_zscore, show_multi_nonpref)
% PLOT_REGION_XVSNONX
% Loop over all neurons whose brain_region (with '_left'/'_right' stripped)
% matches region_name, and call plot_single_cell_XvsNonX for each.
%
% Inputs:
%   nwbAll, all_units, neural_data : your usual data structs
%   bin_size      : bin size in seconds (e.g. 0.1)
%   region_name   : e.g. 'MTL', 'PFC', 'HC', etc.
%   use_zscore    : (optional) true/false for PSTH z-scoring
%   show_multi_nonpref : (optional) true/false for multi-curve vs single orange curve
%
% Example:
%   plot_region_XvsNonX(nwbAll, all_units, neural_data, 0.1, 'MTL');
%   plot_region_XvsNonX(nwbAll, all_units, neural_data, 0.1, 'MTL', true, false);

if nargin < 6 || isempty(use_zscore),       use_zscore = false;        end
if nargin < 7 || isempty(show_multi_nonpref), show_multi_nonpref = false; end

% Normalize requested region name (lowercase, no side info)
region_name_norm = lower(strtrim(region_name));
region_name_norm = regexprep(region_name_norm, '_(left|right)$', '');  % just in case

% Collect indices of neurons in this region
idx_region = [];

for i = 1:numel(neural_data)
    if ~isfield(neural_data, 'brain_region')
        error('neural_data does not contain field "brain_region".');
    end

    br = neural_data(i).brain_region;
    % handle possible cell array
    if iscell(br)
        br = br{1};
    end
    if isempty(br) || ~ischar(br)
        continue;
    end

    % normalize: lowercase, strip trailing '_left'/'_right'
    br_norm = lower(strtrim(br));
    br_norm = regexprep(br_norm, '_(left|right)$', '');

    if strcmp(br_norm, region_name_norm)
        idx_region(end+1) = i; %#ok<AGROW>
    end
end

if isempty(idx_region)
    warning('No neurons found in region "%s".', region_name);
    return;
end

fprintf('Found %d neurons in region "%s".\n', numel(idx_region), region_name);

% Optional: sort by subject_id then unit_id for nicer ordering
subject_ids = arrayfun(@(x) x.patient_id, neural_data(idx_region));
unit_ids    = arrayfun(@(x) x.unit_id,    neural_data(idx_region));
[~, sort_order] = sortrows([subject_ids(:), unit_ids(:)]);
idx_region = idx_region(sort_order);

% Loop through neurons in this region and call your single-cell plotter
for k = 1:numel(idx_region)
    ndx = idx_region(k);
    subj = neural_data(ndx).patient_id;
    unit = neural_data(ndx).unit_id;

    fprintf('Plotting Subject %d, Unit %d (region %s).\n', ...
            subj, unit, neural_data(ndx).brain_region);

    % Assumes your updated version of plot_single_cell_XvsNonX:
    %   plot_single_cell_XvsNonX(nwbAll, all_units, neural_data, ...
    %                             bin_size, subject_id, unit_id, ...
    %                             use_zscore, show_multi_nonpref)
    plot_single_cell_XvsNonX(nwbAll, all_units, neural_data, ...
                             bin_size, subj, unit, ...
                             use_zscore, show_multi_nonpref);
end

end
