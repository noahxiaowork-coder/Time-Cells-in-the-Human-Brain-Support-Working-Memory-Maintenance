function stats = venn_neural_overlap(matFiles, varargin)
% Venn diagram of (patient_id, unit_id) overlaps; no stats

if isstring(matFiles), matFiles = cellstr(matFiles); end
validateattributes(matFiles, {'cell'}, {'vector','nonempty'});

p = inputParser;
p.addParameter('Labels', {}, @(c) iscellstr(c) || isstring(c));
p.addParameter('Plot', true, @(b) islogical(b) && isscalar(b));
p.addParameter('VennFaceAlpha', 0.12, @(x) isnumeric(x) && isscalar(x) && x>=0 && x<=1);
p.addParameter('VennOutlineOnly', false, @(b) islogical(b) && isscalar(b));
p.parse(varargin{:});
labels = p.Results.Labels;
doPlot = p.Results.Plot;
vennFaceAlpha = p.Results.VennFaceAlpha;
vennOutlineOnly = p.Results.VennOutlineOnly;

% derive labels from filenames if empty
if isempty(labels)
    labels = cell(size(matFiles));
    for i = 1:numel(matFiles)
        [~, base, ext] = fileparts(matFiles{i});
        labels{i} = [base ext];
    end
end
labels = cellstr(string(labels));
N = numel(matFiles);
if numel(labels) ~= N
    error('Number of Labels (%d) must match number of files (%d).', numel(labels), N);
end
if N < 2 || N > 3
    error('Venn plot supported only for N=2 or 3 files.');
end

% load sets of (patient_id, unit_id)
sets = cell(1,N);
for i = 1:N
    f = matFiles{i};
    if ~isfile(f), error('File not found: %s', f); end
    S = load(f);
    if ~isfield(S, 'neural_data'), error('File %s missing variable ''neural_data''.', f); end
    nd = S.neural_data;
    if ~all(isfield(nd, {'patient_id','unit_id'}))
        error('In %s, neural_data must have fields patient_id and unit_id.', f);
    end
    pid = vertcat(nd.patient_id);
    uid = vertcat(nd.unit_id);
    sets{i} = unique([pid(:) uid(:)], 'rows');
end

sizes = cellfun(@(A) size(A,1), sets);
interRows = @(A,B) intersect(A,B,'rows','stable');

% region counts for N=2 or N=3
if N == 2
    n12   = size(interRows(sets{1}, sets{2}), 1);
    only1 = sizes(1) - n12;
    only2 = sizes(2) - n12;
    regionTbl = cell2table( ...
        {['only ' labels{1}], only1; ['only ' labels{2}], only2; [labels{1} '+' labels{2}], n12}, ...
        'VariableNames', {'Region','Count'});
else
    n12  = size(interRows(sets{1}, sets{2}), 1);
    n13  = size(interRows(sets{1}, sets{3}), 1);
    n23  = size(interRows(sets{2}, sets{3}), 1);
    n123 = size(interRows(interRows(sets{1},sets{2}), sets{3}), 1);
    ab_only = n12 - n123; ac_only = n13 - n123; bc_only = n23 - n123;
    a_only = sizes(1) - ab_only - ac_only - n123;
    b_only = sizes(2) - ab_only - bc_only - n123;
    c_only = sizes(3) - ac_only - bc_only - n123;
    regionTbl = cell2table( ...
        {['only ' labels{1}], a_only; ['only ' labels{2}], b_only; ['only ' labels{3}], c_only; ...
         [labels{1} '+' labels{2} ' (only)'], ab_only; ...
         [labels{1} '+' labels{3} ' (only)'], ac_only; ...
         [labels{2} '+' labels{3} ' (only)'], bc_only; ...
         [labels{1} '+' labels{2} '+' labels{3}], n123}, ...
        'VariableNames', {'Region','Count'});
end

% plot venn (N=2 or 3)
if doPlot
    figure('Name','Neural Unit Overlap (Venn)');
    switch N
        case 2
            nAB = size(interRows(sets{1}, sets{2}), 1);
            [H,~] = venn([sizes(1) sizes(2)], nAB);
        case 3
            n12  = size(interRows(sets{1}, sets{2}), 1);
            n13  = size(interRows(sets{1}, sets{3}), 1);
            n23  = size(interRows(sets{2}, sets{3}), 1);
            n123 = size(interRows(interRows(sets{1},sets{2}), sets{3}), 1);
            [H,~] = venn([sizes(1) sizes(2) sizes(3)], [n12 n13 n23 n123]);
    end
    faceAlphaUse = vennOutlineOnly * 0.0 + (~vennOutlineOnly) * vennFaceAlpha;
    if exist('H','var') && ~isempty(H)
        cmap = lines(numel(H));
        for ii = 1:numel(H)
            set(H(ii), 'FaceColor', cmap(ii,:), 'FaceAlpha', faceAlphaUse, ...
                'EdgeColor', [0.25 0.25 0.25], 'LineWidth', 1.25);
        end
    end
    axis equal off
    title('Overlap of (patient\_id, unit\_id)');
    legend(labels, 'Location','bestoutside');
end

% output
stats = struct('labels', labels, 'sizes', sizes, 'region_counts', regionTbl);
end
