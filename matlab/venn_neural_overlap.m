function stats = venn_neural_overlap(matFiles, varargin)
% VENN_NEURAL_OVERLAP  Overlap + statistics for (patient_id, unit_id) across .mat files.
%
% stats = venn_neural_overlap(matFiles, 'Labels', labels, 'Universe', 902, ...
%                             'Test', 'enrichment', 'Plot', true, ...
%                             'Method', 'auto', 'VennFaceAlpha', 0.12, 'VennOutlineOnly', false)
%
% INPUT
%   matFiles : cellstr or string array of .mat paths; each must contain variable 'neural_data'
%              struct array with fields patient_id and unit_id.
%
% NAME-VALUE PAIRS
%   'Labels'          : cellstr/string names for each set (default = file basenames)
%   'Universe'        : scalar total number of eligible cells (e.g., 902). REQUIRED for valid enrichment p-values.
%                       If empty (default), uses U = union of all observed pairs (conditional universe).
%   'Test'            : 'enrichment' (default) or 'two-sided'
%                       - enrichment: tests overlap greater than chance (one-tailed)
%                       - two-sided : tests any association (two-tailed; conservative via hypergeom)
%   'Method'          : 'auto' (default), 'venn', or 'upset'
%   'Plot'            : logical (default true)
%   'VennFaceAlpha'   : scalar in [0,1] (default 0.12)
%   'VennOutlineOnly' : logical (default false)
%
% OUTPUT (stats)
%   .labels
%   .sizes
%   .pairwise          table with overlap, similarity metrics, p-values, OR, FDR
%   .triple            triple overlap counts (if N>=3)
%   .all_intersections containers.Map mask->count
%   .region_counts     region table (N<=3) or mask table (N>=4)
%   .universe_U        universe size used
%   .universe_note     description of universe choice
%
% Notes
%   - Effect sizes (Jaccard/Dice/OverlapCoeff) do NOT require Universe.
%   - Enrichment p-values DO require a valid Universe (e.g., 902 eligible cells).

% ---- parse inputs
if isstring(matFiles), matFiles = cellstr(matFiles); end
validateattributes(matFiles, {'cell'}, {'vector','nonempty'});

p = inputParser;
p.addParameter('Labels', {}, @(c) iscellstr(c) || isstring(c));
p.addParameter('Universe', [], @(u) isempty(u) || (isnumeric(u) && isscalar(u) && u>0));
p.addParameter('Test', 'enrichment', @(s) ischar(s) || isstring(s));
p.addParameter('Method', 'auto', @(s) ischar(s) || isstring(s));
p.addParameter('Plot', true, @(b) islogical(b) && isscalar(b));
p.addParameter('VennFaceAlpha', 0.12, @(x) isnumeric(x) && isscalar(x) && x>=0 && x<=1);
p.addParameter('VennOutlineOnly', false, @(b) islogical(b) && isscalar(b));
p.parse(varargin{:});

labels = p.Results.Labels;
U_in = p.Results.Universe;
testType = lower(string(p.Results.Test));
method = lower(string(p.Results.Method));
doPlot = p.Results.Plot;
vennFaceAlpha = p.Results.VennFaceAlpha;
vennOutlineOnly = p.Results.VennOutlineOnly;

% ---- default labels from basenames
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

% ---- load each file, extract unique (patient_id, unit_id) pairs
sets = cell(1,N);
for i = 1:N
    f = matFiles{i};
    if ~isfile(f), error('File not found: %s', f); end
    S = load(f);
    if ~isfield(S, 'neural_data')
        error('File %s does not contain variable ''neural_data''.', f);
    end
    nd = S.neural_data;
    if ~all(isfield(nd, {'patient_id','unit_id'}))
        error('In %s, neural_data must have fields patient_id and unit_id.', f);
    end

    pid = vertcat(nd.patient_id);
    uid = vertcat(nd.unit_id);
    pairs = [pid(:) uid(:)];
    sets{i} = unique(pairs, 'rows'); % [patient_id unit_id]
end

% ---- sizes
sizes = cellfun(@(A) size(A,1), sets);

% ---- helper for row-wise intersection
interRows = @(A,B) intersect(A,B,'rows','stable');

% ---- membership matrix over union of all observed pairs
allPairs = unique(vertcat(sets{:}), 'rows', 'stable');
M = false(size(allPairs,1), N);
for i = 1:N
    M(:,i) = ismember(allPairs, sets{i}, 'rows');
end

% count each unique membership pattern
[patterns, ~, idxPat] = unique(M, 'rows', 'stable');
countsPat = accumarray(idxPat, 1);
maskStr = arrayfun(@(r) char('0' + patterns(r,:)), (1:size(patterns,1))', 'uni', 0);
all_intersections = containers.Map(maskStr, num2cell(countsPat));

% ---- Universe size U for enrichment testing
if isempty(U_in)
    U = size(allPairs,1);
    universe_note = "Universe defaulted to union(all sets). Enrichment p-values are conditional on observed units.";
else
    U = double(U_in);
    universe_note = "Universe provided as scalar total eligible cells.";
end

% ---- Pairwise: overlap + similarity + enrichment test
pw_rows = {};
pvals = [];
for i = 1:N
    for j = i+1:N
        nAB = size(interRows(sets{i}, sets{j}), 1);
        nA  = sizes(i);
        nB  = sizes(j);
        nUnion = nA + nB - nAB;

        % similarity metrics (no universe needed)
        jaccard = nAB / max(nUnion, 1);
        dice    = (2*nAB) / max(nA + nB, 1);
        overlapCoeff = nAB / max(min(nA,nB), 1);

        % enrichment requires universe
        nNeither = U - nUnion;
        if nNeither < 0
            error('Universe U=%d is smaller than |A∪B|=%d for %s vs %s. Fix Universe.', ...
                  U, nUnion, labels{i}, labels{j});
        end

        % Prefer Fisher OR (if available) + hypergeom p (robust)
        % 2x2 table:
        % [both, B_only; A_only, neither]
        nA_only = nA - nAB;
        nB_only = nB - nAB;
        T22 = [nAB,    nB_only;
               nA_only,nNeither];

        odds_ratio = NaN;
        fisher_p = NaN;

        hasFishertest = exist('fishertest','file') == 2;
        if hasFishertest
            try
                tail = "both";
                if testType == "enrichment"
                    % one-tailed enrichment: more overlap than expected -> right tail
                    tail = "right";
                elseif testType == "two-sided"
                    tail = "both";
                end
                [~, fisher_p, stats_f] = fishertest(T22, 'Tail', tail);
                odds_ratio = stats_f.OddsRatio;
            catch
                % fall back below
            end
        end

        % Hypergeometric p-value (always available if Stats toolbox has hygecdf)
        % X ~ Hypergeom(U, nA successes, nB draws); observe k=nAB
        if exist('hygecdf','file') ~= 2
            error('hygecdf not found. Need Statistics and Machine Learning Toolbox for hypergeometric p-values.');
        end

        if testType == "enrichment"
            p_hyper = 1 - hygecdf(nAB-1, U, nA, nB);     % P(X >= k)
        elseif testType == "two-sided"
            p_lo = hygecdf(nAB,   U, nA, nB);           % P(X <= k)
            p_hi = 1 - hygecdf(nAB-1, U, nA, nB);       % P(X >= k)
            p_hyper = min(1, 2*min(p_lo, p_hi));         % conservative two-sided
        else
            error('Unknown Test: %s (use ''enrichment'' or ''two-sided'')', testType);
        end

        % choose p to report (prefer fisher if computed, else hyper)
        p_report = p_hyper;
        p_method = "hypergeom";
        if ~isnan(fisher_p)
            p_report = fisher_p;
            p_method = "fishertest";
        end

        pw_rows(end+1,:) = {labels{i}, labels{j}, nAB, nA, nB, nUnion, ...
                            jaccard, dice, overlapCoeff, ...
                            p_report, p_method, odds_ratio}; %#ok<AGROW>
        pvals(end+1,1) = p_report; %#ok<AGROW>
    end
end

pairwiseTbl = cell2table(pw_rows, 'VariableNames', ...
    {'A','B','Overlap','SizeA','SizeB','Union', ...
     'Jaccard','Dice','OverlapCoeff', ...
     'p_value','p_method','OddsRatio'});

% ---- FDR correction across pairwise tests
pairwiseTbl.FDR_BH = bh_fdr(pairwiseTbl.p_value);

% ---- Significance stars (based on FDR)
sig = repmat({''}, height(pairwiseTbl), 1);
for k = 1:height(pairwiseTbl)
    q = pairwiseTbl.FDR_BH(k);
    if q < 0.001, sig{k} = '***';
    elseif q < 0.01, sig{k} = '**';
    elseif q < 0.05, sig{k} = '*';
    else, sig{k} = 'ns';
    end
end
pairwiseTbl.Significance = sig;

% ---- Print summary
fprintf('\n[PAIRWISE OVERLAP]\n%s\n', universe_note);
fprintf('Test: %s | Universe U=%d\n', testType, U);
fprintf('%-20s %-20s %-7s %-8s %-8s %-8s %-10s %-10s %-10s %-10s\n', ...
    'Set A','Set B','nAB','Jacc','Dice','OvCoef','p','q(FDR)','OR','pMethod');
fprintf('%s\n', repmat('-', 120, 1));
for k = 1:height(pairwiseTbl)
    fprintf('%-20s %-20s %-7d %-8.3f %-8.3f %-8.3f %-10.3g %-10.3g %-10.3g %-10s %s\n', ...
        pairwiseTbl.A{k}, pairwiseTbl.B{k}, pairwiseTbl.Overlap(k), ...
        pairwiseTbl.Jaccard(k), pairwiseTbl.Dice(k), pairwiseTbl.OverlapCoeff(k), ...
        pairwiseTbl.p_value(k), pairwiseTbl.FDR_BH(k), pairwiseTbl.OddsRatio(k), ...
        pairwiseTbl.p_method(k), pairwiseTbl.Significance{k});
end

% ---- triple overlap counts (no inferential test here)
tripleTbl = table;
if N >= 3
    tr_rows = {};
    for i = 1:N
        for j = i+1:N
            for k = j+1:N
                Aij = interRows(sets{i}, sets{j});
                nABC = size(interRows(Aij, sets{k}), 1);
                tr_rows(end+1,:) = {labels{i}, labels{j}, labels{k}, nABC}; %#ok<AGROW>
            end
        end
    end
    if ~isempty(tr_rows)
        tripleTbl = cell2table(tr_rows, 'VariableNames', {'A','B','C','Overlap'});
    end
end

% ---- region-by-region counts (for N<=3) + generic mask table
regionTbl = table;
if N == 2
    n12   = size(interRows(sets{1}, sets{2}), 1);
    only1 = sizes(1) - n12;
    only2 = sizes(2) - n12;
    both  = n12;

    regionTbl = cell2table( ...
        {['only ' labels{1}], only1; ['only ' labels{2}], only2; [labels{1} '+' labels{2}], both}, ...
        'VariableNames', {'Region','Count'});

elseif N == 3
    n12  = size(interRows(sets{1}, sets{2}), 1);
    n13  = size(interRows(sets{1}, sets{3}), 1);
    n23  = size(interRows(sets{2}, sets{3}), 1);
    n123 = size(interRows(interRows(sets{1},sets{2}), sets{3}), 1);

    ab_only = n12 - n123;
    ac_only = n13 - n123;
    bc_only = n23 - n123;

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
else
    keys = all_intersections.keys;
    vals = cellfun(@(k) all_intersections(k), keys);
    [vals, ord] = sort(vals, 'descend');
    keys = keys(ord);
    maskToNames = @(mask) strjoin(labels(mask=='1'), '+');
    names = cellfun(maskToNames, keys, 'uni', 0);
    regionTbl = table(keys(:), string(names(:)), vals(:), ...
        'VariableNames', {'Mask','Sets','Count'});
end

% ---- decide plotting method
if method == "auto"
    useVenn = N <= 3;
elseif method == "venn"
    useVenn = true;
elseif method == "upset"
    useVenn = false;
else
    error('Unknown Method: %s', method);
end

if doPlot
    if useVenn
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

        if vennOutlineOnly
            faceAlphaUse = 0.0;
        else
            faceAlphaUse = vennFaceAlpha;
        end

        if exist('H','var') && ~isempty(H)
            cmap = lines(numel(H));
            for ii = 1:numel(H)
                try
                    set(H(ii), 'FaceColor', cmap(ii,:), ...
                               'FaceAlpha', faceAlphaUse, ...
                               'EdgeColor', [0.25 0.25 0.25], ...
                               'LineWidth', 1.25);
                catch
                end
            end
        end
        axis equal off
        title('Overlap of (patient\_id, unit\_id)');
        legend(labels, 'Location','bestoutside');
    end

    if ~useVenn
        T = array2table(M, 'VariableNames', matlab.lang.makeValidName(labels));
        figure('Name','Neural Unit Overlap (UpSet)');
        upsetplot(T);
        title('Overlap of (patient\_id, unit\_id)');
    end
end

% ---- package output
stats = struct();
stats.labels = labels;
stats.sizes = sizes;
stats.pairwise = pairwiseTbl;
stats.triple = tripleTbl;
stats.all_intersections = all_intersections;
stats.region_counts = regionTbl;
stats.universe_U = U;
stats.universe_note = universe_note;

end

% ---------- local: Benjamini-Hochberg FDR (no toolboxes needed)
function q = bh_fdr(p)
p = p(:);
m = numel(p);
[ps, ord] = sort(p, 'ascend');
qtemp = ps .* m ./ (1:m)';
% enforce monotonicity
for i = m-1:-1:1
    qtemp(i) = min(qtemp(i), qtemp(i+1));
end
q = nan(m,1);
q(ord) = min(qtemp, 1);
end
