function [summaryTbl, nullCountsMat, regionList, nullTotals, nullTimes, nullPerIterTbl] = ...
    run_timecell_nulls(nwbAll, all_units, params, bin_width_analysis, preprocess, Nnull, outCSV)

if nargin < 6 || isempty(Nnull), Nnull = 500; end
if nargin < 7, outCSV = ''; end

paramsObs = params; paramsObs.rngSeed = 42;
[neural_data_obs, ~] = LIS_from_old_Bonf_nov18( ...
    nwbAll, all_units, paramsObs, bin_width_analysis, preprocess, false);

obsRegions = arrayfun(@(nd) normalize_region(nd.brain_region), neural_data_obs, 'UniformOutput', false);
[regionList, ~, idxObs] = unique(obsRegions);
obsCounts = accumarray(idxObs, 1, [numel(regionList), 1]);

if isempty(regionList)
    regionList = {'unknown'};
    obsCounts  = 0;
end

nullCountsMat = zeros(numel(regionList), Nnull);  % rows=region, cols=iteration
nullTotals    = zeros(Nnull,1);
nullTimes     = zeros(Nnull,1);

fprintf('Observed run: %d time cells detected.\n', numel(neural_data_obs));

% ===== 2) NULL RUNS (NullMode = true) =====
for ii = 1:Nnull
    paramsNull = params; paramsNull.rngSeed = 42 + ii;

    tStart = tic;
    [neural_data_null, ~] = LIS_from_old_Bonf_nov18( ...
        nwbAll, all_units, paramsNull, bin_width_analysis, preprocess, true);
    nullTimes(ii) = toc(tStart);
    nullTotals(ii) = numel(neural_data_null);

    if ~isempty(neural_data_null)
        nullRegs = arrayfun(@(nd) normalize_region(nd.brain_region), neural_data_null, 'UniformOutput', false);
        [uNull, ~, idxNull] = unique(nullRegs);
        countsNull = accumarray(idxNull, 1, [numel(uNull), 1]);

        % Ensure master region list and matrices include any new regions
        [regionList, nullCountsMat, obsCounts] = ensureRegionRows(regionList, nullCountsMat, obsCounts, uNull);

      % Map into rows (faster than containers.Map)
        [tf, loc] = ismember(uNull, regionList);
        nullCountsMat(loc(tf), ii) = countsNull(tf);

    end

    fprintf('Null %3d/%3d: detected %4d cells in %.2f s\n', ii, Nnull, nullTotals(ii), nullTimes(ii));
end

% ===== 3) EMPIRICAL p-values per region =====
geqCounts = sum(nullCountsMat >= obsCounts, 2);
p_emp     = (geqCounts + 1) ./ (Nnull + 1);       % one-sided enrichment
nullMean  = mean(nullCountsMat, 2);
nullSD    = std(nullCountsMat, 0, 2);
z_like    = zeros(size(nullSD)); nz = nullSD > 0;
z_like(nz)= (obsCounts(nz) - nullMean(nz)) ./ nullSD(nz);

summaryTbl = table(regionList(:), obsCounts, nullMean, nullSD, p_emp, z_like, ...
    'VariableNames', {'Region','Observed','NullMean','NullSD','p_emp','z_like'});
summaryTbl = sortrows(summaryTbl, 'p_emp');

% ===== 4) Long-format per-iteration-by-region table (optional export) ===
% Builds a table with columns: Iteration, Region, NullCount

nR = numel(regionList);

% Force a column string array for Region names (avoids char-array pitfalls)
regionList = string(regionList(:));

% Column vectors aligned in column-major order:
% Iteration: 1 repeated for all regions, then 2, ..., up to Nnull
ItCol   = repelem((1:Nnull).', nR, 1);

% Region: full region list repeated for each iteration
RegCol  = repmat(regionList, Nnull, 1);

% NullCount: column-major vectorization of the matrix (rows=regions, cols=iterations)
NullCol = nullCountsMat(:);

% All three are Kx1 where K = nR * Nnull
nullPerIterTbl = table(ItCol, RegCol, NullCol, ...
    'VariableNames', {'Iteration','Region','NullCount'});

% Optional CSV
if ~isempty(outCSV)
   
        writetable(summaryTbl, outCSV, 'WriteMode','overwrite');
        [p,f,e] = fileparts(outCSV);
        writetable(nullPerIterTbl, fullfile(p, [f '_perIter' e]), 'WriteMode','overwrite');
        fprintf('Saved: %s and %s\n', outCSV, fullfile(p, [f '_perIter' e]));
end

end % ================= end wrapper =================


function [regionList, nullCountsMat, obsCounts] = ensureRegionRows(regionList, nullCountsMat, obsCounts, newRegions)
% Ensure regionList (and aligned matrices) include any regions in newRegions
% Normalize types and orientations to column cell arrays of char.

    % 1) Coerce to cell array of char
    if isstring(regionList), regionList = cellstr(regionList); end
    if ischar(regionList),   regionList = cellstr(regionList); end
    if isstring(newRegions), newRegions = cellstr(newRegions); end
    if ischar(newRegions),   newRegions = cellstr(newRegions); end

    % 2) Force column orientation
    regionList = regionList(:);
    newRegions = newRegions(:);

    % 3) Compute additions (also column)
    newOnes = setdiff(newRegions, regionList);
    newOnes = newOnes(:);

    if isempty(newOnes), return; end

    % 4) Safe vertical concat (both columns now)
    regionList = [regionList; newOnes];

    % 5) Expand aligned matrices with zero rows
    nExtra = numel(newOnes);
    if isempty(nullCountsMat)
        nullCountsMat = zeros(nExtra, 0);
    else
        nullCountsMat(end+nExtra, size(nullCountsMat,2)) = 0; %#ok<AGROW>
    end
    if isempty(obsCounts)
        obsCounts = zeros(nExtra, 1);
    else
        obsCounts(end+nExtra, 1) = 0; %#ok<AGROW>
    end
end


function nameOut = normalize_region(nameIn)
% Remove _left/_right suffix; fall back to 'unknown'
    if (isstring(nameIn) || ischar(nameIn)), s = char(nameIn); else, s = 'unknown'; end
    s = strtrim(s); if isempty(s), s = 'unknown'; end
    nameOut = regexprep(s, '_(left|right)$', '', 'ignorecase');
end
