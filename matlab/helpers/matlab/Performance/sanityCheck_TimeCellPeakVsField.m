function [peak_z, fieldbin_z, used_idx] = sanityCheck_TimeCellPeakVsField(nwbAll, all_units, neural_data_file, bin_size, useZscore)
% Forces the SAME pipeline as NWB_calcSelective_SB:
%   - 0.1 s bins (25 bins), GaussianKernal(3,1.5)
%   - per-trial smoothing -> z-score across bins (row-wise)
%   - mean across trials -> fr_per_bin_all_tc (mean z per bin)
%   - peak bin = argmax(fr_per_bin_all_tc)
%   - field bin = time_field (must equal peak if from the same run)
% Returns vectors so you can confirm equality numerically.

% --- Load annotations (must include .patient_id, .unit_id, .time_field, and a time-cell flag or p-val) ---
load(neural_data_file,'neural_data');

% Detector settings (HARD-CODED to match NWB_calcSelective_SB)
bin_width = 0.1;
total_bins = 25;
duration = total_bins * bin_width;   % 2.5 s
edges = 0:bin_width:duration;        % 26 edges
nBins = total_bins;
gaussian_kernel = GaussianKernal(3, 1.5);

% Decide which units are "time cells"
is_time_cell = false(numel(neural_data),1);
for k = 1:numel(neural_data)
    if isfield(neural_data,'is_time_cell') && ~isempty(neural_data(k).is_time_cell)
        is_time_cell(k) = logical(neural_data(k).is_time_cell);
    elseif isfield(neural_data,'time_cell_p') && ~isempty(neural_data(k).time_cell_p)
        is_time_cell(k) = neural_data(k).time_cell_p < 0.05;
    else
        is_time_cell(k) = isfield(neural_data,'time_field') && ~isempty(neural_data(k).time_field) ...
                          && neural_data(k).time_field>=1 && neural_data(k).time_field<=total_bins;
    end
end

% Keep only time cells that have a valid time_field index
has_tf = arrayfun(@(x) isfield(x,'time_field') && ~isempty(x.time_field) && x.time_field>=1 && x.time_field<=total_bins, neural_data);
used_idx = find(is_time_cell & has_tf);
if isempty(used_idx)
    warning('No time cells with valid time_field found.'); peak_z=[]; fieldbin_z=[]; return;
end

peak_z     = nan(numel(used_idx),1);
fieldbin_z = nan(numel(used_idx),1);

% Iterate units
for ii = 1:numel(used_idx)
    ndx = used_idx(ii);
    pid = neural_data(ndx).patient_id;
    uid = neural_data(ndx).unit_id;
    tf  = neural_data(ndx).time_field;  % 1..25 from detector

    m = ([all_units.subject_id]==pid) & ([all_units.unit_id]==uid);
    if ~any(m), warning('Unit %d/%d not found, skipping.', pid, uid); continue; end
    SU = all_units(m);

    tsMaint = nwbAll{SU.session_count}.intervals_trials ...
        .vectordata.get('timestamps_Maintenance').data.load();
    nT = numel(tsMaint);

    % Build trial x bin matrix: counts -> smooth -> row-wise z
    Z = zeros(nT, nBins);
    for t = 1:nT
        s = SU.spike_times;
        s = s(s>=tsMaint(t) & s<tsMaint(t)+duration) - tsMaint(t);
        counts = histcounts(s, edges);
        smooth_c = conv(counts, gaussian_kernel, 'same');
        Z(t,:) = zscore_rows(smooth_c);  % match detector
    end

    meanZ = mean(Z,1,'omitnan');

    % Detector's peak and field values on the SAME grid
    [~, peak_bin] = max(meanZ);
    peak_z(ii)    = meanZ(peak_bin);
    fieldbin_z(ii)= meanZ(tf);

    % Optional consistency check
    if peak_bin ~= tf
        fprintf('WARN: peak_bin (%d) != time_field (%d) for unit (%d,%d)\n', peak_bin, tf, pid, uid);
    end
end

% Plot paired distributions (they should be identical if tf==peak)
figure('Name','Sanity: Peak(mean z) vs Time-field(mean z)', 'Position',[120 120 420 520]);
pairedSwarmWithCenterLines([1 3], peak_z, fieldbin_z, ...
    'Time cells: Peak(mean z) vs Time-field(mean z)', 'Mean z-score (a.u.)', true);
set(gca,'XTickLabel',{'Peak','Time-field'});

% Report exact matches
eqMask = abs(peak_z - fieldbin_z) < 1e-9 | (isnan(peak_z) & isnan(fieldbin_z));
fprintf('Exact matches: %d / %d\n', sum(eqMask), numel(eqMask));

end

% ---- helpers ----
function Z = zscore_rows(X)
mu = mean(X,2,'omitnan');
sd = std(X,0,2,'omitnan');
sd(~isfinite(sd) | sd==0) = 1;
Z = (X - mu) ./ sd;
end

function pairedSwarmWithCenterLines(xcats, Y1, Y2, ttl, ylab, useZscore)
assert(isvector(Y1) && isvector(Y2) && numel(Y1)==numel(Y2));
Y1 = Y1(:); Y2 = Y2(:); n = numel(Y1);
xL = xcats(1); xR = xcats(2); xC = mean(xcats);
m1 = mean(Y1,'omitnan'); m2 = mean(Y2,'omitnan');
s1 = std(Y1,'omitnan');  s2 = std(Y2,'omitnan');
n1 = sum(isfinite(Y1));  n2 = sum(isfinite(Y2));
sem1 = s1/sqrt(max(1,n1)); sem2 = s2/sqrt(max(1,n2));
[~, p] = ttest(Y1, Y2); starStr = getStarString(p);
hold on; box on; set(gca,'FontSize',12,'XTick',xcats,'XTickLabel',{'A','B'});
xlabel(''); ylabel(ylab); title(ttl);
ymid = 0.5*(Y1+Y2);
for i=1:n, plot([xL xC xR],[Y1(i) ymid(i) Y2(i)],'-','Color',[0.75 0.75 0.75],'LineWidth',0.5,'HandleVisibility','off'); end
haveSwarm = exist('swarmchart','file')==2; msz=20;
if haveSwarm
    sc1=swarmchart(repmat(xL,n,1),Y1,msz,'filled'); sc1.MarkerFaceAlpha=0.85; sc1.XJitter='density'; sc1.XJitterWidth=0.18; sc1.MarkerEdgeColor='none'; sc1.MarkerFaceColor=[0.2 0.45 0.95];
    sc2=swarmchart(repmat(xR,n,1),Y2,msz,'filled'); sc2.MarkerFaceAlpha=0.85; sc2.XJitter='density'; sc2.XJitterWidth=0.18; sc2.MarkerEdgeColor='none'; sc2.MarkerFaceColor=[0.95 0.25 0.25];
else
    jit=0.12;
    scatter(xL+(rand(n,1)-0.5)*jit,Y1,msz,[0.2 0.45 0.95],'filled','MarkerFaceAlpha',0.85,'MarkerEdgeColor','none');
    scatter(xR+(rand(n,1)-0.5)*jit,Y2,msz,[0.95 0.25 0.25],'filled','MarkerFaceAlpha',0.85,'MarkerEdgeColor','none');
end
drawMeanTicks(xL,m1,sem1,0.18,2.5,1.2); drawMeanTicks(xR,m2,sem2,0.18,2.5,1.2);
yMax = max([Y1; Y2],[],'omitnan'); yMin = min([Y1; Y2],[],'omitnan'); pad = 0.06*max(eps,yMax-yMin);
ySig = max([m1+sem1,m2+sem2,yMax]) + pad; plot([xL xR],[ySig ySig],'k-','LineWidth',1.5);
text(xC,ySig,starStr,'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',14);
xlim([xL-0.6, xR+0.6]); ylim([yMin-pad, ySig+pad]);
if useZscore, yline(0,'--','Color',[0.4 0.4 0.4],'HandleVisibility','off'); end
end

function drawMeanTicks(x, m, sem, dx, lwMean, lwSem)
line([x-dx, x+dx],[m m],'Color','k','LineWidth',lwMean);
line([x-dx, x+dx],[m+sem m+sem],'Color','k','LineWidth',lwSem);
line([x-dx, x+dx],[m-sem m-sem],'Color','k','LineWidth',lwSem);
end

function starStr = getStarString(pVal)
if pVal < 1e-3, starStr = '***';
elseif pVal < 1e-2, starStr = '**';
elseif pVal < 0.05, starStr = '*';
else, starStr = 'n.s.';
end
end
