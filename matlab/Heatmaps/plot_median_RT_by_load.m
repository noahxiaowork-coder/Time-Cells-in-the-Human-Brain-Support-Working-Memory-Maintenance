function plot_median_RT_by_load(filename)
% PLOT_MEDIAN_RT_BY_LOAD
% Load neural_data from .mat file and plot median RT by patient across load conditions.
%
% Input:
%   filename – Path to .mat file containing `neural_data` struct with fields:
%              • patient_id
%              • load (1, 2, 3)
%              • RT (reaction time)

% ---------- 1. Load Data ----------
data = load(filename, 'neural_data');
nd = data.neural_data;

% ---------- 2. Identify Unique Patients and Loads ----------
patients = unique([nd.patient_id]);
loads = [1, 2, 3];

% ---------- 3. Compute Median RTs ----------
medianRTs = nan(numel(patients), numel(loads));

for i = 1:numel(patients)
    for j = 1:numel(loads)
        rt = [nd([nd.patient_id] == patients(i) & [nd.trial_load] == loads(j)).RT];
        if ~isempty(rt)
            medianRTs(i, j) = median(rt);
        end
    end
end

% ---------- 4. Plot ----------
figure; hold on;
colors = lines(numel(loads));

% Plot each patient as a dot
for j = 1:numel(loads)
    scatter(j * ones(size(medianRTs(:, j))), medianRTs(:, j), 40, ...
            'MarkerFaceColor', colors(j,:), 'MarkerEdgeColor', 'k', ...
            'DisplayName', sprintf('Load %d', loads(j)));
end

% Overlay mean RT line across patients per load
meanRTs = nanmean(medianRTs, 1);
plot(1:3, meanRTs, 'k-', 'LineWidth', 2, 'DisplayName', 'Mean RT');

% ---------- 5. Aesthetics ----------
xlim([0.5, 3.5]);
xticks(1:3);
xticklabels({'Load 1', 'Load 2', 'Load 3'});
ylabel('Median RT (s)');
title('Median Reaction Time by Load Condition');
legend('show');
grid on;

end
