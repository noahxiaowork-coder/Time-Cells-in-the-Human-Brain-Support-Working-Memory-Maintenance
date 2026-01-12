% % Data for each group (number of time cells)
% groupData = [0, 6, 11, 10, 31];
% 
% % Similarity matrix representing the number of overlapping time cells
% overlapMatrix = [0 0 0 0 0;
%                  0 0 1 0 2;
%                  0 0 0 1 4;
%                  0 0 0 0 4;
%                  0 0 0 0 0];
% 
% % Number of groups
% numGroups = length(groupData);
% 
% % Initialize figure for subplots
% figure;
% tiledlayout(2,3); % Adjust layout depending on the number of pies and visual spacing
% 
% for i = 1:numGroups
%     % Find overlaps with other groups for current group i
%     overlapsWithOthers = overlapMatrix(i, :);
% 
%     % Calculate the unique cells in this group
%     uniqueCells = groupData(i) - sum(overlapsWithOthers);
% 
%     % Define pie chart slices: [unique cells, overlap with group 2, overlap with group 3, etc.]
%     pieData = [uniqueCells, overlapsWithOthers];
% 
%     % Create a subplot for each group
%     nexttile;
%     pie(pieData);
% 
%     % Create label for each group, showing unique and overlapping cells
%     labels = [{'Unique cells'}, arrayfun(@(x) ['Overlap with Group ' num2str(x)], find(overlapsWithOthers), 'UniformOutput', false)];
%     labels = labels(~cellfun('isempty', labels)); % Remove empty labels
%     legend(labels, 'Location', 'bestoutside');
% 
%     % Add title for each group pie chart
%     title(['Group ' num2str(i)]);
% end
% 
% % General title for the figure
% sgtitle('Venn-like Pie Charts for Each Group');

% function plot_performance_heatmaps(neural_data_file, bin_width)
% %PLOT_LOAD_HEATMAPS Loads neural_data and generates four heatmaps:
% % 1) All trials
% % 2) Correct trials only
% % 3) Error trials only
% %
% % Each row (neuron) is min-max normalized to [0,1].
% % X-axis is shown in actual time (seconds), bin_width is an input argument.
% 
% % Load the neural_data structure
% load(neural_data_file, 'neural_data');
% 
% num_neurons = length(neural_data);
% if num_neurons == 0
%     error('No neurons found in neural_data.');
% end
% 
% % Determine number of bins and create time vectors
% num_bins = size(neural_data(1).firing_rates, 2);
% time_edges = (0:num_bins) * bin_width;
% time_centers = time_edges(1:end-1) + diff(time_edges)/2;
% 
% % Preallocate arrays
% all_psth = zeros(num_neurons, num_bins);
% correct_psth = zeros(num_neurons, num_bins);
% error_psth = zeros(num_neurons, num_bins);
% peak_bins = zeros(num_neurons, 1);
% 
% for i = 1:num_neurons
%     fr = neural_data(i).firing_rates;      % [trials x bins]
%     performance = neural_data(i).trial_correctness; % [trials x 1], values 1, 2, or 3
% 
%     % Separate trials by Performance
%     correct_trials = (performance == 1);
%     error_trials = (performance == 0);
% 
%     % Compute average firing rates
%     avg_fr_all = mean(fr, 1);
%     avg_fr_correct = mean(fr(correct_trials, :), 1, 'omitnan');
%     avg_fr_error = mean(fr(error_trials, :), 1, 'omitnan');
% 
%     % Find peak bin from all trials PSTH
%     [~, peak_bin] = max(avg_fr_all);
%     peak_bins(i) = peak_bin;
% 
%     all_psth(i, :) = avg_fr_all;
%     correct_psth(i, :) = avg_fr_correct;
%     error_psth(i, :) = avg_fr_error;
% end
% 
% % Sort neurons by their peak bin from all trials
% [~, sort_idx] = sort(peak_bins);
% sorted_all_psth = all_psth(sort_idx, :);
% sorted_correct_psth = correct_psth(sort_idx, :);
% sorted_error_psth = error_psth(sort_idx, :);
% 
% % Perform min-max normalization row-wise to ensure values are in [0,1].
% sorted_all_psth = rowwise_minmax_normalize(sorted_all_psth);
% sorted_correct_psth = rowwise_minmax_normalize(sorted_correct_psth);
% sorted_error_psth = rowwise_minmax_normalize(sorted_error_psth);
% 
% % Use parula colormap (blue to yellow)
% cmap = parula;
% 
% % Determine x-axis ticks and labels
% desired_tick_step = 0.5; % seconds between ticks
% max_time = time_centers(end);
% time_ticks = 0:desired_tick_step:max_time;
% xticks = arrayfun(@(t) find_closest_bin(time_centers, t), time_ticks);
% xticklabels = arrayfun(@(t) sprintf('%.1f', t), time_ticks, 'UniformOutput', false);
% 
% % Plot all trials heatmap
% figure;
% imagesc(sorted_all_psth, [0 1]); % Set color range to [0,1]
% colormap(cmap);
% colorbar;
% xlabel('Time (s)');
% ylabel('Neuron (sorted by peak)');
% title('Time Cells: All Trials');
% set(gca, 'TickDir', 'out');
% set(gca, 'XTick', xticks, 'XTickLabel', xticklabels);
% 
% % Plot Correct trials heatmap
% figure;
% imagesc(sorted_correct_psth, [0 1]);
% colormap(cmap);
% colorbar;
% xlabel('Time (s)');
% ylabel('Neuron (sorted by all-trials peak)');
% title('Time Cells: Correct Trials');
% set(gca, 'TickDir', 'out');
% set(gca, 'XTick', xticks, 'XTickLabel', xticklabels);
% 
% % Plot Error trials heatmap
% figure;
% imagesc(sorted_error_psth, [0 1]);
% colormap(cmap);
% colorbar;
% xlabel('Time (s)');
% ylabel('Neuron (sorted by all-trials peak)');
% title('Time Cells: Error  Trials');
% set(gca, 'TickDir', 'out');
% set(gca, 'XTick', xticks, 'XTickLabel', xticklabels);
% 
% end
% 
% function mat = rowwise_minmax_normalize(mat)
% %ROWWISE_MINMAX_NORMALIZE Normalize each row of the matrix using min-max scaling.
% for r = 1:size(mat,1)
%     row = mat(r,:);
%     if all(isnan(row))
%         % If the entire row is NaN, just leave it as NaN
%         continue;
%     end
%     min_val = min(row, [], 'omitnan');
%     max_val = max(row, [], 'omitnan');
%     if max_val > min_val
%         % Replace NaNs with min_val (or handle differently if desired)
%         row(isnan(row)) = min_val;
%         mat(r,:) = (row - min_val) / (max_val - min_val);
%     else
%         % All values are equal or NaN
%         mat(r,:) = 0.5; 
%     end
% end
% end
% 
% function idx = find_closest_bin(time_centers, target_time)
% %FIND_CLOSEST_BIN Finds the bin index in time_centers closest to target_time.
% [~, idx] = min(abs(time_centers - target_time));
% end

function plot_performance_heatmaps(neural_data_file, bin_width)
%PLOT_CORRECT_INCORRECT_HEATMAPS_DOWNSAMPLE Loads neural_data and generates four heatmaps:
% 1) All trials
% 2) Correct trials only
% 3) Incorrect trials only
% 4) Correct trials only (downsampled to match the number of incorrect trials)
%
% Each row (neuron) is min-max normalized to [0,1].
% X-axis is shown in actual time (seconds), bin_width is an input argument.

% Normalization Fixed?

% Load the neural_data structure
load(neural_data_file, 'neural_data'); 

num_neurons = length(neural_data);
if num_neurons == 0
    error('No neurons found in neural_data.');
end

% Determine number of bins and create time vectors
num_bins = size(neural_data(1).firing_rates, 2);
time_edges = (0:num_bins) * bin_width; 
time_centers = time_edges(1:end-1) + diff(time_edges)/2;

% Preallocate arrays
all_psth = zeros(num_neurons, num_bins);
correct_psth = zeros(num_neurons, num_bins);
incorrect_psth = zeros(num_neurons, num_bins);
correct_downsampled_psth = zeros(num_neurons, num_bins); % New array for downsampled correct PSTH
peak_bins = zeros(num_neurons, 1);

for i = 1:num_neurons
    fr = neural_data(i).firing_rates;      % [trials x bins]
    correctness = neural_data(i).trial_correctness; % [trials x 1], 1=correct,0=incorrect

    % Identify correct and incorrect trials
    correct_trials = find(correctness == 1);
    incorrect_trials = find(correctness == 0);

    avg_fr_all = mean(fr, 1);
    avg_fr_correct = mean(fr(correct_trials, :), 1, 'omitnan');
    avg_fr_incorrect = mean(fr(incorrect_trials, :), 1, 'omitnan');

    % Downsample correct trials to match number of incorrect trials
    num_correct = length(correct_trials);
    num_incorrect = length(incorrect_trials);
    if num_incorrect > 0 && num_correct >= num_incorrect
        % Downsample correct trials
        chosen_correct = randsample(correct_trials, num_incorrect);
        avg_fr_correct_downsampled = mean(fr(chosen_correct, :), 1);
    elseif num_incorrect > 0 && num_correct < num_incorrect
        % More incorrect than correct? Just use all correct trials.
        avg_fr_correct_downsampled = mean(fr(correct_trials, :), 1);
    else
        % If no incorrect trials, we can't downsample properly.
        avg_fr_correct_downsampled = nan(1, num_bins);
    end

    [~, peak_bin] = max(avg_fr_all);
    peak_bins(i) = peak_bin;

    all_psth(i, :) = avg_fr_all;
    correct_psth(i, :) = avg_fr_correct;
    incorrect_psth(i, :) = avg_fr_incorrect;
    correct_downsampled_psth(i, :) = avg_fr_correct_downsampled;
end

% Sort neurons by their peak bin from all trials
[~, sort_idx] = sort(peak_bins);
sorted_all_psth = all_psth(sort_idx, :);
sorted_correct_psth = correct_psth(sort_idx, :);
sorted_incorrect_psth = incorrect_psth(sort_idx, :);
sorted_correct_downsampled_psth = correct_downsampled_psth(sort_idx, :);

% Perform min-max normalization row-wise to ensure values are in [0,1].
sorted_all_psth = rowwise_minmax_normalize(sorted_all_psth);
sorted_correct_psth = rowwise_minmax_normalize(sorted_correct_psth);
sorted_incorrect_psth = rowwise_minmax_normalize(sorted_incorrect_psth);
sorted_correct_downsampled_psth = rowwise_minmax_normalize(sorted_correct_downsampled_psth);

% Use parula colormap (blue to yellow)
cmap = parula;

% Determine x-axis ticks and labels
desired_tick_step = 0.5; % seconds between ticks
max_time = time_centers(end); 
time_ticks = 0:desired_tick_step:max_time; 
xticks = arrayfun(@(t) find_closest_bin(time_centers, t), time_ticks);
xticklabels = arrayfun(@(t) sprintf('%.1f', t), time_ticks, 'UniformOutput', false);

% Plot all trials heatmap
figure;
imagesc(sorted_all_psth, [0 1]); % Set color range to [0,1]
colormap(cmap);
colorbar;
xlabel('Time (s)');
ylabel('Neuron (sorted by peak)');
title('Time Cells: All Trials');
set(gca, 'TickDir', 'out');
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels);

% Plot correct trials heatmap
figure;
imagesc(sorted_correct_psth, [0 1]);
colormap(cmap);
colorbar;
xlabel('Time (s)');
ylabel('Neuron (sorted by all-trials peak)');
title('Time Cells: Correct Trials Only (Full)');
set(gca, 'TickDir', 'out');
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels);

% Plot incorrect trials heatmap
figure;
imagesc(sorted_incorrect_psth, [0 1]);
colormap(cmap);
colorbar;
xlabel('Time (s)');
ylabel('Neuron (sorted by all-trials peak)');
title('Time Cells: Incorrect Trials Only');
set(gca, 'TickDir', 'out');
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels);

% Plot correct-downsampled trials heatmap
figure;
imagesc(sorted_correct_downsampled_psth, [0 1]);
colormap(cmap);
colorbar;
xlabel('Time (s)');
ylabel('Neuron (sorted by all-trials peak)');
title('Time Cells: Correct Trials Downsampled');
set(gca, 'TickDir', 'out');
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels);

end

function mat = rowwise_minmax_normalize(mat)
%ROWWISE_MINMAX_NORMALIZE Normalize each row of the matrix using min-max scaling.
for r = 1:size(mat,1)
    row = mat(r,:);
    min_val = min(row);
    max_val = max(row);
    if all(isnan(row))
        % If the row is all NaNs (e.g., no trials), just leave it as NaNs
        continue;
    elseif max_val > min_val
        mat(r,:) = (row - min_val) / (max_val - min_val);
    else
        % If all values are equal or valid but constant, set them to 0.5
        mat(r,:) = 0.5; 
    end
end
end

function idx = find_closest_bin(time_centers, target_time)
%FIND_CLOSEST_BIN Finds the bin index in time_centers closest to target_time.
[~, idx] = min(abs(time_centers - target_time));
end

function mat = global_minmax_normalize(mat)
%GLOBAL_MINMAX_NORMALIZE Normalize the entire matrix using the global min and max values.
global_min = min(mat(:), [], 'omitnan'); % Minimum across all elements
global_max = max(mat(:), [], 'omitnan'); % Maximum across all elements

if global_max > global_min
    mat = (mat - global_min) / (global_max - global_min); % Scale to [0, 1]
else
    % If all values are equal, set all to 0.5
    mat(:) = 0.5;
end
end
