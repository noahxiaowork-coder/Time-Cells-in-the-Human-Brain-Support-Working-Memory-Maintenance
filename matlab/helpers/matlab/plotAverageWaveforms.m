function plotAverageWaveforms(all_units)
    
TimeCellInformation = {
    'PatientID', 'UnitID', 'TimeField';
    1, 1, 3;
    1, 10, 1;
    1, 13, 1;
    1, 14, 2;
    1, 16, 5;
    1, 17, 2;
    1, 20, 2;
    1, 21, 2;
    1, 23, 2;
    1, 24, 3;
    1, 25, 2;
    1, 27, 2;
    1, 28, 1;
    1, 32, 5;
    1, 40, 2;
    2, 4, 1;
    2, 13, 2;
    2, 23, 2;
    2, 24, 5;
    2, 25, 4;
    2, 26, 5;
    3, 1, 2;
    3, 3, 5;
    3, 15, 2;
    3, 17, 5;
    3, 18, 1;
    3, 23, 5;
    3, 24, 5;
    3, 28, 2;
    4, 1, 3;
    4, 3, 1;
    4, 4, 1;
    4, 5, 1;
    4, 7, 1;
    4, 8, 1;
    4, 33, 5;
    4, 58, 5;
    4, 72, 2;
    5, 19, 5;
    5, 43, 2;
    5, 49, 1;
    6, 8, 4;
    6, 30, 4;
    7, 20, 5;
    8, 2, 5;
    8, 3, 1;
    8, 5, 1;
    8, 15, 2;
    8, 26, 5;
    8, 28, 4;
    8, 29, 4;
    8, 32, 1;
    8, 35, 4;
    8, 38, 3;
    8, 39, 3;
    8, 42, 2;
    8, 43, 3;
    8, 46, 3;
    8, 48, 3;
    8, 49, 3;
    8, 52, 2;
    8, 53, 3;
    8, 55, 3;
    8, 56, 4;
    8, 58, 3;
    8, 59, 3;
    8, 60, 3;
    8, 61, 1;
    8, 62, 5;
    8, 64, 4;
    8, 65, 2;
    8, 66, 2;
    8, 69, 5;
    8, 72, 5;
    8, 73, 5;
    9, 12, 3;
    9, 24, 1;
    9, 31, 1;
    10, 10, 1;
    10, 12, 1;
    10, 13, 2;
    10, 14, 2;
    10, 16, 2;
    10, 18, 2;
    10, 19, 2;
    10, 20, 2;
    10, 21, 2;
    10, 23, 2;
    10, 25, 2;
    10, 26, 2;
    11, 6, 4;
    11, 8, 2;
    11, 11, 2;
    11, 20, 2;
    11, 36, 3;
    11, 57, 1;
    11, 67, 5;
    11, 73, 1;
    13, 1, 1;
    13, 4, 1;
    13, 9, 4;
    13, 16, 2;
    13, 19, 1;
    13, 25, 1;
    13, 34, 1;
    13, 49, 5;
    13, 51, 4;
    13, 53, 1;
    14, 28, 4;
    14, 35, 5;
    14, 50, 5;
    14, 54, 4;
    14, 62, 5;
    14, 64, 2;
    15, 1, 1;
    15, 7, 5;
    15, 8, 5;
    15, 14, 2;
    15, 53, 1;
    16, 5, 2;
    16, 6, 2;
    16, 7, 2;
    16, 12, 1;
    16, 17, 1;
    16, 19, 2;
    16, 25, 5;
    16, 28, 2;
    16, 30, 2;
    17, 1, 3;
    17, 13, 5;
    18, 2, 2;
    18, 70, 1;
    18, 73, 1;
    18, 84, 5;
    18, 88, 2;
    18, 102, 4;
    21, 15, 2
};

    % This function plots detailed average spike waveforms for specified units
    % along with their PDF estimates if multiple waveforms exist.

    % Extract the patient and unit IDs from the TimeCellInformation
    patientID = TimeCellInformation{2:end, 1}; % Assuming all are from the same patient in this example
    unitIDs = cell2mat(TimeCellInformation(2:end, 2));

    % Prepare the figure for plotting
    figure;
    hold on;

    % Loop through the list of specified units
    for idx = 1:length(unitIDs)
        % Find the unit in the all_units structure
        unit = all_units([all_units.unit_id] == unitIDs(idx) & [all_units.subject_id] == patientID(idx));
        if isempty(unit)
            warning('Unit %d for Patient %d not found.', unitIDs(idx), patientID(idx));
            continue;
        end

        % Get the waveforms
        waveforms = unit.waveforms;
        t = linspace(0, length(waveforms(1,:)), length(waveforms(1,:))); % Time vector

        % Plot the waveform or its PDF
        if size(waveforms, 1) > 1
            % Calculate the PDF of the waveforms if multiple waveforms are available
            edges = min(waveforms(:)):(max(waveforms(:))-min(waveforms(:)))/100:max(waveforms(:));
            [pdfY, pdfX] = hist(waveforms(:), edges);
            pdfY = pdfY / trapz(pdfX, pdfY); % Normalize to form a probability density
            plot(pdfX, pdfY, 'DisplayName', sprintf('Patient %d Unit %d PDF', patientID(idx), unitIDs(idx)));
        else
            % Plot the single available waveform
            plot(t, waveforms, 'DisplayName', sprintf('Patient %d Unit %d', patientID(idx), unitIDs(idx)));
        end
    end

    hold off;
    xlabel('Time (ms)');
    ylabel('\muV');
    title('Detailed Spike Waveforms with PDF Estimation');
    legend show;
end
