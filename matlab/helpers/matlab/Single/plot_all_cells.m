function plot_all_cells(nwbAll, all_units, neural_data, bin_size)
% Loop through neural_data and plot each unit using plot_single_cell.

for k = 1:numel(neural_data)
    pid = neural_data(k).patient_id;
    uid = neural_data(k).unit_id;

    
        plot_single_cell(nwbAll, all_units, neural_data, bin_size, pid, uid);
        drawnow;   % update the figure right away (optional)

end
end
