%% NWB_SB_import_main
% A sample import script that computes the following:
% - Behavioral metrics
% - Spike sorting metrics
% - SU selectivity metrics (SB)
% - SU selectivity metrics (SC)
%
% SB refers to sternberg main task, and SC to sternberg screen task
% throughout.
%
%
% Michael Kyzar 6/30/23
 
clear; clc; close all
fs = filesep;
%% Parameters

% Operation Flags: Should either be  '1' (SCREENING), '2' (STERNBERG), or '3' (BOTH)
taskFlag = 2;

% subject IDs for dataset.
importRange = 1:21; % Full Dataset: Kyzar et al 2023
%importRange = 1; % Sternberg Examples
% importRange = [4 7 15 16 21]; % Screening Examples
% importRange = 6:19; % Dataset: Kaminski et al 2017 
% importRange = 1:20; % Dataset: Kaminski et al 2020

%% Initializing and pathing
paths.baseData = '/Users/xiangxuankong/Desktop/Human TC Support WM/000469'; % Dataset directory
paths.nwb_sb = paths.baseData; % Dandiset Directoryr
paths.nwb_sc = paths.baseData; % Dandiset Directory
paths.code = '/Users/xiangxuankong/Desktop/HHuman TC Support WM/workingmem-release-NWB-main/NWB_SB';
paths.matnwb = '/Users/xiangxuankong/Desktop/Human TC Support WM/matnwb-master';
% paths.figOut = ['/Users/xiangxuankong/Desktop/HMB496_WM/Figures' fs 'figures'];
% 

% Helpers
if(~isdeployed) 
  cd(fileparts(matlab.desktop.editor.getActiveFilename));
  addpath(genpath([pwd fs 'helpers'])) % Should be in same folder as active script. 
else
    error('Unexpected error.')
end

pathCell = struct2cell(paths);
for i = 1:length(pathCell)
    addpath(genpath(pathCell{i}))
end

% Initialize NWB Package
%generateCore() for first instantiation of matnwb API
fprintf('Checking generateCore() ... ')
if isfile([paths.matnwb fs '+types' fs '+core' fs 'NWBFile.m'])
     fprintf('generateCore() already initialized.\n') %only need to do once
else 
    cd(paths.matnwb)
    generateCore();
    fprintf('generateCore() initialized.\n')
end 

%% Importing Datasets From Folder
switch taskFlag
    case 1 % Screening
        internalFlag = taskFlag;
        [nwbAll_sc, importLog_sc] = NWB_importFromFolder(paths.nwb_sc, importRange,internalFlag);
    case 2 % Sternberg
        internalFlag = taskFlag;
        [nwbAll_sb, importLog_sb] = NWB_importFromFolder(paths.nwb_sb, importRange, internalFlag);    
    case 3 % Both
        internalFlag = 1;
        [nwbAll_sc, importLog_sc] = NWB_importFromFolder(paths.nwb_sc, importRange, internalFlag);
        internalFlag = 2;
        [nwbAll_sb, importLog_sb] = NWB_importFromFolder(paths.nwb_sb, importRange, internalFlag);
    otherwise
        error('Task flag not properly specified')
end

%% Extracting Single Units
load_all_waveforms = 1; % Extracts all by default. Set to '0' to only extract the mean waveform. 
switch taskFlag
    case 1 % Screening
        fprintf('Loading Screening\n')
        all_units_sc = NWB_SB_extractUnits(nwbAll_sc,load_all_waveforms);
    case 2 % Sternberg
        fprintf('Loading Sternberg\n')
        all_units_sb = NWB_SB_extractUnits(nwbAll_sb,load_all_waveforms);    
    case 3 % Both
        fprintf('Loading Screening\n')
        all_units_sc = NWB_SB_extractUnits(nwbAll_sc,load_all_waveforms);
        fprintf('Loading Sternberg\n')
        all_units_sb = NWB_SB_extractUnits(nwbAll_sb,load_all_waveforms);
    otherwise
        error('Task flag not properly specified')
end

%%
% plot_correct_incorrect_heatmaps(nwbAll_sb, all_units_sb, '3sig15_data.mat', 0.1)

% plot_concept_ratecurve(nwbAll_sb, all_units_sb, 'Nov27_Nov18.mat', 0.1)
% neural_data = create_concept_neural_data(nwbAll_sb, all_units_sb, paramsSB, 0.1, true);
%plot_LIS_integratedHM(nwbAll_sb, all_units_sb, 'Nov27_Nov18.mat', 0.1, 2)
%out = showPreferredTF_CorrVsIncorr_ByRegionIntegrated_Loads(nwbAll_sb, all_units_sb, 'Nov27_Nov18.mat', 0.1, true, [1]);

% Load neural_data first before calling plot_rt_heatmaps_subgroup
% data = load('3sig15_data.mat', 'neural_data');
% neural_data = data.neural_data;
% plot_rt_heatmaps_subgroup(neural_data, 0.1, true);
%plot_LIS_Load123_prefHeatmaps(nwbAll_sb, all_units_sb,'Nov27_Nov18.mat', 1)
% plot_LIS_Load123_prefHeatmaps(nwbAll_sb, all_units_sb,'Dec1_load2.mat', 2, false);
% plot_LIS_Load123_prefHeatmaps(nwbAll_sb, all_units_sb,'Dec1_Load3.mat', 3, false);

% showTimeCellAverages_ByRegionLoadIntegrated(nwbAll_sb, all_units_sb,'3sig15_data.mat', 0.1, true)

[neural_data, time_cell_info, unit_stats] = Find_cue_cells( ...
    nwbAll_sb, all_units_sb, 0.100, false, false);