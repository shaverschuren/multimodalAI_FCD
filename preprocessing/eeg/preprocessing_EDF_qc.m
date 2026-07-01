%% --- Step 1: Add FieldTrip to MATLAB path ---

if ~contains(path, 'L:\her_knf_golf\Wetenschap\newtransport\Sjors\ext\MATLAB\fieldtrip')
    run('L:\her_knf_golf\Wetenschap\newtransport\Sjors\ext\MATLAB\startup_ft.m')
end

%% --- Step 2: Read the EDF file ---
% cfg = [];
% cfg.dataset = "L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\preprocessing\eeg\persyst\RESP0483\RESP0483.edf";
% raw = ft_preprocessing(cfg);

cfg = [];
cfg.dataset       = 'L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\raw\eeg\EDFdata\RESP0483\RESP0483.edf';
cfg.headerformat  = 'edf';
cfg.eventformat   = 'edf_annot';
path = cfg.dataset;

event = ft_read_event(path, 'headerformat', 'edf'); %, 'eventformat', 'edf_annot');

%% --- Step 3: Basic preprocessing ---
% cfg = [];
% cfg.demean     = 'yes';            % remove mean
% cfg.detrend    = 'yes';            % linear detrending
% cfg.bpfilter = 'yes';
% cfg.bpfreq   = [1 40];             % 1–40 Hz
% data = ft_preprocessing(cfg, raw);
% data = raw;

%% --- Step 4: Visual inspection / browsing ---
% cfg = [];
% cfg.viewmode = 'vertical';        % or 'vertical'
% ft_databrowser(cfg, data);
