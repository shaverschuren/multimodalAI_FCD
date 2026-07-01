function conversion_table = convert_trc_to_edf_raw(input_path, output_path, varargin)
%CONVERT_TRC_TO_EDF_RAW Convert Micromed .TRC EEG files to raw-preserving EDF.
%
%   conversion_table = convert_trc_to_edf_raw(input_path, output_path)
%
%   This is a minimal TRC -> EDF converter intended for a raw
%   BIDS conversion workflow. One input .trc file becomes one output .edf
%   file. The converter DOES NOT concatenate recordings, resample, filter,
%   rereference, remove bad channels, or interpolate channels.
%   Assumes that the parent folder of each TRC file is the patient id.
%
%   Required:
%     input_path   Path to one .trc file, or to a directory containing .trc files.
%     output_path  Output .edf file if input_path is a file, or output directory
%                  if input_path is a directory.
%
%   Name-value options:
%     'fieldtrip_startup'       Path to FieldTrip startup script. Default: ''.
%     'normalize_labels'        Rename labels to FieldTrip standard_1005 case
%                               when possible. Original labels are logged.
%                               Default: true.
%     'record_duration_sec'     EDF data-record duration in seconds. Default: 10.
%     'pad_final_record'        Pad final incomplete EDF record with zeros rather
%                               than trimming data. Padding is logged. Default: true.
%     'overwrite'               Overwrite existing EDF files. Default: false.
%     'recursive'               Search input directory recursively. Default: true.
%     'log_dir'                 Directory for conversion logs. Default:
%                               <output_dir>/conversion_logs.
%
%   Example, one file:
%     convert_trc_to_edf_raw('C:\data\sub-001\rec01.TRC', ...
%                            'C:\out\sub-001_run-01_eeg.edf', ...
%                            'fieldtrip_startup', 'L:\path\startup_ft.m');
%
%   Example, folder:
%     convert_trc_to_edf_raw('C:\data\sub-001', 'C:\out\sub-001', ...
%                            'recursive', false);
%
%   Notes:
%     - EDF is written in physical units using MATLAB's edfwrite.
%     - For BIDS, create the final BIDS sidecars separately from the output EDFs.
%       The TSV written here is only a conversion/provenance log.
%

% ------------------------------
% Parse inputs
% ------------------------------
p = inputParser;
p.addRequired('input_path', @(x) ischar(x) || isstring(x));
p.addRequired('output_path', @(x) ischar(x) || isstring(x));
p.addParameter('fieldtrip_startup', 'L:\her_knf_golf\Wetenschap\newtransport\Sjors\ext\MATLAB\startup_ft.m', @(x) ischar(x) || isstring(x));
p.addParameter('normalize_labels', true, @(x) islogical(x) && isscalar(x));
p.addParameter('record_duration_sec', 10, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('pad_final_record', true, @(x) islogical(x) && isscalar(x));
p.addParameter('overwrite', false, @(x) islogical(x) && isscalar(x));
p.addParameter('recursive', false, @(x) islogical(x) && isscalar(x));
p.addParameter('log_dir', '', @(x) ischar(x) || isstring(x));
p.parse(input_path, output_path, varargin{:});
opt = p.Results;

input_path = char(input_path);
output_path = char(output_path);

% ------------------------------
% Load FieldTrip if requested
% ------------------------------
if strlength(string(opt.fieldtrip_startup)) > 0
    ft_startup = char(opt.fieldtrip_startup);
    if exist(ft_startup, 'file') ~= 2
        error('FieldTrip startup script not found: %s', ft_startup);
    end
    fprintf('Loading FieldTrip startup script: %s\n', ft_startup);
    run(ft_startup);
end

required_ft = {'ft_read_header', 'ft_preprocessing'};
for i = 1:numel(required_ft)
    if exist(required_ft{i}, 'file') ~= 2
        error(['Required FieldTrip function not found on path: %s. ', ...
               'Add FieldTrip to the MATLAB path or pass ''fieldtrip_startup''.'], required_ft{i});
    end
end

if exist('edfwrite', 'file') ~= 2 || exist('edfheader', 'file') ~= 2
    error('MATLAB edfwrite/edfheader not found. This script requires MATLAB EDF writing support.');
end

% ------------------------------
% Resolve input/output files
% ------------------------------
[input_files, output_files, output_root] = resolve_io_paths(input_path, output_path, opt.recursive);

if isempty(input_files)
    error('No .TRC/.trc files found at input_path: %s', input_path);
end

if strlength(string(opt.log_dir)) > 0
    log_dir = char(opt.log_dir);
else
    log_dir = fullfile(output_root, 'conversion_logs');
end
if ~exist(log_dir, 'dir')
    mkdir(log_dir);
end

log_txt = fullfile(log_dir, 'trc2edf_raw_conversion_log.txt');
diary(log_txt);
cleanup_diary = onCleanup(@() diary('off'));

fprintf('============= Start raw .TRC -> .EDF conversion =============\n');
fprintf('Input path:  %s\n', input_path);
fprintf('Output path: %s\n', output_path);
fprintf('Log dir:     %s\n', log_dir);
fprintf('Files:       %d\n', numel(input_files));
fprintf('==============================================================\n\n');

% Load standard electrode configuration only when needed.
elec1005 = [];
if opt.normalize_labels
    try
        elec1005 = ft_read_sens('standard_1005.elc');
    catch ME
        warning('Could not load standard_1005.elc. Channel labels will not be normalized: %s', ME.message);
        opt.normalize_labels = false;
    end
end

% ------------------------------
% Convert files
% ------------------------------
rows = repmat(empty_conversion_row(), numel(input_files), 1);

for iFile = 1:numel(input_files)
    in_file = input_files{iFile};
    out_file = output_files{iFile};

    fprintf('\n----- File %d/%d -----\n', iFile, numel(input_files));
    fprintf('Input:  %s\n', in_file);
    fprintf('Output: %s\n', out_file);

    out_dir = fileparts(out_file);
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    if exist(out_file, 'file') && ~opt.overwrite
        warning('Output exists and overwrite=false. Skipping: %s', out_file);
        rows(iFile).input_file = string(in_file);
        rows(iFile).output_file = string(out_file);
        rows(iFile).status = "skipped_exists";
        continue;
    end

    try
        result = convert_one_trc_to_edf(in_file, out_file, elec1005, opt);
        rows(iFile) = result;
        fprintf('Done: %s\n', out_file);
    catch ME
        warning('Failed to convert %s: %s', in_file, ME.message);
        rows(iFile).input_file = string(in_file);
        rows(iFile).output_file = string(out_file);
        rows(iFile).status = "failed";
        rows(iFile).error_message = string(ME.message);
    end
end

conversion_table = struct2table(rows);

% Write TSV conversion log.
tsv_file = fullfile(log_dir, 'trc2edf_raw_conversion.tsv');
fprintf('\nWriting conversion TSV: %s\n', tsv_file);
writetable(conversion_table, tsv_file, 'FileType', 'text', 'Delimiter', '\t');

fprintf('\n================ End raw .TRC -> .EDF conversion ================\n');
end

% ========================================================================
% Conversion of one file
% ========================================================================
function row = convert_one_trc_to_edf(in_file, out_file, elec1005, opt)
row = empty_conversion_row();
row.input_file = string(in_file);
row.output_file = string(out_file);
row.status = "started";

% Header and events. Event reading is allowed to fail.
hdr = ft_read_header(in_file);
evt = try_read_events(in_file);

% Load only EEG channels. This mirrors the original TRC path but avoids all
% downstream preprocessing. Data are kept at native sampling frequency.
cfg = [];
cfg.dataset = in_file;
cfg.channel = 'all';
raw = ft_preprocessing(cfg);

if ~isfield(raw, 'trial') || isempty(raw.trial)
    error('FieldTrip returned no data trials.');
end

% Infer BIDS-like channel types and units
[chantype, chanunit] = infer_chantype_chanunit(raw.label);

hdr.chantype = chantype(:);
hdr.chanunit = chanunit(:);

% Some files may contain multiple FieldTrip trials/segments. For a single
% source file, append these into one continuous EDF timeline. This is not
% concatenation across source recordings; it only linearizes segments inside
% one TRC file for EDF writing.
internal_segments = numel(raw.trial);
if internal_segments > 1
    fprintf('Internal FieldTrip segments detected in one TRC: %d. Appending within file.\n', internal_segments);
    cfg = [];
    raw = ft_appenddata(cfg, raw);
end

fs = double(raw.fsample);
if isempty(fs) || ~isfinite(fs) || fs <= 0
    if isfield(hdr, 'Fs') && ~isempty(hdr.Fs)
        fs = double(hdr.Fs);
    else
        error('Could not determine sampling frequency.');
    end
end

orig_labels = raw.label(:);
labels = orig_labels;
label_normalization_applied = false;
unmatched_labels = {};

if opt.normalize_labels
    [labels, unmatched_labels] = normalize_eeg_labels(orig_labels, elec1005);
    raw.label = labels;
    label_normalization_applied = true;
end

% Data matrix: channels x samples. Do not resample/filter/reref/repair.
data = raw.trial{1};
if ~isa(data, 'single')
    data = single(data);
end
[nCh, nSamplesOriginal] = size(data);

% EDF data-record handling. Prefer not to lose data. If the sample count is
% not divisible by samplesPerRecord, zero-pad only the final record and log it.
recordDuration = double(opt.record_duration_sec);
samplesPerRecord = fs * recordDuration;
if abs(samplesPerRecord - round(samplesPerRecord)) > 1e-9
    error(['record_duration_sec * sampling_frequency must be an integer for EDF writing. ', ...
           'Got %g * %g = %g. Choose a different record_duration_sec.'], recordDuration, fs, samplesPerRecord);
end
samplesPerRecord = round(samplesPerRecord);

remainder = mod(nSamplesOriginal, samplesPerRecord);
paddingSamples = 0;
if remainder ~= 0
    if ~opt.pad_final_record
        error(['Data length (%d samples) is not divisible by EDF samplesPerRecord (%d), ', ...
               'and pad_final_record=false.'], nSamplesOriginal, samplesPerRecord);
    end
    paddingSamples = samplesPerRecord - remainder;
    fprintf('Padding final EDF record with %d zero samples (%.6f sec).\n', paddingSamples, paddingSamples / fs);
    data(:, end+1:end+paddingSamples) = 0;
end

nSamplesWritten = size(data, 2);
numRecords = nSamplesWritten / samplesPerRecord;

% Start date/time from TRC header when possible.
[dtStart, hasTimeOfDay] = trc_get_start_datetime(hdr);
[startDateEDF, startTimeEDF] = datetime_to_edf_strings(dtStart);

% Annotation timetable: source-file marker, internal segments, events if available,
% and final-record padding if present.
annotations = build_annotation_timetable(evt, fs, nSamplesOriginal, paddingSamples, in_file, internal_segments);

% Physical min/max. Avoid zero-width ranges.
physMin = floor(double(min(data, [], 2))).' - 1;
physMax = ceil(double(max(data, [], 2))).' + 1;
flat = physMin >= physMax;
physMin(flat) = physMin(flat) - 1;
physMax(flat) = physMax(flat) + 1;

% Get patient ID
[folder_path, ~, ~] = fileparts(in_file);
[~, patient_folder] = fileparts(folder_path);

% Construct EDF header.
edf_hdr = edfheader('EDF');
edf_hdr.Patient = sanitize_edf_text(patient_folder);
edf_hdr.Recording = sanitize_edf_text(sprintf('Startdate %s', datestr(dtStart, 'dd-mmm-yyyy')));
edf_hdr.StartDate = startDateEDF;
edf_hdr.StartTime = startTimeEDF;
edf_hdr.Reserved = '';
edf_hdr.NumSignals = nCh;
edf_hdr.SignalLabels = string(labels(:)).';
edf_hdr.DataRecordDuration = seconds(recordDuration);
edf_hdr.NumDataRecords = numRecords;
edf_hdr.PhysicalMin = physMin;
edf_hdr.PhysicalMax = physMax;
edf_hdr.DigitalMin = repmat(-32768, 1, nCh);
edf_hdr.DigitalMax = repmat(32767, 1, nCh);
edf_hdr.PhysicalDimensions = hdr.chanunit';
% Set edf-specific channel types
for i = 1:numel(raw.label)
    switch hdr.chantype{i}
        case 'EEG'
            edf_hdr.TransducerTypes{i} = 'EEG electrode';
        case 'ECG'
            edf_hdr.TransducerTypes{i} = 'ECG electrode';
        case 'EMG'
            edf_hdr.TransducerTypes{i} = 'EMG electrode';
        case 'EOG'
            edf_hdr.TransducerTypes{i} = 'EOG electrode';
        case 'TRIG'
            edf_hdr.TransducerTypes{i} = 'Trigger/marker channel';
        case 'RESP'
            edf_hdr.TransducerTypes{i} = 'Respiration channel';
        otherwise
            edf_hdr.TransducerTypes{i} = 'Auxiliary channel';
    end
end

% Write EDF. MATLAB expects samples x signals for physical input.
fprintf('Writing EDF with %d channels, %d original samples, %d written samples, Fs=%g Hz.\n', ...
        nCh, nSamplesOriginal, nSamplesWritten, fs);
% edfwrite(out_file, edf_hdr, data.', annotations, 'InputSampleType', 'physical');
edfwrite(out_file, edf_hdr, data.', 'InputSampleType', 'physical');

% Populate conversion row.
row.status = "ok";
row.error_message = "";
row.input_file = string(in_file);
row.output_file = string(out_file);
row.start_date = string(datestr(dtStart, 'yyyy-mm-dd'));
if hasTimeOfDay
    row.start_time = string(datestr(dtStart, 'HH:MM:SS'));
else
    row.start_time = "";
end
row.has_time_of_day = hasTimeOfDay;
row.f_sample = fs;
row.n_channels = nCh;
row.n_samples_original = nSamplesOriginal;
row.n_samples_written = nSamplesWritten;
row.padding_samples = paddingSamples;
row.duration_original_sec = nSamplesOriginal / fs;
row.duration_written_sec = nSamplesWritten / fs;
row.record_duration_sec = recordDuration;
row.internal_segments = internal_segments;
row.n_events_read = numel(evt);
row.label_normalization_applied = label_normalization_applied;
row.original_labels = string(strjoin(orig_labels(:).', ','));
row.output_labels = string(strjoin(labels(:).', ','));
row.unmatched_labels = string(strjoin(unmatched_labels, ','));
end

% ========================================================================
% I/O helpers
% ========================================================================
function [input_files, output_files, output_root] = resolve_io_paths(input_path, output_path, recursive)
if exist(input_path, 'file') == 2
    input_files = {input_path};
    [out_dir, ~, ext] = fileparts(output_path);
    if isempty(ext)
        [~, base] = fileparts(input_path);
        if ~exist(output_path, 'dir')
            mkdir(output_path);
        end
        output_files = {fullfile(output_path, [base '.edf'])};
        output_root = output_path;
    else
        output_files = {output_path};
        if isempty(out_dir)
            output_root = pwd;
        else
            output_root = out_dir;
        end
    end
elseif exist(input_path, 'dir') == 7
    if recursive
        listing = [dir(fullfile(input_path, '**', '*.TRC')); dir(fullfile(input_path, '**', '*.trc'))];
    else
        listing = [dir(fullfile(input_path, '*.TRC')); dir(fullfile(input_path, '*.trc'))];
    end
    listing = unique_dir_entries(listing);

    input_files = cell(numel(listing), 1);
    output_files = cell(numel(listing), 1);
    for i = 1:numel(listing)
        in_file = fullfile(listing(i).folder, listing(i).name);
        input_files{i} = in_file;
        [~, base] = fileparts(in_file);

        if recursive
            rel_folder = erase(string(listing(i).folder), string(input_path));
            rel_folder = regexprep(char(rel_folder), ['^' regexptranslate('escape', filesep)], '');
            out_dir = fullfile(output_path, rel_folder);
        else
            out_dir = output_path;
        end
        output_files{i} = fullfile(out_dir, [base '.edf']);
    end
    output_root = output_path;
else
    error('input_path does not exist: %s', input_path);
end
end

function entries_out = unique_dir_entries(entries_in)
if isempty(entries_in)
    entries_out = entries_in;
    return;
end
fullnames = strings(numel(entries_in), 1);
for i = 1:numel(entries_in)
    fullnames(i) = lower(string(fullfile(entries_in(i).folder, entries_in(i).name)));
end
[~, ia] = unique(fullnames, 'stable');
entries_out = entries_in(sort(ia));
[~, order] = sort({entries_out.name});
entries_out = entries_out(order);
end

function name = filename_without_ext(path_in)
[~, name] = fileparts(path_in);
end

function s = sanitize_edf_text(s)
s = char(string(s));
s = regexprep(s, '[^\x20-\x7E]', '');
s = strtrim(s);
if isempty(s)
    s = 'unknown';
end
end

% ========================================================================
% Annotation/event helpers
% ========================================================================
function events = try_read_events(block_path)
try
    events = ft_read_event(block_path);
catch ME
    warning('ft_read_event failed for %s: %s', block_path, ME.message);
    events = [];
end
end

function annotations = build_annotation_timetable(evt, fs, nSamplesOriginal, paddingSamples, in_file, internal_segments)
Onset = seconds(0);
Annotations = "SOURCE_FILE: " + string(filename_without_ext(in_file));
Duration = seconds(0);

if internal_segments > 1
    Onset(end+1,1) = seconds(0);
    Annotations(end+1,1) = "WARNING: multiple internal FieldTrip segments appended within source file";
    Duration(end+1,1) = seconds(0);
end

if ~isempty(evt)
    for e = 1:numel(evt)
        if isfield(evt(e), 'sample') && ~isempty(evt(e).sample) && isnumeric(evt(e).sample)
            sample = double(evt(e).sample);
            if isfinite(sample) && sample >= 1
                Onset(end+1,1) = seconds((sample - 1) / fs); %#ok<AGROW>
                Annotations(end+1,1) = "TRC_EVENT: " + event_to_string(evt(e)); %#ok<AGROW>
                Duration(end+1,1) = seconds(0); %#ok<AGROW>
            end
        end
    end
end

if paddingSamples > 0
    Onset(end+1,1) = seconds(nSamplesOriginal / fs);
    Annotations(end+1,1) = "CONVERSION_PADDING_ZERO_SAMPLES: " + string(paddingSamples);
    Duration(end+1,1) = seconds(paddingSamples / fs);
end

annotations = timetable(Onset, Annotations, Duration);
end

function s = event_to_string(ev)
parts = strings(0);
if isfield(ev, 'type') && ~isempty(ev.type)
    parts(end+1) = "type=" + string(ev.type);
end
if isfield(ev, 'value') && ~isempty(ev.value)
    try
        parts(end+1) = "value=" + string(ev.value);
    catch
        parts(end+1) = "value=<unprintable>";
    end
end
if isfield(ev, 'sample') && ~isempty(ev.sample)
    parts(end+1) = "sample=" + string(ev.sample);
end
if isempty(parts)
    s = "event";
else
    s = strjoin(parts, '; ');
end
% EDF annotations should remain simple ASCII-ish text.
s = string(sanitize_edf_text(s));
end

% ========================================================================
% Channel label helpers
% ========================================================================
function [labels_out, unmatched_labels] = normalize_eeg_labels(labels_in, elec1005)
standard_labels = elec1005.label;
labels_out = cell(size(labels_in));
unmatched_labels = {};

for i = 1:numel(labels_in)
    lbl = labels_in{i};
    if isstring(lbl)
        lbl = char(lbl);
    end
    lbl = strtrim(lbl);

    if isempty(lbl)
        labels_out{i} = lbl;
        continue;
    end

    matching_label_idx = find(strcmpi(lbl, standard_labels), 1);
    if ~isempty(matching_label_idx)
        labels_out{i} = standard_labels{matching_label_idx};
    else
        labels_out{i} = lbl;
        unmatched_labels{end+1} = lbl; %#ok<AGROW>
    end
end

if ~isempty(unmatched_labels)
    warning('The following labels did not match the standard 1005 electrode format: %s', ...
            strjoin(unmatched_labels, ', '));
end
end

function [chantype, chanunit] = infer_chantype_chanunit(labels)
% Infer FieldTrip/BIDS-like channel types and units from channel labels.
% Keeps this non-destructive: it only annotates channels, it does not remove them.

    n = numel(labels);
    chantype = cell(n, 1);
    chanunit = cell(n, 1);

    for i = 1:n
        lab = upper(strtrim(labels{i}));

        if contains_any_label(lab, {'ECG', 'EKG'})
            chantype{i} = 'ECG';
            chanunit{i} = 'uV';

        elseif contains_any_label(lab, {'EMG'})
            chantype{i} = 'EMG';
            chanunit{i} = 'uV';

        elseif contains_any_label(lab, {'EOG'})
            chantype{i} = 'EOG';
            chanunit{i} = 'uV';

        elseif contains_any_label(lab, {'MKR', 'MRK', 'MARK', 'MARKER', ...
                                        'TRIG', 'TRIGGER', ...
                                        'STATUS', 'EVENT', 'ANNOT'})
            chantype{i} = 'TRIG';
            chanunit{i} = 'Boolean';

        elseif contains_any_label(lab, {'RESP', 'BREATH', 'AIRFLOW', 'ADEM'})
            chantype{i} = 'RESP';
            chanunit{i} = 'n/a';

        elseif contains_any_label(lab, {'SPO2', 'SAO2', 'PLETH', 'PULSE', ...
                                        'PHOTIC', 'LIGHT', 'SOUND', ...
                                        'DC', 'AUX', 'X', 'ARM', 'SCH'})
            chantype{i} = 'MISC';
            chanunit{i} = 'n/a';

        elseif contains_any_label(lab, {'F', 'C', 'T', 'P', 'O', 'Sp', 'A', 'M', 'G'})
            chantype{i} = 'EEG';
            chanunit{i} = 'uV';

        else
            chantype{i} = 'MISC';
            chanunit{i} = 'n/a';
        end
    end
end


function tf = contains_any_label(label, patterns)
    tf = false;
    for j = 1:numel(patterns)
        if contains(label, patterns{j})
            tf = true;
            return;
        end
    end
end

% ========================================================================
% TRC datetime helpers, copied/adapted from previous pipeline
% ========================================================================
function [dt, hasTimeOfDay] = trc_get_start_datetime(hdr)
hasTimeOfDay = false;

if isfield(hdr, 'orig') && isfield(hdr.orig, 'NewSubFileHeader')
    H = hdr.orig.NewSubFileHeader;
    if isfield(H,'SamplingDate') && isfield(H,'SamplingTime')
        dateStr = cleanup(H.SamplingDate);
        timeStr = cleanup(H.SamplingTime);
        dt = parse_datetime(dateStr, timeStr);
        hasTimeOfDay = true;
        return;
    end
end

if isfield(hdr, 'orig') && isfield(hdr.orig, 'SubFileHeader')
    H = hdr.orig.SubFileHeader;
    if isfield(H,'DataStartDate') && isfield(H,'DataStartTime')
        dateStr = cleanup(H.DataStartDate);
        timeStr = cleanup(H.DataStartTime);
        dt = parse_datetime(dateStr, timeStr);
        hasTimeOfDay = true;
        return;
    end
end

if isfield(hdr, 'orig') && isfield(hdr.orig, 'H1')
    H = hdr.orig.H1;
    if isfield(H,'StartDate') && isfield(H,'StartTime')
        dateStr = cleanup(H.StartDate);
        timeStr = cleanup(H.StartTime);
        dt = parse_datetime(dateStr, timeStr);
        hasTimeOfDay = true;
        return;
    end
end

if isfield(hdr, 'orig') && isfield(hdr.orig, 'HeadboxDate') && isfield(hdr.orig, 'HeadboxTime')
    dateStr = cleanup(hdr.orig.HeadboxDate);
    timeStr = cleanup(hdr.orig.HeadboxTime);
    dt = parse_datetime(dateStr, timeStr);
    hasTimeOfDay = true;
    return;
end

if isfield(hdr, 'orig') && all(isfield(hdr.orig, {'day','month','year'}))
    day   = cleanup(hdr.orig.day);
    month = cleanup(hdr.orig.month);
    year  = cleanup(hdr.orig.year);

    try
        dt = datetime([day '-' month '-' year], 'InputFormat','dd-MMM-yyyy');
    catch
        dt = datetime([day '-' month '-' year]);
    end
    dt.Hour = 0;
    dt.Minute = 0;
    dt.Second = 0;
    hasTimeOfDay = false;
    return;
end

warning('Cannot determine TRC start datetime from hdr.orig. Falling back to 2000-01-01 00:00:00.');
dt = datetime(2000,1,1,0,0,0);
hasTimeOfDay = false;
end

function s = cleanup(s)
s = strrep(char(string(s)), '\0', '');
s = strtrim(s);
end

function dt = parse_datetime(dateStr, timeStr)
possibleFormats = {
    'dd/MM/yyyy HH:mm:ss'
    'dd/MM/yy HH:mm:ss'
    'MM/dd/yyyy HH:mm:ss'
    'yyyy-MM-dd HH:mm:ss'
    'dd-MMM-yyyy HH:mm:ss'
    'dd-MMM-yy HH:mm:ss'
    'dd.MM.yyyy HH.mm.ss'
    'dd.MM.yy HH.mm.ss'
    };

for f = 1:numel(possibleFormats)
    try
        dt = datetime([dateStr ' ' timeStr], 'InputFormat', possibleFormats{f});
        return;
    catch
    end
end

% Last-resort parser.
try
    dt = datetime([dateStr ' ' timeStr]);
    return;
catch
end

error('Failed to parse datetime strings: "%s" "%s"', dateStr, timeStr);
end

function [dateStr, timeStr] = datetime_to_edf_strings(dt)
% EDF+ uses dd.mm.yy and hh.mm.ss in MATLAB edfheader fields.
dateStr = datestr(dt, 'dd.mm.yy');
timeStr = datestr(dt, 'HH.MM.SS');
end

% ========================================================================
% Empty output row
% ========================================================================
function row = empty_conversion_row()
row = struct();
row.status = "";
row.error_message = "";
row.input_file = "";
row.output_file = "";
row.start_date = "";
row.start_time = "";
row.has_time_of_day = false;
row.f_sample = NaN;
row.n_channels = NaN;
row.n_samples_original = NaN;
row.n_samples_written = NaN;
row.padding_samples = NaN;
row.duration_original_sec = NaN;
row.duration_written_sec = NaN;
row.record_duration_sec = NaN;
row.internal_segments = NaN;
row.n_events_read = NaN;
row.label_normalization_applied = false;
row.original_labels = "";
row.output_labels = "";
row.unmatched_labels = "";
end
