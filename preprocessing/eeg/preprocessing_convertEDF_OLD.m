% TRC -> EDF conversion using FieldTrip
clearvars;
close all;
clc;

% Set directories 
input_dir = 'C:\Users\sversch6\Documents\tmp\edfconversion_in';
output_dir = 'C:\Users\sversch6\Documents\tmp\edfconversion_out';
conversion_tsv_dir = [output_dir, '\conversion_logs'];
L_drive_in_dir = 'L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\raw\eeg\_tmp\raw_EEG_SEIN\SEIN_Zwolle';
L_drive_out_dir = 'L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\raw\eeg\EDFdata\SEIN_Zwolle';
L_drive_conversion_tsv_dir = [L_drive_out_dir, '\conversion_logs'];

% Clear local in- and output directories if exist
% if exist(input_dir, 'dir')
%     rmdir(input_dir, 's');
% end
% if exist(output_dir, 'dir')
%     rmdir(output_dir, 's');
% end
% Make log dir
if ~exist(conversion_tsv_dir, 'dir')
    mkdir(conversion_tsv_dir);
end
% Setup logfile
diary(fullfile(conversion_tsv_dir, 'conversion_log.txt'))

% Start
fprintf("============= Start .TRC -> .EDF conversion script =============\n\n")
fprintf("Local input:    %s\n", input_dir)
fprintf("Local output:   %s\n", output_dir)
fprintf("L-drive input:  %s\n", L_drive_in_dir)
fprintf("L-drive output: %s\n", L_drive_out_dir)
fprintf("Logging to:     %s\n\n", fullfile(conversion_tsv_dir, 'conversion_log.txt'))
fprintf("================================================================\n\n")

% Load fieldtrip
if ~contains(path, 'L:\her_knf_golf\Wetenschap\newtransport\Sjors\ext\MATLAB\fieldtrip')
    fprintf('  >>> Loading FieldTrip\n');
    run('L:\her_knf_golf\Wetenschap\newtransport\Sjors\ext\MATLAB\startup_ft.m')
end

% % Get files locally for I/O efficiency
% syncFolders(L_drive_in_dir, input_dir)

% Desired resampling frequency
targetFs = 256;

% Standard electrode configuration
elec1005 = ft_read_sens('standard_1005.elc');

% Subject folders
subj_list = dir(fullfile(L_drive_in_dir,'RESP*'));

% Loop over subjects
for iP = 1:numel(subj_list)

    subj_id = subj_list(iP).name;
    fprintf('\n----- Processing subject %d/%d: %s -----\n\n', iP, numel(subj_list), subj_id);
    % Set directories and sync
    L_drive_subj_in_dir = fullfile(L_drive_in_dir, subj_id);
    L_drive_subj_out_dir = fullfile(L_drive_out_dir, subj_id);
    local_subj_in_dir = fullfile(input_dir, subj_id);
    local_subj_out_dir = fullfile(output_dir, subj_id);
    
    syncFolders(L_drive_subj_in_dir, local_subj_in_dir);
    
    % Get trace files
    trc_list = dir(fullfile(local_subj_in_dir,'*.TRC'));

    if isempty(trc_list)
        warning('No TRC files for %s - skipping', subj_id);
        continue;
    end

    % Set output .edf
    output_edf = fullfile(local_subj_out_dir, subj_id + ".edf");
    if exist(output_edf)
        warning('Output already exists for %s - skipping', subj_id)
        continue
    end

    % clear per-subject containers
    hdr = cell(1,numel(trc_list));
    evt = cell(1,numel(trc_list));
    raw = cell(1,numel(trc_list));
    nsamples = zeros(1,numel(trc_list));

    block_labels = cell(1, numel(trc_list));
    block_bads = cell(1, numel(trc_list));
    block_interp = cell(1, numel(trc_list));

    % Catch errors to move on to next patient
    try
        % --- Load each TRC block, resample, reref, collect labels & events ---
        for iT = 1:numel(trc_list)
            trc_path = fullfile(trc_list(iT).folder, trc_list(iT).name);
            fprintf('\n  >>> Loading TRC block %d/%d: %s\n', iT, numel(trc_list), trc_list(iT).name);
    
            % read original header & events (orig Fs needed for event resampling)
            hdr_orig = ft_read_header(trc_path);
            hdr{iT} = hdr_orig;
            % events sometimes fail, catch for now, TODO: Maybe fix if
            % necessary later
            try
                evt{iT} = ft_read_event(trc_path);
            catch ME
                warning('ft_read_event failed for %s: %s', trc_list(iT).name, ME.message);
                evt{iT} = [];
            end
    
            % Preprocessing: load EEG channels
            fprintf("   >> Loading EEG channels\n")
            cfg = [];
            cfg.dataset = trc_path;
            cfg.channel = 'EEG';
            raw_block = ft_preprocessing(cfg);  % raw_block is a FT structure
    
            % RESAMPLE to targetFs
            fprintf("   >> Resampling to %s Hz\n", string(targetFs))
            cfg = [];
            cfg.resamplefs = targetFs;
            raw_block = ft_resampledata(cfg, raw_block);
    
            % Update event sample indices according to resampling
            if ~isempty(evt{iT})
                origFs = hdr_orig.Fs;
                if isempty(origFs) || origFs == 0
                    warning('Original Fs missing in header, assuming %g Hz', targetFs);
                    origFs = targetFs;
                end
                ratio = targetFs / origFs;
                for e = 1:numel(evt{iT})
                    if isfield(evt{iT}(e),'sample') && ~isempty(evt{iT}(e).sample)
                        evt{iT}(e).sample = round(evt{iT}(e).sample * ratio);
                    end
                end
            end
    
            % In some TRC files, data may already be continuous; if multiple segments
            % are found in raw_block.trial, append them into one trial for clarity
            if isfield(raw_block,'trial') && numel(raw_block.trial) > 1
                fprintf("   >> Concatenating %s trials\n", string(numel(raw_block.trial)))
                cfg = [];
                raw_block = ft_appenddata(cfg, raw_block);
            end
    
            % Normalize channel names (not homogeneous across centers...)
            fprintf('   >> Normalizing channel names\n');
            fprintf('Pre:  %s\n', strjoin(raw_block.label, ', '))
            raw_block.label = normalize_eeg_labels(raw_block.label, elec1005);
            fprintf('Post: %s\n', strjoin(raw_block.label, ', '))

            % Automatically select and remove bad channels 
            % Based on very low or high variance
            chan_std = std(raw_block.trial{1},0,2);
            z = (chan_std - mean(chan_std)) / std(chan_std);
            bad_ch = raw_block.label(abs(z) > 3);  % z-score threshold
            % Remove them
            if ~isempty(bad_ch)
                fprintf('   >> Excluding bad channels: %s\n', strjoin(bad_ch, ', '));
                cfg = [];
                cfg.channel = setdiff(raw_block.label, bad_ch);  % keep only good channels
                raw_block = ft_selectdata(cfg, raw_block);
            end

            % Rereference: use implicitref + average of listed channels
            fprintf('   >> Rereferencing\n');
            % Re-ref based on all available standard electrode positions
            refList = intersect(elec1005.label, raw_block.label, 'stable');  % only keep present channels
            % Rereference
            cfg = [];
            cfg.implicitref = 'G1';
            cfg.refchannel = refList;
            cfg.reref = 'yes';
            raw_block = ft_preprocessing(cfg, raw_block);

            % Remove G1 if present
            fprintf('   >> Removing G1\n');
            cfg = [];
            cfg.channel = setdiff(raw_block.label,'G1');
            raw_block = ft_selectdata(cfg, raw_block);
    
            % store final block
            raw{iT} = raw_block;
            nsamples(iT) = size(raw_block.trial{1},2);
    
            % store labels for later union
            block_labels{iT} = raw_block.label(:)';  % row cell
            block_bads{iT} = bad_ch;
        end % block loop
    
        % --- Combine/shift block events so they refer to concatenated samples ---
        % compute cumulative offsets and shift event samples of subsequent blocks
        evt_all = [];
        cumulative = 0;
        for b = 1:numel(evt)
            if ~isempty(evt{b})
                for e = 1:numel(evt{b})
                    if isfield(evt{b}(e),'sample') && ~isempty(evt{b}(e).sample)
                        evt{b}(e).sample = evt{b}(e).sample + cumulative;
                    end
                end
                evt_all = [evt_all, evt{b}]; %#ok<AGROW>
            end
            cumulative = cumulative + nsamples(b);
        end
    
        % --- Determine master channel list (union of all block labels) ---
        % ensure block_labels exists
        if exist('block_labels','var')
            all_labels = unique([block_labels{:}],'stable'); % keep stable order
            all_labels(strcmp(all_labels,'G1')) = [];
        else
            error('No block_labels detected for subject %s', subj_list(iP).name);
        end
    
        % --- For blocks missing channels, interpolate them ---
        fprintf('  >>> Interpolating missing/bad channels\n');
        chan_rep = cell(1,numel(raw));
        for iT = 1:numel(raw)
            present = raw{iT}.label(:)';
            % missing channels = those in all_labels but not in present
            miss_channels = setdiff(all_labels, present, 'stable');
            block_interp{iT} = miss_channels;
    
            if isempty(miss_channels)
                fprintf('   >> [BLOCK %s] No missing/bad channels\n', string(iT));
                chan_rep{iT} = raw{iT};
            else
                fprintf('   >> [BLOCK %s] Interpolating missing/bad channels: %s\n', ...
                string(iT), strjoin(miss_channels, ', '));

                % read standard electrode positions
                elec = ft_read_sens('standard_1005.elc');
    
                % prepare neighbours using triangulation for spline interpolation
                cfgn = [];
                cfgn.elec = elec;
                cfgn.channel = all_labels;
                cfgn.method = 'triangulation';
                cfgn.compress = 'yes';
                neighbours = ft_prepare_neighbours(cfgn);
    
                % repair missing channels using spline interpolation
                cfgr = [];
                cfgr.missingchannel = miss_channels;
                cfgr.method = 'spline';
                cfgr.elec = elec;
                cfgr.neighbours = neighbours;
                cfgr.senstype = 'eeg';
                chan_rep{iT} = ft_channelrepair(cfgr, raw{iT});
            end
        end
    
        % After interpolation, check whether this was successful.
        % Sometimes, with "outer" channels, no neighbours are found and are
        % skipped, leading to concat errors. If this is the case, we drop
        % these channels for now.
        fprintf('  >>> Verifying interpolation success across blocks\n');
        % collect channel labels after interpolation
        post_labels = cellfun(@(x) x.label(:)', chan_rep, 'UniformOutput', false);
        % find channels present in all blocks
        common_labels = post_labels{1};
        for iT = 2:numel(post_labels)
            common_labels = intersect(common_labels, post_labels{iT}, 'stable');
        end
        % identify channels that failed interpolation
        failed_channels = setdiff(all_labels, common_labels, 'stable');
        if ~isempty(failed_channels)
            fprintf('   >> Dropping channels with failed interpolation: %s\n', ...
                strjoin(failed_channels, ', '));
            % drop failed channels from all blocks
            cfgsel = [];
            cfgsel.channel = common_labels;
            for iT = 1:numel(chan_rep)
                chan_rep{iT} = ft_selectdata(cfgsel, chan_rep{iT});
            end
        else
            fprintf('   >> All channels successfully interpolated\n');
        end
        
        % Reorder channels after interpolation etc.
        fprintf('  >>> Reordering channels across blocks\n');
        % Define order
        canonical_order = all_labels(ismember(all_labels, common_labels));
        % Define config
        cfgsel = [];
        cfgsel.channel = canonical_order;
        % Reorder
        for iT = 1:numel(chan_rep)
            chan_rep{iT} = ft_selectdata(cfgsel, chan_rep{iT});
        end

        % final sanity check
        assert(all(cellfun(@(x) isempty(setxor(x.label(:), canonical_order(:))), chan_rep)), ...
            'Channel set mismatch remains after cleanup');

        % --- Concatenate repaired blocks into single structure ---
        % cfg = [];
        % cfg.keepsampleinfo = 'no';
        % cfg.appenddim = 'rpt';
        % concat_data = ft_appenddata(cfg, chan_rep{:});  % merged
    
        fprintf('\n  >>> Concatenating data blocks...\n');
    
        % Number of channels (assume all blocks have same channels after repair)
        nCh = numel(chan_rep{1}.label);
        % Total number of samples across all blocks
        totalSamples = sum(cellfun(@(x) size(x.trial{1},2), chan_rep));
    
        % Preallocate final data matrix (channels x total samples)
        fprintf('Clearing raw data from memory...\n')
        raw_cp = cell(1, numel(raw));
        for i = 1:numel(raw)
            raw_cp{i}.hdr = raw{i}.hdr;
            raw_cp{i}.fsample = raw{i}.fsample;
            raw_cp{i}.sampleinfo = raw{i}.sampleinfo;
            raw_cp{i}.label = raw{i}.label';
            raw_cp{i}.cfg = raw{i}.cfg;
            raw_cp{i}.size = size(raw{i}.trial{:});
        end
        clear raw raw_block;
        raw = raw_cp;
        fprintf('Preallocating final data...\n')
        final_data = zeros(nCh, totalSamples, 'single');
    
        % Preallocate sampleinfo if it exists
        if isfield(chan_rep{1}, 'sampleinfo')
            totalSamplesBlocks = size(chan_rep{1}.sampleinfo,1);
            totalSampleinfoRows = sum(cellfun(@(x) size(x.sampleinfo,1), chan_rep));
            sampleinfo_all = zeros(totalSampleinfoRows, 2);
            hasSampleinfo = true;
        else
            hasSampleinfo = false;
        end
        
        % Copy blocks into preallocated array
        fprintf('Copying blocks into final data array...\n')
        currentIndex = 1;        % running index along time
        currentSampleinfoRow = 1; % running row for sampleinfo
        
        for i = 1:numel(chan_rep)
            block = chan_rep{i};
            nS = size(block.trial{1}, 2);
            
            % Copy data into preallocated matrix
            final_data(:, currentIndex:currentIndex+nS-1) = block.trial{1};
            
            % Copy & shift sampleinfo if present
            if hasSampleinfo
                nRows = size(block.sampleinfo,1);
                % shift start/end indices by current total samples
                if currentSampleinfoRow == 1
                    offset = 0;
                else
                    offset = sampleinfo_all(currentSampleinfoRow-1,2);
                end
                sampleinfo_all(currentSampleinfoRow:currentSampleinfoRow+nRows-1, :) = ...
                    block.sampleinfo + offset;
                
                currentSampleinfoRow = currentSampleinfoRow + nRows;
            end
            
            currentIndex = currentIndex + nS;
        end
        
        % Build the FieldTrip structure
        concat_data = chan_rep{1};
        concat_data.trial{1} = final_data;
    
        if hasSampleinfo
            concat_data.sampleinfo = sampleinfo_all;
        end
    
        % Prepare data for chopping into "records" (for which we choose 10 seconds),
        % which is apparently required for most EDF+C readers. Did this
        % because Persyst didn't like the 1-record-of-5-days data.
        % Number of channels and samples
        [numSignals, numSamples] = size(final_data);
        % Choose DataRecord duration
        recordDuration = 10; % seconds
        % Number of samples per signal per record
        samplesPerSignal = targetFs * recordDuration;
        % Number of DataRecords
        numRecords = floor(numSamples / samplesPerSignal);
        % Trim or warn if data does not divide evenly
        final_data = final_data(:, 1 : numRecords * samplesPerSignal);

        % Calculate block timings for annotation in EDF+
        % ------ BLOCK START DATETIMES ------
        block_start_time = datetime.empty(1,0);
        
        for iT = 1:numel(hdr)
            block_start_time(iT) = trc_get_start_datetime(hdr{iT});
        end
        % ------ SAMPLE INDICES WHERE BLOCKS START IN CONCATENATED DATA ------
        block_start_samples = zeros(1, numel(raw));
        block_start_times_rel = zeros(1, numel(raw));
        block_duration_samples = zeros(1, numel(raw));
        block_duration_times_rel = zeros(1, numel(raw));
        current_index = 1;
        Onset = seconds(zeros(numel(raw), 1));
        Annotations = strings(numel(raw), 1);
        Duration = seconds(zeros(numel(raw), 1));
        for iT = 1:numel(raw)
            % number of samples in this block
            nS = raw{iT}.size(2); % size(raw{iT}.trial{1}, 2);
            % Relative times and n samples
            block_start_samples(iT) = current_index;
            block_start_times_rel(iT) = (current_index - 1) / targetFs;
            block_duration_samples(iT) = nS;
            block_duration_times_rel(iT) = nS / targetFs;
            % Generate annotations
            Onset(iT) = seconds(block_start_times_rel(iT));
            Annotations(iT) = "BREAK: Start " + trc_list(iT).name;

            % Update index
            current_index = current_index + nS;
        end

        % Generate annotationTable
        tsal = timetable(Onset, Annotations, Duration);

        % Construct new header
        new_hdr = edfheader("EDF+");
        new_hdr.Patient = strjoin([subj_id, ""]);
        new_hdr.Recording = strjoin(["Startdate", '01-Jan-2000']);
        new_hdr.StartDate = '01.01.00';
        new_hdr.StartTime = '00.00.00';
        new_hdr.Reserved = "EDF+C";
        new_hdr.NumSignals = numSignals;
        new_hdr.SignalLabels = string(concat_data.label);
        new_hdr.DataRecordDuration = seconds(recordDuration);
        new_hdr.NumDataRecords = numRecords;
        new_hdr.PhysicalMin = floor(min(final_data, [], 2)).' - 1;
        new_hdr.PhysicalMax = ceil(max(final_data, [], 2)).' + 1;
        new_hdr.DigitalMin = repmat(-32768, 1, new_hdr.NumSignals);
        new_hdr.DigitalMax = repmat(32768, 1, new_hdr.NumSignals);
        new_hdr.TransducerTypes = repmat("EEG electrode", 1, new_hdr.NumSignals);
        new_hdr.PhysicalDimensions = repmat("uV", 1, new_hdr.NumSignals);

        % --- Optionally attach combined events into header (if ft_write_data supports) ---
        % keep evt_all around if you want to write triggers separately later
        % evt_all contains events with sample indices adjusted for resampling + block offsets
    
        clear concat_data

        % --- Write EDF file ---
        out_subjectdir = [output_dir, char("\"), subj_list(iP).name];
        if ~exist(out_subjectdir, 'dir')
            mkdir(out_subjectdir)
        end
        out_fname = fullfile(out_subjectdir, [subj_list(iP).name, '.edf']);
        fprintf('\n  >>> Writing EDF: %s\n', out_fname);
        edfwrite(out_fname, new_hdr, final_data.', tsal, 'InputSampleType', 'physical');
        % edfwrite_streaming(out_fname, new_hdr, final_data, tsal, targetFs);

        % ------ WRITE BLOCK METADATA TO TSV ------
        tsv_file = fullfile(conversion_tsv_dir, [subj_list(iP).name '_trc2edf_conversion.tsv']);
        fid = fopen(tsv_file, 'w');
        fprintf('\n  >>> Writing block timing TSV: %s\n', tsv_file);
        
        % Header
        fprintf(fid, 'block_index\tstart_sample\tn_samples\tstart_relative_time\tduration\tstart_absolute_date\tstart_absolute_time\ttrc_file\tf_sample\tn_samples\tn_good_channels\tgood_channels\tbad_channels\tinterpolated_channels\n');
        
        for iT = 1:numel(raw)
        
            % Extract datetime (handles both old + new TRC headers)
            [dt_block, hasTime] = trc_get_start_datetime(hdr{iT});
        
            % Convert components
            date_str = datestr(dt_block, 'yyyy-mm-dd');
        
            if hasTime
                time_str = datestr(dt_block, 'HH:MM:SS');
            else
                time_str = '';
            end
    
            % Write row
            fprintf(fid, '%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', ...
                iT, ...
                block_start_samples(iT), ...
                string(block_duration_samples(iT)), ...
                seconds2hms(block_start_times_rel(iT)), ...
                seconds2hms(block_duration_times_rel(iT)), ...
                date_str, ...
                time_str, ...
                trc_list(iT).name, ...
                string(raw{iT}.fsample), ...
                string(raw{iT}.size(2)), ...
                string(length(block_labels{iT})), ...
                strjoin(block_labels{iT}, ','), ...
                strjoin(block_bads{iT}, ','), ...
                strjoin(block_interp{iT}, ','));
        end
        
        fclose(fid);
    
        % clean per-subject variables
        clear hdr evt raw raw_cp chan_rep concat_data final_data new_hdr evt_all nsamples block_labels block_bads;

        % Sync to L-drive
        fprintf("\n  >>> Syncing to L-drive and cleaning up local data...\n")
        syncFolders(local_subj_out_dir, L_drive_subj_out_dir);
        syncFolders(conversion_tsv_dir, L_drive_conversion_tsv_dir);
        try
            rmdir(local_subj_in_dir, 's');
            rmdir(local_subj_out_dir, 's');
        catch ME
            warning("Error removing local directories:\n%s", string(ME.message))
        end
 
    % Move on to next patient (and log) if error
    catch ME
        warning("Error processing patient %s:\n%s: %s", subj_id, ME.identifier, ME.message)
        fid = fopen(fullfile(conversion_tsv_dir, 'failed_subjects.txt'), 'a');
        fprintf(fid, '%s\n', subj_id);
        fclose(fid);
        clear hdr evt raw raw_cp chan_rep concat_data final_data new_hdr evt_all nsamples block_labels block_bads;
        % Sync to L-drive
        fprintf("\n  >>> Syncing to L-drive and cleaning up local data...\n")
        syncFolders(local_subj_out_dir, L_drive_subj_out_dir);
        syncFolders(conversion_tsv_dir, L_drive_conversion_tsv_dir);
        try
            rmdir(local_subj_in_dir, 's');
            rmdir(local_subj_out_dir, 's');
        catch ME
            warning("Error removing local directories:\n%s", string(ME.message))
        end
        continue
    end
end

% Done. Sync to L-drive
fprintf('All done. Starting final L-drive sync...\n');
syncFolders(output_dir, L_drive_out_dir);

% Clean up local directories
fprintf('Done. Exiting logs and removing local directories...\n')
fprintf("\n========================== End logfile =========================\n")
diary("off")
fprintf('Cleaning up local directories...\n');
rmdir(input_dir, 's')
rmdir(output_dir, 's')
fprintf('Done, exiting...\n')

% ---- Helper functions ----
function labels_out = normalize_eeg_labels(labels_in, elec1005)
%NORMALIZE_EEG_LABELS Normalize EEG channel labels based on the standard 1005 montage format.
%
% This function ensures that the labels match the standard 1005 format (used in FieldTrip), adjusting 
% the capitalization to fit the correct EEG labeling conventions (e.g., Fp1 -> Fp1, CP4 -> CP4).
% A warning is issued if no labels from the input match the standard 1005 montage.
    
    standard_labels = elec1005.label;

    labels_out = cell(size(labels_in));
    unmatched_labels = {};  % To store unmatched labels

    for i = 1:numel(labels_in)
        lbl = labels_in{i};

        % Convert string -> char if needed
        if isstring(lbl)
            lbl = char(lbl);
        end

        % Trim whitespace around label
        lbl = strtrim(lbl);

        if isempty(lbl)
            labels_out{i} = lbl;
            continue;
        end

        % Check if the label matches any of the standard labels
        matching_label_idx = find(strcmpi(lbl, standard_labels), 1);

        if ~isempty(matching_label_idx)
            % If we find a match, we use the standardized label format
            labels_out{i} = standard_labels{matching_label_idx};
        else
            % If no match is found, store it in the unmatched list
            labels_out{i} = lbl;
            unmatched_labels{end+1} = lbl;  % Add the unmatched label to the list
        end
    end

    % Warn if any labels didn't match
    if ~isempty(unmatched_labels)
        warning('normalize_eeg_labels:unmatchedLabels', ...
            'The following labels did not match the standard 1005 electrode format: %s', ...
            strjoin(unmatched_labels, ', '));
    end
end

function [dt, hasTimeOfDay] = trc_get_start_datetime(hdr)
%TRC_GET_START_DATETIME  Extracts datetime from Micromed TRC header.
% Supports both old TRC (date only) and newer TRC (date + time).
%
% OUTPUTS:
%   dt            = datetime object (time = 00:00:00 if missing)
%   hasTimeOfDay  = true if TRC contains real HH:MM:SS; false otherwise

    hasTimeOfDay = false;  % assume no time-of-day until proven otherwise

    % -------------------------------
    % 1. NEWER TRC FORMAT (recommended)
    % -------------------------------
    if isfield(hdr.orig, 'NewSubFileHeader')
        H = hdr.orig.NewSubFileHeader;

        if isfield(H,'SamplingDate') && isfield(H,'SamplingTime')
            dateStr = cleanup(H.SamplingDate);
            timeStr = cleanup(H.SamplingTime);

            dt = parse_datetime(dateStr, timeStr);
            hasTimeOfDay = true;
            return;
        end
    end

    % -------------------------------
    % 2. ALTERNATIVE NEWER FORMAT
    % -------------------------------
    if isfield(hdr.orig, 'SubFileHeader')
        H = hdr.orig.SubFileHeader;

        if isfield(H,'DataStartDate') && isfield(H,'DataStartTime')
            dateStr = cleanup(H.DataStartDate);
            timeStr = cleanup(H.DataStartTime);

            dt = parse_datetime(dateStr, timeStr);
            hasTimeOfDay = true;
            return;
        end
    end

    % -------------------------------
    % 3. H1 structure (some TRC revisions)
    % -------------------------------
    if isfield(hdr.orig, 'H1')
        H = hdr.orig.H1;

        if isfield(H,'StartDate') && isfield(H,'StartTime')
            dateStr = cleanup(H.StartDate);
            timeStr = cleanup(H.StartTime);

            dt = parse_datetime(dateStr, timeStr);
            hasTimeOfDay = true;
            return;
        end
    end

    % -------------------------------
    % 4. SOME FILES USE HeadboxDate / HeadboxTime
    % -------------------------------
    if isfield(hdr.orig, 'HeadboxDate') && isfield(hdr.orig, 'HeadboxTime')
        dateStr = cleanup(hdr.orig.HeadboxDate);
        timeStr = cleanup(hdr.orig.HeadboxTime);

        dt = parse_datetime(dateStr, timeStr);
        hasTimeOfDay = true;
        return;
    end

    % -------------------------------
    % 5. OLD TRC FORMAT (date only)
    % -------------------------------
    if all(isfield(hdr.orig, {'day','month','year'}))
        day   = cleanup(hdr.orig.day);
        month = cleanup(hdr.orig.month);
        year  = cleanup(hdr.orig.year);

        % Month may be in text form (e.g., 'JUN')
        % Guarantee valid datetime interpretation
        try
            dt = datetime([day '-' month '-' year], 'InputFormat','dd-MMM-yyyy');
        catch
            dt = datetime([day '-' month '-' year]);
        end

        % No time available → assume midnight
        dt.Hour = 0;
        dt.Minute = 0;
        dt.Second = 0;

        hasTimeOfDay = false;
        return;
    end

    % -------------------------------
    % 6. If none match → FAIL
    % -------------------------------
    error('Cannot determine TRC start datetime from hdr.orig. Examine hdr.orig manually.');
end


% =========================================================
% Cleanup helper (removes null chars and trims whitespace)
% =========================================================
function s = cleanup(s)
    s = strrep(s, '\0', '');
    s = strtrim(s);
end

% =========================================================
% Robust datetime parsing (multiple formats)
% =========================================================
function dt = parse_datetime(dateStr, timeStr)

    possibleFormats = {
        'dd/MM/yyyy HH:mm:ss'
        'dd/MM/yy HH:mm:ss'
        'MM/dd/yyyy HH:mm:ss'
        'yyyy-MM-dd HH:mm:ss'
        'dd-MMM-yyyy HH:mm:ss'
        'dd-MMM-yy HH:mm:ss'
    };

    dt = [];
    for f = 1:numel(possibleFormats)
        try
            dt = datetime([dateStr ' ' timeStr], 'InputFormat', possibleFormats{f});
            return;
        catch
        end
    end

    error('Failed to parse datetime strings: "%s" "%s"', dateStr, timeStr);
end

function out = seconds2hms(sec)
    % Convert seconds to HH:MM:SS string
    h = floor(sec / 3600);
    m = floor(mod(sec, 3600) / 60);
    s = round(mod(sec, 60), 2);

    out = sprintf('%02d:%02d:%05.2f', h, m, s);
end

function edfwrite_streaming(out_fname, hdr, final_data, annotations, fs)
% Stream large EDF data with diary-safe progress logging.

    % --------------------------------------
    % Basic signal sizes
    % --------------------------------------
    [numSignals, numSamples] = size(final_data);

    samplesPerRecord = fs * seconds(hdr.DataRecordDuration);
    samplesPerRecord = double(samplesPerRecord);

    numRecords = hdr.NumDataRecords;

    expectedSamples = samplesPerRecord * numRecords;
    if expectedSamples > numSamples
        error("Not enough samples for the number of DataRecords in the header.");
    elseif expectedSamples < numSamples
        warning("Extra samples in final_data ignored (beyond header length).");
        final_data = final_data(:,1:expectedSamples);
    end

    % --------------------------------------
    % Write header and annotations only
    % --------------------------------------
    edfwrite(out_fname, hdr, annotations);

    % --------------------------------------
    % Streaming parameters
    % --------------------------------------
    batchSize = 25;      % DataRecords per batch
    nBatches  = ceil(numRecords / batchSize);

    fprintf("Writing EDF data (%d records in %d batches):\n", ...
        numRecords, nBatches);

    % Logging frequency (one line every N batches)
    printEvery = max(1, ceil(nBatches / 100));   % ~100 lines, diary-safe

    % --------------------------------------
    % Stream-write each batch
    % --------------------------------------
    for b = 1:nBatches

        rStart = (b-1)*batchSize + 1;
        rEnd   = min(b*batchSize, numRecords);
        nRec   = rEnd - rStart + 1;

        % Allocate chunk: samplesPerRecord × nRec × numSignals
        chunk = zeros(samplesPerRecord, nRec, numSignals, 'double');

        for ch = 1:numSignals
            idx = (rStart-1)*samplesPerRecord + (1:(nRec*samplesPerRecord));
            sigChunk = reshape(final_data(ch, idx), samplesPerRecord, nRec);
            chunk(:,:,ch) = sigChunk;
        end

        % Append chunk
        edfwrite(out_fname, [], chunk, 'InputSampleType','physical');

        % --------------------------------------
        % Diary-safe progress update
        % --------------------------------------
        if mod(b, printEvery) == 0 || b == nBatches
            pct = (b / nBatches) * 100;
            fprintf("  Progress: %6.2f%%  (%d / %d batches written)\n", ...
                    pct, b, nBatches);
        end
    end

    fprintf("EDF streaming write complete: %s\n", out_fname);
end


function syncFolders(src, dst)
% syncFolders Recursively sync only changed or new files from src → dst
%
%   syncFolders(src, dst)
%   - Creates missing subfolders
%   - Copies files only if:
%         * They do not exist in dst, OR
%         * They are newer in src than dst
%
%   Example:
%       syncFolders('C:\data\source', 'C:\data\backup')

    fprintf("  >>> Syncing %s -> %s\n", src, dst);

    if ~isfolder(src)
        error('Source folder does not exist: %s', src);
    end

    if ~isfolder(dst)
        mkdir(dst);
    end

    % Process all items in the source folder
    items = dir(src);

    for k = 1:length(items)
        name = items(k).name;

        % Skip . and ..
        if strcmp(name, '.') || strcmp(name, '..')
            continue;
        end

        srcPath = fullfile(src, name);
        dstPath = fullfile(dst, name);

        if items(k).isdir
            % --- Recursively sync subfolder ---
            if ~exist(dstPath, 'dir')
                mkdir(dstPath);
            end
            syncFolders(srcPath, dstPath);

        else
            % --- File: copy only if new or updated ---
            copyNeeded = false;

            if ~exist(dstPath, 'file')
                copyNeeded = true;
            else
                if dir(srcPath).datenum > dir(dstPath).datenum
                    copyNeeded = true;
                end
            end

            if copyNeeded
                copyfile(srcPath, dstPath);
                fprintf('Updated: %s\n', dstPath);
            end
        end
    end
end

