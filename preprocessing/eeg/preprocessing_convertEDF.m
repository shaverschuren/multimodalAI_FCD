% TRC/EDF -> EDF conversion using FieldTrip
clearvars;
close all;
clc;

% Set directories 
input_dir = 'C:\Users\sversch6\Documents\tmp\edfconversion_in';
output_dir = 'C:\Users\sversch6\Documents\tmp\edfconversion_out';
conversion_tsv_dir = [output_dir, '\conversion_logs'];
L_drive_in_dir = 'L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\raw\eeg\sig2edf';
L_drive_out_dir = 'L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\raw\eeg\EDFdata_SIG';
L_drive_conversion_tsv_dir = [L_drive_out_dir, '\conversion_logs'];

% Set-up bad channel z-score cut-off
BAD_CH_Z_CUTOFF = 3.0;

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
fprintf("============= Start .TRC/.EDF -> .EDF conversion script =============\n\n")
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
    % Set directories
    L_drive_subj_in_dir = fullfile(L_drive_in_dir, subj_id);
    L_drive_subj_out_dir = fullfile(L_drive_out_dir, subj_id);
    local_subj_in_dir = fullfile(input_dir, subj_id);
    local_subj_out_dir = fullfile(output_dir, subj_id);
    
    % Set edf paths and skip if already present
    L_drive_edf = fullfile(L_drive_subj_out_dir, subj_id + ".edf");
    output_edf = fullfile(local_subj_out_dir, subj_id + ".edf");
    
    if exist(L_drive_edf, 'file')
        warning('Output already exists for %s - skipping', subj_id)
        continue
    end

    % Sync (move on to next patient if fails e.g. due to disk space issue)
    try
        syncFolders(L_drive_subj_in_dir, local_subj_in_dir);
    catch ME
        warning("Error syncing:\n%s", string(ME.message))
        continue
    end
    
    % Get input EEG files. Prefer native TRC if present; otherwise use EDF
    % files, e.g. converted Stellate/Harmonie SIG/STS recordings.
    [block_list, input_filetype] = get_subject_input_blocks(local_subj_in_dir);

    if isempty(block_list)
        warning('No TRC or EDF files for %s - skipping', subj_id);
        continue;
    end

    nBlocks = numel(block_list);
    fprintf('Detected %d %s file(s) for %s\n', nBlocks, upper(input_filetype), subj_id);

    % clear per-subject containers
    hdr = cell(1,nBlocks);
    evt = cell(1,nBlocks);
    raw = cell(1,nBlocks);
    nsamples = zeros(1,nBlocks);

    block_labels = cell(1, nBlocks);
    block_labels_orig_norm = cell(1, nBlocks);
    block_bads = cell(1, nBlocks);
    block_interp = cell(1, nBlocks);

    % Catch errors to move on to next patient
    try
        % --- Load each block, resample, reref, collect labels & events ---
        for iT = 1:nBlocks
            block_path = fullfile(block_list(iT).folder, block_list(iT).name);
            fprintf('\n  >>> Loading %s block %d/%d: %s\n', upper(input_filetype), iT, nBlocks, block_list(iT).name);
    
            % read original header & events (orig Fs needed for event resampling)
            hdr_orig = ft_read_header(block_path);
            hdr{iT} = hdr_orig;
            % events sometimes fail, catch for now, TODO: Maybe fix if
            % necessary later
            evt{iT} = try_read_events(block_path, block_list(iT).name);
    
            % Preprocessing: load EEG channels
            fprintf("   >> Loading EEG channels\n")
            raw_block = load_eeg_block_fieldtrip(block_path, input_filetype);  % raw_block is a FT structure
    
            % RESAMPLE to targetFs
            fprintf("   >> Resampling to %s Hz\n", string(targetFs))
            cfg = [];
            cfg.resamplefs = targetFs;
            raw_block = ft_resampledata_chunked(cfg, raw_block);
    
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
    
            % In some input files, data may already be continuous; if multiple segments
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
            labels_orig_norm = raw_block.label;
            fprintf('Post: %s\n', strjoin(raw_block.label, ', '))

            % Automatically select and remove bad channels
            % Conservative robust amplitude outlier detection, PREP/PyPREP-inspired
            data = raw_block.trial{1};
            % Use robust channel amplitude instead of plain std.
            % This follows the PREP/PyPREP idea more closely:
            % channel amplitude = IQR of signal, scaled to approximate SD.
            IQR_TO_SD = 0.7413;
            chan_amp = iqr(data, 2) * IQR_TO_SD;
            
            med_amp = median(chan_amp, 'omitnan');
            amp_iqr = iqr(chan_amp);
            
            if amp_iqr == 0 || ~isfinite(amp_iqr)
                warning('Bad-channel detection skipped: IQR of channel amplitudes is zero or invalid');
                bad_ch = {};
            else
                robust_scale = amp_iqr * IQR_TO_SD;
                robust_z = (chan_amp - med_amp) / robust_scale;
                % Get high-z channels
                bad_high = robust_z > BAD_CH_Z_CUTOFF;
                % Flat/dead channels: handle separately.
                % 5% of median amplitude is conservative but catches near-flat channels.
                bad_low = chan_amp < max(1e-12, 0.05 * med_amp);
                % Select badd channels
                bad_ch = raw_block.label(bad_high | bad_low);
                % Safety guard: do not automatically reject too many channels.
                % This prevents pathological recordings/references from causing mass rejection.
                max_bad_fraction = 0.3;
                if numel(bad_ch) > max_bad_fraction * numel(raw_block.label)
                    % Re-try with higher z cut-off
                    warning('Setting cut-off to z-score > 4.0, too many excluded')
                    bad_high = robust_z > 4.0;
                    bad_ch = raw_block.label(bad_high | bad_low);
                    if numel(bad_ch) > max_bad_fraction * numel(raw_block.label)
                        warning('Bad-channel detection flagged %d/%d channels; skipping automatic rejection for this block', ...
                            numel(bad_ch), numel(raw_block.label));
                        bad_ch = {};
                    end
                end
                % Z-score info
                fprintf('Highest remaining z-score in selected channels: %d\n', max(robust_z(~bad_high)))
            end
            
            % Remove them
            if ~isempty(bad_ch)
                fprintf('   >> Excluding bad channels: %s\n', strjoin(bad_ch, ', '));
                cfg = [];
                cfg.channel = setdiff(raw_block.label, bad_ch, 'stable');
                raw_block = ft_selectdata(cfg, raw_block);
            else
                fprintf('   >> No bad channels excluded\n');
            end

            % Rereference: use implicitref + average of listed channels
            fprintf('   >> Rereferencing\n');
            % Re-ref based on all available standard electrode positions
            all1005 = string(elec1005.label);
            labels = string(raw_block.label);
            commonReferences = ["A1","A2","M1","M2","TP9","TP10", "G1", "G2"]; % Exclude common reference electrodes
            refList = intersect(all1005, labels, 'stable');
            refList = setdiff(refList, commonReferences, 'stable');
            fprintf('Using ref channels (available, good 10-05): %s\n', strjoin(refList, ', '));
            % Rereference
            cfg = [];
            % cfg.implicitref = 'G1';
            cfg.refchannel = refList;
            cfg.reref = 'yes';
            raw_block = ft_preprocessing(cfg, raw_block);

            % Remove reference-like channels as we don't need them anymore.
            fprintf('   >> Removing reference-like channels\n');
            fprintf('Removing %s\n', strjoin(intersect(labels, commonReferences), ', '))
            cfg = [];
            cfg.channel = cellstr(setdiff(labels, commonReferences, 'stable'));
            raw_block = ft_selectdata(cfg, raw_block);
    
            % store final block
            raw{iT} = raw_block;
            nsamples(iT) = size(raw_block.trial{1},2);
    
            % store labels for later union
            block_labels_orig_norm{iT} = labels_orig_norm;
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
            % Master channel order should be based on original normalized labels,
            % before bad-channel removal, but after excluding reference-like channels.
            all_labels = {};
            for iT = 1:numel(block_labels_orig_norm)
                labels_this = setdiff(block_labels_orig_norm{iT}, cellstr(commonReferences), 'stable');
                all_labels = [all_labels, labels_this(:)']; %#ok<AGROW>
            end
            all_labels = unique(all_labels, 'stable');
        else
            error('No block_labels detected for subject %s', subj_list(iP).name);
        end
    
        % --- For blocks missing channels, interpolate them ---
        fprintf('  >>> Interpolating missing/bad channels\n');
        chan_rep = cell(1,numel(raw));
        for iT = 1:numel(raw)
            labels = raw{iT}.label(:)';
            % missing channels = those in all_labels but not in present
            miss_channels = setdiff(all_labels, labels, 'stable');
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
        canonical_1005_order = get_canonical_1005_order();
        available_labels = common_labels(:)';
        
        % First: hard-coded 10-05 order
        ordered_known = canonical_1005_order(ismember(canonical_1005_order, available_labels));
        % Second: any remaining labels not in hard-coded order
        ordered_extra = available_labels(~ismember(available_labels, canonical_1005_order));
        % Merge
        canonical_order = [ordered_known, ordered_extra];
        
        % Explicitly reorder each FieldTrip raw structure.
        % Doing this because ft_selectdata doesn't do this automatically
        % and afraid that the channels will mismatch at the concat step.
        for iT = 1:numel(chan_rep)
            chan_rep{iT} = reorder_ft_raw_channels(chan_rep{iT}, canonical_order);
        end
        
        % Strong sanity check: exact order, not just same set
        assert(all(cellfun(@(x) isequal(x.label(:), canonical_order(:)), chan_rep)), ...
            'Channel order mismatch remains after cleanup');

        % % Reorder channels after interpolation etc.
        % fprintf('  >>> Reordering channels across blocks\n');
        % % Define order
        % canonical_order = all_labels(ismember(all_labels, common_labels));
        % % Define config
        % cfgsel = [];
        % cfgsel.channel = canonical_order;
        % % Reorder
        % for iT = 1:numel(chan_rep)
        %     chan_rep{iT} = ft_selectdata(cfgsel, chan_rep{iT});
        % end
        % 
        % % final sanity check
        % assert(all(cellfun(@(x) isempty(setxor(x.label(:), canonical_order(:))), chan_rep)), ...
        %     'Channel set mismatch remains after cleanup');

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
            [block_start_time(iT), ~] = get_block_start_datetime(hdr{iT}, input_filetype);
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
            Annotations(iT) = "BREAK: Start " + block_list(iT).name;

            % Update index
            current_index = current_index + nS;
        end

        % Generate annotationTable
        tsal = timetable(Onset, Annotations, Duration);

        % Construct new header
        new_hdr = edfheader('EDF');
        new_hdr.Patient = sanitize_edf_text(strjoin([subj_id, ""]));
        new_hdr.Recording = sanitize_edf_text(strjoin(["Startdate", '01-Jan-2000']));
        new_hdr.StartDate = '01.01.00';
        new_hdr.StartTime = '00.00.00';
        new_hdr.Reserved = '';
        new_hdr.NumSignals = numSignals;
        new_hdr.SignalLabels = string(concat_data.label);
        new_hdr.DataRecordDuration = seconds(recordDuration);
        new_hdr.NumDataRecords = numRecords;
        new_hdr.PhysicalMin = floor(min(final_data, [], 2)).' - 1;
        new_hdr.PhysicalMax = ceil(max(final_data, [], 2)).' + 1;
        new_hdr.DigitalMin = repmat(-32768, 1, new_hdr.NumSignals);
        new_hdr.DigitalMax = repmat(32767, 1, new_hdr.NumSignals);
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
        tsv_file = fullfile(conversion_tsv_dir, [subj_list(iP).name '_' char(input_filetype) '2edf_conversion.tsv']);
        fid = fopen(tsv_file, 'w');
        fprintf('\n  >>> Writing block timing TSV: %s\n', tsv_file);
        
        % Header
        fprintf(fid, 'block_index\tstart_sample\tn_samples\tstart_relative_time\tduration\tstart_absolute_date\tstart_absolute_time\tinput_file\tinput_filetype\tf_sample\tn_samples\tn_good_channels\tgood_channels\tbad_channels\tinterpolated_channels\n');
        
        for iT = 1:numel(raw)
        
            % Extract datetime (handles TRC and EDF headers)
            [dt_block, hasTime] = get_block_start_datetime(hdr{iT}, input_filetype);
        
            % Convert components
            date_str = datestr(dt_block, 'yyyy-mm-dd');
        
            if hasTime
                time_str = datestr(dt_block, 'HH:MM:SS');
            else
                time_str = '';
            end
    
            % Write row
            fprintf(fid, '%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', ...
                iT, ...
                block_start_samples(iT), ...
                string(block_duration_samples(iT)), ...
                seconds2hms(block_start_times_rel(iT)), ...
                seconds2hms(block_duration_times_rel(iT)), ...
                date_str, ...
                time_str, ...
                block_list(iT).name, ...
                char(input_filetype), ...
                string(raw{iT}.fsample), ...
                string(raw{iT}.size(2)), ...
                string(length(block_labels{iT})), ...
                strjoin(block_labels{iT}, ','), ...
                strjoin(block_bads{iT}, ','), ...
                strjoin(block_interp{iT}, ','));
        end
        
        fclose(fid);
    
        % clean per-subject variables
        clear hdr evt raw raw_cp chan_rep concat_data final_data new_hdr evt_all nsamples block_labels block_bads block_interp block_list input_filetype nBlocks;

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
        clear hdr evt raw raw_cp chan_rep concat_data final_data new_hdr evt_all nsamples block_labels block_bads block_interp block_list input_filetype nBlocks;
        % Try to sync to L-drive, remove local data
        try
            fprintf("\n  >>> Syncing to L-drive and cleaning up local data...\n")
            syncFolders(local_subj_out_dir, L_drive_subj_out_dir);
            syncFolders(conversion_tsv_dir, L_drive_conversion_tsv_dir);
            rmdir(local_subj_in_dir, 's');
            rmdir(local_subj_out_dir, 's');
        catch ME
            warning("Error syncing/removing local directories:\n%s", string(ME.message))
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
function [block_list, input_filetype] = get_subject_input_blocks(local_subj_in_dir)
%GET_SUBJECT_INPUT_BLOCKS Return ordered TRC or EDF input blocks for one subject.
% Prefer native TRC if present; otherwise use EDF. This keeps the original
% TRC pathway unchanged while allowing converted SIG/STS -> EDF files to use
% the same downstream processing.

    trc_list = [dir(fullfile(local_subj_in_dir, '*.TRC')); ...
                dir(fullfile(local_subj_in_dir, '*.trc'))];
    edf_list = [dir(fullfile(local_subj_in_dir, '*.EDF')); ...
                dir(fullfile(local_subj_in_dir, '*.edf'))];

    trc_list = unique_dir_entries(trc_list);
    edf_list = unique_dir_entries(edf_list);

    if ~isempty(trc_list)
        block_list = trc_list;
        input_filetype = "trc";
    elseif ~isempty(edf_list)
        block_list = edf_list;
        input_filetype = "edf";
    else
        block_list = [];
        input_filetype = "";
    end
end

function entries_out = unique_dir_entries(entries_in)
%UNIQUE_DIR_ENTRIES Remove duplicate dir entries and sort by name.
% On case-insensitive file systems, *.TRC and *.trc can return overlapping
% results. This helper prevents processing the same file twice.

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

function data_out = ft_resampledata_chunked(cfg, data_in)
%FT_RESAMPLEDATA_CHUNKED Resample a continuous FieldTrip raw structure in chunks.
%
% Drop-in helper for:
%   data_out = ft_resampledata(cfg, data_in)
%
% Intended to avoid very large temporary arrays when resampling long
% continuous recordings. Keeps the external script architecture unchanged.

% Settings
chunkDurSec = 3600;

origFs = data_in.fsample;
targetFs = cfg.resamplefs;

if isequal(origFs, targetFs)
    fprintf('   >> Resample skipped: original Fs (%g Hz) equals target Fs (%g Hz)\n', origFs, targetFs);
    data_out = data_in;
    return;
end

nSamples = size(data_in.trial{1}, 2);
chunkSamples = round(chunkDurSec * origFs);
nChunks = ceil(nSamples / chunkSamples);

fprintf("   >> Chunked resampling: %d chunks of about %d seconds\n", ...
    nChunks, chunkDurSec);

chunk_trials = cell(1, nChunks);
totalSamplesOut = 0;

for c = 1:nChunks

    begsample = (c-1) * chunkSamples + 1;
    endsample = min(c * chunkSamples, nSamples);

    % Create lightweight chunk FT structure
    chunk = data_in;
    chunk.trial = {data_in.trial{1}(:, begsample:endsample)};
    chunk.time = {(0:(endsample-begsample)) / origFs};

    if isfield(data_in, 'sampleinfo')
        chunk.sampleinfo = [begsample endsample];
    end

    % Resample this chunk using FieldTrip's own machinery
    chunk_rs = ft_resampledata(cfg, chunk);

    chunk_trials{c} = single(chunk_rs.trial{1});
    totalSamplesOut = totalSamplesOut + size(chunk_trials{c}, 2);

    clear chunk chunk_rs
end

% Reassemble into one continuous FT structure
data_out = data_in;
data_out.trial = {zeros(numel(data_in.label), totalSamplesOut, 'single')};
data_out.fsample = targetFs;
data_out.time = {(0:totalSamplesOut-1) / targetFs};
data_out.sampleinfo = [1 totalSamplesOut];

idx = 1;
for c = 1:nChunks
    nS = size(chunk_trials{c}, 2);
    data_out.trial{1}(:, idx:idx+nS-1) = chunk_trials{c};
    idx = idx + nS;
end

end

function events = try_read_events(block_path, block_name)
%TRY_READ_EVENTS Read FieldTrip events but allow event-less files to proceed.

    try
        events = ft_read_event(block_path);
    catch ME
        warning('ft_read_event failed for %s: %s', block_name, ME.message);
        events = [];
    end
end

function raw_block = load_eeg_block_fieldtrip(block_path, input_filetype)
%LOAD_EEG_BLOCK_FIELDTRIP Load one block into a FieldTrip raw structure.
% TRC files keep the original cfg.channel = 'EEG' behavior. EDF files are
% loaded as 'all' first because EDF channel-type metadata is often unreliable;
% obvious annotation/status/non-EEG channels are then removed by label.

    cfg = [];
    cfg.dataset = block_path;

    switch lower(string(input_filetype))
        case "trc"
            cfg.channel = 'EEG';
            raw_block = ft_preprocessing(cfg);

        case "edf"
            cfg.channel = 'all';
            raw_block = ft_preprocessing(cfg);
            raw_block = remove_likely_non_eeg_channels(raw_block);

        otherwise
            error('Unknown input filetype: %s', input_filetype);
    end
end

function raw_block = remove_likely_non_eeg_channels(raw_block)
%REMOVE_LIKELY_NON_EEG_CHANNELS Drop common EDF annotation/status channels.
% This deliberately happens before label normalization and preserves all
% remaining channels for the existing bad-channel and interpolation logic.

    if ~isfield(raw_block, 'label') || isempty(raw_block.label)
        error('Loaded EDF block contains no channel labels.');
    end

    non_eeg_patterns = {'EDF Annotations', 'Annotations', 'Annotation', ...
                        'Status', 'Trigger', 'Triggers', 'Marker', ...
                        'DC', 'ECG', 'EKG', 'EMG', 'EOG', ...
                        'Photic', 'Pulse', 'Resp', 'Adem', 'X', 'oog'};

    keep = true(size(raw_block.label));
    for k = 1:numel(non_eeg_patterns)
        keep = keep & ~contains(raw_block.label, non_eeg_patterns{k}, 'IgnoreCase', true);
    end

    if ~any(keep)
        error('All EDF channels were rejected as non-EEG/annotation channels. Check EDF labels.');
    end

    dropped = raw_block.label(~keep);
    if ~isempty(dropped)
        fprintf('   >> Removing likely non-EEG EDF channels: %s\n', strjoin(dropped, ', '));
        cfg = [];
        cfg.channel = raw_block.label(keep);
        raw_block = ft_selectdata(cfg, raw_block);
    end
end

function data_out = reorder_ft_raw_channels(data_in, desired_order)
%REORDER_FT_RAW_CHANNELS Explicitly reorder channels in a FieldTrip raw structure.
%
% ft_selectdata can select channels, but depending on FieldTrip version and
% cfg.channel handling, it may preserve the original channel order. This
% helper forces the requested order by direct indexing.

    data_out = data_in;

    current_labels = data_in.label(:)';
    desired_order = desired_order(:)';

    [is_present, idx] = ismember(desired_order, current_labels);

    if ~all(is_present)
        missing = desired_order(~is_present);
        error('Cannot reorder: requested channels missing from data: %s', ...
            strjoin(missing, ', '));
    end

    % Reorder labels
    data_out.label = current_labels(idx)';

    % Reorder trial matrices: channels x samples
    if isfield(data_out, 'trial')
        for tr = 1:numel(data_out.trial)
            data_out.trial{tr} = data_out.trial{tr}(idx, :);
        end
    end

    % Reorder channel-level metadata where present
    if isfield(data_out, 'elec') && isfield(data_out.elec, 'label')
        [elec_present, elec_idx] = ismember(desired_order, data_out.elec.label(:)');
        if all(elec_present)
            data_out.elec.label = data_out.elec.label(elec_idx);

            if isfield(data_out.elec, 'chanpos') && size(data_out.elec.chanpos, 1) == numel(elec_idx)
                data_out.elec.chanpos = data_out.elec.chanpos(elec_idx, :);
            end
            if isfield(data_out.elec, 'elecpos') && size(data_out.elec.elecpos, 1) == numel(elec_idx)
                data_out.elec.elecpos = data_out.elec.elecpos(elec_idx, :);
            end
        end
    end

    % Remove stale cfg.channel info if present; it can be misleading later
    if isfield(data_out, 'cfg') && isfield(data_out.cfg, 'channel')
        data_out.cfg = rmfield(data_out.cfg, 'channel');
    end
end

function [dt, hasTimeOfDay] = get_block_start_datetime(hdr, input_filetype)
%GET_BLOCK_START_DATETIME Dispatch date parsing according to source filetype.

    switch lower(string(input_filetype))
        case "trc"
            [dt, hasTimeOfDay] = trc_get_start_datetime(hdr);
        case "edf"
            [dt, hasTimeOfDay] = edf_get_start_datetime(hdr);
        otherwise
            error('Unknown input filetype: %s', input_filetype);
    end
end

function [dt, hasTimeOfDay] = edf_get_start_datetime(hdr)
%EDF_GET_START_DATETIME Minimal start-datetime extraction from EDF header.
% Prefer numeric hdr.orig.T0 (e.g. [2013 1 17 11 24 49]).
% If not available or invalid, return fixed fallback 2000-01-01 00:00:00.

    hasTimeOfDay = false;

    if isfield(hdr, 'orig') && isfield(hdr.orig, 'T0')
        val = hdr.orig.T0;
        if isnumeric(val) && numel(val) >= 3
            v = double(val(:).');           % ensure row vector
            if numel(v) < 6
                v = [v, zeros(1, 6 - numel(v))];  % pad to [Y M D H MI S]
            end
            try
                dt = datetime(v(1), v(2), v(3), v(4), v(5), v(6));
                hasTimeOfDay = any(v(4:6) ~= 0);
                return;
            catch
                % fall through to fallback
            end
        end
    end

    % Immediate fallback (no other parsing attempted)
    dt = datetime(2000,1,1,0,0,0);
    hasTimeOfDay = false;
end

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

function order = get_canonical_1005_order()
%GET_CANONICAL_1005_ORDER Hard-coded scalp EEG channel order.
%
% Ordered roughly anterior-to-posterior, left-to-right within rows,
% using common 10-05 / extended 10-10 labels.
%
% Unknown labels are handled outside this function and appended at the end.

order = { ...
    ... % Frontopolar / prefrontal
    'Fp1','Fpz','Fp2', ...
    'AF9','AF7','AF5','AF3','AF1','AFz','AF2','AF4','AF6','AF8','AF10', ...
    ... % Frontal
    'F9','F7','F5','F3','F1','Fz','F2','F4','F6','F8','F10', ...
    ... % Frontocentral
    'FT9','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','FT10', ...
    ... % Central / temporal
    'T9','T7','C5','C3','C1','Cz','C2','C4','C6','T8','T10', ...
    ... % Centroparietal
    'TP9','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','TP10', ...
    ... % Parietal
    'P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8','P10', ...
    ... % Parieto-occipital
    'PO9','PO7','PO5','PO3','PO1','POz','PO2','PO4','PO6','PO8','PO10', ...
    ... % Occipital
    'O1','Oz','O2', ...
    'I1','Iz','I2', ...
    ... % Common mastoid / auricular / reference-like labels
    'M1','M2','A1','A2' ...
};
end

function s = sanitize_edf_text(s)
s = char(string(s));
s = regexprep(s, '[^\x20-\x7E]', '');
s = strtrim(s);
if isempty(s)
    s = 'unknown';
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
