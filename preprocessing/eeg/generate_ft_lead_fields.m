
%% Volumetric EEG sensitivity maps using FieldTrip BEM
% Each map represents the norm of the average-referenced EEG lead field
% across three orthogonal dipole orientations.
% Using this for construction and projection of EEG model v2 targets and
% predictions to MRI MNI-152 space.

%% Configuration
SCRIPT_DIR = fileparts(mfilename('fullpath'));           % root/src/preprocessing/eeg
ROOT_DIR = fileparts(fileparts(fileparts(SCRIPT_DIR)));  % root

TEMPLATE = fullfile(SCRIPT_DIR, 'MNI152_T1_1mm_brain.nii.gz');
OUTPUT_DIR = fullfile(ROOT_DIR, 'data', 'preprocessing', ...
    'eeg', 'sensitivity_maps');

CHANNELS = {'Fp1','Fp2','F9','F10','F7','F3','Fz','F4','F8', ...
    'T7','C3','Cz','C4','T8','P7','P3','Pz','P4','P8','O1','O2'};

GRID_STEP_VOX = 1;       % ~1 mm for the 1 mm template
INWARD_SHIFT_MM = 1;     % Avoid dipoles on the BEM boundary

[~, ft_dir] = ft_version;
mkdir(OUTPUT_DIR);

%% 1. Load the template head model and electrodes

headmodel = ft_read_headmodel(fullfile(ft_dir, ...
    'template', 'headmodel', 'standard_bem.mat'));
elec = ft_read_sens(fullfile(ft_dir, ...
    'template', 'electrode', 'standard_1005.elc'));

headmodel = ft_convert_units(headmodel, 'mm');
elec = ft_convert_units(elec, 'mm');

assert(all(ismember(CHANNELS, elec.label)), ...
    'Some requested EEG channels are missing from the montage.');

%% 2. Construct a grid aligned to the output NIfTI

mri = ft_read_mri(TEMPLATE);
info = niftiinfo(TEMPLATE);
brain_mask = mri.anatomy > 0;
dim = size(brain_mask);

assert(isequal(dim, info.ImageSize(1:3)), ...
    'FieldTrip and MATLAB NIfTI dimensions differ.');

ix = unique([1:GRID_STEP_VOX:dim(1), dim(1)]);
iy = unique([1:GRID_STEP_VOX:dim(2), dim(2)]);
iz = unique([1:GRID_STEP_VOX:dim(3), dim(3)]);

[I, J, K] = ndgrid(ix, iy, iz);
vox = [I(:)'; J(:)'; K(:)'; ones(1, numel(I))];
world = mri.transform * vox;

cfg = [];
cfg.method = 'basedonpos';
cfg.headmodel = headmodel;
cfg.sourcemodel.pos = world(1:3, :)';
cfg.sourcemodel.dim = size(I);
cfg.unit = 'mm';
cfg.inwardshift = INWARD_SHIFT_MM;
sourcemodel = ft_prepare_sourcemodel(cfg);

fprintf('Source points inside brain: %d / %d\n', ...
    nnz(sourcemodel.inside), numel(sourcemodel.inside));

assert(nnz(sourcemodel.inside) > 0, 'No valid source points.');

%% 3. Compute EEG lead fields

cfg = [];
cfg.elec = elec;
cfg.headmodel = headmodel;
cfg.sourcemodel = sourcemodel;
cfg.channel = CHANNELS;
cfg.reducerank = 'no';
cfg.normalize = 'no';

lf = ft_prepare_leadfield(cfg);

assert(isequal(lf.dim, size(I)), 'Lead-field grid dimensions changed.');
assert(max(abs(lf.pos(:) - sourcemodel.pos(:))) < 1e-6, ...
    'Lead-field source positions changed unexpectedly.');

%% 4. Compute orientation-independent sensitivity

n_channels = numel(CHANNELS);
sensitivity = zeros(numel(lf.inside), n_channels, 'single');
channel_idx = cellfun(@(c) find(strcmp(lf.label, c), 1), CHANNELS);

for idx = find(lf.inside(:))'
    L = double(lf.leadfield{idx});  % Channels x 3 orientations

    % Common-average reference across our 21 channels
    L = L - mean(L, 1);

    % Euclidean norm across dipole orientations
    sensitivity(idx, :) = single( ...
        sqrt(sum(L(channel_idx, :).^2, 2)))';
end

%% 5. Interpolate to the full MNI grid and export NIfTIs

info.Datatype = 'single';
info.BitsPerPixel = 32;
if isfield(info, 'MultiplicativeScaling')
    info.MultiplicativeScaling = 1;
end
if isfield(info, 'AdditiveOffset')
    info.AdditiveOffset = 0;
end

[X, Y] = ndgrid(1:dim(1), 1:dim(2));

for c = 1:n_channels
    coarse = reshape(sensitivity(:, c), size(I));

    F = griddedInterpolant({double(ix), double(iy), double(iz)}, ...
        double(coarse), 'linear', 'none');

    volume = zeros(dim, 'single');
    for z = 1:dim(3)
        slice = F(X, Y, z * ones(size(X)));
        slice(~isfinite(slice)) = 0;
        volume(:, :, z) = single(slice);
    end

    volume(~brain_mask) = 0;

    % Same peak normalisation as the previous Gaussian maps
    maximum = max(volume(:));
    assert(maximum > 0, 'Empty sensitivity map for %s.', CHANNELS{c});
    volume = volume / maximum;

    filename = fullfile(OUTPUT_DIR, ...
        ['sensitivity_map_MNI_' CHANNELS{c} '.nii']);
    niftiwrite(volume, filename, info, 'Compressed', true);

    fprintf('Saved %s | max %.3f\n', CHANNELS{c}, max(volume(:)));
end

fprintf('Finished generating %d sensitivity maps.\n', n_channels);
