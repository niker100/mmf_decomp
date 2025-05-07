% MMF_UTILS - Utility functions for MMF decomposition pipeline
% This file contains shared functions used across the MMF pipeline

function utils = mmf_utils()
    % Return a struct of function handles to keep namespace clean
    utils.getOrCreateModelWithModes = @getOrCreateModelWithModes;
    utils.createComplexWeights = @createComplexWeights;
    utils.dlCorr = @dlCorr;
    utils.calculateSignAccuracy = @calculateSignAccuracy;
    utils.calculateRelativeSignAccuracy = @calculateRelativeSignAccuracy;
    utils.thresholdL2Norm = @thresholdL2Norm;
end

function P = getOrCreateModelWithModes(number_of_modes, image_size, use_cached)
    % Creates or reuses a cached BPMmatlab model with precomputed modes
    % Inputs:
    %   number_of_modes - Number of modes to find
    %   image_size - Size of the simulation grid
    %   use_cached - Whether to use the cached model or create a new one
    
    persistent cached_P;
    
    % Return cached model if available and requested
    if use_cached && ~isempty(cached_P) && isfield(cached_P, 'modes') && length(cached_P.modes) >= number_of_modes
        P = cached_P;
        return;
    end
    
    % Create new model
    fprintf('Creating new BPMmatlab model with %d modes...\n', number_of_modes);
    P = BPMmatlab.model;
    P.name = 'mmf_pipeline_shared';
    P.useAllCPUs = true;
    P.useGPU = true;
    P.Lx_main = 50e-6;
    P.Ly_main = 50e-6;
    P.Nx_main = image_size;
    P.Ny_main = image_size;
    P.padfactor = 1.5;
    P.dz_target = 1e-6;
    P.lambda = 1000e-9;
    P.n_background = 1.45;
    P.n_0 = 1.46;
    P.Lz = 10e-4;
    P.updates = 1;
    core_radius = 25e-6;
    n_core = P.n_0;
    n_clad = P.n_background;
    P = initializeRIfromFunction(P, @(X,Y,~,~) n_clad + (n_core-n_clad)*(X.^2+Y.^2 < core_radius^2));
    P = findModes(P, number_of_modes, 'plotModes', false);
    
    % Cache the model for future use
    cached_P = P;
end

function [weights, full_phases] = createComplexWeights(amplitudes, phases, phase_signs, number_of_modes)
    % Create complex weights from amplitudes and phases
    % Inputs:
    %   amplitudes - Mode amplitudes [number_of_modes x batch_size]
    %   phases - Phase magnitudes [number_of_modes-1 x batch_size]
    %   phase_signs - Phase signs [number_of_modes-1 x batch_size]
    %   number_of_modes - Number of modes
    %
    % Returns:
    %   weights - Complex weights for all modes [number_of_modes x batch_size]
    %   full_phases - Phases for all modes [number_of_modes x batch_size]
    
    % Initialize outputs
    weights = zeros(number_of_modes, size(amplitudes, 2), 'like', amplitudes);
    full_phases = zeros(number_of_modes, size(amplitudes, 2), 'like', phases);
    
    % First mode has zero phase (reference)
    weights(1, :) = amplitudes(1, :);
    
    % Remaining modes
    for m = 2:number_of_modes
        phase_idx = m-1;
        phase_value = phases(phase_idx, :) .* phase_signs(phase_idx, :);
        full_phases(m, :) = phase_value;
        weights(m, :) = amplitudes(m, :) .* exp(1i * phase_value * pi);
    end
end

function corr = dlCorr(A, B)
    % Calculate correlation between dlarrays efficiently
    % Reshape to 2D while maintaining dlarray format
    imgSize = size(A, 1) * size(A, 2);
    batchSize = size(A, 4);
    
    % Reshape to [pixels, batch] - crucial to keep as dlarray
    A_flat = reshape(A, [imgSize, batchSize]);
    B_flat = reshape(B, [imgSize, batchSize]);
    
    % Fast mean calculation along pixel dimension
    A_mean = mean(A_flat, 1);  
    B_mean = mean(B_flat, 1);
    
    % Center the data (subtract mean)
    A_centered = A_flat - A_mean;
    B_centered = B_flat - B_mean;
    
    % Compute correlation efficiently
    % Numerator: covariance
    numerator = sum(A_centered .* B_centered, 1);
    
    % Denominator: product of standard deviations
    A_std = sqrt(sum(A_centered.^2, 1) + 1e-8);  % Add epsilon for numerical stability
    B_std = sqrt(sum(B_centered.^2, 1) + 1e-8);
    denominator = A_std .* B_std;
    
    % Compute correlation and average across batch
    batch_corr = numerator ./ denominator;
    corr = mean(batch_corr);
end

function accuracy = calculateSignAccuracy(pred_signs, true_signs)
    % Calculate sign accuracy (exact match)
    match_count = sum(sign(pred_signs) == true_signs, 'all');
    total_elements = numel(pred_signs);
    
    accuracy = match_count / total_elements;
end

function [accuracy, per_mode_accuracy] = calculateRelativeSignAccuracy(pred_phases, true_phases)
    % Calculate sign accuracy allowing for global phase ambiguity
    num_samples = size(pred_phases, 2);
    num_modes = size(pred_phases, 1);
    
    % Initialize tracking
    per_mode_accuracy = zeros(1, num_modes);
    total_correct = 0;

    pred_phases = extract(pred_phases);
    true_phases = extract(true_phases);
    
    for i = 1:num_samples
        pred_signs = sign(pred_phases(:, i));
        true_signs = sign(true_phases(:, i));
        
        % Compare original vs flipped accuracy
        match_original = pred_signs == true_signs;
        match_flipped = -pred_signs == true_signs;
        
        % Determine which has better overall match
        if sum(match_flipped) > sum(match_original)
            correct_for_sample = match_flipped;
        else
            correct_for_sample = match_original;
        end
        
        % Update counters
        per_mode_accuracy = per_mode_accuracy + correct_for_sample';
        total_correct = total_correct + sum(correct_for_sample);
    end
    
    % Normalize
    per_mode_accuracy = per_mode_accuracy / num_samples;
    accuracy = total_correct / (num_samples * num_modes);
end

function grad = thresholdL2Norm(grad, threshold)
    % Apply gradient clipping by L2 norm
    gradNorm = sqrt(sum(grad(:).^2));
    if gradNorm > threshold
        grad = (threshold/gradNorm) * grad;
    end
end