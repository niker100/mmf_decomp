% filepath: c:\Users\nicks\MATLAB\Projects\MT2_prak\Versuch 3\generate_residuals.m
% generate_residuals.m - Generates residuals for global classifier training: nonlinear image minus linear reconstruction from amplitude+phase sign model predictions

function generate_residuals(mmf_train, labels_train, mmf_val, labels_val, mmf_test, labels_test)
% Generate residuals for global classifier training: nonlinear image minus linear recon from amplitude+phase sign model predictions

% Load amplitude and phase sign models
ampModel = load('amplitude_model.mat');
dlnet_amplitude = ampModel.dlnet_amplitude;
number_of_modes = ampModel.number_of_modes;
clear ampModel;

phaseModel = load('phase_sign_model.mat');
dlnet_phase = phaseModel.dlnet_phase;
clear phaseModel;

% Use smaller batch size for GPU memory efficiency
batchSize = 32;

% TRAIN SET
fprintf('Generating residuals for training set...\n');
[residuals_train] = compute_residuals(mmf_train, dlnet_amplitude, dlnet_phase, number_of_modes, batchSize);

% VAL SET
fprintf('Generating residuals for validation set...\n');
[residuals_val] = compute_residuals(mmf_val, dlnet_amplitude, dlnet_phase, number_of_modes, batchSize);

% TEST SET
fprintf('Generating residuals for test set...\n');
[residuals_test] = compute_residuals(mmf_test, dlnet_amplitude, dlnet_phase, number_of_modes, batchSize);

save('residuals.mat', 'residuals_train', 'residuals_val', 'residuals_test', '-v7.3');
end

function residuals = compute_residuals(X, ampNet, phaseNet, number_of_modes, batchSize)
    numSamples = size(X, 4);
    numBatches = ceil(numSamples / batchSize);
    image_size = size(X, 1);
    
    % Create residuals in CPU memory first for large datasets
    useGPU = false;
    try
        % Check if we can use GPU without memory issues
        if canUseGPU
            % Check available GPU memory
            gpuInfo = gpuDevice();
            memRequired = prod(size(X)) * 8 * 2; % Rough estimate for complex values
            if gpuInfo.AvailableMemory > memRequired
                useGPU = true;
            else
                fprintf('GPU memory insufficient for full residuals, computing in batches on CPU.\n');
            end
        end
    catch
        fprintf('GPU not available, computing on CPU.\n');
    end
    
    % Initialize residuals on CPU to avoid GPU memory overflow
    residuals = zeros(size(X), 'like', gather(X));
    
    % Process in very small batches to avoid GPU memory issues
    processBatchSize = min(batchSize, 16); % Even smaller batch for GPU processing
    
    for b = 1:numBatches
        startIdx = (b-1)*batchSize + 1;
        endIdx = min(b*batchSize, numSamples);
        currentBatch = startIdx:endIdx;
        
        % Process this batch
        fprintf('Processing batch %d/%d (samples %d-%d)\n', b, numBatches, startIdx, endIdx);
        
        % Further divide this batch into mini-batches for prediction
        numMiniBatches = ceil(length(currentBatch) / processBatchSize);
        
        % Initialize batch predictions
        Y_amps_batch = zeros(length(currentBatch), number_of_modes);
        Y_phases_batch = zeros(length(currentBatch), number_of_modes-1);
        
        for mb = 1:numMiniBatches
            mb_start = (mb-1)*processBatchSize + 1;
            mb_end = min(mb*processBatchSize, length(currentBatch));
            mb_indices = currentBatch(mb_start:mb_end);
            
            % Get nonlinear image for this mini-batch
            miniX = X(:,:,:,mb_indices);
            
            % Move to GPU if possible
            dlX = dlarray(miniX, 'SSCB');
            if useGPU
                try
                    dlX = gpuArray(dlX);
                catch
                    useGPU = false;
                    fprintf('GPU memory overflow, switching to CPU.\n');
                end
            end
            
            % Predict amplitudes and phase magnitudes
            YPred_amps = predict(ampNet, dlX);
            YPred_amps = extractdata(YPred_amps);
            
            % Gather to CPU for memory efficiency
            YPred_amps = gather(YPred_amps);
            
            % Predict phase signs
            if useGPU
                dlX = gpuArray(dlX);
            end
            phase_signs_pred = predict(phaseNet, dlX);
            phase_signs_pred = extractdata(phase_signs_pred);
            phase_signs_pred = gather(phase_signs_pred);
            
            % Process predictions for this mini-batch
            amps_pred = YPred_amps(1:number_of_modes, :);
            phase_magnitudes = abs(YPred_amps(number_of_modes+1:end, :));
            
            phase_signs = sign(tanh(phase_signs_pred));
            
            % Canonical form handling
            if size(phase_signs, 1) == number_of_modes - 2
                all_signs = [ones(1, size(phase_signs, 2)); phase_signs];
            else
                all_signs = phase_signs;
                all_signs(1, :) = ones(1, size(phase_signs, 2));
            end
            
            % Apply signs to phase magnitudes
            phase_values = phase_magnitudes .* all_signs;
            
            % Store in batch arrays
            mb_rel_indices = mb_start:mb_end;
            Y_amps_batch(mb_rel_indices, :) = amps_pred';
            Y_phases_batch(mb_rel_indices, :) = phase_values';
        end
        
        % Now generate reconstructions for the entire batch at once
        % Create complex weights for the full batch
        amps_full = Y_amps_batch';
        phases_full = Y_phases_batch';
        full_phases = [zeros(1, size(phases_full, 2)); phases_full];
        weights_pred = amps_full .* exp(1i * full_phases * pi);
        
        % Generate reconstructions in mini-batches to avoid memory issues
        for mb = 1:numMiniBatches
            mb_start = (mb-1)*processBatchSize + 1;
            mb_end = min(mb*processBatchSize, length(currentBatch));
            mb_indices = mb_start:mb_end;
            mb_global_indices = currentBatch(mb_indices);
            
            % Create weights for this mini-batch
            mb_weights = weights_pred(:, mb_indices);
            
            % Build reconstruction
            [recon_mb, ~] = mmf_build_image(number_of_modes, image_size, length(mb_indices), mb_weights, false);
            
            % Get corresponding nonlinear images
            nonlinear_mb = X(:,:,:,mb_global_indices);
            
            % Calculate and store residuals
            residuals(:,:,:,mb_global_indices) = nonlinear_mb - recon_mb;
        end
        
        % Report progress
        if mod(b, max(1, floor(numBatches/10))) == 0 || b == numBatches
            fprintf('Processed %d/%d batches (%.1f%%)\n', b, numBatches, b/numBatches*100);
        end
    end
end

function result = canUseGPU()
    % Check if GPU is available
    persistent useGPU;
    if isempty(useGPU)
        useGPU = false;
        try
            % Check for CUDA GPU
            gpuArray(1);
            useGPU = true;
        catch
            useGPU = false;
        end
    end
    result = useGPU;
end