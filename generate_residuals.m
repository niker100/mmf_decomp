% filepath: c:\Users\nicks\MATLAB\Projects\MT2_prak\mmf_decomp\generate_residuals.m
% generate_residuals.m - Generates residuals for global classifier training: nonlinear image minus linear reconstruction from amplitude+phase sign model predictions

function generate_residuals(mmf_train, labels_train, mmf_val, labels_val, mmf_test, labels_test, P_precomputed)
    % Generate residuals for global classifier training
    % 
    % Inputs:
    %   mmf_train - Training images [height x width x 1 x samples]
    %   labels_train - Training labels [samples x (2*number_of_modes-1)]
    %   mmf_val - Validation images [height x width x 1 x samples]
    %   labels_val - Validation labels [samples x (2*number_of_modes-1)]
    %   mmf_test - Test images [height x width x 1 x samples]
    %   labels_test - Test labels [samples x (2*number_of_modes-1)]
    %   P_precomputed - (Optional) Precomputed BPMmatlab model with modes
    %
    % Outputs:
    %   Saves residuals to 'residuals.mat'
    
    % Get utility functions
    utils = mmf_utils();

    % Load amplitude and phase sign models
    ampModel = load('absolute_model.mat');
    dlnet = ampModel.dlnet;
    number_of_modes = ampModel.number_of_modes;
    clear ampModel;

    phaseModel = load('phase_sign_model.mat');
    dlnet_phase = phaseModel.dlnet;
    clear phaseModel;

    % Get or create BPMmatlab model with precomputed modes
    if nargin < 7 || isempty(P_precomputed)
        inputSize = [size(mmf_train, 1), size(mmf_train, 2)];
        P = utils.getOrCreateModelWithModes(number_of_modes, inputSize(1), true);
    else
        P = P_precomputed;
    end

    % Use smaller batch size for GPU memory efficiency
    batchSize = 32;

    % TRAIN SET
    fprintf('Generating residuals for training set...\n');
    [residuals_train] = compute_residuals(mmf_train, dlnet, dlnet_phase, number_of_modes, P, batchSize, utils);

    % VAL SET
    fprintf('Generating residuals for validation set...\n');
    [residuals_val] = compute_residuals(mmf_val, dlnet, dlnet_phase, number_of_modes, P, batchSize, utils);

    % TEST SET
    fprintf('Generating residuals for test set...\n');
    [residuals_test] = compute_residuals(mmf_test, dlnet, dlnet_phase, number_of_modes, P, batchSize, utils);

    save('residuals.mat', 'residuals_train', 'residuals_val', 'residuals_test', '-v7.3');
end

function residuals = compute_residuals(X, ampNet, phaseNet, number_of_modes, P, batchSize, utils)
    numSamples = size(X, 4);
    numBatches = ceil(numSamples / batchSize);
    image_size = size(X, 1);
    
    % Create residuals in CPU memory first for large datasets
    useGPU = canUseGPU;
    if useGPU
        try
            % Check available GPU memory
            gpuInfo = gpuDevice();
            memRequired = prod(size(X)) * 8 * 2; % Rough estimate for complex values
            if gpuInfo.AvailableMemory < memRequired
                fprintf('GPU memory insufficient for full residuals, computing in batches on CPU.\n');
                useGPU = false;
            end
        catch
            useGPU = false;
            fprintf('Error checking GPU memory, switching to CPU.\n');
        end
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
                try
                    dlX = gpuArray(dlX);
                catch
                    useGPU = false;
                end
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
        % Create complex weights for the full batch using utility function
        Y_amps_batch = Y_amps_batch';
        Y_phases_batch = Y_phases_batch';
        weights_pred = utils.createComplexWeights(Y_amps_batch, Y_phases_batch);
        
        % Generate reconstructions in mini-batches to avoid memory issues
        for mb = 1:numMiniBatches
            mb_start = (mb-1)*processBatchSize + 1;
            mb_end = min(mb*processBatchSize, length(currentBatch));
            mb_indices = mb_start:mb_end;
            mb_global_indices = currentBatch(mb_indices);
            
            % Create weights for this mini-batch
            mb_weights = weights_pred(:, mb_indices);
            
            % Build reconstruction using precomputed P
            [recon_mb, ~] = mmf_build_image(number_of_modes, image_size, length(mb_indices), mb_weights', false, 0, P);
            
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