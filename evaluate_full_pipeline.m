% filepath: c:\Users\nicks\MATLAB\Projects\MT2_prak\mmf_decomp\evaluate_full_pipeline.m
% evaluate_full_pipeline.m - Evaluates the full MMF decomposition pipeline

function evaluate_full_pipeline(mmf_test, labels_test, options, P_precomputed)
    % EVALUATE_FULL_PIPELINE - Evaluate the complete MMF decomposition pipeline
    % Evaluates the full pipeline including:
    % 1. Amplitude and phase magnitude prediction
    % 2. Phase sign prediction
    % 3. Global sign flip classification
    % 4. Full reconstruction quality
    %
    % Inputs:
    %   mmf_test - Test images [height x width x 1 x samples]
    %   labels_test - Test labels [samples x (2*number_of_modes-1)]
    %   options - (Optional) Evaluation options struct
    %   P_precomputed - (Optional) Precomputed BPMmatlab model with modes
    %
    % Outputs:
    %   Display evaluation metrics and plots
    %   Saves results to 'pipeline_evaluation.mat'
    
    % Get utility functions
    utils = mmf_utils();
    
    % Parse options or use defaults
    if nargin < 3
        options = struct();
    end
    
    % Default options
    if ~isfield(options, 'showPlots'), options.showPlots = true; end
    if ~isfield(options, 'batchSize'), options.batchSize = 64; end
    if ~isfield(options, 'numSamplesToShow'), options.numSamplesToShow = 8; end
    if ~isfield(options, 'executionEnvironment'), options.executionEnvironment = "gpu"; end
    
    % Load models
    try
        % Load amplitude and phase model
        ampModel = load('absolute_model.mat');
        dlnet_amp = ampModel.dlnet;
        number_of_modes = ampModel.number_of_modes;
        
        % Load phase sign model
        phaseModel = load('phase_sign_model.mat');
        dlnet_phase = phaseModel.dlnet;
        
        % Load global sign classifier
        globalModel = load('phase_sign_classifier.mat');
        global_classifier = globalModel.global_classifier;
    catch ME
        error('Failed to load models: %s', ME.message);
    end
    
    % Get or create BPMmatlab model with precomputed modes
    if nargin < 4 || isempty(P_precomputed)
        inputSize = [size(mmf_test, 1), size(mmf_test, 2)];
        P = utils.getOrCreateModelWithModes(number_of_modes, inputSize(1), true);
    else
        P = P_precomputed;
    end
    
    % Get ground truth
    amps_true = labels_test(:, 1:number_of_modes);
    phases_true = labels_test(:, number_of_modes+1:end);
    signs_true = sign(phases_true);
    
    % Run complete evaluation pipeline
    [metrics, reconstructions] = evaluatePipeline(mmf_test, amps_true, phases_true, signs_true, ...
                                dlnet_amp, dlnet_phase, global_classifier, ...
                                number_of_modes, P, options, utils);
    
    % Display results
    displayResults(metrics, reconstructions, options);
    
    % Save results
    save('pipeline_evaluation.mat', 'metrics', 'reconstructions');
    disp('Pipeline evaluation results saved to pipeline_evaluation.mat');
end

function [metrics, reconstructions] = evaluatePipeline(X_test, amps_true, phases_true, signs_true, ...
                                dlnet_amp, dlnet_phase, global_classifier, ...
                                number_of_modes, P, options, utils)
    % Run all evaluation steps of the pipeline
    batchSize = options.batchSize;
    numSamples = size(X_test, 4);
    
    % Initialize metrics struct
    metrics = struct();
    
    % 1. Evaluate amplitude and phase magnitude model
    fprintf('Step 1: Evaluating amplitude and phase model...\n');
    
    % Initialize arrays for predictions
    amps_pred = zeros(numSamples, number_of_modes);
    phase_mags_pred = zeros(numSamples, number_of_modes-1);
    
    % Process in batches
    numBatches = ceil(numSamples/batchSize);
    for i = 1:numBatches
        startIdx = (i-1)*batchSize + 1;
        endIdx = min(i*batchSize, numSamples);
        
        % Extract batch data
        batch_X = X_test(:,:,:,startIdx:endIdx);
        
        % Predict amplitudes and phase magnitudes
        dlX = dlarray(batch_X, 'SSCB');
        if (options.executionEnvironment == "auto" && utils.canUseGPU()) || options.executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end
        
        % Get amplitude model predictions
        amp_pred_batch = predict(dlnet_amp, dlX);
        amp_pred_batch = extractdata(amp_pred_batch);
        
        % Separate predictions
        amps_batch = amp_pred_batch(1:number_of_modes, :)';
        phase_mags_batch = abs(amp_pred_batch(number_of_modes+1:end, :))';
        
        % Store predictions
        amps_pred(startIdx:endIdx, :) = amps_batch;
        phase_mags_pred(startIdx:endIdx, :) = phase_mags_batch;
    end
    
    % Calculate amplitude metrics
    amp_mse = mean((amps_pred - amps_true).^2, 'all');
    amp_rmse = sqrt(amp_mse);
    amp_mae = mean(abs(amps_pred - amps_true), 'all');
    amp_rel_error = amp_rmse / mean(abs(amps_true(:)));
    
    % Calculate phase magnitude metrics
    phase_mag_true = abs(phases_true);
    phase_mag_mse = mean((phase_mags_pred - phase_mag_true).^2, 'all');
    phase_mag_rmse = sqrt(phase_mag_mse);
    phase_mag_mae = mean(abs(phase_mags_pred - phase_mag_true), 'all');
    phase_mag_rel_error = phase_mag_rmse / mean(phase_mag_true(:));
    
    % Store in metrics
    metrics.amp_mse = amp_mse;
    metrics.amp_rmse = amp_rmse;
    metrics.amp_mae = amp_mae;
    metrics.amp_rel_error = amp_rel_error;
    metrics.phase_mag_mse = phase_mag_mse;
    metrics.phase_mag_rmse = phase_mag_rmse;
    metrics.phase_mag_mae = phase_mag_mae;
    metrics.phase_mag_rel_error = phase_mag_rel_error;
    
    % 2. Evaluate phase sign model
    fprintf('Step 2: Evaluating phase sign model...\n');
    
    % Initialize arrays for predictions
    phase_signs_pred = zeros(numSamples, number_of_modes-1);
    
    % Process in batches
    for i = 1:numBatches
        startIdx = (i-1)*batchSize + 1;
        endIdx = min(i*batchSize, numSamples);
        
        % Extract batch data
        batch_X = X_test(:,:,:,startIdx:endIdx);
        
        % Predict phase signs
        dlX = dlarray(batch_X, 'SSCB');
        if (options.executionEnvironment == "auto" && utils.canUseGPU()) || options.executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end
        
        % Get phase sign model predictions
        signs_pred_batch = predict(dlnet_phase, dlX);
        signs_pred_batch = extractdata(sign(tanh(signs_pred_batch)));
        
        % Handle canonical form
        if size(signs_pred_batch, 1) == number_of_modes - 2
            % Add the positive sign for mode 2
            all_signs = [ones(1, size(signs_pred_batch, 2)); signs_pred_batch];
        else
            all_signs = signs_pred_batch;
            all_signs(1, :) = 1; % Force canonical form
        end
        
        % Store predictions
        phase_signs_pred(startIdx:endIdx, :) = all_signs';
    end
    
    % Calculate sign accuracy per mode
    sign_matches = (phase_signs_pred == signs_true);
    mode_sign_accuracy = mean(sign_matches, 1);
    overall_sign_accuracy = mean(sign_matches, 'all');
    
    % Calculate weighted accuracy (weighting by phase magnitude)
    weighted_matches = sign_matches .* phase_mag_true;
    weighted_sign_accuracy = sum(weighted_matches(:)) / sum(phase_mag_true(:));
    
    % Store in metrics
    metrics.mode_sign_accuracy = mode_sign_accuracy;
    metrics.overall_sign_accuracy = overall_sign_accuracy;
    metrics.weighted_sign_accuracy = weighted_sign_accuracy;
    
    % 3. Evaluate global sign classifier
    fprintf('Step 3: Evaluating global sign classifier...\n');
    
    % Generate residuals from current (potentially wrong) predictions
    residuals = generateResiduals(X_test, amps_pred, phase_mags_pred, phase_signs_pred, number_of_modes, P, batchSize, utils);
    
    % Initialize arrays for predictions
    global_flip_pred = zeros(numSamples, 1);
    
    % Process in batches for memory efficiency
    for i = 1:numBatches
        startIdx = (i-1)*batchSize + 1;
        endIdx = min(i*batchSize, numSamples);
        
        % Extract batch residuals
        batch_res = residuals(:,:,:,startIdx:endIdx);
        
        % Predict global sign flip
        dlX = dlarray(batch_res, 'SSCB');
        if (options.executionEnvironment == "auto" && utils.canUseGPU()) || options.executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end
        
        % Get global flip predictions (0=keep, 1=flip)
        flip_pred_batch = predict(global_classifier, dlX);
        flip_pred_batch = extractdata(flip_pred_batch > 0.5);
        
        % Store predictions
        global_flip_pred(startIdx:endIdx) = flip_pred_batch;
    end
    
    % Apply global sign flips to get final signs
    final_signs_pred = phase_signs_pred;
    for i = 1:numSamples
        if global_flip_pred(i)
            final_signs_pred(i,:) = -phase_signs_pred(i,:);
        end
    end
    
    % Calculate final sign accuracy
    final_sign_matches = (final_signs_pred == signs_true);
    final_mode_sign_accuracy = mean(final_sign_matches, 1);
    final_overall_sign_accuracy = mean(final_sign_matches, 'all');
    
    % Calculate weighted accuracy
    final_weighted_matches = final_sign_matches .* phase_mag_true;
    final_weighted_sign_accuracy = sum(final_weighted_matches(:)) / sum(phase_mag_true(:));
    
    % Calculate improvement from global classifier
    sign_accuracy_improvement = final_overall_sign_accuracy - overall_sign_accuracy;
    weighted_accuracy_improvement = final_weighted_sign_accuracy - weighted_sign_accuracy;
    
    % Store in metrics
    metrics.global_sign_accuracy = mean(global_flip_pred == getGroundTruthFlips(phase_signs_pred, signs_true));
    metrics.final_mode_sign_accuracy = final_mode_sign_accuracy;
    metrics.final_overall_sign_accuracy = final_overall_sign_accuracy;
    metrics.final_weighted_sign_accuracy = final_weighted_sign_accuracy;
    metrics.sign_accuracy_improvement = sign_accuracy_improvement;
    metrics.weighted_accuracy_improvement = weighted_accuracy_improvement;
    
    % 4. Generate and evaluate final reconstructions
    fprintf('Step 4: Evaluating final reconstructions...\n');
    
    % Create complex weights from final predictions
    final_phases_pred = phase_mags_pred .* final_signs_pred;
    weights_pred = utils.createComplexWeights(amps_pred, final_phases_pred);
    weights_true = utils.createComplexWeights(amps_true, phases_true);
    
    % Create ground truth reconstructions once
    image_size = size(X_test, 1);
    recons_true = zeros(size(X_test), 'like', X_test);
    
    % Generate ground truth reconstructions in batches
    for i = 1:numBatches
        startIdx = (i-1)*batchSize + 1;
        endIdx = min(i*batchSize, numSamples);
        
        % Extract batch weights
        batch_weights_true = weights_true(startIdx:endIdx, :);
        
        % Build reconstruction
        [recon_true_batch, ~] = mmf_build_image(number_of_modes, image_size, length(startIdx:endIdx), batch_weights_true, false, 0, P);
        
        % Store reconstructions
        recons_true(:,:,:,startIdx:endIdx) = recon_true_batch;
    end
    
    % Generate predicted reconstructions in batches
    recons_pred = zeros(size(X_test), 'like', X_test);
    correlations = zeros(numSamples, 1);
    
    for i = 1:numBatches
        startIdx = (i-1)*batchSize + 1;
        endIdx = min(i*batchSize, numSamples);
        
        % Extract batch weights
        batch_weights_pred = weights_pred(startIdx:endIdx, :);
        
        % Build reconstruction
        [recon_pred_batch, ~] = mmf_build_image(number_of_modes, image_size, length(startIdx:endIdx), batch_weights_pred, false, 0, P);
        
        % Store reconstructions
        recons_pred(:,:,:,startIdx:endIdx) = recon_pred_batch;
        
        % Calculate correlations with original images
        for j = startIdx:endIdx
            correlations(j) = corr2(extract(X_test(:,:,:,j)), extract(recon_pred_batch(:,:,:,j-startIdx+1)));
        end
    end
    
    % Calculate reconstruction metrics
    recon_mse = mean((recons_pred - X_test).^2, 'all');
    recon_rmse = sqrt(recon_mse);
    recon_corr = mean(correlations);
    
    % Calculate relative improvement over ground truth reconstruction
    gt_mse = mean((recons_true - X_test).^2, 'all');
    gt_rmse = sqrt(gt_mse);
    gt_correlations = zeros(numSamples, 1);
    
    for i = 1:numSamples
        gt_correlations(i) = corr2(extract(X_test(:,:,:,i)), extract(recons_true(:,:,:,i)));
    end
    gt_corr = mean(gt_correlations);
    
    recon_rel_error = recon_rmse / mean(abs(X_test(:)));
    gt_rel_error = gt_rmse / mean(abs(X_test(:)));
    
    % Store reconstruction metrics
    metrics.recon_mse = recon_mse;
    metrics.recon_rmse = recon_rmse;
    metrics.recon_corr = recon_corr;
    metrics.gt_mse = gt_mse;
    metrics.gt_rmse = gt_rmse;
    metrics.gt_corr = gt_corr;
    metrics.recon_rel_error = recon_rel_error;
    metrics.gt_rel_error = gt_rel_error;
    
    % Store reconstructions for a subset of samples
    n_samples = min(options.numSamplesToShow, numSamples);
    sample_indices = randperm(numSamples, n_samples);
    
    reconstructions = struct();
    reconstructions.indices = sample_indices;
    reconstructions.original = X_test(:,:,:,sample_indices);
    reconstructions.predicted = recons_pred(:,:,:,sample_indices);
    reconstructions.ground_truth = recons_true(:,:,:,sample_indices);
    reconstructions.correlations = correlations(sample_indices);
    reconstructions.gt_correlations = gt_correlations(sample_indices);
end

function residuals = generateResiduals(X, amps_pred, phase_mags_pred, phase_signs_pred, number_of_modes, P, batchSize, utils)
    % Generate residuals for global sign classifier
    numSamples = size(X, 4);
    image_size = size(X, 1);
    
    % Initialize residuals
    residuals = zeros(size(X), 'like', X);
    
    % Create complex weights
    phases_pred = phase_mags_pred .* phase_signs_pred;
    weights_pred = utils.createComplexWeights(amps_pred, phases_pred);
    
    % Generate reconstructions in batches
    numBatches = ceil(numSamples/batchSize);
    
    for i = 1:numBatches
        startIdx = (i-1)*batchSize + 1;
        endIdx = min(i*batchSize, numSamples);
        
        % Extract batch weights
        batch_weights = weights_pred(startIdx:endIdx, :);
        
        % Build reconstruction
        [recon_batch, ~] = mmf_build_image(number_of_modes, image_size, length(startIdx:endIdx), batch_weights, false, 0, P);
        
        % Calculate residuals
        residuals(:,:,:,startIdx:endIdx) = X(:,:,:,startIdx:endIdx) - recon_batch;
    end
end

function ground_truth_flips = getGroundTruthFlips(pred_signs, true_signs)
    % Determine if global sign flips are needed
    % 1 = flip needed, 0 = keep as is
    numSamples = size(pred_signs, 1);
    ground_truth_flips = zeros(numSamples, 1);
    
    for i = 1:numSamples
        % Count matches in original vs flipped version
        matches_original = sum(pred_signs(i,:) == true_signs(i,:));
        matches_flipped = sum(-pred_signs(i,:) == true_signs(i,:));
        
        % Determine if flip is needed
        if matches_flipped > matches_original
            ground_truth_flips(i) = 1; % Flip needed
        end
    end
end

function displayResults(metrics, reconstructions, options)
    % Display evaluation results
    
    % 1. Display metrics
    fprintf('\n=== MMF Decomposition Pipeline Evaluation Results ===\n\n');
    
    fprintf('Amplitude Metrics:\n');
    fprintf('  RMSE: %.4f\n', metrics.amp_rmse);
    fprintf('  MAE: %.4f\n', metrics.amp_mae);
    fprintf('  Relative Error: %.2f%%\n', metrics.amp_rel_error * 100);
    
    fprintf('\nPhase Magnitude Metrics:\n');
    fprintf('  RMSE: %.4f\n', metrics.phase_mag_rmse);
    fprintf('  MAE: %.4f\n', metrics.phase_mag_mae);
    fprintf('  Relative Error: %.2f%%\n', metrics.phase_mag_rel_error * 100);
    
    fprintf('\nPhase Sign Metrics:\n');
    fprintf('  Initial Overall Accuracy: %.2f%%\n', metrics.overall_sign_accuracy * 100);
    fprintf('  Initial Weighted Accuracy: %.2f%%\n', metrics.weighted_sign_accuracy * 100);
    fprintf('  Global Classifier Accuracy: %.2f%%\n', metrics.global_sign_accuracy * 100);
    fprintf('  Final Overall Accuracy: %.2f%%\n', metrics.final_overall_sign_accuracy * 100);
    fprintf('  Final Weighted Accuracy: %.2f%%\n', metrics.final_weighted_sign_accuracy * 100);
    fprintf('  Accuracy Improvement: %.2f%%\n', metrics.sign_accuracy_improvement * 100);
    fprintf('  Weighted Accuracy Improvement: %.2f%%\n', metrics.weighted_accuracy_improvement * 100);
    
    fprintf('\nReconstruction Metrics:\n');
    fprintf('  Prediction RMSE: %.4f\n', metrics.recon_rmse);
    fprintf('  Ground Truth RMSE: %.4f\n', metrics.gt_rmse);
    fprintf('  Prediction Correlation: %.4f\n', metrics.recon_corr);
    fprintf('  Ground Truth Correlation: %.4f\n', metrics.gt_corr);
    fprintf('  Prediction Relative Error: %.2f%%\n', metrics.recon_rel_error * 100);
    fprintf('  Ground Truth Relative Error: %.2f%%\n', metrics.gt_rel_error * 100);
    
    % 2. Display visualizations if enabled
    if options.showPlots
        % Create visualization of results for sampled images
        n_samples = length(reconstructions.indices);
        
        % Plot format will depend on number of samples
        if n_samples <= 4
            rows = 1;
            cols = n_samples;
        else
            rows = ceil(n_samples/4);
            cols = min(4, n_samples);
        end
        
        % Plot reconstructions
        figure('Name', 'MMF Decomposition Pipeline Evaluation', 'Position', [100, 100, 1200, 800]);
        
        for i = 1:n_samples
            % Original image
            subplot(3, n_samples, i);
            imagesc(extract(reconstructions.original(:,:,:,i)));
            axis image off;
            if i == 1
                ylabel('Original');
            end
            title(sprintf('Sample #%d', reconstructions.indices(i)));
            
            % Ground truth reconstruction
            subplot(3, n_samples, i + n_samples);
            imagesc(extract(reconstructions.ground_truth(:,:,:,i)));
            axis image off;
            if i == 1
                ylabel('Ground Truth');
            end
            title(sprintf('r=%.3f', reconstructions.gt_correlations(i)));
            
            % Predicted reconstruction
            subplot(3, n_samples, i + 2*n_samples);
            imagesc(extract(reconstructions.predicted(:,:,:,i)));
            axis image off;
            if i == 1
                ylabel('Predicted');
            end
            title(sprintf('r=%.3f', reconstructions.correlations(i)));
        end
        
        % Plot phase sign accuracies
        figure('Name', 'Phase Sign Accuracies', 'Position', [100, 500, 800, 400]);
        
        % Mode-specific accuracies
        subplot(1, 2, 1);
        bar(metrics.final_mode_sign_accuracy * 100);
        xlabel('Mode Number');
        ylabel('Accuracy (%)');
        title('Phase Sign Accuracy by Mode');
        grid on;
        ylim([0, 100]);
        
        % Sign accuracy improvement
        subplot(1, 2, 2);
        bar([metrics.overall_sign_accuracy, metrics.final_overall_sign_accuracy] * 100);
        set(gca, 'XTickLabel', {'Before Global Classifier', 'After Global Classifier'});
        ylabel('Accuracy (%)');
        title('Phase Sign Accuracy Improvement');
        grid on;
        ylim([0, 100]);
    end
end