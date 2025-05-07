function evaluate_full_pipeline(modelFile)
    % EVALUATE_FULL_PIPELINE - Comprehensive evaluation of the MMF pipeline
    %
    % Usage:
    %   evaluate_full_pipeline() - Evaluates with default phase sign model
    %   evaluate_full_pipeline('high_mode_pinn_model.mat') - Evaluates with custom model
    
    % Handle model file input
    if nargin < 1
        phaseModelFile = 'phase_sign_model.mat';
    else
        phaseModelFile = modelFile;
    end
    
    % Check if all required models exist
    if ~exist('amplitude_model.mat', 'file')
        error('Absolute model not found. Please run train_absolute_model.m first.');
    end
    
    if ~exist(phaseModelFile, 'file')
        error('Phase sign model not found: %s', phaseModelFile);
    end
    
    if ~exist('phase_sign_classifier.mat', 'file')
        warning('Global phase sign classifier not found. Skipping global classifier evaluation.');
        evaluateGlobalClassifier = false;
    else
        evaluateGlobalClassifier = true;
    end
    
    % Load dataset
    fprintf('Loading test dataset...\n');
    load('mmf_dataset_multi_sign.mat', 'mmf_test', 'labels_test');
    XTest = mmf_test;
    YTest = labels_test;

    % Load or create precomputed P (mode basis)
    persistent P;
    if isempty(P)
        P = BPMmatlab.model;
        P.name = 'pipeline_eval';
        P.useAllCPUs = true;
        P.useGPU = true;
        P.Lx_main = 50e-6;
        P.Ly_main = 50e-6;
        P.Nx_main = size(XTest,1);
        P.Ny_main = size(XTest,2);
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
    end
    
    % Load models
    fprintf('Loading models...\n');
    load('amplitude_model.mat', 'dlnet_amplitude', 'number_of_modes');
    load(phaseModelFile, 'dlnet_phase');
    
    % Check if using high mode count PINN
    isHighModeCount = contains(lower(phaseModelFile), 'high_mode');
    
    if evaluateGlobalClassifier
        load('phase_sign_classifier.mat', 'global_classifier');
    end
    
    % Determine batch size based on available memory
    if isHighModeCount
        batchSize = min(32, size(XTest, 4)); % Smaller batch size for high mode count
    else
        batchSize = min(64, size(XTest, 4));
    end
    
    % Print info
    fprintf('\n===== MMF Pipeline Evaluation =====\n');
    fprintf('Number of modes: %d\n', number_of_modes);
    fprintf('Test samples: %d\n', size(XTest, 4));
    fprintf('Phase model: %s\n', phaseModelFile);
    fprintf('Using high mode count model: %s\n', string(isHighModeCount));
    fprintf('================================\n\n');
    
    % APPROACH 1: Evaluate amplitude model only (with positive phases)
    fprintf('\nApproach 1: Amplitude model with positive phases\n');
    [YPred_abs, recon_abs, metrics_abs] = evaluateAbsoluteModel(dlnet_amplitude, XTest, YTest, number_of_modes, batchSize);
    
    % APPROACH 2: Evaluate phase sign model
    fprintf('\nApproach 2: Phase sign model\n');
    [YPred_phase, recon_phase, metrics_phase] = evaluatePhaseSignModel(dlnet_amplitude, dlnet_phase, XTest, YTest, number_of_modes, batchSize);
    
    % APPROACH 3: Evaluate with global classifier (if available)
    if evaluateGlobalClassifier
        fprintf('\nApproach 3: Global sign classifier\n');
        % residuals_test = generateNewResiduals(XTest, YPred_phase, number_of_modes, batchSize);
        load('residuals.mat', 'residuals_test');
        [YPred_classifier, recon_classifier, metrics_classifier] = evaluateWithGlobalClassifier(dlnet_amplitude, dlnet_phase, global_classifier, XTest, YTest, residuals_test, number_of_modes, batchSize);
    else
        YPred_classifier = [];
        recon_classifier = [];
        metrics_classifier = struct();
    end
    
    % Compare all approaches
    fprintf('\nComparing all approaches...\n');
    if evaluateGlobalClassifier
        compareApproaches(XTest, YTest, recon_abs, recon_phase, recon_classifier, metrics_abs, metrics_phase, metrics_classifier, number_of_modes);
    else
        compareApproaches(XTest, YTest, recon_abs, recon_phase, [], metrics_abs, metrics_phase, [], number_of_modes);
    end
    
    % Save results
    results = struct();
    results.metrics_abs = metrics_abs;
    results.metrics_phase = metrics_phase;
    if evaluateGlobalClassifier
        results.metrics_classifier = metrics_classifier;
    end
    
    % Use appropriate filename based on model
    if isHighModeCount
        outputFile = 'high_mode_evaluation_results.mat';
    else
        outputFile = 'evaluation_results.mat';
    end
    
    save(outputFile, 'results');
    fprintf('Evaluation results saved to %s\n', outputFile);
    
    % Show examples
    visualizeExamples(XTest, YTest, recon_abs, recon_phase, recon_classifier, number_of_modes);
end

function [YPred, reconstructed, metrics] = evaluateAbsoluteModel(dlnet, XTest, YTest, number_of_modes, batchSize)
    fprintf(' - Evaluating model with absolute values only...\n');
    
    % Initialize
    numBatches = ceil(size(XTest, 4) / batchSize);
    YPred = zeros(size(XTest, 4), 2*number_of_modes - 1);
    reconstructed = zeros(size(XTest));
    
    % Process in batches
    for b = 1:numBatches
        batchIdx = ((b-1)*batchSize + 1):min(b*batchSize, size(XTest, 4));
        
        % Prepare input
        dlX = dlarray(XTest(:,:,:,batchIdx), 'SSCB');
        if canUseGPU()
            dlX = gpuArray(dlX);
        end
        
        % Get predictions
        YPred_raw = predict(dlnet, dlX);
        YPred_raw = extractdata(YPred_raw);
        
        % Extract amplitudes and phases
        amps_pred = YPred_raw(1:number_of_modes, :);
        phase_abs_pred = YPred_raw(number_of_modes+1:end, :);
        
        % ABSOLUTE MODEL APPROACH:
        % - Use the absolute values of predicted amplitudes (normalized to 0...1)
        % - Use the absolute values of phases (normalized to 0...1)
        % - First phase (mode 1) is always zero (reference)
        
        % Ensure amplitudes are positive and properly normalized
        amps_pred = abs(amps_pred);
        
        % Use absolute values of phases (convert from -1...1 to 0...1)
        phase_abs_pred = abs(phase_abs_pred);
        
        % Create full phase array with first phase set to zero (reference mode)
        phase_values = [zeros(1, size(phase_abs_pred, 2)); phase_abs_pred];
        
        % Create complex weights with absolute phase values
        weights = amps_pred .* exp(1i * phase_values * pi);
        
        % Reconstruct
        [recon_batch, ~] = mmf_build_image(number_of_modes, size(XTest,1), length(batchIdx), weights', false, P);
        
        % Store results
        YPred(batchIdx, 1:number_of_modes) = amps_pred';
        YPred(batchIdx, number_of_modes+1:end) = phase_abs_pred';
        reconstructed(:,:,:,batchIdx) = recon_batch;
    end
    
    % Calculate metrics
    metrics = calculateMetrics(XTest, YTest, YPred, reconstructed, number_of_modes);
    
    % Report key metrics
    fprintf('   Mean correlation: %.4f ± %.4f\n', metrics.mean_correlation, metrics.std_correlation);
    fprintf('   Phase sign accuracy: %.2f%%\n', metrics.phase_sign_accuracy*100);
    fprintf('   Relative sign accuracy: %.2f%%\n', metrics.relative_sign_accuracy*100);
end

function [YPred, reconstructed, metrics] = evaluatePhaseSignModel(ampNet, phaseNet, XTest, YTest, number_of_modes, batchSize)
    fprintf(' - Evaluating dedicated phase sign model...\n');
    
    % Initialize
    numBatches = ceil(size(XTest, 4) / batchSize);
    YPred = zeros(size(XTest, 4), 2*number_of_modes - 1);
    reconstructed = zeros(size(XTest));
    
    % Process in batches
    for b = 1:numBatches
        batchIdx = ((b-1)*batchSize + 1):min(b*batchSize, size(XTest, 4));
        
        % Prepare input
        dlX = dlarray(XTest(:,:,:,batchIdx), 'SSCB');
        if canUseGPU()
            dlX = gpuArray(dlX);
        end
        
        % Get amplitude predictions from amplitude model
        YPred_amps = predict(ampNet, dlX);
        YPred_amps = extractdata(YPred_amps);
        
        % Extract amplitudes and phase magnitudes
        amps_pred = YPred_amps(1:number_of_modes, :);
        phase_magnitudes = abs(YPred_amps(number_of_modes+1:end, :));
        
        % Get phase sign predictions from phase sign model
        phase_signs_pred = forward(phaseNet, dlX);
        phase_signs_pred = extractdata(phase_signs_pred);
        
        % Apply tanh and then take sign to convert raw outputs to -1/+1 signs
        phase_signs = sign(tanh(phase_signs_pred));
        
        % CANONICAL FORM HANDLING:
        % 1. Mode 1 has phase 0 (reference mode)
        % 2. Mode 2 has phase sign always +1 in canonical form
        % 3. Modes 3...N have their signs predicted by the network
        
        % Check if we're predicting signs for modes 3...N (N-2 signs in total)
        if size(phase_signs, 1) == number_of_modes - 2
            % Phase signs model predicts only for modes 3...N
            % Mode 2's sign is fixed to +1 in canonical form
            all_signs = [ones(1, size(phase_signs, 2)); phase_signs]; % [1, pred_signs]
        else
            % The model predicts all signs, but we enforce canonical constraints
            all_signs = phase_signs;
            all_signs(1, :) = ones(1, size(phase_signs, 2)); % Force mode 2 sign to +1
        end
        
        % Apply signs to phase magnitudes
        phase_values = phase_magnitudes .* all_signs;

        % TEMP: Use ground truth signs for evaluation
        % phase_values = phase_magnitudes .* sign(YTest(batchIdx, number_of_modes+1:end)');
        
        % Complete phase array with 0 for reference mode (mode 1)
        full_phases = [zeros(1, size(phase_values, 2)); phase_values];
        
        % Create complex weights
        weights = amps_pred .* exp(1i * full_phases * pi);
        
        % Reconstruct
        [recon_batch, ~] = mmf_build_image(number_of_modes, size(XTest,1), length(batchIdx), weights', false, P);
        
        % Store results
        YPred(batchIdx, 1:number_of_modes) = amps_pred';
        YPred(batchIdx, number_of_modes+1:end) = phase_values';
        reconstructed(:,:,:,batchIdx) = recon_batch;
    end
    
    % Calculate metrics
    metrics = calculateMetrics(XTest, YTest, YPred, reconstructed, number_of_modes);
    
    % Report key metrics
    fprintf('   Mean correlation: %.4f ± %.4f\n', metrics.mean_correlation, metrics.std_correlation);
    fprintf('   Phase sign accuracy: %.2f%%\n', metrics.phase_sign_accuracy*100);
    fprintf('   Relative sign accuracy: %.2f%%\n', metrics.relative_sign_accuracy*100);
end

function [YPred, reconstructed, metrics] = evaluateWithGlobalClassifier(ampNet, phaseNet, global_classifier, XTest, YTest, residuals, number_of_modes, batchSize)
    fprintf(' - Evaluating with global sign classifier...\n');
    
    % Initialize
    numBatches = ceil(size(XTest, 4) / batchSize);
    YPred = zeros(size(XTest, 4), 2*number_of_modes - 1);
    reconstructed = zeros(size(XTest));
    
    % Process in batches
    for b = 1:numBatches
        batchIdx = ((b-1)*batchSize + 1):min(b*batchSize, size(XTest, 4));
        
        % Prepare input
        dlX = dlarray(XTest(:,:,:,batchIdx), 'SSCB');
        if canUseGPU()
            dlX = gpuArray(dlX);
        end
        
        % Step 1: Get amplitude predictions from amplitude model
        YPred_amps = forward(ampNet, dlX);
        YPred_amps = extractdata(YPred_amps);
        
        % Extract amplitudes and phase magnitudes
        amps_pred = YPred_amps(1:number_of_modes, :);
        phase_magnitudes = abs(YPred_amps(number_of_modes+1:end, :));
        
        % Step 2: Get phase sign predictions from phase sign model
        phase_signs_pred = forward(phaseNet, dlX);
        phase_signs_pred = extractdata(phase_signs_pred);
        
        % Apply tanh and then take sign to convert raw outputs to -1/+1 signs
        phase_signs = sign(tanh(phase_signs_pred));
        
        % Handle canonical form (identical to evaluatePhaseSignModel)
        if size(phase_signs, 1) == number_of_modes - 2
            % Phase signs model predicts only for modes 3...N
            % Mode 2's sign is fixed to +1 in canonical form
            all_signs = [ones(1, size(phase_signs, 2)); phase_signs]; % [1, pred_signs]
        else
            % The model predicts all signs, but we enforce canonical constraints
            all_signs = phase_signs;
            all_signs(1, :) = ones(1, size(phase_signs, 2)); % Force mode 2 sign to +1
        end
        
        % Apply signs to phase magnitudes
        phase_values = phase_magnitudes .* all_signs;

        
        YPred(batchIdx, 1:number_of_modes) = amps_pred';
        YPred(batchIdx, number_of_modes+1:end) = phase_values';
        
        % Step 3: Get global sign flip decision from global classifier
        dlResiduals = generateNewResiduals(dlX, YPred(batchIdx, :), number_of_modes, batchSize);
        dlResiduals = dlarray(dlResiduals, 'SSCB');
        global_flip = forward(global_classifier, dlResiduals);
        global_flip = extractdata(global_flip > 0.5);

        % TEMP: Use ground truth sign for evaluation
        % global_flip = (sign(YTest(batchIdx, number_of_modes+1)') < 0);

        % Complete phase array with 0 for reference mode
        full_phases = [zeros(1, size(phase_values, 2)); phase_values];
        
        % Apply global flip to all non-reference modes (modes 2...N)
        for i = 1:size(global_flip, 2)
            if global_flip(i)
                % Flip all phases
                full_phases(:, i) = -full_phases(:, i);
            end
        end        

        % Create complex weights
        weights = amps_pred .* exp(1i * full_phases * pi);
        
        % Reconstruct
        [recon_batch, ~] = mmf_build_image(number_of_modes, size(XTest,1), length(batchIdx), weights', false, P);
        
        % Store results
        YPred(batchIdx, 1:number_of_modes) = amps_pred';
        YPred(batchIdx, number_of_modes+1:end) = full_phases(2:end, :)';
        reconstructed(:,:,:,batchIdx) = recon_batch;
    end
    
    % Calculate metrics
    metrics = calculateMetrics(XTest, YTest, YPred, reconstructed, number_of_modes);
    
    % Report key metrics
    fprintf('   Mean correlation: %.4f ± %.4f\n', metrics.mean_correlation, metrics.std_correlation);
    fprintf('   Phase sign accuracy: %.2f%%\n', metrics.phase_sign_accuracy*100);
    fprintf('   Relative sign accuracy: %.2f%%\n', metrics.relative_sign_accuracy*100);
end

function residuals = generateNewResiduals(XTest, YPred_phase, number_of_modes, batchSize)
    % Generate residuals as the difference between nonlinear images in XTest and linear reconstructions of phase sign model predictions
    numSamples = size(XTest, 4);
    numBatches = ceil(numSamples / batchSize);
    residuals = zeros(size(XTest));
    image_size = size(XTest, 1);
    
    for b = 1:numBatches
        batchIdx = ((b-1)*batchSize + 1):min(b*batchSize, numSamples);
        % --- Nonlinear image from XTest ---
        nonlinear_img = XTest(:,:,:,batchIdx);
        % --- Phase sign model prediction reconstruction (linear) ---
        amps_pred = YPred_phase(batchIdx, 1:number_of_modes)';
        phases_pred = YPred_phase(batchIdx, number_of_modes+1:end)';
        phases_pred_full = [zeros(1, size(phases_pred,2)); phases_pred];
        weights_pred = amps_pred .* exp(1i * phases_pred_full * pi);
        [recon_pred, ~] = mmf_build_image(number_of_modes, image_size, length(batchIdx), weights_pred, false, P);
        % --- Residuals ---
        residuals(:,:,:,batchIdx) = nonlinear_img - recon_pred;
    end
end

function metrics = calculateMetrics(XTest, YTest, YPred, reconstructed, number_of_modes)
    % Initialize metrics struct
    metrics = struct();
    
    % Calculate image correlation
    num_test = size(XTest, 4);
    correlations = zeros(num_test, 1);
    
    for i = 1:num_test
        correlations(i) = corr2(XTest(:,:,1,i), reconstructed(:,:,1,i));
    end
    
    % Extract amplitudes and phases
    pred_amplitudes = YPred(:, 1:number_of_modes);
    true_amplitudes = YTest(:, 1:number_of_modes);
    pred_phases = YPred(:, number_of_modes+1:end);
    true_phases = YTest(:, number_of_modes+1:end);
    
    % Calculate amplitude errors
    amplitude_rmse = sqrt(mean((pred_amplitudes - true_amplitudes).^2, 'all'));
    
    % Calculate phase errors
    phase_error = abs(pred_phases - true_phases);
    phase_rmse = sqrt(mean(phase_error.^2, 'all'));
    
    % Calculate phase sign accuracy
    phase_sign_correct = (sign(pred_phases) == sign(true_phases));
    phase_sign_accuracy = mean(phase_sign_correct(:));
    per_mode_sign_accuracy = mean(phase_sign_correct, 1);
    
    % Calculate relative sign accuracy (allowing global ambiguity)
    [relative_sign_accuracy, per_mode_rel_accuracy] = calculateRelativeSignAccuracy(pred_phases, true_phases);
    
    % Store results in metrics struct
    metrics.correlations = correlations;
    metrics.mean_correlation = mean(correlations);
    metrics.median_correlation = median(correlations);
    metrics.std_correlation = std(correlations);
    metrics.amplitude_rmse = amplitude_rmse;
    metrics.phase_rmse = phase_rmse;
    metrics.phase_sign_accuracy = phase_sign_accuracy;
    metrics.per_mode_sign_accuracy = per_mode_sign_accuracy;
    metrics.relative_sign_accuracy = relative_sign_accuracy;
    metrics.per_mode_rel_accuracy = per_mode_rel_accuracy;
    
    % Store the predicted and true values for further analysis
    metrics.pred_amplitudes = pred_amplitudes;
    metrics.true_amplitudes = true_amplitudes;
    metrics.pred_phases = pred_phases;
    metrics.true_phases = true_phases;
end

function [accuracy, per_mode_accuracy] = calculateRelativeSignAccuracy(pred_phases, true_phases)
    % Calculate sign accuracy allowing for global phase ambiguity
    num_samples = size(pred_phases, 1);
    num_modes = size(pred_phases, 2);
    
    % Initialize tracking
    per_mode_accuracy = zeros(1, num_modes);
    total_correct = 0;
    
    for i = 1:num_samples
        pred_signs = sign(pred_phases(i, :));
        true_signs = sign(true_phases(i, :));
        
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
        per_mode_accuracy = per_mode_accuracy + correct_for_sample;
        total_correct = total_correct + sum(correct_for_sample);
    end
    
    % Normalize
    per_mode_accuracy = per_mode_accuracy / num_samples;
    accuracy = total_correct / (num_samples * num_modes);
end

function compareApproaches(XTest, YTest, recon_abs, recon_phase, recon_classifier, metrics_abs, metrics_phase, metrics_classifier, number_of_modes)
    % Create comparison figure
    figure('Name', 'Pipeline Comparison', 'Position', [100, 100, 1200, 800]);
    
    % 1. Correlation comparison
    subplot(2, 3, 1);
    if ~isempty(metrics_classifier)
        boxplot([metrics_abs.correlations, metrics_phase.correlations, metrics_classifier.correlations], ...
            'Labels', {'Abs Only', 'Phase Model', 'Global Classifier'});
        colors = {'r', 'g', 'b'};
    else
        boxplot([metrics_abs.correlations, metrics_phase.correlations], ...
            'Labels', {'Abs Only', 'Phase Model'});
        colors = {'r', 'g'};
    end
    
    % Apply box colors
    h = findobj(gca, 'Tag', 'Box');
    for j = 1:length(h)
        patch(get(h(j), 'XData'), get(h(j), 'YData'), colors{j}, 'FaceAlpha', 0.5);
    end
    
    title('Image Correlation');
    ylabel('Correlation');
    grid on;
    
    % 2. Phase sign accuracy by mode
    subplot(2, 3, 2);
    hold on;
    
    % Collect data for bar chart
    bar_data = [metrics_abs.per_mode_sign_accuracy' * 100, ...
               metrics_phase.per_mode_sign_accuracy' * 100];
    
    if ~isempty(metrics_classifier)
        bar_data = [bar_data, metrics_classifier.per_mode_sign_accuracy' * 100];
    end
    
    bar(bar_data);
    
    title('Phase Sign Accuracy by Mode');
    xlabel('Mode');
    ylabel('Accuracy (%)');
    
    if ~isempty(metrics_classifier)
        legend({'Abs Only', 'Phase Model', 'Global Classifier'}, 'Location', 'best');
    else
        legend({'Abs Only', 'Phase Model'}, 'Location', 'best');
    end
    
    ylim([0 100]);
    grid on;
    hold off;
    
    % 3. Relative improvement
    subplot(2, 3, 3);
    
    % Define metrics to compare
    metrics_labels = {'Correlation', 'Amp RMSE', 'Phase RMSE', 'Sign Acc.', 'Rel. Sign Acc.'};
    
    % Collect absolute metrics
    abs_values = [metrics_abs.mean_correlation, ...
                 metrics_abs.amplitude_rmse, ...
                 metrics_abs.phase_rmse, ...
                 metrics_abs.phase_sign_accuracy, ...
                 metrics_abs.relative_sign_accuracy];
    
    % Calculate improvement from phase sign model
    phase_values = [metrics_phase.mean_correlation, ...
                   metrics_phase.amplitude_rmse, ...
                   metrics_phase.phase_rmse, ...
                   metrics_phase.phase_sign_accuracy, ...
                   metrics_phase.relative_sign_accuracy];
    
    % Calculate relative improvement - handle different directions
    phase_improvement = zeros(1, 5);
    
    % For correlation and accuracies - higher is better
    phase_improvement(1) = (phase_values(1) - abs_values(1)) / abs(abs_values(1)) * 100;
    phase_improvement(4) = (phase_values(4) - abs_values(4)) / abs(abs_values(4)) * 100;
    phase_improvement(5) = (phase_values(5) - abs_values(5)) / abs(abs_values(5)) * 100;
    
    % For error metrics - lower is better
    phase_improvement(2) = -(phase_values(2) - abs_values(2)) / abs(abs_values(2)) * 100;
    phase_improvement(3) = -(phase_values(3) - abs_values(3)) / abs(abs_values(3)) * 100;
    
    % Add classifier improvement if available
    if ~isempty(metrics_classifier)
        classifier_values = [metrics_classifier.mean_correlation, ...
                            metrics_classifier.amplitude_rmse, ...
                            metrics_classifier.phase_rmse, ...
                            metrics_classifier.phase_sign_accuracy, ...
                            metrics_classifier.relative_sign_accuracy];
        
        classifier_improvement = zeros(1, 5);
        
        % Same logic as above
        classifier_improvement(1) = (classifier_values(1) - abs_values(1)) / abs(abs_values(1)) * 100;
        classifier_improvement(4) = (classifier_values(4) - abs_values(4)) / abs(abs_values(4)) * 100;
        classifier_improvement(5) = (classifier_values(5) - abs_values(5)) / abs(abs_values(5)) * 100;
        
        classifier_improvement(2) = -(classifier_values(2) - abs_values(2)) / abs(abs_values(2)) * 100;
        classifier_improvement(3) = -(classifier_values(3) - abs_values(3)) / abs(abs_values(3)) * 100;
        
        % Plot both improvements
        bar([phase_improvement; classifier_improvement]');
        legend({'Phase Model', 'Global Classifier'});
    else
        % Plot only phase model improvement
        bar(phase_improvement);
        legend({'Phase Model'});
    end
    
    title('Improvement over Absolute Model (%)');
    xticklabels(metrics_labels);
    ylabel('Improvement (%)');
    grid on;
    
    % Add text annotations for improvements
    if ~isempty(metrics_classifier)
        for i = 1:length(phase_improvement)
            text(i-0.15, phase_improvement(i) + sign(phase_improvement(i))*3, ...
                sprintf('%+.1f%%', phase_improvement(i)), ...
                'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 8);
            
            text(i+0.15, classifier_improvement(i) + sign(classifier_improvement(i))*3, ...
                sprintf('%+.1f%%', classifier_improvement(i)), ...
                'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 8);
        end
    else
        for i = 1:length(phase_improvement)
            text(i, phase_improvement(i) + sign(phase_improvement(i))*3, ...
                sprintf('%+.1f%%', phase_improvement(i)), ...
                'HorizontalAlignment', 'center', 'FontWeight', 'bold');
        end
    end
    
    % 4. Error distributions
    subplot(2, 3, 4);
    hold on;
    histogram(metrics_abs.correlations, 20, 'FaceColor', 'r', 'FaceAlpha', 0.5);
    histogram(metrics_phase.correlations, 20, 'FaceColor', 'g', 'FaceAlpha', 0.5);
    
    if ~isempty(metrics_classifier)
        histogram(metrics_classifier.correlations, 20, 'FaceColor', 'b', 'FaceAlpha', 0.5);
        legend({'Abs Only', 'Phase Model', 'Global Classifier'}, 'Location', 'northwest');
    else
        legend({'Abs Only', 'Phase Model'}, 'Location', 'northwest');
    end
    
    title('Correlation Distribution');
    xlabel('Correlation');
    ylabel('Count');
    grid on;
    hold off;
    
    % 5. Create summary text
    subplot(2, 3, 5);
    
    if ~isempty(metrics_classifier)
        summary_text = sprintf(['Pipeline Performance Summary:\n\n', ...
            'Absolute Only Model:\n', ...
            '  Correlation: %.4f ± %.4f\n', ...
            '  Amp RMSE: %.4f\n', ...
            '  Phase RMSE: %.4f\n', ...
            '  Sign Acc: %.2f%%\n', ...
            '  Rel Sign Acc: %.2f%%\n\n', ...
            'Phase Sign Model:\n', ...
            '  Correlation: %.4f ± %.4f\n', ...
            '  Amp RMSE: %.4f\n', ...
            '  Phase RMSE: %.4f\n', ...
            '  Sign Acc: %.2f%%\n', ...
            '  Rel Sign Acc: %.2f%%\n\n', ...
            'Global Classifier:\n', ...
            '  Correlation: %.4f ± %.4f\n', ...
            '  Amp RMSE: %.4f\n', ...
            '  Phase RMSE: %.4f\n', ...
            '  Sign Acc: %.2f%%\n', ...
            '  Rel Sign Acc: %.2f%%\n'], ...
            metrics_abs.mean_correlation, metrics_abs.std_correlation, ...
            metrics_abs.amplitude_rmse, metrics_abs.phase_rmse, ...
            metrics_abs.phase_sign_accuracy*100, metrics_abs.relative_sign_accuracy*100, ...
            metrics_phase.mean_correlation, metrics_phase.std_correlation, ...
            metrics_phase.amplitude_rmse, metrics_phase.phase_rmse, ...
            metrics_phase.phase_sign_accuracy*100, metrics_phase.relative_sign_accuracy*100, ...
            metrics_classifier.mean_correlation, metrics_classifier.std_correlation, ...
            metrics_classifier.amplitude_rmse, metrics_classifier.phase_rmse, ...
            metrics_classifier.phase_sign_accuracy*100, metrics_classifier.relative_sign_accuracy*100);
    else
        summary_text = sprintf(['Pipeline Performance Summary:\n\n', ...
            'Absolute Only Model:\n', ...
            '  Correlation: %.4f ± %.4f\n', ...
            '  Amp RMSE: %.4f\n', ...
            '  Phase RMSE: %.4f\n', ...
            '  Sign Acc: %.2f%%\n', ...
            '  Rel Sign Acc: %.2f%%\n\n', ...
            'Phase Sign Model:\n', ...
            '  Correlation: %.4f ± %.4f\n', ...
            '  Amp RMSE: %.4f\n', ...
            '  Phase RMSE: %.4f\n', ...
            '  Sign Acc: %.2f%%\n', ...
            '  Rel Sign Acc: %.2f%%\n'], ...
            metrics_abs.mean_correlation, metrics_abs.std_correlation, ...
            metrics_abs.amplitude_rmse, metrics_abs.phase_rmse, ...
            metrics_abs.phase_sign_accuracy*100, metrics_abs.relative_sign_accuracy*100, ...
            metrics_phase.mean_correlation, metrics_phase.std_correlation, ...
            metrics_phase.amplitude_rmse, metrics_phase.phase_rmse, ...
            metrics_phase.phase_sign_accuracy*100, metrics_phase.relative_sign_accuracy*100);
    end
    
    text(0.5, 0.5, summary_text, 'HorizontalAlignment', 'center', ...
        'FontSize', 9, 'VerticalAlignment', 'middle');
    axis off;
    
    % 6. Modal intensity patterns for high-mode fibers
    subplot(2, 3, 6);
    
    % Calculate mode power distribution
    mode_powers_true = mean(YTest(:, 1:number_of_modes).^2, 1);
    mode_powers_pred_abs = mean(metrics_abs.pred_amplitudes.^2, 1);
    mode_powers_pred_phase = mean(metrics_phase.pred_amplitudes.^2, 1);
    
    % Plot mode intensities
    hold on;
    plot(1:number_of_modes, mode_powers_true, 'k-', 'LineWidth', 2);
    plot(1:number_of_modes, mode_powers_pred_abs, 'r--', 'LineWidth', 1.5);
    plot(1:number_of_modes, mode_powers_pred_phase, 'g-.', 'LineWidth', 1.5);
    
    if ~isempty(metrics_classifier)
        mode_powers_pred_classifier = mean(metrics_classifier.pred_amplitudes.^2, 1);
        plot(1:number_of_modes, mode_powers_pred_classifier, 'b:', 'LineWidth', 1.5);
        legend({'Ground Truth', 'Abs Only', 'Phase Model', 'Global Classifier'}, 'Location', 'best');
    else
        legend({'Ground Truth', 'Abs Only', 'Phase Model'}, 'Location', 'best');
    end
    
    title('Mode Power Distribution');
    xlabel('Mode Index');
    ylabel('Mean Power');
    grid on;
    hold off;
    
    % Set title for the entire figure
    sgtitle('MMF Pipeline Evaluation', 'FontSize', 14);
end

function visualizeExamples(XTest, YTest, recon_abs, recon_phase, recon_classifier, number_of_modes)
    % Select a few examples to visualize
    numExamples = 3;
    figure('Name', 'Example Reconstructions', 'Position', [150, 150, 1200, 400]);
    
    % Pick examples with different performance
    if ~isempty(recon_classifier)
        % We have all three methods
        rows = 4;
        labels = {'Original', 'Abs Only', 'Phase Model', 'Global Classifier'};
    else
        % We have only abs and phase models
        rows = 3;
        labels = {'Original', 'Abs Only', 'Phase Model'};
    end
    
    % Try to pick diverse examples (good, medium, poor reconstruction)
    example_indices = findDiverseExamples(XTest, recon_phase, numExamples);
    
    for i = 1:numExamples
        idx = example_indices(i);
        
        % Original image
        subplot(rows, numExamples, i);
        imagesc(XTest(:,:,1,idx));
        axis image off;
        title(sprintf('Original #%d', idx));
        
        % Absolute model reconstruction
        subplot(rows, numExamples, i+numExamples);
        imagesc(recon_abs(:,:,1,idx));
        axis image off;
        corr_abs = corr2(XTest(:,:,1,idx), recon_abs(:,:,1,idx));
        title(sprintf('Abs Only (r=%.3f)', corr_abs));
        
        % Phase model reconstruction
        subplot(rows, numExamples, i+2*numExamples);
        imagesc(recon_phase(:,:,1,idx));
        axis image off;
        corr_phase = corr2(XTest(:,:,1,idx), recon_phase(:,:,1,idx));
        title(sprintf('Phase Model (r=%.3f)', corr_phase));
        
        % Global classifier reconstruction (if available)
        if ~isempty(recon_classifier)
            subplot(rows, numExamples, i+3*numExamples);
            imagesc(recon_classifier(:,:,1,idx));
            axis image off;
            corr_classifier = corr2(XTest(:,:,1,idx), recon_classifier(:,:,1,idx));
            title(sprintf('Global Class (r=%.3f)', corr_classifier));
        end
    end
    
        % ...existing code...
    for i = 1:rows
        pos = get(subplot(rows, numExamples, (i-1)*numExamples+1), 'Position');
        % Clamp position values to [0, 1]
        x = max(min(pos(1)-0.15, 1), 0);
        y = max(min(pos(2), 1), 0);
        w = max(min(0.1, 1-x), 0);
        h = max(min(pos(4), 1-y), 0);
        annotation('textbox', [x, y, w, h], ...
            'String', labels{i}, 'EdgeColor', 'none', ...
            'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'right');
    end
    
    % Overall title
    sgtitle('Example Reconstructions', 'FontSize', 14);
end

function indices = findDiverseExamples(XTest, reconstructed, numExamples)
    % Find examples with varying reconstruction quality
    numTotal = size(XTest, 4);
    correlations = zeros(numTotal, 1);
    
    for i = 1:numTotal
        correlations(i) = corr2(XTest(:,:,1,i), reconstructed(:,:,1,i));
    end
    
    % Sort correlations
    [sorted_corrs, sorted_idx] = sort(correlations);
    
    % Pick examples: one bad, one medium, one good
    indices = zeros(numExamples, 1);
    
    % Find the worst example (but not too bad)
    worst_idx = find(sorted_corrs > 0.1, 1, 'first');
    if isempty(worst_idx)
        worst_idx = 1;
    end
    indices(1) = sorted_idx(worst_idx);
    
    % Find a medium example
    med_idx = round(length(sorted_corrs) / 2);
    indices(2) = sorted_idx(med_idx);
    
    % Find a good example (but not too perfect)
    best_idx = find(sorted_corrs < 0.98, 1, 'last');
    if isempty(best_idx)
        best_idx = length(sorted_corrs);
    end
    indices(3) = sorted_idx(best_idx);
    
    % Ensure we have unique examples
    if length(unique(indices)) < numExamples
        % Just pick evenly spaced examples
        step = floor(numTotal / (numExamples + 1));
        indices = step * (1:numExamples)';
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