% filepath: c:\Users\nicks\MATLAB\Projects\MT2_prak\Versuch 3\train_phase_sign_model.m
% train_phase_sign_model.m - Trains a specialized network for predicting canonical phase signs

function dlnet = train_phase_sign_model(XTrain, YTrain_amps, YTrain_phases, XValidation, YValidation_amps, YValidation_phases, options)
    % This function trains a specialized network that focuses only on predicting phase signs
    % where the goal is to maximize correlation between the reconstructed field and the input.
    %
    % Inputs:
    %   XTrain - Training images [height x width x 1 x samples]
    %   YTrain_amps - Training amplitude values [samples x modes]
    %   YTrain_phases - Training phase values [samples x (modes-1)]
    %   XValidation - Validation images [height x width x 1 x samples]
    %   YValidation_amps - Validation amplitude values [samples x modes]
    %   YValidation_phases - Validation phase values [samples x (modes-1)]
    %   options - Struct containing training parameters
    %
    % Output:
    %   dlnet - Trained network for phase sign prediction
    
    % Parse options or use defaults
    if nargin < 7
        options = struct();
    end
    
    % Default options
    if ~isfield(options, 'initialLearnRate'), options.initialLearnRate = 1e-4; end
    if ~isfield(options, 'maxEpochs'), options.maxEpochs = 300; end
    if ~isfield(options, 'miniBatchSize'), options.miniBatchSize = 64; end
    if ~isfield(options, 'validationFrequency'), options.validationFrequency = 50; end
    if ~isfield(options, 'validationPatience'), options.validationPatience = 40; end
    if ~isfield(options, 'executionEnvironment'), options.executionEnvironment = "gpu"; end
    if ~isfield(options, 'modelType'), options.modelType = "PhaseSignCNN"; end
    if ~isfield(options, 'plotProgress'), options.plotProgress = "gui"; end
    if ~isfield(options, 'evaluationFrequency'), options.evaluationFrequency = 200; end
    if ~isfield(options, 'useFixedAmplitudes'), options.useFixedAmplitudes = true; end
    
    % Define network parameters from data
    inputSize = [size(XTrain,1) size(XTrain,2)];
    number_of_modes = size(YTrain_amps, 2);
    % For N modes: N-1 phases, but only N-2 phase signs are predicted (canonical form)
    outputSize = number_of_modes - 2; % Only predicting phase signs for modes 3..N

    fprintf('Creating phase sign network for %d modes (%d phase signs to predict)...\n', number_of_modes, outputSize);
    
    % Create network based on specified type
    if strcmpi(options.modelType, "PhaseSignCNN")
        dlnet = createPhaseSignCNN(inputSize, outputSize);
    elseif strcmpi(options.modelType, "PhaseSignMLP")
        dlnet = createPhaseSignMLP(inputSize, outputSize);
    elseif strcmpi(options.modelType, "PhaseSignResNet")
        dlnet = createPhaseSignResNet(inputSize, outputSize);
    elseif strcmpi(options.modelType, "HighModePINN")
        dlnet = createHighModePINN(inputSize, outputSize, number_of_modes);
    elseif strcmpi(options.modelType, "load")
        load('phase_sign_model.mat', 'dlnet_phase');
        dlnet = dlnet_phase;
        disp('Loaded existing phase sign model from phase_sign_model.mat');
    else
        error('Unsupported model type: %s', options.modelType);
    end

    % Analyze the network structure
    analyzeNetwork(dlnet);
    
    % Initialize training progress visualization
    if strcmpi(options.plotProgress, "gui")
        metrics = ["TotalLoss", "ValidationLoss", "SignAccuracy", "ValSignAccuracy", "Correlation", "ValCorrelation"];
        
        monitor = trainingProgressMonitor(Metrics=metrics, ...
            Info=["Epoch", "LearningRate"], ...
            XLabel="Iteration");

        % Group plots
        groupSubPlot(monitor, "Losses", ["TotalLoss", "ValidationLoss"]);
        groupSubPlot(monitor, "Accuracy", ["SignAccuracy", "ValSignAccuracy"]);
        groupSubPlot(monitor, "Correlation", ["Correlation", "ValCorrelation"]);
    end
    
    % Initialize training state
    iteration = 0;
    epoch = 0;
    bestValidationLoss = Inf;
    bestNet = dlnet;
    patienceCounter = 0;
    stopTraining = false;
    
    % Initialize optimization variables
    averageGrad = [];
    averageSqGrad = [];
    
    % Create data indices for batching
    numTrain = size(XTrain, 4);
    trainIdx = 1:numTrain;
    
    % Training loop
    while epoch < options.maxEpochs && patienceCounter < options.validationPatience && ~stopTraining
        epoch = epoch + 1;
        
        % Shuffle training data
        trainIdx = trainIdx(randperm(length(trainIdx)));
        
        % Loop over mini-batches
        for i = 1:floor(numTrain/options.miniBatchSize)
            iteration = iteration + 1;
            
            % Get mini-batch
            batchIdx = trainIdx((i-1)*options.miniBatchSize+1:min(i*options.miniBatchSize,numTrain));
            X = XTrain(:,:,:,batchIdx);
            Y_amps = YTrain_amps(batchIdx,:);
            Y_phases = YTrain_phases(batchIdx, :); 
            
            % Convert to dlarray and transfer to GPU if needed
            dlX = dlarray(X, 'SSCB');
            if (options.executionEnvironment == "auto" && canUseGPU) || options.executionEnvironment == "gpu"
                dlX = gpuArray(dlX);
            end
            
            % Train with model gradients focused on phase sign prediction
            if strcmpi(options.modelType, "HighModePINN")
                [gradients, loss, signAccuracy, correlation] = dlfeval(@modelGradients_highModePINN, dlnet, dlX, Y_amps, Y_phases, number_of_modes, options);
            else
                [gradients, loss, signAccuracy, correlation] = dlfeval(@modelGradients_phaseSign, dlnet, dlX, Y_amps, Y_phases, number_of_modes);
            end
            
            % Apply gradient clipping
            gradientThreshold = 5e-3;
            gradients = dlupdate(@(g) thresholdL2Norm(g, gradientThreshold), gradients);
            
            % Update weights
            [dlnet, averageGrad, averageSqGrad] = adamupdate(dlnet, gradients, ...
                averageGrad, averageSqGrad, iteration, options.initialLearnRate);
            
            % Log training progress
            if strcmpi(options.plotProgress, "gui")
                recordMetrics(monitor, iteration, ...
                    TotalLoss=extractdata(loss), ...
                    SignAccuracy=extractdata(signAccuracy), ...
                    Correlation=extractdata(correlation));
                
                updateInfo(monitor, Epoch=epoch, LearningRate=options.initialLearnRate);
                stopTraining = monitor.Stop;
            end
            
            % Validation check
            if mod(iteration, options.validationFrequency) == 0
                [validationLoss, valSignAccuracy, valCorrelation] = validatePhaseSignModel(dlnet, XValidation, YValidation_amps, YValidation_phases, ...
                    options.miniBatchSize, options.executionEnvironment, number_of_modes);
                
                % Early stopping check
                if validationLoss < bestValidationLoss
                    bestValidationLoss = validationLoss;
                    patienceCounter = 0;
                    bestNet = dlnet;
                    fprintf('Iteration %d: Best validation loss so far: %.4f (sign acc: %.2f%%, corr: %.4f)\n', ...
                        iteration, validationLoss, valSignAccuracy*100, valCorrelation);
                else
                    patienceCounter = patienceCounter + 1;
                end
                
                % Log validation metrics
                if strcmpi(options.plotProgress, "gui")
                    recordMetrics(monitor, iteration, ...
                        ValidationLoss=validationLoss, ...
                        ValSignAccuracy=valSignAccuracy, ...
                        ValCorrelation=valCorrelation);
                end
            end
            
            % Intermediate evaluation
            if options.evaluationFrequency > 0 && mod(iteration, options.evaluationFrequency) == 0
                performPhaseSignEvaluation(dlnet, XValidation, YValidation_amps, YValidation_phases, number_of_modes);
            end
        end
        
        if patienceCounter >= options.validationPatience || stopTraining
            break;
        end
    end
    
    % Use best network
    dlnet = bestNet;
    
    % Save model
    save('phase_sign_model.mat', 'dlnet', 'number_of_modes');
    disp('Phase sign model saved to phase_sign_model.mat');
end

function [gradients, loss, signAccuracy, correlation] = modelGradients_phaseSign(dlnet, dlX, Y_amps, Y_phases, number_of_modes)
    % Forward pass to get phase sign predictions
    dlY_pred = forward(dlnet, dlX);
    dlY_pred = real(dlY_pred); % Ensure real values for backprop
    
    Y_amps = Y_amps';
    Y_phases = Y_phases' * pi;

    % Get ground truth sign values - assume input phases are in canonical form
    true_signs = sign(Y_phases);
    
    % Use tanh activation to constrain predictions to [-1, 1]
    pred_signs = tanh(dlY_pred);
    
    % Convert to binary signs for accuracy calculation
    pred_signs_binary = sign(pred_signs);
    
    % Make sure we have the right number of signs for the canonical form
    % If we're predicting signs for modes 3..N (N-2 signs), add the +1 sign for mode 2
    if size(pred_signs_binary, 1) == number_of_modes - 2
        % First mode (reference) has 0 phase, not included
        % Second mode's sign is always +1 in canonical form
        % We're predicting signs for modes 3..N (which is N-2 signs)
        pred_signs_canonical = [ones(1, size(pred_signs_binary, 2), 'like', pred_signs_binary); 
                              pred_signs_binary];
    else
        % If we're predicting all N-1 signs but working in canonical form,
        % force the first sign to be +1
        pred_signs_canonical = pred_signs_binary;
        pred_signs_canonical(1, :) = ones(1, size(pred_signs_binary, 2), 'like', pred_signs_binary);
    end
    
    % Calculate sign accuracy - considering all phase signs (canonical form)
    correct = sum(pred_signs_canonical == true_signs, 'all');
    total = numel(true_signs);
    signAccuracy = correct / total;
    
    % Create complex weights for reconstructions
    % True weights from ground truth
    true_weights = zeros(number_of_modes, size(Y_amps, 2), 'like', Y_amps);
    true_weights(1, :) = Y_amps(1, :); % Reference mode (always real)
    
    for m = 2:number_of_modes
        phase_idx = m - 1;
        true_weights(m, :) = Y_amps(m, :) .* exp(1i * Y_phases(phase_idx, :));
    end
    
    % Predicted weights using predicted signs with true amplitudes and phase magnitudes
    pred_weights = zeros(number_of_modes, size(Y_amps, 2), 'like', Y_amps);
    pred_weights(1, :) = Y_amps(1, :); % Reference mode (always real)
    
    for m = 2:number_of_modes
        phase_idx = m - 1;
        if phase_idx <= size(pred_signs_canonical, 1)
            % Use magnitude of true phase with predicted sign
            phase_magnitude = abs(Y_phases(phase_idx, :));
            phase_with_sign = pred_signs_canonical(phase_idx, :) .* phase_magnitude;
            pred_weights(m, :) = Y_amps(m, :) .* exp(1i * phase_with_sign);
        else
            % Fallback (should not happen in canonical form)
            pred_weights(m, :) = Y_amps(m, :) .* exp(1i * Y_phases(phase_idx, :));
        end
    end

    
    % Build reconstructions using weights
    %[recon_pred, ~] = mmf_build_image(number_of_modes, size(dlX, 1), size(dlX, 4), pred_weights', false);
    %[recon_true, ~] = mmf_build_image(number_of_modes, size(dlX, 1), size(dlX, 4), true_weights', false);
    
    % Calculate correlation between predictions and ground truth reconstructions
    %correlation = dlCorr(recon_pred, recon_true);
    
    % Calculate original image correlation too (helpful for debugging)
    %input_correlation = dlCorr(recon_true, dlX);
    
    % Calculate reconstruction loss - maximize correlation
    %reconLoss = 1 - correlation;
    
        % *** Important: Keep a small contribution from BCE loss to maintain gradient flow ***
    % Calculate sign prediction loss (binary cross-entropy loss for signs)
    bce_loss = calculateBCELoss(pred_signs, true_signs(2:end, :));
    
    % Weighted loss combination - keep a tiny weight for BCE to maintain tracing
    correlationWeight = 0.6;
    signLossWeight = 0.4; % Tiny weight just to maintain tracing

    correlation = zeros(1, 1, 'like', bce_loss);
    reconLoss = zeros(1, 1, 'like', bce_loss);
    
    % Combined loss with small BCE component to maintain tracing
    loss = correlationWeight * reconLoss + signLossWeight * bce_loss;

    % Calculate gradients
    gradients = dlgradient(loss, dlnet.Learnables);
end

function [gradients, loss, signAccuracy, correlation] = modelGradients_highModePINN(dlnet, dlX, Y_amps, Y_phases, number_of_modes, options)
    % Specialized gradient function for high mode count PINNs with advanced physics constraints
    % Additional physics-informed constraints include:
    % 1. Modal dispersion physics with accurate dispersion relations
    % 2. Phase matching constraints for nonlinear interactions 
    % 3. Inter-modal energy transfer dynamics
    % 4. Group velocity effects
    
    % Get options or use defaults
    if nargin < 6
        options = struct();
    end
    
    % Default weights for different physics components
    if ~isfield(options, 'correlationWeight'), options.correlationWeight = 0.6; end
    if ~isfield(options, 'signLossWeight'), options.signLossWeight = 0.4; end
    if ~isfield(options, 'physicsWeight'), options.physicsWeight = 0.2; end
    if ~isfield(options, 'dispersionWeight'), options.dispersionWeight = 0.15; end
    if ~isfield(options, 'modalCouplingWeight'), options.modalCouplingWeight = 0.15; end
    
    % Forward pass to get phase sign predictions
    dlY_pred = forward(dlnet, dlX);
    dlY_pred = real(dlY_pred); % Ensure real values for backprop
    
    Y_amps = Y_amps';
    Y_phases = Y_phases' * pi;

    % Get ground truth sign values - assume input phases are in canonical form
    true_signs = sign(Y_phases);
    
    % Use tanh activation to constrain predictions to [-1, 1]
    pred_signs = tanh(dlY_pred);
    
    % Convert to binary signs for accuracy calculation
    pred_signs_binary = sign(pred_signs);
    
    % Make sure we have the right number of signs for the canonical form
    % If we're predicting signs for modes 3..N (N-2 signs), add the +1 sign for mode 2
    if size(pred_signs_binary, 1) == number_of_modes - 2
        % First mode (reference) has 0 phase, not included
        % Second mode's sign is always +1 in canonical form
        % We're predicting signs for modes 3..N (which is N-2 signs)
        pred_signs_canonical = [ones(1, size(pred_signs_binary, 2), 'like', pred_signs_binary); 
                              pred_signs_binary];
    else
        % If we're predicting all N-1 signs but working in canonical form,
        % force the first sign to be +1
        pred_signs_canonical = pred_signs_binary;
        pred_signs_canonical(1, :) = ones(1, size(pred_signs_binary, 2), 'like', pred_signs_binary);
    end
    
    % Calculate sign accuracy - considering all phase signs (canonical form)
    correct = sum(pred_signs_canonical == true_signs, 'all');
    total = numel(true_signs);
    signAccuracy = correct / total;
    
    % Create complex weights for reconstructions
    % True weights from ground truth
    true_weights = zeros(number_of_modes, size(Y_amps, 2), 'like', Y_amps);
    true_weights(1, :) = Y_amps(1, :); % Reference mode (always real)
    
    for m = 2:number_of_modes
        phase_idx = m - 1;
        true_weights(m, :) = Y_amps(m, :) .* exp(1i * Y_phases(phase_idx, :));
    end
    
    % Predicted weights using predicted signs with true amplitudes and phase magnitudes
    pred_weights = zeros(number_of_modes, size(Y_amps, 2), 'like', Y_amps);
    pred_weights(1, :) = Y_amps(1, :); % Reference mode (always real)
    
    for m = 2:number_of_modes
        phase_idx = m - 1;
        if phase_idx <= size(pred_signs_canonical, 1)
            % Use magnitude of true phase with predicted sign
            phase_magnitude = abs(Y_phases(phase_idx, :));
            phase_with_sign = pred_signs_canonical(phase_idx, :) .* phase_magnitude;
            pred_weights(m, :) = Y_amps(m, :) .* exp(1i * phase_with_sign);
        else
            % Fallback (should not happen in canonical form)
            pred_weights(m, :) = Y_amps(m, :) .* exp(1i * Y_phases(phase_idx, :));
        end
    end

    % Build reconstructions using weights
    [recon_pred, ~] = mmf_build_image(number_of_modes, size(dlX, 1), size(dlX, 4), pred_weights', false);
    [recon_true, ~] = mmf_build_image(number_of_modes, size(dlX, 1), size(dlX, 4), true_weights', false);
    
    % Calculate correlation between predictions and ground truth reconstructions
    correlation = dlCorr(recon_pred, recon_true);
    
    % Calculate reconstruction loss - maximize correlation
    reconLoss = 1 - correlation;
    
    % Calculate sign prediction loss (binary cross-entropy loss for signs)
    bce_loss = calculateBCELoss(pred_signs, true_signs(2:end, :));
    
    % PHYSICS-INFORMED COMPONENTS for high mode count fibers
    % These enhance the network's understanding of physical constraints
    
    % 1. Modal Dispersion Physics - enforce physical relationships between mode propagation constants
    dispersion_loss = calculateModalDispersionConstraint(pred_signs_canonical, number_of_modes);
    
    % 2. Phase Matching Constraint - for FWM and other nonlinear processes
    phase_matching_loss = calculatePhaseMatchingConstraint(pred_signs_canonical, true_weights, number_of_modes);
    
    % 3. Modal Energy Conservation - conservation laws across mode groups
    energy_conservation_loss = calculateEnergyConservationConstraint(pred_weights, Y_amps);
    
    % Combine physics-based losses
    physics_loss = options.dispersionWeight * dispersion_loss + ...
                  options.modalCouplingWeight * phase_matching_loss + ...
                  (1 - options.dispersionWeight - options.modalCouplingWeight) * energy_conservation_loss;
    
    % Combined weighted loss with physics-informed components
    loss = options.correlationWeight * reconLoss + ...
           options.signLossWeight * bce_loss + ...
           options.physicsWeight * physics_loss;
    
    % Calculate gradients
    gradients = dlgradient(loss, dlnet.Learnables);
end

% Physics-informed constraints specialized for high mode count fibers

function dispersion_loss = calculateModalDispersionConstraint(pred_signs, number_of_modes)
    % Enforce physical relationships between mode propagation constants
    % In real fibers, adjacent mode groups typically maintain consistent phase relationships
    
    % Modal groups and their theoretical relationships
    % For a step-index fiber, each mode group l has 2*l+1 positions in the LP modes
    num_groups = floor(sqrt(number_of_modes));
    
    % Get propagation constants from theory - β_lm ≈ β_00 * (1 - 2Δ((l+2m)λ/2πa)²)
    % where l is azimuthal index, m is radial index
    mode_indices = 1:number_of_modes-1;
    beta_relative = zeros(1, number_of_modes-1, 'like', pred_signs);
    
    for idx = 1:number_of_modes-1
        % Approximate LP mode indices from linear index
        l = floor(sqrt(idx+1)) - 1;  % Azimuthal mode number approximation
        m = idx - l*(l+1) - 1;       % Radial mode number approximation
        beta_relative(idx) = 1 - 0.01*((l+2*m)^2); % Simplified relative propagation constant
    end
    
    % For adjacent mode groups, phase signs tend to be correlated
    group_violations = 0;
    group_count = 0;
    
    % Check sign patterns within mode groups
    current_group = 1;
    group_start = 1;
    
    for l = 0:num_groups-1
        % Each mode group l has (2l+1) positions 
        group_size = 2*l + 1;
        group_end = group_start + group_size - 1;
        
        if group_end > number_of_modes-1
            group_end = number_of_modes-1;
        end
        
        % If we have at least 2 modes in this group
        if group_end > group_start
            % Get signs for this mode group
            group_signs = pred_signs(group_start:group_end, :);
            
            % Check for sign consistency within group
            for b = 1:size(group_signs, 2)
                % Calculate sign variety within group
                sign_variety = (sum(group_signs(:,b) > 0) / size(group_signs, 1)) - 0.5;
                group_violations = group_violations + abs(sign_variety) * 2; % Penalize mixed signs in group
            end
            
            group_count = group_count + 1;
        end
        
        % Move to next group
        group_start = group_end + 1;
        if group_start > number_of_modes-1
            break;
        end
    end
    
    % Normalize violations
    if group_count > 0
        dispersion_loss = group_violations / group_count;
    else
        dispersion_loss = dlarray(0);
    end
end

function phase_matching_loss = calculatePhaseMatchingConstraint(pred_signs, true_weights, number_of_modes)
    % For nonlinear FWM processes, phases must satisfy matching conditions:
    % φᵢ + φⱼ = φₖ + φₗ for specific mode combinations
    
    % Initialize loss
    phase_matching_violation = 0;
    condition_count = 0;
    
    % Consider main phase matching conditions (most significant combinations)
    % Focus on degenerate FWM: 2φᵢ = φᵢ₋₁ + φᵢ₊₁
    for i = 2:(number_of_modes-2)
        if i+1 <= size(pred_signs, 1)
            % Degenerate FWM condition
            left_side = 2 * pred_signs(i,:); 
            right_side = pred_signs(i-1,:) + pred_signs(i+1,:);
            
            % Calculate phase mismatch - penalize differently based on magnitude
            mismatch = abs(left_side - right_side);
            
            % Only count significant mismatches (odd number differences)
            significant_mismatch = (mismatch > 1.5);
            phase_matching_violation = phase_matching_violation + sum(significant_mismatch);
            condition_count = condition_count + size(pred_signs, 2);
        end
    end
    
    % Normalize by number of conditions checked
    if condition_count > 0
        phase_matching_loss = phase_matching_violation / condition_count;
    else
        phase_matching_loss = dlarray(0);
    end
end

function energy_loss = calculateEnergyConservationConstraint(pred_weights, Y_amps)
    % In fiber physics, energy conservation applies across mode groups
    % This constraint ensures predicted weights maintain appropriate energy distribution
    
    % Total power in true amplitudes (per sample)
    true_power = sum(Y_amps.^2, 1);
    
    % Estimate effective power in predicted weights
    pred_power = sum(abs(pred_weights).^2, 1);
    
    % Calculate normalized power deviation
    power_deviation = abs(pred_power - true_power) ./ true_power;
    
    % Mean deviation across batch
    energy_loss = mean(power_deviation);
end

function bce_loss = calculateBCELoss(pred_signs, true_signs)
    % Convert from [-1,1] to [0,1] range for BCE loss
    pred_probs = (pred_signs + 1) / 2;
    true_probs = (true_signs + 1) / 2;
    
    % Binary cross-entropy loss
    epsilon = 1e-8; % To prevent log(0)
    bce = -mean(true_probs .* log(pred_probs + epsilon) + ...
               (1 - true_probs) .* log(1 - pred_probs + epsilon), 'all');
    
    bce_loss = bce;
end

function complex_weights = createComplexWeights(amplitudes, phases, phase_signs, number_of_modes)
    % Creates complex weights using magnitudes of amplitudes and phases and predicted phase signs
    % For canonical form:
    % - amplitudes: [N, batch]
    % - phases: Contains phase magnitudes for modes 2-N
    % - phase_signs: Contains signs for modes 2-N, with mode 2's sign always +1 (canonical constraint)

    complex_weights = dlarray(zeros(number_of_modes, size(amplitudes, 2), 'like', amplitudes), 'CB');

    % Reference mode (mode 1) has zero phase
    complex_weights(1, :) = amplitudes(1, :);

    % For modes 2 to N
    for m = 2:number_of_modes
        if m-1 <= size(phases, 1) && m-1 <= size(phase_signs, 1)
            % Use the magnitude of the phase with the sign
            phase_magnitude = abs(phases(m-1, :));
            sign_value = phase_signs(m-1, :);

            % Combine magnitude and sign to get the complex weight
            complex_weights(m, :) = amplitudes(m, :) .* exp(1i * pi * (phase_magnitude .* sign_value));
        else
            % Fallback if phase data is missing (just use amplitude)
            complex_weights(m, :) = amplitudes(m, :);
        end
    end
end

function [validationLoss, signAccuracy, correlation] = validatePhaseSignModel(dlnet, X, Y_amps, Y_phases, batchSize, executionEnvironment, number_of_modes)
    % Validates the phase sign model on validation data
    numValidation = size(X, 4);
    numBatches = ceil(numValidation/batchSize);
    
    totalLoss = 0;
    totalAccuracy = 0;
    totalCorrelation = 0;
    
    for i = 1:numBatches
        batchIdx = (i-1)*batchSize+1:min(i*batchSize, numValidation);
        dlX = dlarray(X(:,:,:,batchIdx), 'SSCB');
        batch_Y_amps = Y_amps(batchIdx, :);
        batch_Y_phases = Y_phases(batchIdx, 1:end); % Only use phase signs for modes 3..N
        
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end
        
        [~, batchLoss, batchAccuracy, batchCorrelation] = dlfeval(@modelGradients_phaseSign, dlnet, dlX, batch_Y_amps, batch_Y_phases, number_of_modes);
        
        totalLoss = totalLoss + extractdata(batchLoss);
        totalAccuracy = totalAccuracy + extractdata(batchAccuracy);
        totalCorrelation = totalCorrelation + extractdata(batchCorrelation);
    end
    
    validationLoss = totalLoss / numBatches;
    signAccuracy = totalAccuracy / numBatches;
    correlation = totalCorrelation / numBatches;
end

function performPhaseSignEvaluation(dlnet, X_val, Y_amps, Y_phases, number_of_modes)
    % Evaluate the phase sign model with detailed analysis
    % Select a small subset for visualization
    evalSize = min(8, size(X_val, 4));
    indices = randperm(size(X_val, 4), evalSize);
    X_eval = X_val(:,:,:,indices);
    Y_eval_amps = Y_amps(indices, :);
    Y_eval_phases = Y_phases(indices, :);

    % Prepare data in the same format as modelGradients_phaseSign
    Y_eval_amps = Y_eval_amps';
    Y_eval_phases = Y_eval_phases' * pi;
    
    % Get predictions from network
    dlX = dlarray(X_eval, 'SSCB');
    if canUseGPU
        dlX = gpuArray(dlX);
    end
    
    % Forward pass to get predictions and apply tanh to constrain to [-1, 1]
    pred_signs_raw = forward(dlnet, dlX);
    pred_signs = tanh(pred_signs_raw);
    pred_signs_binary = sign(pred_signs);
    
    % Handle canonical form exactly as in modelGradients_phaseSign
    if size(pred_signs_binary, 1) == number_of_modes - 2
        % First mode has 0 phase (reference)
        % Second mode's sign is +1 in canonical form
        % We're predicting signs for modes 3..N
        pred_signs_canonical = [ones(1, size(pred_signs_binary, 2), 'like', pred_signs_binary); 
                              pred_signs_binary];
    else
        % If predicting all N-1 signs but working in canonical form
        pred_signs_canonical = pred_signs_binary;
        pred_signs_canonical(1, :) = ones(1, size(pred_signs_binary, 2), 'like', pred_signs_binary);
    end
    
    % Get ground truth signs in canonical form
    true_signs = sign(Y_eval_phases);
    
    % Create complex weights for reconstruction
    true_weights = zeros(number_of_modes, size(Y_eval_amps, 2), 'like', Y_eval_amps);
    true_weights(1, :) = Y_eval_amps(1, :); % Reference mode (always real)
    
    pred_weights = zeros(number_of_modes, size(Y_eval_amps, 2), 'like', Y_eval_amps);
    pred_weights(1, :) = Y_eval_amps(1, :); % Reference mode (always real)
    
    % Create weights in same manner as modelGradients_phaseSign
    for m = 2:number_of_modes
        phase_idx = m - 1;
        
        % True weights - use the original complex values directly
        true_weights(m, :) = Y_eval_amps(m, :) .* exp(1i * Y_eval_phases(phase_idx, :));
        
        % Predicted weights
        if phase_idx <= size(pred_signs_canonical, 1)
            % Use magnitude of true phase with predicted sign
            phase_magnitude = abs(Y_eval_phases(phase_idx, :));
            phase_with_sign = pred_signs_canonical(phase_idx, :) .* phase_magnitude;
            pred_weights(m, :) = Y_eval_amps(m, :) .* exp(1i * phase_with_sign);
        else
            % Fallback (should not happen in canonical form)
            pred_weights(m, :) = Y_eval_amps(m, :) .* exp(1i * Y_eval_phases(phase_idx, :));
        end
    end
    
    % Build reconstructions - ensure we're using the correct dimensions
    [recons_true, ~] = mmf_build_image(number_of_modes, size(X_eval, 1), evalSize, true_weights', false);
    [recons_pred, ~] = mmf_build_image(number_of_modes, size(X_eval, 1), evalSize, pred_weights', false);
    
    % Calculate correlations consistently with training approach
    correlations_pred_true = zeros(evalSize, 1); % Correlation between pred and true recon
    correlations_pred_orig = zeros(evalSize, 1); % Correlation between pred and original image
    correlations_true_orig = zeros(evalSize, 1); % Correlation between true recon and original image
    
    for i = 1:evalSize
        % Convert to regular arrays for calculation
        orig_img = extract(X_eval(:,:,:,i));
        true_recon = extract(recons_true(:,:,:,i));
        pred_recon = extract(recons_pred(:,:,:,i));
        
        % Calculate correlations
        correlations_pred_true(i) = corr2(pred_recon, true_recon);
        correlations_pred_orig(i) = corr2(orig_img, pred_recon);
        correlations_true_orig(i) = corr2(orig_img, true_recon);
    end
    
    % Only consider the signs we're actually predicting (modes 3..N)
    % since mode 2's sign is fixed to +1 in canonical form
    if size(pred_signs_binary, 1) == number_of_modes - 2
        % We're predicting signs for modes 3..N
        correct_signs = (true_signs == pred_signs_canonical);
    else
        % We're predicting all signs but comparing only those that matter
        correct_signs = (true_signs(2:end, :) == pred_signs_canonical(2:end, :));
    end
    
    % Calculate per-sample accuracy
    sign_matches = sum(correct_signs, 1) ./ size(correct_signs, 1);
    overall_accuracy = mean(sign_matches);
    
    % Additional per-mode accuracy (for analysis)
    mode_accuracy = sum(correct_signs, 2) ./ size(correct_signs, 2);
    
    % Display results
    fprintf('\nPhase Sign Evaluation:\n');
    fprintf('  Mean correlation (pred vs true recon): %.4f\n', mean(correlations_pred_true));
    fprintf('  Mean correlation (pred vs original): %.4f\n', mean(correlations_pred_orig));
    fprintf('  Mean correlation (true vs original): %.4f\n', mean(correlations_true_orig));
    fprintf('  Sign accuracy: %.2f%%\n', overall_accuracy*100);
    
    % Find existing figure or create a new one
    fig_recon = findobj('Type', 'figure', 'Tag', 'PhaseSignEvaluationFigure');
    if isempty(fig_recon)
        fig_recon = figure('Name', 'Phase Sign Evaluation', 'Position', [100, 100, 1200, 500], ...
                     'Tag', 'PhaseSignEvaluationFigure');
    else
        figure(fig_recon);
        clf;
    end
    
    % Update title with the correlation metrics
    sgtitle(sprintf('Phase Sign Reconstruction - Pred vs True: %.4f, Pred vs Orig: %.4f', ...
           mean(correlations_pred_true), mean(correlations_pred_orig)), 'FontSize', 14);
    
    % Plot the comparisons
    for i = 1:min(4, evalSize)
        % Original image
        subplot(3, 4, i);
        imagesc(extract(X_eval(:,:,:,i)));
        axis image off;
        title(sprintf('Original #%d', i));
        
        % True reconstruction
        subplot(3, 4, i+4);
        imagesc(extract(recons_true(:,:,:,i)));
        axis image off;
        title(sprintf('True Recon (r=%.3f)', correlations_true_orig(i)));
        
        % Predicted reconstruction
        subplot(3, 4, i+8);
        imagesc(extract(recons_pred(:,:,:,i)));
        axis image off;
        title(sprintf('Pred Recon (r=%.3f)', correlations_pred_orig(i)));
    end
    
    % Create figure for phase sign analysis
    fig_signs = findobj('Type', 'figure', 'Tag', 'PhaseSignPatternAnalysis');
    if isempty(fig_signs)
        fig_signs = figure('Name', 'Phase Sign Analysis', 'Position', [100, 600, 1200, 500], ...
                     'Tag', 'PhaseSignPatternAnalysis');
    else
        figure(fig_signs);
        clf;
    end
    
    % Plot sign patterns
    sgtitle(sprintf('Phase Sign Pattern Analysis - Accuracy: %.2f%%', overall_accuracy*100), 'FontSize', 14);
    
    % Create a visual representation of the phase sign patterns
    for i = 1:min(6, evalSize)
        subplot(2, 3, i);
        visualizePhaseSignPattern(true_signs(:,i), pred_signs_canonical(:,i), correct_signs(:,i));
        title(sprintf('Sample #%d (Acc: %.1f%%)', i, sign_matches(i)*100));
    end
    
    % Update both figures
    drawnow;
end

function visualizePhaseSignPattern(true_signs, pred_signs, correct_signs)
    % Create a visual representation of phase sign patterns with canonical form
    n_modes = length(true_signs) + 1; % Add one for reference mode
    
    % Create layout grid
    hold on;
    
    % Plot reference line (always 0 phase)
    line([1, 1], [0, 1], 'Color', 'k', 'LineWidth', 2);
    text(1, -0.1, '1', 'FontSize', 10, 'HorizontalAlignment', 'center');
    
    % Plot phase signs for modes 2..N
    for m = 2:n_modes
        % Get the index in the signs array
        phase_idx = m - 1;
        
        % Plot true sign (top)
        if true_signs(phase_idx) > 0
            line([m, m], [0.5, 1], 'Color', 'blue', 'LineWidth', 2);
        else
            line([m, m], [0.5, 1], 'Color', 'red', 'LineWidth', 2);
        end
        
        % Plot predicted sign (bottom)
        if pred_signs(phase_idx) > 0
            line([m, m], [0, 0.5], 'Color', 'blue', 'LineWidth', 2);
        else
            line([m, m], [0, 0.5], 'Color', 'red', 'LineWidth', 2);
        end
        
        % Add accuracy indicator
        if correct_signs(phase_idx)
            plot(m, 0.5, 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
        else
            plot(m, 0.5, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
        end
        
        % Highlight mode 2 (always +1 in canonical form)
        if m == 2
            plot(m, -0.05, '^', 'MarkerSize', 8, 'MarkerFaceColor', 'y', 'Color', 'k');
        end
        
        % Add mode number
        text(m, -0.1, num2str(m), 'FontSize', 10, 'HorizontalAlignment', 'center');
    end
    
    % Add labels
    text(0.5, 0.75, 'True', 'FontSize', 12);
    text(0.5, 0.25, 'Pred', 'FontSize', 12);
    
    % Set axis properties
    xlim([0.5, n_modes+0.5]);
    ylim([-0.2, 1.2]);
    axis off;
    
    % Add legend
    legend_elements = [
        line([0,0], [0,0], 'Color', 'blue', 'LineWidth', 2),
        line([0,0], [0,0], 'Color', 'red', 'LineWidth', 2),
        plot(0, 0, 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g'),
        plot(0, 0, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'),
        plot(0, 0, '^', 'MarkerSize', 8, 'MarkerFaceColor', 'y', 'Color', 'k')
    ];
    legend(legend_elements, {'Positive Sign', 'Negative Sign', 'Correct', 'Incorrect', 'Canonical (+1)'}, ...
        'Location', 'southoutside', 'Orientation', 'horizontal');
end

function corr = dlCorr(A, B)
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

% Network architectures for phase sign prediction
function dlnet = createPhaseSignCNN(inputSize, outputSize)
    % Create a specialized CNN for phase sign prediction
    % This architecture is designed to focus on global phase relationships
    
    layers = [
        imageInputLayer([inputSize 1], 'Name', 'input', 'Normalization', 'none')
        
        % Feature extraction path
        convolution2dLayer(5, 32, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
        
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
        
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
        
        % Special phase-sensitive layers
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv_phase1')
        batchNormalizationLayer('Name', 'bn_phase1')
        reluLayer('Name', 'relu_phase1')
        
        % Add global average pooling for translation invariance
        globalAveragePooling2dLayer('Name', 'gap')
        
        % Decision layers
        fullyConnectedLayer(128, 'Name', 'fc1')
        reluLayer('Name', 'relu_fc1')
        dropoutLayer(0.5, 'Name', 'drop1')
        
        fullyConnectedLayer(outputSize, 'Name', 'fc_out')
    ];
    
    lgraph = layerGraph(layers);
    dlnet = dlnetwork(lgraph);
end

function dlnet = createPhaseSignMLP(inputSize, outputSize)
    % Create a simplified MLP for phase sign prediction
    layers = [
        imageInputLayer([inputSize 1], 'Name', 'input', 'Normalization', 'none')
        flattenLayer('Name', 'flatten')
        fullyConnectedLayer(256, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        dropoutLayer(0.3, 'Name', 'drop1')
        fullyConnectedLayer(128, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        dropoutLayer(0.2, 'Name', 'drop2')
        fullyConnectedLayer(64, 'Name', 'fc3')
        reluLayer('Name', 'relu3')
        fullyConnectedLayer(outputSize, 'Name', 'fc_out')
    ];
    
    lgraph = layerGraph(layers);
    dlnet = dlnetwork(lgraph);
end

function dlnet = createPhaseSignResNet(inputSize, outputSize)
    % Create a ResNet-based architecture for phase sign prediction
    % This network uses residual connections for better gradient flow
    
    layers = [
        imageInputLayer([inputSize 1], 'Name', 'input', 'Normalization', 'none')
        
        % Initial convolution
        convolution2dLayer(7, 32, 'Stride', 2, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(3, 'Stride', 2, 'Padding', 'same', 'Name', 'pool1')
    ];
    
    lgraph = layerGraph(layers);
    
    % Residual block function
    function lgraph = addResidualBlock(lgraph, numFilters, blockName, inName)
        % Create main path
        mainPath = [
            convolution2dLayer(3, numFilters, 'Padding', 'same', 'Name', [blockName '_conv1'])
            batchNormalizationLayer('Name', [blockName '_bn1'])
            reluLayer('Name', [blockName '_relu1'])
            convolution2dLayer(3, numFilters, 'Padding', 'same', 'Name', [blockName '_conv2'])
            batchNormalizationLayer('Name', [blockName '_bn2'])
        ];
        
        lgraph = addLayers(lgraph, mainPath);
        
        % Connect from input to first layer of block
        lgraph = connectLayers(lgraph, inName, [blockName '_conv1']);
        
        % Add skip connection - simplified approach without findLayerIndex
        skipOutput = inName;
        
        % If dimensions need to match (like in downsampling)
        if contains(blockName, 'downsample') || ~strcmp(inName, 'pool1')
            skipPath = [
                convolution2dLayer(1, numFilters, 'Stride', 1, 'Name', [blockName '_skip'])
                batchNormalizationLayer('Name', [blockName '_skip_bn'])
            ];
            lgraph = addLayers(lgraph, skipPath);
            lgraph = connectLayers(lgraph, inName, [blockName '_skip']);
            skipOutput = [blockName '_skip_bn'];
        end
        
        % Add layer for combining main and skip paths
        add = additionLayer(2, 'Name', [blockName '_add']);
        relu = reluLayer('Name', [blockName '_relu_out']);
        
        lgraph = addLayers(lgraph, [add; relu]);
        lgraph = connectLayers(lgraph, [blockName '_bn2'], [blockName '_add/in1']);
        lgraph = connectLayers(lgraph, skipOutput, [blockName '_add/in2']);
        % lgraph = connectLayers(lgraph, [blockName '_add'], [blockName '_relu_out']);
    end
    
    % Add residual blocks
    % First block group (32 filters)
    lgraph = addResidualBlock(lgraph, 32, 'block1a', 'pool1');
    lgraph = addResidualBlock(lgraph, 32, 'block1b', 'block1a_relu_out');
    
    % Second block group (64 filters) with downsampling
    lgraph = addResidualBlock(lgraph, 64, 'block2a_downsample', 'block1b_relu_out');
    lgraph = addResidualBlock(lgraph, 64, 'block2b', 'block2a_downsample_relu_out');
    
    % Third block group (128 filters) with downsampling
    lgraph = addResidualBlock(lgraph, 128, 'block3a_downsample', 'block2b_relu_out');
    lgraph = addResidualBlock(lgraph, 128, 'block3b', 'block3a_downsample_relu_out');
    
    % Global average pooling and output layers
    finalLayers = [
        globalAveragePooling2dLayer('Name', 'gap')
        fullyConnectedLayer(outputSize, 'Name', 'fc')
    ];
    
    lgraph = addLayers(lgraph, finalLayers);
    lgraph = connectLayers(lgraph, 'block3b_relu_out', 'gap');
    
    % Create network
    dlnet = dlnetwork(lgraph);
end

function dlnet = createHighModePINN(inputSize, outputSize, number_of_modes)
    % Creates an advanced Physics-Informed Neural Network for high mode count fibers
    % This network uses advanced architecture techniques to model complex inter-modal
    % relationships in high-dimensional mode spaces
    %
    % Parameters:
    %   inputSize - Input image dimensions [height width]
    %   outputSize - Output size (2*(number_of_modes-1) for complex phases)
    %   number_of_modes - Number of fiber modes to model
    
    fprintf('Creating High Mode Count PINN with input size %dx%d and output size %d for %d modes\n', ...
            inputSize(1), inputSize(2), outputSize, number_of_modes);
    
    % Higher capacity backbone for feature extraction
    layers = [
        imageInputLayer([inputSize 1], 'Name', 'input', 'Normalization', 'none')
        
        % Initial feature extraction with large receptive field
        convolution2dLayer(9, 64, 'Padding', 'same', 'Name', 'conv1', 'WeightsInitializer', 'he')
        batchNormalizationLayer('Name', 'bn1')
        leakyReluLayer(0.2, 'Name', 'lrelu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    ];
    
    lgraph = layerGraph(layers);
    
    % Add multi-scale perception path to capture different spatial frequencies
    % This helps with detecting both low and high-order mode patterns
    multiScalePaths = {};
    
    % Four parallel paths with different kernel sizes
    kernelSizes = [3, 5, 7, 9];
    for k = 1:length(kernelSizes)
        pathName = sprintf('scale%d', k);
        ks = kernelSizes(k);
        
        scalePath = [
            convolution2dLayer(ks, 64, 'Padding', 'same', 'Name', [pathName '_conv1'])
            batchNormalizationLayer('Name', [pathName '_bn1'])
            leakyReluLayer(0.2, 'Name', [pathName '_lrelu1'])
            convolution2dLayer(ks, 64, 'Padding', 'same', 'Name', [pathName '_conv2'])
            batchNormalizationLayer('Name', [pathName '_bn2'])
            leakyReluLayer(0.2, 'Name', [pathName '_lrelu2'])
        ];
        
        lgraph = addLayers(lgraph, scalePath);
        lgraph = connectLayers(lgraph, 'pool1', [pathName '_conv1']);
        multiScalePaths{k} = [pathName '_lrelu2'];
    end
    
    % Concatenate multi-scale paths
    concatLayer = depthConcatenationLayer(length(multiScalePaths), 'Name', 'multi_scale_concat');
    lgraph = addLayers(lgraph, concatLayer);
    
    % Connect all scale paths to concatenation
    for k = 1:length(multiScalePaths)
        lgraph = connectLayers(lgraph, multiScalePaths{k}, ['multi_scale_concat/in' num2str(k)]);
    end
    
    % Add mode interaction module using self-attention to model complex relationships
    % between modes in high-dimensional spaces
    attentionPath = [
        convolution2dLayer(1, 128, 'Name', 'att_dim_reduction')
        batchNormalizationLayer('Name', 'att_bn1')
        leakyReluLayer(0.2, 'Name', 'att_lrelu1')
        
        % Custom attention layer - can be adapted to fit specific modal relationships
        globalAveragePooling2dLayer('Name', 'gap')
        fullyConnectedLayer(512, 'Name', 'att_fc1')
        batchNormalizationLayer('Name', 'att_bn2')
        leakyReluLayer(0.2, 'Name', 'att_lrelu2')
        
        % Simulate multi-head attention with parallel paths
        fullyConnectedLayer(256, 'Name', 'att_fc2a')
        fullyConnectedLayer(256, 'Name', 'att_fc2b')
    ];
    
    lgraph = addLayers(lgraph, attentionPath);
    lgraph = connectLayers(lgraph, 'multi_scale_concat', 'att_dim_reduction');
    
    % Add pathway specialized for mode dispersion physics
    dispersionPath = [
        fullyConnectedLayer(128, 'Name', 'disp_fc1')
        batchNormalizationLayer('Name', 'disp_bn1')
        leakyReluLayer(0.2, 'Name', 'disp_lrelu1')
        dropoutLayer(0.3, 'Name', 'disp_drop1')
        fullyConnectedLayer(number_of_modes*2, 'Name', 'disp_fc2') % 2 parameters per mode for dispersion model
    ];
    
    lgraph = addLayers(lgraph, dispersionPath);
    lgraph = connectLayers(lgraph, 'att_fc2a', 'disp_fc1');
    
    % Add pathway specialized for nonlinear interactions (FWM, XPM)
    nonlinearPath = [
        fullyConnectedLayer(256, 'Name', 'nl_fc1')
        batchNormalizationLayer('Name', 'nl_bn1')
        leakyReluLayer(0.2, 'Name', 'nl_lrelu1')
        dropoutLayer(0.3, 'Name', 'nl_drop1')
        fullyConnectedLayer(256, 'Name', 'nl_fc2')
        batchNormalizationLayer('Name', 'nl_bn2')
        leakyReluLayer(0.2, 'Name', 'nl_lrelu2')
    ];
    
    lgraph = addLayers(lgraph, nonlinearPath);
    lgraph = connectLayers(lgraph, 'att_fc2b', 'nl_fc1');
    
    % Add modal coupling module - allows explicit modeling of pair-wise mode interactions
    % Add modal coupling module - allows explicit modeling of pair-wise mode interactions
    modalCouplingLayer = createModalCouplingLayer(number_of_modes);
    lgraph = addLayers(lgraph, modalCouplingLayer);

    % Connect directly to the first layer of the coupling module (the FC layer)
    lgraph = connectLayers(lgraph, 'nl_lrelu2', 'coupling_matrix_fc');
    
    % Merge physics-specialized pathways
    mergeLayers = [
        depthConcatenationLayer(3, 'Name', 'physics_merge')
        fullyConnectedLayer(512, 'Name', 'merge_fc1')
        batchNormalizationLayer('Name', 'merge_bn1')
        leakyReluLayer(0.2, 'Name', 'merge_lrelu1')
        
        dropoutLayer(0.3, 'Name', 'merge_drop1')
        fullyConnectedLayer(512, 'Name', 'merge_fc2')
        leakyReluLayer(0.2, 'Name', 'merge_lrelu2')
        
        % Final output layer - real/imaginary pairs for each mode's phase
        fullyConnectedLayer(outputSize, 'Name', 'final_fc')
    ];
    
    lgraph = addLayers(lgraph, mergeLayers);
    lgraph = connectLayers(lgraph, 'disp_fc2', 'physics_merge/in1');
    lgraph = connectLayers(lgraph, 'modal_coupling_output', 'physics_merge/in2');
    lgraph = connectLayers(lgraph, 'att_lrelu2', 'physics_merge/in3');
    
    % Create network
    dlnet = dlnetwork(lgraph);
end

function layers = createModalCouplingLayer(numModes)
    % Creates a specialized layer for modeling pair-wise interactions between modes
    % This custom layer allows the network to learn physical relationships between modes
    
    % Instead of using featureInputLayer, use regular layers that can accept inputs
    layers = [
        % Use a fully connected layer to transform input features
        fullyConnectedLayer(numModes^2, 'Name', 'coupling_matrix_fc')
        
        % Reshape to modal coupling matrix
        %functionLayer(@(x) reshape(x, [numModes, numModes, []]), 'Name', 'coupling_matrix_reshape')
        addDimensionsLayer('SS')
        depthToSpace2dLayer([numModes, numModes], 'Name', 'coupling_matrix_reshape')
        
        % Process with symmetric constraints using the correct functionLayer syntax
        functionLayer(@(X) ModalCouplingFunction(X, numModes), 'Name', 'modal_coupling_sym')
        
        % Output processing
        fullyConnectedLayer(256, 'Name', 'modal_coupling_output')
    ];
end

% Modal coupling function - adapted to receive numModes directly
function Y = ModalCouplingFunction(X, numModes)
    % Apply physics-based constraints to the mode coupling matrix
    % X input shape is [numModes, numModes, 1, batch]
    
    % Make coupling matrix symmetric (reciprocal coupling)
    % Use proper permutation order for a 4D tensor (SSCB format)
    X_sym = 0.5 * (X + permute(X, [2, 1, 3, 4]));
    
    % Apply physical constraints:
    % 1. Higher-order modes couple less effectively (apply distance-based decay)
    mode_indices = 1:numModes;
    [i, j] = ndgrid(mode_indices, mode_indices);
    mode_distance = abs(i - j);
    
    % Distance-based coupling decay (1/r law approximation)
    coupling_mask = 1./(1 + 0.5*mode_distance);
    
    % Expand coupling_mask to match X dimensions for broadcasting
    % [numModes, numModes] → [numModes, numModes, 1, 1]
    coupling_mask = reshape(coupling_mask, [numModes, numModes, 1, 1]);
    
    % Apply mask to enforce physics-informed coupling strengths
    % Broadcasting will apply across all batches automatically
    X_constrained = X_sym .* coupling_mask;
    
    % Output in the original format
    Y = X_constrained;
end

% Gradient clipping utility
function grad = thresholdL2Norm(grad, threshold)
    gradNorm = sqrt(sum(grad(:).^2));
    if gradNorm > threshold
        grad = (threshold/gradNorm) * grad;
    end
end