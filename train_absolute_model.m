% filepath: c:\Users\nicks\MATLAB\Projects\MT2_prak\Versuch 3\train_absolute_model.m
% train_absolute_model.m - Trains a network to predict absolute amplitudes and phases

function dlnet = train_absolute_model(XTrain, YTrain, XValidation, YValidation, options, P_precomputed)
    % Parse options or use defaults
    if nargin < 5
        options = struct();
    end
    
    % Get utility functions
    utils = mmf_utils();
    
    % Default options
    if ~isfield(options, 'initialLearnRate'), options.initialLearnRate = 5e-4; end
    if ~isfield(options, 'maxEpochs'), options.maxEpochs = 10000; end
    if ~isfield(options, 'miniBatchSize'), options.miniBatchSize = 64; end
    if ~isfield(options, 'validationFrequency'), options.validationFrequency = 100; end
    if ~isfield(options, 'validationPatience'), options.validationPatience = 20000; end
    if ~isfield(options, 'executionEnvironment'), options.executionEnvironment = "gpu"; end
    if ~isfield(options, 'modelType'), options.modelType = "CNN"; end
    if ~isfield(options, 'plotProgress'), options.plotProgress = "gui"; end
    if ~isfield(options, 'useReconLoss'), options.useReconLoss = false; end
    if ~isfield(options, 'showSignAccuracy'), options.showSignAccuracy = false; end
    if ~isfield(options, 'evaluationFrequency'), options.evaluationFrequency = 0; end
    if ~isfield(options, 'adaptiveLossWeights'), options.adaptiveLossWeights = false; end
    
    % Define network parameters from data
    inputSize = [size(XTrain,1) size(XTrain,2)];
    outputSize = size(YTrain,2);
    number_of_modes = (outputSize+1) / 2;
    
    % Get or create BPMmatlab model with precomputed modes
    if nargin < 6 || isempty(P_precomputed)
        P = utils.getOrCreateModelWithModes(number_of_modes, inputSize(1), true);
    else
        P = P_precomputed;
    end
    
    % Create network based on specified type if not loading existing model
    if strcmpi(options.modelType, "CNN")
        dlnet = createCNNModel(inputSize, outputSize);
    elseif strcmpi(options.modelType, "MLP")
        dlnet = createMLPModel(inputSize, outputSize);
    elseif strcmpi(options.modelType, "VGG")
        dlnet = createVGGModel(inputSize, outputSize);
    elseif strcmpi(options.modelType, "Custom")
        dlnet = createCustomModel(inputSize, outputSize);
    elseif strcmpi(options.modelType, "ResNet")
        dlnet = createResNet(inputSize, outputSize);
    elseif strcmpi(options.modelType, "load")
        load('absolute_model.mat', 'dlnet');
        disp('Loaded existing absolute value model from absolute_model.mat');        
    else
        error('Unsupported model type: %s', options.modelType);
    end

    analyzeNetwork(dlnet); % Analyze the network structure
    
    % Initialize training progress visualization
    if strcmpi(options.plotProgress, "gui")
        metrics = ["TotalLoss", "AmpLoss", "PhaseLoss", ...
                    "ValidationLoss", "ValAmpLoss", "ValPhaseLoss", "SignLoss"];
        
        if options.useReconLoss
            metrics = [metrics, "ReconLoss", "NormLoss"];
        end

        if options.showSignAccuracy
            metrics = [metrics, "SignAccuracy", "ValSignAccuracy"]; 
        end
        
        monitor = trainingProgressMonitor(Metrics=metrics, ...
            Info=["Epoch", "LearningRate", "TrainingPhase"], ...
            XLabel="Iteration");

        % Set y-scale to log for metrics
        for i = 1:length(metrics)
            yscale(monitor, metrics(i), 'log');
        end

        % Group plots
        groupSubPlot(monitor, "Losses", ["TotalLoss", "ValidationLoss"]);
        if options.useReconLoss
            groupSubPlot(monitor, "Other Losses", ["ReconLoss", "NormLoss"]);
        end
        if options.showSignAccuracy
            groupSubPlot(monitor, "Sign Accuracy", ["SignAccuracy", "ValSignAccuracy"]);
        end
        groupSubPlot(monitor, "Training Losses", ["AmpLoss", "PhaseLoss"]);
        groupSubPlot(monitor, "Validation Losses", ["ValAmpLoss", "ValPhaseLoss"]);
    end
    
    % Initialize training state
    iteration = 0;
    epoch = 0;
    bestValidationLoss = Inf;
    bestDlnet = dlnet;
    patienceCounter = 0;
    stopTraining = false;
    
    % Initialize optimization variables
    averageGrad = [];
    averageSqGrad = [];
    
    % Initialize dynamic amplitude coefficient
    global ampCoeff;
    ampCoeff = 10;
    
    % Create data indices for batching
    numTrain = size(XTrain, 4);
    trainInd = 1:numTrain;
    
    % Training loop
    while epoch < options.maxEpochs && patienceCounter < options.validationPatience && ~stopTraining  
        epoch = epoch + 1;
        
        % Shuffle training data
        trainInd = trainInd(randperm(length(trainInd)));
        
        % Loop over mini-batches
        for i = 1:floor(numTrain/options.miniBatchSize)
            iteration = iteration + 1;
            
            % Get mini-batch
            batchInd = trainInd((i-1)*options.miniBatchSize+1:min(i*options.miniBatchSize,numTrain));
            X = XTrain(:,:,:,batchInd);
            Y = YTrain(batchInd,:);
            
            % Convert to dlarray and transfer to GPU if needed
            dlX = dlarray(X, 'SSCB');
            dlY = dlarray(Y', 'CB');
            if (options.executionEnvironment == "auto" && canUseGPU) || options.executionEnvironment == "gpu"
                dlX = gpuArray(dlX);
                dlY = gpuArray(dlY);
            end
            
            % Train with optimized model for absolute value estimation
            [genGradients, genLoss, ampLoss, phaseLoss, reconLoss, normLoss, signLoss] = ...
                dlfeval(@modelGradients_abs, dlnet, dlX, dlY, P, options.useReconLoss, options.adaptiveLossWeights, utils);
            
            % Apply gradient clipping
            gradientThreshold = 1e-3;
            genGradients = dlupdate(@(g) utils.thresholdL2Norm(g, gradientThreshold), genGradients);
            
            % Update weights
            [dlnet, averageGrad, averageSqGrad] = adamupdate(dlnet, genGradients, ...
                averageGrad, averageSqGrad, iteration, options.initialLearnRate);
            
            % Log training progress
            if strcmpi(options.plotProgress, "gui")
                if options.useReconLoss
                    recordMetrics(monitor, iteration, ...
                        TotalLoss=extractdata(genLoss), ...
                        AmpLoss=extractdata(ampLoss), ...
                        PhaseLoss=extractdata(phaseLoss), ...
                        ReconLoss=extractdata(reconLoss), ...
                        NormLoss=extractdata(normLoss), ...
                        SignLoss=extractdata(signLoss));
                else
                    recordMetrics(monitor, iteration, ...
                        TotalLoss=extractdata(genLoss), ...
                        AmpLoss=extractdata(ampLoss), ...
                        PhaseLoss=extractdata(phaseLoss), ...
                        SignLoss=extractdata(signLoss));
                end
                
                updateInfo(monitor, Epoch=epoch, LearningRate=options.initialLearnRate);
                stopTraining = monitor.Stop;
            end
            
            % Validation check
            if mod(iteration, options.validationFrequency) == 0
                [validationLoss, valAmpLoss, valPhaseLoss] = modelValidation_abs(dlnet, XValidation, YValidation, ...
                    P, options.miniBatchSize, options.executionEnvironment, options.useReconLoss, options.adaptiveLossWeights, utils);
                
                % Early stopping check
                if validationLoss < bestValidationLoss
                    bestValidationLoss = validationLoss;
                    patienceCounter = 0;
                    bestDlnet = dlnet;
                else
                    patienceCounter = patienceCounter + 1;
                end

                dlX_val = dlarray(XValidation, 'SSCB');
                if (options.executionEnvironment == "auto" && canUseGPU) || options.executionEnvironment == "gpu"
                    dlX_val = gpuArray(dlX_val);
                end
                
                YValPred = predict(dlnet, dlX_val);
                YValTrue = YValidation';
                
                if options.showSignAccuracy
                    % Calculate phase sign accuracy (with ambiguity allowed)
                    phase_true = YValidation(:, number_of_modes+1:end)';
                    phase_pred = YValPred(number_of_modes+1:end,:);
                    
                    % Convert normalized values to signs
                    true_signs = sign(phase_true);
                    pred_signs = sign(phase_pred);
                    
                    valSignAccuracy = utils.calculateRelativeSignAccuracy(pred_signs, true_signs);
                    
                    % Log it to monitor
                    if strcmpi(options.plotProgress, "gui")
                        recordMetrics(monitor, iteration, ValSignAccuracy=valSignAccuracy);
                    end
                end
                
                % Log validation metrics
                if strcmpi(options.plotProgress, "gui")
                    recordMetrics(monitor, iteration, ...
                        ValidationLoss=validationLoss, ...
                        ValAmpLoss=valAmpLoss, ...
                        ValPhaseLoss=valPhaseLoss);
                end
            end

            % Intermediate evaluation
            if options.evaluationFrequency > 0 && mod(iteration, options.evaluationFrequency) == 0
                performIntermediateEvaluation(dlnet, XValidation, YValidation, number_of_modes, P);
            end
        end
        
        if patienceCounter >= options.validationPatience || stopTraining
            break;
        end
    end
    
    % Use best network
    dlnet = bestDlnet;
    
    save('absolute_model.mat', 'dlnet', 'ampCoeff', 'number_of_modes');
    disp('Final absolute value model saved to absolute_model.mat');
end

function performIntermediateEvaluation(dlnet, X_val, Y_val, number_of_modes, P)
    % Get utils
    utils = mmf_utils();
    
    % Select a small subset for visualization
    evalSize = min(8, size(X_val, 4));
    indices = randperm(size(X_val, 4), evalSize);
    X_eval = X_val(:,:,:,indices);
    Y_eval = Y_val(indices,:)';
    
    % Get predictions
    dlX = dlarray(X_eval, 'SSCB');
    if canUseGPU
        dlX = gpuArray(dlX);
    end
    
    Y_pred = predict(dlnet, dlX);
    Y_pred = extractdata(Y_pred);
    
    % Extract amplitudes and phases
    amp_pred = Y_pred(1:number_of_modes, :);
    phase_pred = Y_pred(number_of_modes+1:end, :);
    
    % Extract true values
    amp_true = Y_eval(1:number_of_modes, :);
    phase_true = Y_eval(number_of_modes+1:end, :);
    
    % Create complex weights for reconstruction
    weights_pred = zeros(evalSize, number_of_modes);
    weights_true = zeros(evalSize, number_of_modes);
    
    for i = 1:evalSize
        % First mode (reference mode)
        weights_pred(i, 1) = amp_pred(1, i);
        weights_true(i, 1) = amp_true(1, i);
        
        % Remaining modes with phase
        for m = 2:number_of_modes
            phase_pred_val = phase_pred(m-1, i);
            phase_true_val = phase_true(m-1, i);
            
            % Convert normalized phase to radians
            phase_pred_radians = phase_pred_val * pi;
            phase_true_radians = phase_true_val * pi;
            
            weights_pred(i, m) = amp_pred(m, i) * exp(1i * phase_pred_radians);
            weights_true(i, m) = amp_true(m, i) * exp(1i * phase_true_radians);
        end
    end
    
    % Create reconstructions using precomputed P
    [recons_pred, ~] = mmf_build_image(number_of_modes, size(X_eval, 1), evalSize, weights_pred, false, 0, P);
    [recons_true, ~] = mmf_build_image(number_of_modes, size(X_eval, 1), evalSize, weights_true, false, 0, P);
    
    % Calculate correlations
    correlations = zeros(evalSize, 1);
    for i = 1:evalSize
        correlations(i) = corr2(extract(X_eval(:,:,:,i)), extract(recons_pred(:,:,:,i)));
    end
    
    % Display results
    fprintf('\nIntermediate Evaluation:\n');
    fprintf('  Mean correlation: %.4f\n', mean(correlations));
    
    % Find existing figure or create a new one with a specific tag
    fig_recon = findobj('Type', 'figure', 'Tag', 'IntermediateEvaluationFigure');
    if isempty(fig_recon)
        fig_recon = figure('Name', 'Intermediate Evaluation', 'Position', [100, 100, 1200, 500], ...
                     'Tag', 'IntermediateEvaluationFigure');
    else
        figure(fig_recon); % Make the existing figure current
        clf; % Clear the figure
    end
    
    % Update title with the mean correlation
    sgtitle(sprintf('Reconstruction Evaluation - Mean Correlation: %.4f', mean(correlations)), 'FontSize', 14);
    
    % Extract data from dlarrays for visualization
    X_eval_data = extract(X_eval);
    recons_pred_data = extract(recons_pred);
    recons_true_data = extract(recons_true);
    
    % Plot the comparisons
    for i = 1:min(4, evalSize)
        % Original image
        subplot(3, 4, i);
        imagesc(X_eval_data(:,:,:,i));
        axis image off;
        title(sprintf('Original #%d', i));
        
        % True reconstruction
        subplot(3, 4, i+4);
        imagesc(recons_true_data(:,:,:,i));
        axis image off;
        title('True Recon');
        
        % Predicted reconstruction
        subplot(3, 4, i+8);
        imagesc(recons_pred_data(:,:,:,i));
        axis image off;
        title(sprintf('Pred Recon (r=%.3f)', correlations(i)));
    end
    
    % Create new figure for phase sign analysis
    fig_signs = findobj('Type', 'figure', 'Tag', 'PhaseSignAnalysisFigure');
    if isempty(fig_signs)
        fig_signs = figure('Name', 'Phase Sign Analysis', 'Position', [100, 600, 1200, 500], ...
                     'Tag', 'PhaseSignAnalysisFigure');
    else
        figure(fig_signs);
        clf;
    end
    
    % Convert phases to signs for easier visualization
    true_signs = sign(phase_true*pi);
    pred_signs = sign(phase_pred*pi);
    
    % Calculate sign accuracy without global ambiguity
    sign_matches = sum(true_signs == pred_signs, 1) / (number_of_modes-1);
    overall_accuracy = mean(sign_matches);
    
    % Plot sign patterns and accuracy
    sgtitle(sprintf('Phase Sign Analysis - Overall Accuracy: %.2f%%', overall_accuracy*100), 'FontSize', 14);
    
    % Show sign patterns for each sample
    for i = 1:min(4, evalSize)
        % Create sign pattern visualization
        subplot(2, 4, i);
        hold on;
        
        % Plot true signs (circles)
        y_true = ones(number_of_modes-1, 1) * 0.7;
        scatter(1:(number_of_modes-1), y_true, 80, sign_to_color(true_signs(:,i)), 'o', 'filled', 'MarkerEdgeColor', 'k');
        
        % Plot predicted signs (squares)
        y_pred = ones(number_of_modes-1, 1) * 0.3;
        scatter(1:(number_of_modes-1), y_pred, 80, sign_to_color(pred_signs(:,i)), 's', 'filled', 'MarkerEdgeColor', 'k');
        
        % Add connecting lines, green for correct, red for incorrect
        for m = 1:number_of_modes-1
            if true_signs(m,i) == pred_signs(m,i)
                line([m, m], [0.3, 0.7], 'Color', 'g', 'LineWidth', 2);
            else
                line([m, m], [0.3, 0.7], 'Color', 'r', 'LineWidth', 2);
            end
        end
        
        % Set plot limits and labels
        xlim([0.5, number_of_modes-0.5]);
        ylim([0, 1]);
        title(sprintf('Sample #%d (Acc: %.2f%%)', i, sign_matches(i)*100));
        xticks(1:(number_of_modes-1));
        xticklabels(arrayfun(@(x) sprintf('%d', x+1), 1:(number_of_modes-1), 'UniformOutput', false));
        xlabel('Mode Index');
        yticks([]);
    end
    
    % Compute mode-specific sign accuracy - without global flip
    mode_accuracy = zeros(number_of_modes-1, 1);
    for m = 1:number_of_modes-1
        mode_accuracy(m) = mean(true_signs(m,:) == pred_signs(m,:));
    end
    
    % Plot mode-specific accuracy
    subplot(2, 4, 5:8);
    bar(mode_accuracy * 100);
    hold on;
    yline(overall_accuracy * 100, 'r--', 'LineWidth', 2);
    xlabel('Mode Index');
    ylabel('Accuracy (%)');
    title('Per-Mode Sign Accuracy');
    xticks(1:(number_of_modes-1));
    xticklabels(arrayfun(@(x) sprintf('%d', x+1), 1:(number_of_modes-1), 'UniformOutput', false));
    ylim([0, 100]);
    grid on;
    
    % Add legend
    legend({'Per-mode Accuracy', 'Average'}, 'Location', 'best');
    
    % Update both figures
    drawnow;
end

function colors = sign_to_color(signs)
    % Convert signs to colors (blue for positive, red for negative)
    colors = zeros(length(signs), 3);
    for i = 1:length(signs)
        if signs(i) > 0
            colors(i,:) = [0.1, 0.6, 0.9];  % Blue for positive
        else
            colors(i,:) = [0.9, 0.1, 0.1];  % Red for negative
        end
    end
end

% Create CNN model
function dlnet = createCNNModel(inputSize, outputSize)
    numModes = (outputSize + 1) / 2;
    % Shared feature extraction
    layers = [
        imageInputLayer([inputSize 1], 'Name', 'input', 'Normalization', 'none')
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1')
        leakyReluLayer(0.2, 'Name', 'relu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
        convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv3')
        leakyReluLayer(0.2, 'Name', 'relu3')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
        convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'conv4')
        leakyReluLayer(0.2, 'Name', 'relu4')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool4')
        fullyConnectedLayer(512, 'Name', 'fc1')
        leakyReluLayer(0.2, 'Name', 'relu5')
        dropoutLayer(0.3, 'Name', 'drop1')
        fullyConnectedLayer(256, 'Name', 'fc2')
        leakyReluLayer(0.2, 'Name', 'relu6')
        dropoutLayer(0.2, 'Name', 'drop2')
        fullyConnectedLayer(128, 'Name', 'fc3')
        leakyReluLayer(0.2, 'Name', 'relu7')
    ];
    lgraph = layerGraph(layers);
    % Amplitude branch
    ampBranch = [
        fullyConnectedLayer(numModes, 'Name', 'AmpOut')
        sigmoidLayer('Name', 'AmpSigmoid')
    ];
    lgraph = addLayers(lgraph, ampBranch);
    lgraph = connectLayers(lgraph, 'relu7', 'AmpOut');
    % Phase branch
    phaseBranch = [
        fullyConnectedLayer(numModes-1, 'Name', 'PhaseOut')
        tanhLayer('Name', 'PhaseTanh')
    ];
    lgraph = addLayers(lgraph, phaseBranch);
    lgraph = connectLayers(lgraph, 'relu7', 'PhaseOut');
    % Concatenate outputs
    lgraph = addLayers(lgraph, depthConcatenationLayer(2, 'Name', 'OutputConcat'));
    lgraph = connectLayers(lgraph, 'AmpSigmoid', 'OutputConcat/in1');
    lgraph = connectLayers(lgraph, 'PhaseTanh', 'OutputConcat/in2');
    dlnet = dlnetwork(lgraph);
end

% Create MLP model
function dlnet = createMLPModel(inputSize, outputSize)
    numModes = (outputSize + 1) / 2;
    % Shared feature extraction
    layers = [
        imageInputLayer([inputSize 1],'Name','input')
        fullyConnectedLayer(1024)
        leakyReluLayer(0.2)
        fullyConnectedLayer(512)
        leakyReluLayer(0.2)
        fullyConnectedLayer(256)
        leakyReluLayer(0.2)
        fullyConnectedLayer(128)
        leakyReluLayer(0.2)
        fullyConnectedLayer(64)
        leakyReluLayer(0.2)
    ];
    lgraph = layerGraph(layers);
    % Amplitude branch
    ampBranch = [
        fullyConnectedLayer(numModes, 'Name', 'AmpOut')
        sigmoidLayer('Name', 'AmpSigmoid')
    ];
    lgraph = addLayers(lgraph, ampBranch);
    lgraph = connectLayers(lgraph, 'leakyrelu_5', 'AmpOut');
    % Phase branch
    phaseBranch = [
        fullyConnectedLayer(numModes-1, 'Name', 'PhaseOut')
        tanhLayer('Name', 'PhaseTanh')
    ];
    lgraph = addLayers(lgraph, phaseBranch);
    lgraph = connectLayers(lgraph, 'leakyrelu_5', 'PhaseOut');
    % Concatenate outputs
    lgraph = addLayers(lgraph, depthConcatenationLayer(2, 'Name', 'OutputConcat'));
    lgraph = connectLayers(lgraph, 'AmpSigmoid', 'OutputConcat/in1');
    lgraph = connectLayers(lgraph, 'PhaseTanh', 'OutputConcat/in2');
    dlnet = dlnetwork(lgraph);
end

function [gradients, loss, ampLoss, phaseLoss, reconLoss, normLoss, signLoss] = modelGradients_abs(dlnet, dlX, dlY, P, useReconLoss, adaptiveLossWeights, utils)
    % Persistent variables for Dynamic Loss Normalization using EMA
    persistent loss_ema weights_initialized iteration_count loss_names_internal num_tasks_internal;

    if isempty(weights_initialized)
        weights_initialized = false;
        iteration_count = 0;
        loss_ema = [];
        % Define task names internally for consistency
        loss_names_internal = {'Amp', 'Phase'}; % Include Sign task
        if useReconLoss
            loss_names_internal{end+1} = 'Recon';
        end
        num_tasks_internal = length(loss_names_internal);
    end
    iteration_count = iteration_count + 1;

    % Forward pass - get initial predictions
    dlY_pred = forward(dlnet, dlX);
    dlY_pred = real(dlY_pred);

    % Determine number of modes
    totalOut = size(dlY_pred, 1);
    number_of_modes = (totalOut + 1) / 2;

    % Split predictions and ground truth
    amps_raw = dlY_pred(1:number_of_modes, :);
    phase_pred = dlY_pred(number_of_modes + 1:end, :);
    amps_true = dlY(1:number_of_modes, :);
    phase_true = dlY(number_of_modes + 1:end, :); % Ground truth normalized to [-1,1] (assuming canonical)

    % --- Calculate Individual Task Losses ---
    % 1. Amplitude Loss (L2 Fidelity)
    ampLoss = l2loss(amps_raw, amps_true);
    
    % 2. Phase Loss - both magnitude and direction
    % Use combined L2 loss on the full phase values instead of just magnitude
    phaseLoss = l2loss(abs(phase_pred), abs(phase_true));
    
    % 3. Phase Sign Loss placeholder
    signLoss = zeros(1, 'like', phaseLoss);
    
    % Store current losses
    current_losses = [ampLoss, phaseLoss];

    % 4. Reconstruction Loss (if enabled)
    reconLoss = ones(1, 'like', ampLoss); % Initialize
    if useReconLoss
        % Create complex weights using predicted amplitudes and phases
        weights_pred = dlarray(zeros(number_of_modes, size(amps_raw, 2), 'like', amps_raw), 'CB');
        weights_true = dlarray(zeros(number_of_modes, size(amps_true, 2), 'like', amps_true), 'CB');

        weights_pred(1,:) = amps_raw(1,:); % First mode phase is 0
        weights_true(1,:) = amps_true(1,:);

        for m = 2:number_of_modes
            weights_pred(m,:) = amps_raw(m,:) .* exp(1i * (phase_pred(m-1,:) * pi));
            weights_true(m,:) = amps_true(m,:) .* exp(1i * (phase_true(m-1,:) * pi));
        end

        % Build linear image using precomputed P
        [linear_recon, ~] = mmf_build_image(number_of_modes, size(dlX, 1), size(dlX, 4), weights_pred', false, 0, P);
        [dlX_linear, ~] = mmf_build_image(number_of_modes, size(dlX, 1), size(dlX, 4), weights_true', false, 0, P);

        % Correlation-based loss using utility function
        reconLoss = 1 - utils.dlCorr(dlX_linear, linear_recon);

        current_losses = [current_losses, reconLoss]; % Add recon loss to the list
    end

    % Combine losses based on adaptiveLossWeights flag
    if adaptiveLossWeights
        % --- Combine Losses using Dynamic Loss Normalization (EMA-based) ---
        alpha = 0.9; % EMA decay factor (closer to 1 = slower adaptation)
        epsilon = 1e-8; % For numerical stability

        current_losses_data = extract(current_losses); % Extract data for EMA update, ensure on CPU

        if ~weights_initialized || any(isnan(loss_ema)) || isempty(loss_ema)
            loss_ema = max(current_losses_data, epsilon); % Initialize EMA, ensure positive
            weights_initialized = true;
            % Use equal weights for the first iteration
            task_weights = ones(1, num_tasks_internal, 'like', current_losses) / num_tasks_internal;
            if isa(current_losses, 'gpuArray') % Ensure weights are on GPU if losses are
                task_weights = gpuArray(task_weights);
            end
        else
            % Update EMA
            loss_ema = alpha * loss_ema + (1 - alpha) * max(current_losses_data, epsilon); % Ensure EMA stays positive

            % Calculate weights inversely proportional to EMA loss magnitude
            inv_ema = 1 ./ loss_ema; % EMA is already kept > 0
            task_weights = inv_ema / sum(inv_ema); % Normalize weights to sum to 1

            % Convert weights back to dlarray and move to GPU if necessary
            task_weights = dlarray(task_weights, 'CB'); % Assuming 'CB' format matches losses
            if isa(current_losses, 'gpuArray')
                task_weights = gpuArray(task_weights);
            end
        end

        % Calculate final weighted loss
        loss = sum(task_weights .* current_losses);

        % Log weights and EMA for debugging (optional)
        if mod(iteration_count, 100) == 1 % Log occasionally
            fprintf('Dynamic Loss Weights (Iter %d):\n', iteration_count);
            temp_weights = extractdata(gather(task_weights)); % Ensure on CPU for printing
            for i = 1:num_tasks_internal
                fprintf('  %s: Loss=%.4f, EMA=%.4f, Weight=%.4f\n', ...
                    loss_names_internal{i}, current_losses_data(i), loss_ema(i), temp_weights(i));
            end
            fprintf('  Total Loss = %.4f\n', extractdata(gather(loss)));
        end
    else
        % Simple weighted average with fixed weights
        amp_weight = 200.0;
        phase_weight = 1.0;
        sign_weight = 1.0;
        recon_weight = 1.0;
        
        if useReconLoss
            loss = (amp_weight * ampLoss + phase_weight * phaseLoss + sign_weight * signLoss + recon_weight * reconLoss) / (amp_weight + phase_weight + sign_weight + recon_weight);
        else
            loss = (amp_weight * ampLoss + phase_weight * phaseLoss + sign_weight * signLoss) / (amp_weight + phase_weight + sign_weight);
        end
        
        if mod(iteration_count, 100) == 1 % Log occasionally
            fprintf('Fixed Loss Weights (Iter %d): Amp=%.1f, Phase=%.1f, Sign=%.1f\n', ...
                iteration_count, amp_weight, phase_weight, sign_weight);
            fprintf('  Losses: Amp=%.4f, Phase=%.4f, Sign=%.4f, Total=%.4f\n', ...
                extractdata(gather(ampLoss)), extractdata(gather(phaseLoss)), ...
                extractdata(gather(signLoss)), extractdata(gather(loss)));
        end
    end

    % Placeholder for normLoss if needed elsewhere
    normLoss = zeros(1, 'like', ampLoss);

    % Compute gradients
    gradients = dlgradient(loss, dlnet.Learnables);
end

function [totalLoss, ampLoss, phaseLoss] = modelValidation_abs(dlnet, X, Y, P, batchSize, executionEnvironment, useReconLoss, adaptiveLossWeights, utils)
    numValidation = size(X, 4);
    totalLoss = 0;
    ampLoss = 0;
    phaseLoss = 0;

    for i = 1:floor(numValidation/batchSize)
        batchInd = (i-1)*batchSize+1:min(i*batchSize,numValidation);
        dlX = dlarray(X(:,:,:,batchInd), 'SSCB');
        dlY = dlarray(Y(batchInd,:)', 'CB');

        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
            dlY = gpuArray(dlY);
        end

        [~, batchLoss, batchAmpLoss, batchPhaseLoss, ~, ~, ~] = dlfeval(@modelGradients_abs, dlnet, dlX, dlY, P, useReconLoss, adaptiveLossWeights, utils);
        totalLoss = totalLoss + extractdata(batchLoss);
        ampLoss = ampLoss + extractdata(batchAmpLoss);
        phaseLoss = phaseLoss + extractdata(batchPhaseLoss);
    end

    n = floor(numValidation/batchSize);
    totalLoss = totalLoss / n;
    ampLoss = ampLoss / n;
    phaseLoss = phaseLoss / n;
end

% Gradient clipping utility
function grad = thresholdL2Norm(grad, threshold)
    gradNorm = sqrt(sum(grad(:).^2));
    if gradNorm > threshold
        grad = (threshold/gradNorm) * grad;
    end
end