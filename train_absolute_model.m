% filepath: c:\Users\nicks\MATLAB\Projects\MT2_prak\Versuch 3\train_absolute_model.m
% train_absolute_model.m - Trains a network to predict absolute amplitudes and phases

function dlnet = train_absolute_model(XTrain, YTrain, XValidation, YValidation, options)
    % Parse options or use defaults
    if nargin < 5
        options = struct();
    end
    
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
    alphaCoeff = 0.1;  % Smoothing factor for phase 2
    
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
                dlfeval(@modelGradients_abs, dlnet, dlX, dlY, options.useReconLoss, options.adaptiveLossWeights);
            
            % Apply gradient clipping
            gradientThreshold = 1e-3;
            genGradients = dlupdate(@(g) thresholdL2Norm(g, gradientThreshold), genGradients);
            
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
                    options.miniBatchSize, options.executionEnvironment, options.useReconLoss, options.adaptiveLossWeights);
                
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
                    
                    valSignAccuracy = calculateRelativeSignAccuracy(pred_signs, true_signs);
                    
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
                performIntermediateEvaluation(dlnet, XValidation, YValidation, number_of_modes);
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

function accuracy = calculateRelativeSignAccuracy(pred_signs, true_signs)
    % Calculate sign accuracy without allowing for global phase ambiguity
    % (since the labels now use a canonical representation)
    match_count = sum(sign(pred_signs) == true_signs, 'all');
    total_elements = numel(pred_signs);
    
    accuracy = match_count / total_elements;
end

function performIntermediateEvaluation(dlnet, X_val, Y_val, number_of_modes)
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
            phase_pred_radians = phase_pred_val * pi; % (phase_pred_val * 2 - 1) * pi;
            phase_true_radians = phase_true_val * pi;
            
            weights_pred(i, m) = amp_pred(m, i) * exp(1i * phase_pred_radians);
            weights_true(i, m) = amp_true(m, i) * exp(1i * phase_true_radians);
        end
    end
    
    % Create reconstructions
    [recons_pred, ~] = mmf_build_image(number_of_modes, size(X_eval, 1), evalSize, weights_pred, true);
    [recons_true, ~] = mmf_build_image(number_of_modes, size(X_eval, 1), evalSize, weights_true, true);
    
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

% Create VGG model (simplified version with reduced parameters)
function dlnet = createVGGModel(inputSize, outputSize)
    % Determine numModes from outputSize
    numModes = (outputSize + 1) / 2;

    layers = [
        imageInputLayer([inputSize 1],'Name','input','Normalization','none') % Grayscale input

        % Block 1 (Reduced Filters)
        convolution2dLayer(3,32,'Padding','same','Name','conv1_1') % Reduced from 64
        reluLayer('Name','relu1_1')
        convolution2dLayer(3,32,'Padding','same','Name','conv1_2') % Reduced from 64
        reluLayer('Name','relu1_2')
        maxPooling2dLayer(2,'Stride',2,'Name','pool1')

        % Block 2 (Reduced Filters)
        convolution2dLayer(3,64,'Padding','same','Name','conv2_1') % Reduced from 128
        reluLayer('Name','relu2_1')
        convolution2dLayer(3,64,'Padding','same','Name','conv2_2') % Reduced from 128
        reluLayer('Name','relu2_2')
        maxPooling2dLayer(2,'Stride',2,'Name','pool2')

        % Block 3 (Reduced Filters)
        convolution2dLayer(3,128,'Padding','same','Name','conv3_1') % Reduced from 256
        reluLayer('Name','relu3_1')
        convolution2dLayer(3,128,'Padding','same','Name','conv3_2') % Reduced from 256
        reluLayer('Name','relu3_2')
        convolution2dLayer(3,128,'Padding','same','Name','conv3_3') % Reduced from 256
        reluLayer('Name','relu3_3')
        % Removed conv3_4 for shallower network
        maxPooling2dLayer(2,'Stride',2,'Name','pool3')

        % Block 4 (Reduced Filters)
        convolution2dLayer(3,256,'Padding','same','Name','conv4_1') % Reduced from 512
        reluLayer('Name','relu4_1')
        convolution2dLayer(3,256,'Padding','same','Name','conv4_2') % Reduced from 512
        reluLayer('Name','relu4_2')
        convolution2dLayer(3,256,'Padding','same','Name','conv4_3') % Reduced from 512
        reluLayer('Name','relu4_3')
        % Removed conv4_4 for shallower network
        maxPooling2dLayer(2,'Stride',2,'Name','pool4')

        % Block 5 (Reduced Filters)
        convolution2dLayer(3,256,'Padding','same','Name','conv5_1') % Reduced from 512
        reluLayer('Name','relu5_1')
        convolution2dLayer(3,256,'Padding','same','Name','conv5_2') % Reduced from 512
        reluLayer('Name','relu5_2')
        convolution2dLayer(3,256,'Padding','same','Name','conv5_3') % Reduced from 512
        reluLayer('Name','relu5_3')
        % Removed conv5_4 for shallower network
        maxPooling2dLayer(2,'Stride',2,'Name','pool5')

        % --- Replace Large FC layers with Global Average Pooling ---
        globalAveragePooling2dLayer('Name','gap')
        % Removed fc6, relu6, drop6, fc7, relu7, drop7
    ];

    lgraph = layerGraph(layers);

    % --- Amplitude Branch ---
    ampBranch = [
        fullyConnectedLayer(256, 'Name', 'AmpFC1') % Reduced from 512
        reluLayer('Name', 'AmpReLU1')
        dropoutLayer(0.3, 'Name', 'AmpDrop1')
        fullyConnectedLayer(numModes, 'Name', 'AmpOut')
        functionLayer(@normalizeAmplitudesFunc, 'Name', 'AmpNorm', 'Formattable', true, 'Acceleratable', true)
    ];
    lgraph = addLayers(lgraph, ampBranch);
    lgraph = connectLayers(lgraph, 'gap', 'AmpFC1'); % Connect from GAP layer

    % --- Phase Branch ---
    phaseBranch = [
        fullyConnectedLayer(256, 'Name', 'PhaseFC1') % Reduced from 512
        reluLayer('Name', 'PhaseReLU1')
        dropoutLayer(0.3, 'Name', 'PhaseDrop1')
        fullyConnectedLayer(numModes - 1, 'Name', 'PhaseOut') % numModes-1 phases
        %sigmoidLayer('Name', 'PhaseSigmoid')
        tanhLayer('Name', 'PhaseSigmoid') % Use tanh for phase output
    ];
    lgraph = addLayers(lgraph, phaseBranch);
    lgraph = connectLayers(lgraph, 'gap', 'PhaseFC1'); % Connect from GAP layer

    % --- Concatenate Outputs ---
    lgraph = addLayers(lgraph, depthConcatenationLayer(2, 'Name', 'OutputConcat'));
    lgraph = connectLayers(lgraph, 'AmpNorm', 'OutputConcat/in1');
    lgraph = connectLayers(lgraph, 'PhaseSigmoid', 'OutputConcat/in2');

    dlnet = dlnetwork(lgraph);
    disp('Reduced VGG-based model with split output created.');
end

function dlnet = createResNet(inputSize, outputSize)
    % Create a ResNet-based architecture for phase sign prediction
    % This network uses residual connections for better gradient flow

    numModes = (outputSize + 1) / 2;

    layers = [
        imageInputLayer([inputSize 1], 'Name', 'input', 'Normalization', 'none')
        % Initial convolution
        convolution2dLayer(7, 32, 'Stride', 2, 'Padding', 'same', 'Name', 'conv1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(3, 'Stride', 2, 'Padding', 'same', 'Name', 'pool1')
    ];

    lgraph = layerGraph(layers);

    % Residual block function
    function lgraph = addResidualBlock(lgraph, numFilters, blockName, inName)
        % Create main path
        mainPath = [
            convolution2dLayer(3, numFilters, 'Padding', 'same', 'Name', [blockName '_conv1'])
            reluLayer('Name', [blockName '_relu1'])
            convolution2dLayer(3, numFilters, 'Padding', 'same', 'Name', [blockName '_conv2'])
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
            ];
            lgraph = addLayers(lgraph, skipPath);
            lgraph = connectLayers(lgraph, inName, [blockName '_skip']);
            skipOutput = [blockName '_skip'];
        end

        % Add layer for combining main and skip paths
        add = additionLayer(2, 'Name', [blockName '_add']);
        relu = reluLayer('Name', [blockName '_relu_out']);

        lgraph = addLayers(lgraph, [add; relu]);
        lgraph = connectLayers(lgraph, [blockName '_conv2'], [blockName '_add/in1']);
        lgraph = connectLayers(lgraph, skipOutput, [blockName '_add/in2']);
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

    % Global average pooling
    lgraph = addLayers(lgraph, globalAveragePooling2dLayer('Name', 'gap'));
    lgraph = connectLayers(lgraph, 'block3b_relu_out', 'gap');

    % Amplitude branch
    ampBranch = [
        fullyConnectedLayer(numModes, 'Name', 'AmpOut')
        sigmoidLayer('Name', 'AmpSigmoid')
    ];
    lgraph = addLayers(lgraph, ampBranch);
    lgraph = connectLayers(lgraph, 'gap', 'AmpOut');

    % Phase branch
    phaseBranch = [
        fullyConnectedLayer(numModes-1, 'Name', 'PhaseOut')
        tanhLayer('Name', 'PhaseTanh')
    ];
    lgraph = addLayers(lgraph, phaseBranch);
    lgraph = connectLayers(lgraph, 'gap', 'PhaseOut');

    % Concatenate outputs
    lgraph = addLayers(lgraph, depthConcatenationLayer(2, 'Name', 'OutputConcat'));
    lgraph = connectLayers(lgraph, 'AmpSigmoid', 'OutputConcat/in1');
    lgraph = connectLayers(lgraph, 'PhaseTanh', 'OutputConcat/in2');

    % Create network
    dlnet = dlnetwork(lgraph);
end

function dlnet = createCustomModel(inputSize, outputSize)
    % Determine numModes from outputSize
    numModes = (outputSize + 1) / 2;
    
    % Handle input size - ensure it's compatible with EfficientNet
    requiredInputSize = [224 224]; % Standard EfficientNet input size
    if ~isequal(inputSize, requiredInputSize)
        warning('Input size not %dx%d. Resizing input layer. Ensure data matches.', requiredInputSize(1), requiredInputSize(2));
        inputSize = requiredInputSize;
    end
    
    % Create EfficientNet-B0 backbone
    lgraph = efficientnetb0('Weights', 'none');
    
    % --- Modification for Grayscale Input ---
    % Get the input layer
    inputLayer = lgraph.Layers(1);
    % Create a new input layer for grayscale
    newInputLayer = imageInputLayer([inputSize 1], 'Name', inputLayer.Name, 'Normalization', inputLayer.Normalization);
    lgraph = replaceLayer(lgraph, inputLayer.Name, newInputLayer);
    
    % Get the first convolution layer
    firstConvLayer = lgraph.Layers(2); % Usually the first conv layer
    % Create a new conv layer with adjusted weights for 1 channel input
    originalWeights = firstConvLayer.Weights;
    % Average weights across the color channels (simple approach)
    newWeights = mean(originalWeights, 3); 
    newConvLayer = convolution2dLayer(firstConvLayer.FilterSize, firstConvLayer.NumFilters, ...
        'Name', firstConvLayer.Name, ...
        'Padding', firstConvLayer.PaddingSize, ...
        'Stride', firstConvLayer.Stride, ...
        'BiasLearnRateFactor', firstConvLayer.BiasLearnRateFactor, ...
        'WeightLearnRateFactor', firstConvLayer.WeightLearnRateFactor);
    newConvLayer.Weights = newWeights;
    newConvLayer.Bias = firstConvLayer.Bias;
    lgraph = replaceLayer(lgraph, firstConvLayer.Name, newConvLayer);
    % --- End Grayscale Modification ---

    % Remove classification layers
    lgraph = removeLayers(lgraph, {'efficientnet-b0|model|head|dense|MatMul', 'Softmax', 'classification'});
    
    % Find the last layer name after removals (likely the pooling or dropout before FC)
    lastBackboneLayerName = lgraph.Layers(end).Name; 
    
    % Define feature dimension from EfficientNet-B0 output
    featureDim = 1280; % EfficientNet-B0 output features before classifier

    % Add Flatten layer if the output is not already flat
    % Check the output size of the lastBackboneLayerName if necessary
    % Assuming the output is [1, 1, featureDim, BatchSize] after global pooling
    lgraph = addLayers(lgraph, flattenLayer('Name', 'Flatten'));
    lgraph = connectLayers(lgraph, lastBackboneLayerName, 'Flatten');
    lastProcessedLayerName = 'Flatten'; % Keep track of the last layer

    % --- Transformer Block ---
    numHeads = 8; % Number of attention heads
    headDim = featureDim / numHeads;
    if mod(featureDim, numHeads) ~= 0
        error('featureDim must be divisible by numHeads');
    end

    % Input Layer Normalization (optional but common)
    lgraph = addLayers(lgraph, layerNormalizationLayer('Name', 'TransformerInputNorm'));
    lgraph = connectLayers(lgraph, lastProcessedLayerName, 'TransformerInputNorm');
    transformerInputName = 'TransformerInputNorm';

    % Multi-Head Self-Attention Simulation
    % 1. QKV Projections
    lgraph = addLayers(lgraph, fullyConnectedLayer(featureDim, 'Name', 'fc_q'));
    lgraph = addLayers(lgraph, fullyConnectedLayer(featureDim, 'Name', 'fc_k'));
    lgraph = addLayers(lgraph, fullyConnectedLayer(featureDim, 'Name', 'fc_v'));
    lgraph = connectLayers(lgraph, transformerInputName, 'fc_q');
    lgraph = connectLayers(lgraph, transformerInputName, 'fc_k');
    lgraph = connectLayers(lgraph, transformerInputName, 'fc_v');

    % 2. Reshape for Multi-Head and Attention Calculation (using functionLayer)
    lgraph = addLayers(lgraph, functionLayer(@multiHeadAttentionFunc, ...
        'Name', 'MultiHeadAttention', ...
        'NumInputs', 3, ... % Q, K, V
        'OutputNames', {'attended_features'}, ...
        'Formattable', true, ... % Indicate support for dlarray formats
        'Acceleratable', true)); % Enable acceleration if possible
        
    % Connect Q, K, V to the function layer
    lgraph = connectLayers(lgraph, 'fc_q', 'MultiHeadAttention/in1');
    lgraph = connectLayers(lgraph, 'fc_k', 'MultiHeadAttention/in2');
    lgraph = connectLayers(lgraph, 'fc_v', 'MultiHeadAttention/in3');

    % 3. Output Projection
    lgraph = addLayers(lgraph, fullyConnectedLayer(featureDim, 'Name', 'AttentionOut'));
    lgraph = connectLayers(lgraph, 'MultiHeadAttention/attended_features', 'AttentionOut');
    
    % 4. Add & Norm (Skip Connection 1)
    lgraph = addLayers(lgraph, additionLayer(2, 'Name', 'SkipConnection1'));
    lgraph = addLayers(lgraph, layerNormalizationLayer('Name', 'AttentionNorm'));
    lgraph = connectLayers(lgraph, 'AttentionOut', 'SkipConnection1/in1');
    lgraph = connectLayers(lgraph, transformerInputName, 'SkipConnection1/in2'); % Connect back to input of attention block
    lgraph = connectLayers(lgraph, 'SkipConnection1', 'AttentionNorm');
    
    % Feed-Forward Network (FFN)
    ffnInputName = 'AttentionNorm';
    lgraph = addLayers(lgraph, fullyConnectedLayer(featureDim * 4, 'Name', 'FFN1'));
    lgraph = addLayers(lgraph, functionLayer(@geluFunc, 'Name', 'GELU', 'Formattable', true, 'Acceleratable', true)); % Use functionLayer for GELU
    lgraph = addLayers(lgraph, dropoutLayer(0.1, 'Name', 'FFNDropout'));
    lgraph = addLayers(lgraph, fullyConnectedLayer(featureDim, 'Name', 'FFN2'));
    lgraph = connectLayers(lgraph, ffnInputName, 'FFN1');
    lgraph = connectLayers(lgraph, 'FFN1', 'GELU');
    lgraph = connectLayers(lgraph, 'GELU', 'FFNDropout');
    lgraph = connectLayers(lgraph, 'FFNDropout', 'FFN2');

    % Add & Norm (Skip Connection 2)
    lgraph = addLayers(lgraph, additionLayer(2, 'Name', 'SkipConnection2'));
    lgraph = addLayers(lgraph, layerNormalizationLayer('Name', 'FFNNorm'));
    lgraph = connectLayers(lgraph, 'FFN2', 'SkipConnection2/in1');
    lgraph = connectLayers(lgraph, ffnInputName, 'SkipConnection2/in2'); % Connect back to input of FFN block
    lgraph = connectLayers(lgraph, 'SkipConnection2', 'FFNNorm');
    
    lastProcessedLayerName = 'FFNNorm'; % Output of the transformer block
    % --- End Transformer Block ---

    % --- Decoder Heads ---
    % Amplitude branch
    lgraph = addLayers(lgraph, [
        fullyConnectedLayer(512, 'Name', 'AmpFC1')
        layerNormalizationLayer('Name', 'AmpNorm1') % Use LayerNorm instead of BatchNorm after Transformer
        reluLayer('Name', 'AmpReLU1')
        dropoutLayer(0.3, 'Name', 'AmpDrop1')
        fullyConnectedLayer(256, 'Name', 'AmpFC2')
        layerNormalizationLayer('Name', 'AmpNorm2')
        reluLayer('Name', 'AmpReLU2')
        dropoutLayer(0.2, 'Name', 'AmpDrop2')
        fullyConnectedLayer(numModes, 'Name', 'AmpOut')
        functionLayer(@normalizeAmplitudesFunc, 'Name', 'AmpNorm', 'Formattable', true, 'Acceleratable', true)
    ]);
    lgraph = connectLayers(lgraph, lastProcessedLayerName, 'AmpFC1'); % Connect from transformer output

    % Phase branch
    lgraph = addLayers(lgraph, [
        fullyConnectedLayer(512, 'Name', 'PhaseFC1')
        layerNormalizationLayer('Name', 'PhaseNorm1') % Use LayerNorm
        reluLayer('Name', 'PhaseReLU1')
        dropoutLayer(0.3, 'Name', 'PhaseDrop1')
        fullyConnectedLayer(256, 'Name', 'PhaseFC2')
        layerNormalizationLayer('Name', 'PhaseNorm2')
        reluLayer('Name', 'PhaseReLU2')
        dropoutLayer(0.2, 'Name', 'PhaseDrop2')
        fullyConnectedLayer(numModes - 1, 'Name', 'PhaseOut') % numModes-1 phases
        sigmoidLayer('Name', 'PhaseSigmoid')
    ]);
    lgraph = connectLayers(lgraph, lastProcessedLayerName, 'PhaseFC1'); % Connect from transformer output
    % --- End Decoder Heads ---

    % Concatenate outputs
    lgraph = addLayers(lgraph, depthConcatenationLayer(2, 'Name', 'OutputConcat')); % Use depth concatenation
    lgraph = connectLayers(lgraph, 'AmpNorm', 'OutputConcat/in1');
    lgraph = connectLayers(lgraph, 'PhaseSigmoid', 'OutputConcat/in2');
    
    % Create final network
    dlnet = dlnetwork(lgraph);
    disp('Custom model created successfully.');
end

% --- Helper Functions for functionLayer ---

function Z = multiHeadAttentionFunc(Q, K, V)
    % Simple self-attention without explicit multi-head implementation
    % Get dimensions
    featureDim = size(Q, 1);
    batchSize = size(Q, 2);
    
    % Compute attention scores for each batch element
    scores = zeros(1, batchSize, 'like', Q);
    for b = 1:batchSize
        % Simple dot product attention
        qk = sum(Q(:, b) .* K(:, b));
        scores(1, b) = qk / sqrt(featureDim);
    end
    
    % Apply attention weights
    Z = zeros(featureDim, batchSize, 'like', V);
    for b = 1:batchSize
        Z(:, b) = V(:, b) * scores(1, b);
    end
    
    % Ensure correct format
    Z = dlarray(Z, 'CB');
end


function Y = geluFunc(X)
    % GELU activation function for dlarray
    % GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    coeff = sqrt(2/pi);
    Y = 0.5 * X .* (1 + tanh(coeff * (X + 0.044715 * X.^3)));
end

function Y = normalizeAmplitudesFunc(X)
    % Ensure amplitudes are positive and normalized with L2 norm along channel dim
    % Assumes input X has format 'CB' or 'BC'
    channelDim = find(dims(X) == 'C');
    if isempty(channelDim)
         channelDim = ndims(X); % Assume last dimension if 'C' is not present
    end

    X_pos = max(X, 0); % Ensure positive
    normFactor = sqrt(sum(X_pos.^2, channelDim) + 1e-8);
    Y = X_pos ./ normFactor;
end

function [gradients, loss, ampLoss, phaseLoss, reconLoss, normLoss, signLoss] = modelGradients_abs(dlnet, dlX, dlY, useReconLoss, adaptiveLossWeights)
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

    % Convert predicted phases to [-1, 1] to match canonical true phases
    % phase_pred = (phase_pred - 0.5) * 2;

    % --- Calculate Individual Task Losses ---
    % 1. Amplitude Loss (L2 Fidelity) - Keep diversity separate if needed
    ampLoss = l2loss(amps_raw, amps_true);
    %ampLoss = dlarray(zeros(1, 'like', amps_raw), 'CB'); % Initialize to zero
    
    % 2. Phase Loss - both magnitude and direction
    % Use combined L2 loss on the full phase values instead of just magnitude
    phaseLoss = l2loss(abs(phase_pred), abs(phase_true));
    %phaseLoss = zeros(1, 'like', ampLoss); % Initialize to zero
    
    % 3. Phase Sign Loss (Using BCE for stronger gradients on sign prediction)
    %signLoss = calculatePhaseSignLoss(phase_pred, phase_true);
    %signLoss = l2loss(phase_pred, phase_true); % Using L2 loss for simplicity
    signLoss = zeros(1, 'like', phaseLoss); % Initialize to zero
    
    % Store current losses
    current_losses = [ampLoss, phaseLoss];

    % 4. Reconstruction Loss (if enabled)
    reconLoss = ones(1, 'like', ampLoss); % Initialize
    linear_recon = []; % Initialize for scope check later
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

        % Build linear image
        [linear_recon, ~] = mmf_build_image(number_of_modes, size(dlX, 1), size(dlX, 4), weights_pred, false);
        [dlX_linear, ~] = mmf_build_image(number_of_modes, size(dlX, 1), size(dlX, 4), weights_true, false);

        % Correlation-based loss
        reconLoss = 1 - dlCorr(dlX_linear, linear_recon);
        % --- End Reconstruction Loss ---

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
        % Simple equal weighting - just average the losses
        % You can also use fixed weights if certain tasks need more emphasis
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

function signLoss = calculatePhaseSignLoss(phase_pred, phase_true)
    % Multi-component loss with strong gradient for canonical representation
    
    % 1. Binary Cross-Entropy with balanced weighting
    epsilon = 1e-8;
    pred_prob = (tanh(5 * phase_pred) + 1) / 2;  % Steeper tanh for sharper decision boundary
    true_prob = (tanh(5 * phase_true) + 1) / 2;
    
    % Count positive/negative examples for balance weighting
    pos_weight = sum(true_prob < 0.5, 'all') / numel(true_prob);
    neg_weight = 1 - pos_weight;
    
    % Weighted BCE with stronger penalty for sign flips
    bce_loss = -(pos_weight .* true_prob .* log(pred_prob + epsilon) + ...
                 neg_weight .* (1 - true_prob) .* log(1 - pred_prob + epsilon));
    bce_component = mean(bce_loss, 'all');
    
    % 2. Cosine similarity on signs
    sign_pred = sign(phase_pred);
    sign_true = sign(phase_true);
    sign_match_loss = 1 - mean(sign_pred .* sign_true, 'all');
    
    % 3. Phase-aware margin loss (larger penalty for phase values near zero)
    certainty_factor = 1 - exp(-5 * abs(phase_true)); % Higher weight for confident true phases
    margin = 0.8;
    margin_loss = max(0, margin - phase_pred .* sign_true) .* certainty_factor;
    margin_component = mean(margin_loss, 'all');
    
    % 4. NEW: Mode relationship consistency loss
    mode_relationship_loss = 0;
    num_phases = size(phase_pred, 1);
    
    if num_phases > 1
        for i = 1:num_phases-1
            for j = i+1:num_phases
                % True relationship between modes i and j
                true_relationship = sign_true(i,:) .* sign_true(j,:);
                
                % Predicted relationship between modes i and j
                pred_relationship = sign_pred(i,:) .* sign_pred(j,:);
                
                % Penalize inconsistent relationships
                mode_relationship_loss = mode_relationship_loss + ...
                    mean((true_relationship - pred_relationship).^2, 'all');
            end
        end
        mode_relationship_loss = mode_relationship_loss / (num_phases * (num_phases-1) / 2);
    end
    
    % Weighted sum of all components
    signLoss = 0.35 * bce_component + ...
               0.25 * sign_match_loss + ...
               0.20 * margin_component + ...
               0.20 * mode_relationship_loss;
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

function [totalLoss, ampLoss, phaseLoss] = modelValidation_abs(dlnet, X, Y, batchSize, executionEnvironment, useReconLoss, adaptiveLossWeights)
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

        [~, batchLoss, batchAmpLoss, batchPhaseLoss, ~, ~, ~] = dlfeval(@modelGradients_abs, dlnet, dlX, dlY, useReconLoss, adaptiveLossWeights);
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