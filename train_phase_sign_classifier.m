% filepath: c:\Users\nicks\MATLAB\Projects\MT2_prak\mmf_decomp\train_phase_sign_classifier.m
function train_phase_sign_classifier(mmf_train, labels_train, mmf_val, labels_val, mmf_test, labels_test, options, P_precomputed)
    % TRAIN_PHASE_SIGN_CLASSIFIER - Train a global phase sign classifier to resolve sign ambiguity
    % This classifier predicts whether to flip all signs or keep them as-is
    %
    % Inputs:
    %   mmf_train - Training images [height x width x 1 x samples]
    %   labels_train - Training labels [samples x (2*number_of_modes-1)]
    %   mmf_val - Validation images [height x width x 1 x samples]
    %   labels_val - Validation labels [samples x (2*number_of_modes-1)]
    %   mmf_test - Test images [height x width x 1 x samples]
    %   labels_test - Test labels [samples x (2*number_of_modes-1)]
    %   options - Struct containing training parameters (optional)
    %   P_precomputed - Precomputed BPMmatlab model with modes (optional)
    
    % Get utility functions
    utils = mmf_utils();
    
    % Parse options or use defaults
    if nargin < 7
        options = struct();
    end
    
    % Default options
    if ~isfield(options, 'miniBatchSize'), options.miniBatchSize = 128; end
    if ~isfield(options, 'maxEpochs'), options.maxEpochs = 100; end
    if ~isfield(options, 'initialLearnRate'), options.initialLearnRate = 1e-4; end
    if ~isfield(options, 'validationFrequency'), options.validationFrequency = 50; end
    if ~isfield(options, 'validationPatience'), options.validationPatience = 20000; end
    if ~isfield(options, 'executionEnvironment'), options.executionEnvironment = "gpu"; end
    if ~isfield(options, 'plotProgress'), options.plotProgress = "gui"; end
    
    % Check if residuals have been generated
    if ~exist('residuals.mat', 'file')
        error('Residuals not found. Please run generate_residuals.m first.');
    end
    
    % Load residuals
    residData = load('residuals.mat');
    residuals_train = residData.residuals_train;
    residuals_val = residData.residuals_val;
    residuals_test = residData.residuals_test;
    clear residData; % Free memory
    
    % Load pre-trained model
    modelData = load('phase_sign_model.mat');   
    dlnet = modelData.dlnet_phase;
    number_of_modes = modelData.number_of_modes;
    clear modelData; % Free memory
    
    % Extract phase signs from labels
    train_signs = sign(labels_train(:, number_of_modes+1:end));
    val_signs = sign(labels_val(:, number_of_modes+1:end));
    test_signs = sign(labels_test(:, number_of_modes+1:end));
    
    % Get or create BPMmatlab model with precomputed modes (for batch predictions)
    if nargin < 8 || isempty(P_precomputed)
        inputSize = [size(mmf_train, 1), size(mmf_train, 2)];
        P = utils.getOrCreateModelWithModes(number_of_modes, inputSize(1), true);
    else
        P = P_precomputed;
    end
    
    % Process in batches to reduce memory usage
    fprintf('Generating predictions from phase sign model for training set...\n');
    train_pred = predictInBatches(dlnet, mmf_train, options, utils);
    
    fprintf('Generating predictions from phase sign model for validation set...\n');
    val_pred = predictInBatches(dlnet, mmf_val, options, utils);
    
    fprintf('Generating predictions from phase sign model for test set...\n');
    test_pred = predictInBatches(dlnet, mmf_test, options, utils);
    
    % Apply tanh to get values in [-1, 1] range and then take sign
    train_signs_pred = sign(tanh(train_pred));
    val_signs_pred = sign(tanh(val_pred));
    test_signs_pred = sign(tanh(test_pred));

    % Add ones as first column to match canonical representation
    % NOTE: If the model returns all signs for modes 2..N, skip adding the first column
    if size(train_signs_pred, 2) == number_of_modes - 2
        % If model predicts signs for modes 3...N, add the fixed +1 sign for mode 2
        train_signs_pred = [ones(size(train_signs_pred, 1), 1) train_signs_pred];
        val_signs_pred = [ones(size(val_signs_pred, 1), 1) val_signs_pred];
        test_signs_pred = [ones(size(test_signs_pred, 1), 1) test_signs_pred];
    end
    
    % Clear memory
    clear train_pred val_pred test_pred;

    % Generate binary labels for classification: 0=keep signs, 1=flip signs
    train_labels = generateBinaryLabels(train_signs_pred, train_signs);
    val_labels = generateBinaryLabels(val_signs_pred, val_signs);
    test_labels = generateBinaryLabels(test_signs_pred, test_signs);
    
    % Clear more memory
    clear train_signs_pred val_signs_pred;
    
    % Create and train the global sign classifier with reduced memory footprint
    fprintf('Training global phase sign classifier...\n');
    global_classifier = trainGlobalSignClassifier(residuals_train, train_labels, ...
        residuals_val, val_labels, options);
    
    % Evaluate classifier on test set
    evaluateGlobalSignClassifier(global_classifier, residuals_test, test_labels, test_signs, test_signs_pred);
    
    % Save the classifier
    save('phase_sign_classifier.mat', 'global_classifier', 'number_of_modes');
    disp('Global phase sign classifier saved to phase_sign_classifier.mat');
end

function predictions = predictInBatches(net, data, options, utils)
    % Process predictions in small batches to avoid memory issues
    numSamples = size(data, 4);
    batchSize = min(options.miniBatchSize, 32); % Use smaller batch size for prediction to save memory
    
    % Make one prediction to determine output size
    dlX_sample = dlarray(data(:,:,:,1), 'SSCB');
    if canUseGPU
        dlX_sample = gpuArray(dlX_sample);
    end
    outputSize = size(predict(net, dlX_sample), 2);
    predictions = zeros(numSamples, outputSize);
    
    for i = 1:ceil(numSamples/batchSize)
        startIdx = (i-1)*batchSize + 1;
        endIdx = min(i*batchSize, numSamples);
        batchData = data(:,:,:,startIdx:endIdx);
        
        % Move to GPU if available
        dlX = dlarray(batchData, 'SSCB');
        if canUseGPU
            dlX = gpuArray(dlX);
        end
        
        % Get predictions and move back to CPU
        batchPred = predict(net, dlX);
        predictions(startIdx:endIdx, :) = extractdata(batchPred)';
    end
end

function binary_labels = generateBinaryLabels(predicted_signs, actual_signs)
    % Generate binary labels for global sign classifier:
    % 1 = flip all signs, 0 = keep signs as is
    
    num_samples = size(predicted_signs, 1);
    binary_labels = zeros(num_samples, 1);
    
    % Process each sample
    for i = 1:num_samples
        % Extract predicted and actual phase signs for the sample
        pred_signs = predicted_signs(i, :);
        true_signs = actual_signs(i, :);
        
        % Calculate how many signs match in original vs flipped version
        matches_original = sum(pred_signs == true_signs);
        matches_flipped = sum(-pred_signs == true_signs);
        
        % Assign binary label: 1 if flipped version has more matches
        if matches_flipped > matches_original
            binary_labels(i) = 1; % Flip needed
        else
            binary_labels(i) = 0; % Keep as is
        end
    end
end

function classifier = trainGlobalSignClassifier(X_train, y_train, X_val, y_val, options)
    % Create classifier network with single output for global sign decision
    % Optimized training with efficient memory usage
    
    % Get utility functions
    utils = mmf_utils();
    
    % Create input size from residuals
    inputSize = [size(X_train, 1) size(X_train, 2)];
    
    % Create an appropriate network architecture for binary classification
    classifier = createGlobalSignClassifier(inputSize);
    
    % Convert labels to dlarray format - single output per sample
    y_train_dl = dlarray(y_train', 'CB'); % [1 x batch_size]
    y_val_dl = dlarray(y_val', 'CB');     % [1 x batch_size]
    
    % Initialize training monitor if GUI display is enabled
    if strcmpi(options.plotProgress, "gui")
        metrics = ["Loss", "Accuracy", "ValidationLoss", "ValidationAccuracy"];
        
        monitor = trainingProgressMonitor(Metrics=metrics, ...
            Info=["Epoch", "LearningRate"], ...
            XLabel="Iteration");

        % Group plots
        groupSubPlot(monitor, "Loss", ["Loss", "ValidationLoss"]);
        groupSubPlot(monitor, "Accuracy", ["Accuracy", "ValidationAccuracy"]);
    end
    
    % Initialize training state
    iteration = 0;
    epoch = 0;
    bestValAccuracy = 0;
    bestClassifier = classifier;
    patienceCounter = 0;
    stopTraining = false;
    
    % Initialize optimization variables
    averageGrad = [];
    averageSqGrad = [];
    
    % Create data indices for batching
    numTrain = size(X_train, 4);
    trainInd = 1:numTrain;
    
    % Training loop
    while epoch < options.maxEpochs && patienceCounter < options.validationPatience && ~stopTraining
        epoch = epoch + 1;
        
        % Shuffle training data
        shuffleIdx = randperm(length(trainInd));
        trainInd = shuffleIdx;
        
        % Loop over mini-batches
        for i = 1:floor(numTrain/options.miniBatchSize)
            iteration = iteration + 1;
            
            % Get mini-batch
            batchInd = trainInd((i-1)*options.miniBatchSize+1:min(i*options.miniBatchSize,numTrain));
            X = X_train(:,:,:,batchInd);
            y = y_train_dl(:,batchInd);
            
            % Convert to dlarray
            dlX = dlarray(X, 'SSCB');
            
            % Move to GPU if needed
            if (options.executionEnvironment == "auto" && canUseGPU) || options.executionEnvironment == "gpu"
                dlX = gpuArray(dlX);
                y = gpuArray(y);
            end
            
            % Calculate gradients with balanced loss for potential class imbalance
            [gradients, loss, accuracy] = dlfeval(@binarySignGradients, classifier, dlX, y);
            
            % Apply gradient clipping
            gradientThreshold = 1e-3;
            gradients = dlupdate(@(g) utils.thresholdL2Norm(g, gradientThreshold), gradients);
            
            % Update network parameters
            [classifier, averageGrad, averageSqGrad] = adamupdate(classifier, gradients, ...
                averageGrad, averageSqGrad, iteration, options.initialLearnRate);
            
            % Log progress
            if strcmpi(options.plotProgress, "gui")
                recordMetrics(monitor, iteration, Loss=extractdata(loss), Accuracy=extractdata(accuracy));
                updateInfo(monitor, Epoch=epoch, LearningRate=options.initialLearnRate);
                
                % Check for stop button
                stopTraining = monitor.Stop;
                if stopTraining
                    break;
                end
            end
            
            % Validation check
            if mod(iteration, options.validationFrequency) == 0
                [valLoss, valAccuracy] = validateGlobalSignClassifier(classifier, X_val, y_val_dl, options, utils);
                
                % Log validation metrics
                if strcmpi(options.plotProgress, "gui")
                    recordMetrics(monitor, iteration, ValidationLoss=extractdata(valLoss), ValidationAccuracy=extractdata(valAccuracy));
                end
                
                % Early stopping check - save best model based on validation accuracy
                if valAccuracy > bestValAccuracy
                    bestValAccuracy = valAccuracy;
                    bestClassifier = classifier;
                    patienceCounter = 0;
                    fprintf('Iteration %d: New best model (accuracy = %.2f%%)\n', iteration, valAccuracy*100);
                else
                    patienceCounter = patienceCounter + 1;
                end
            end
        end
        
        if stopTraining
            break;
        end
    end
    
    % Use best network
    classifier = bestClassifier;
    
    % Print final accuracy
    fprintf('Global sign classifier - Best validation accuracy: %.2f%%\n', bestValAccuracy*100);
end

function classifier = createGlobalSignClassifier(inputSize)
    % Creates a network with a single binary output for global sign classification
    % Network has efficient architecture with attention mechanism to focus on key features
    
    % Create the base feature extraction layers
    layers = [
        % Input layer with normalization
        imageInputLayer([inputSize 1], 'Name', 'input', 'Normalization', 'zscore')
        
        % Feature extraction block 1
        convolution2dLayer(7, 8, 'Padding', 'same', 'Name', 'conv1a')
        leakyReluLayer(0.3, 'Name', 'relu1a')
        convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv1b')
        additionLayer(2, 'Name', 'add1')
        leakyReluLayer(0.3, 'Name', 'relu1b')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
        
        % Feature extraction block 2
        convolution2dLayer(5, 16, 'Padding', 'same', 'Name', 'conv2a')
        leakyReluLayer(0.3, 'Name', 'relu2a')
        convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv2b')
        additionLayer(2, 'Name', 'add2')
        leakyReluLayer(0.3, 'Name', 'relu2b')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
        
        % Multi-scale feature extraction
        convolution2dLayer(3, 32, 'Padding', 'same', 'Dilation', [1 1], 'Name', 'conv3a')
        leakyReluLayer(0.3, 'Name', 'relu3a')
        
        convolution2dLayer(3, 32, 'Padding', 'same', 'Dilation', [2 2], 'Name', 'conv3b')
        leakyReluLayer(0.3, 'Name', 'relu3b')
        
        convolution2dLayer(3, 32, 'Padding', 'same', 'Dilation', [4 4], 'Name', 'conv3c')
        leakyReluLayer(0.3, 'Name', 'relu3c')
        
        additionLayer(3, 'Name', 'multi_scale_add')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
    ];

    % Create layer graph with skip connections
    lgraph = layerGraph();
    lgraph = addLayers(lgraph, layers);

    % Skip connections
    skipConv1 = convolution2dLayer(1, 8, 'Name', 'skip_conv1');
    skipConv2 = convolution2dLayer(1, 16, 'Name', 'skip_conv2');
    lgraph = addLayers(lgraph, skipConv1);
    lgraph = addLayers(lgraph, skipConv2);

    % Connect skip paths
    lgraph = connectLayers(lgraph, 'input', 'skip_conv1');
    lgraph = connectLayers(lgraph, 'pool1', 'skip_conv2');
    lgraph = connectLayers(lgraph, 'skip_conv1', 'add1/in2');
    lgraph = connectLayers(lgraph, 'skip_conv2', 'add2/in2');

    % Connect multi-scale outputs
    lgraph = connectLayers(lgraph, 'relu3b', 'multi_scale_add/in2');
    lgraph = connectLayers(lgraph, 'relu3c', 'multi_scale_add/in3');

    % Attention mechanism
    attentionLayers = [
        globalAveragePooling2dLayer('Name', 'gap')
        fullyConnectedLayer(16, 'Name', 'att_fc1')
        reluLayer('Name', 'att_relu')
        fullyConnectedLayer(32, 'Name', 'att_fc2')
        sigmoidLayer('Name', 'att_sigmoid')
    ];

    lgraph = addLayers(lgraph, attentionLayers);
    lgraph = connectLayers(lgraph, 'pool3', 'gap');

    multiplyLayer = functionLayer(@(x,weights) applyChannelAttention(x,weights), ...
        'Name', 'att_multiply', ...
        'NumInputs', 2, 'Formattable', true);
    lgraph = addLayers(lgraph, multiplyLayer);
    lgraph = connectLayers(lgraph, 'pool3', 'att_multiply/in1');
    lgraph = connectLayers(lgraph, 'att_sigmoid', 'att_multiply/in2');

    % Classification layers - now outputs a single value for binary decision
    classificationLayers = [
        fullyConnectedLayer(256, 'Name', 'fc1')
        leakyReluLayer(0.3, 'Name', 'relu_fc1')
        dropoutLayer(0.5, 'Name', 'drop1')
        
        fullyConnectedLayer(128, 'Name', 'fc2')
        leakyReluLayer(0.3, 'Name', 'relu_fc2')
        dropoutLayer(0.3, 'Name', 'drop2')
        
        fullyConnectedLayer(1, 'Name', 'fc_out')
        sigmoidLayer('Name', 'output')
    ];

    lgraph = addLayers(lgraph, classificationLayers);
    lgraph = connectLayers(lgraph, 'att_multiply', 'fc1');

    classifier = dlnetwork(lgraph);
end

function y = applyChannelAttention(x, weights)
    % Reshape weights for broadcasting across spatial dimensions
    [h, w, c, n] = size(x);
    weights_reshaped = reshape(weights, [1, 1, c, n]);
    y = x .* weights_reshaped;
end

function [gradients, loss, accuracy] = binarySignGradients(net, X, Y)
    % Forward pass
    y_pred = forward(net, X);
    
    % Get class distribution in this batch
    pos_samples = sum(Y > 0.5);
    neg_samples = sum(Y <= 0.5);
    total_samples = pos_samples + neg_samples;
    
    % Calculate weights to balance classes
    if pos_samples > 0 && neg_samples > 0
        pos_weight = total_samples / (2 * pos_samples);
        neg_weight = total_samples / (2 * neg_samples);
    else
        pos_weight = 1.0;
        neg_weight = 1.0;
    end
    
    % Create weight matrix
    weights = ones(size(Y));
    weights(Y > 0.5) = pos_weight;
    weights(Y <= 0.5) = neg_weight;
    
    % Weighted binary cross-entropy loss
    loss = -mean(weights .* (Y.*log(y_pred + 1e-8) + (1-Y).*log(1-y_pred + 1e-8)), 'all');
    
    % Calculate accuracy
    y_binary = y_pred > 0.5;
    y_true = Y > 0.5;
    accuracy = mean(y_binary == y_true, 'all');
    
    % Calculate gradients
    gradients = dlgradient(loss, net.Learnables);
end

function [valLoss, valAccuracy] = validateGlobalSignClassifier(net, X_val, Y_val, options, utils)
    % Validate global sign classifier efficiently in batches
    numValidation = size(X_val, 4);
    batchSize = options.miniBatchSize;
    executionEnvironment = options.executionEnvironment;
    
    % Initialize accumulators
    totalLoss = 0;
    correctPredictions = 0;
    totalPredictions = 0;
    
    % Process in batches
    for i = 1:ceil(numValidation/batchSize)
        startIdx = (i-1)*batchSize + 1;
        endIdx = min(i*batchSize, numValidation);
        
        % Extract batch
        X_batch = X_val(:,:,:,startIdx:endIdx);
        Y_batch = Y_val(:, startIdx:endIdx);
        
        % Convert to dlarray
        dlX_batch = dlarray(X_batch, 'SSCB');
        dlY_batch = dlarray(Y_batch, 'CB');
        
        % Move to GPU if needed
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX_batch = gpuArray(dlX_batch);
            dlY_batch = gpuArray(dlY_batch);
        end
        
        % Forward pass
        Y_pred = predict(net, dlX_batch);
        
        % Loss - binary cross-entropy
        batchLoss = -mean(dlY_batch.*log(Y_pred + 1e-8) + (1-dlY_batch).*log(1-Y_pred + 1e-8), 'all');
        totalLoss = totalLoss + extractdata(batchLoss) * size(X_batch, 4);
        
        % Accuracy
        Y_binary = Y_pred > 0.5;
        Y_true = dlY_batch > 0.5;
        batchCorrect = sum(Y_binary == Y_true, 'all');
        correctPredictions = correctPredictions + extractdata(batchCorrect);
        totalPredictions = totalPredictions + numel(Y_binary);
    end
    
    % Compute final metrics
    valLoss = totalLoss / numValidation;
    valAccuracy = correctPredictions / totalPredictions;
end

function evaluateGlobalSignClassifier(classifier, X_test, global_test_labels, test_signs_true, test_signs_pred)
    % Get utility functions
    utils = mmf_utils();
    
    % Setup for evaluation
    numTest = size(X_test, 4);
    numModes = size(test_signs_true, 2);
    
    fprintf('Evaluating global sign classifier on test set...\n');
    
    % Process in batches to avoid memory issues
    batch_size = 64;
    all_predictions = zeros(1, numTest);
    all_probabilities = zeros(1, numTest);
    
    % Make predictions
    for i = 1:ceil(numTest/batch_size)
        startIdx = (i-1)*batch_size + 1;
        endIdx = min(i*batch_size, numTest);
        
        % Get batch
        X_batch = X_test(:,:,:,startIdx:endIdx);
        
        % Predict global sign flip decision
        dlX_batch = dlarray(X_batch, 'SSCB');
        if canUseGPU
            dlX_batch = gpuArray(dlX_batch);
        end
        
        % Get predictions and probabilities
        pred_probs = predict(classifier, dlX_batch);
        all_probabilities(startIdx:endIdx) = extractdata(pred_probs);
        all_predictions(startIdx:endIdx) = extractdata(pred_probs > 0.5);
    end
    
    % Convert to standard arrays
    test_signs_pred = extractdata(test_signs_pred);
    test_signs_true = extractdata(test_signs_true);
    all_predictions = extractdata(all_predictions);
    
    % Ensure global_test_labels is a row vector for comparison
    global_test_labels = global_test_labels(:)';
    
    % Calculate global sign accuracy
    global_accuracy = mean(all_predictions == global_test_labels);
    fprintf('Global phase sign accuracy: %.2f%%\n', global_accuracy*100);
    
    % Simulate how this would affect individual mode signs
    mode_accuracies = zeros(numModes, 1);
    
    % For each sample
    for sample = 1:numTest
        % Get the predicted global sign decision (1 = flip all, 0 = keep all)
        flip_decision = all_predictions(sample);

        pred_signs = test_signs_pred(sample, :);
        
        % Apply the flip decision
        if flip_decision > 0
            pred_signs = -pred_signs;
        end
        
        % True signs for this sample
        true_signs = test_signs_true(sample, :);
        
        % Calculate accuracy per mode
        mode_accuracies = mode_accuracies + (pred_signs == true_signs)';
    end
    
    % Normalize mode accuracies
    mode_accuracies = mode_accuracies / numTest;
    
    % Calculate confusion matrix for global decision
    TP = sum(all_predictions == 1 & global_test_labels == 1);
    FP = sum(all_predictions == 1 & global_test_labels == 0);
    FN = sum(all_predictions == 0 & global_test_labels == 1);
    TN = sum(all_predictions == 0 & global_test_labels == 0);
    
    precision = TP / (TP + FP + 1e-10);
    recall = TP / (TP + FN + 1e-10);
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10);
    
    % Create visualization
    figure('Name', 'Global Phase Sign Classification Results', 'Position', [100, 100, 1200, 800]);
    
    % Plot 1: Global sign accuracy
    subplot(2, 2, 1);
    bar(global_accuracy * 100);
    title('Global Phase Sign Detection Accuracy');
    ylim([0, 100]);
    ylabel('Accuracy (%)');
    grid on;
    
    % Plot 2: Per-mode accuracy when using global sign decision
    subplot(2, 2, 2);
    bar(mode_accuracies * 100);
    title('Per-Mode Accuracy using Global Decision');
    xlabel('Mode Number');
    ylabel('Accuracy (%)');
    ylim([0, 100]);
    grid on;
    
    % Plot 3: Confusion matrix
    subplot(2, 2, 3);
    cm = [TN, FP; FN, TP];
    cm_norm = cm ./ sum(cm, 2);
    
    imagesc(cm_norm);
    colormap('cool');
    title('Global Sign Confusion Matrix');
    xlabel('Predicted Decision');
    ylabel('True Decision');
    xticks([1 2]);
    yticks([1 2]);
    xticklabels({'Keep Signs', 'Flip Signs'});
    yticklabels({'Keep Signs', 'Flip Signs'});
    colorbar;
    
    % Add text to confusion matrix
    text(1, 1, sprintf('%.1f%%', cm_norm(1,1)*100), 'HorizontalAlignment', 'center', 'Color', 'white');
    text(2, 1, sprintf('%.1f%%', cm_norm(1,2)*100), 'HorizontalAlignment', 'center', 'Color', 'white');
    text(1, 2, sprintf('%.1f%%', cm_norm(2,1)*100), 'HorizontalAlignment', 'center', 'Color', 'white');
    text(2, 2, sprintf('%.1f%%', cm_norm(2,2)*100), 'HorizontalAlignment', 'center', 'Color', 'white');
    
    % Plot 4: Overall metrics
    subplot(2, 2, 4);
    text(0.5, 0.8, sprintf('Global Phase Sign Decision Metrics'), ...
        'FontSize', 16, 'HorizontalAlignment', 'center');
    text(0.5, 0.6, sprintf('Accuracy: %.2f%%', global_accuracy*100), ...
        'FontSize', 14, 'HorizontalAlignment', 'center');
    text(0.5, 0.4, sprintf('Precision: %.2f%%', precision*100), ...
        'FontSize', 14, 'HorizontalAlignment', 'center');
    text(0.5, 0.2, sprintf('Recall: %.2f%%', recall*100), ...
        'FontSize', 14, 'HorizontalAlignment', 'center');
    text(0.5, 0.0, sprintf('F1 Score: %.2f%%', f1_score*100), ...
        'FontSize', 14, 'HorizontalAlignment', 'center');
    
    axis off;
    
    % Save results
    results = struct();
    results.all_predictions = all_predictions;
    results.all_probabilities = all_probabilities;
    results.all_true_labels = global_test_labels;
    results.global_accuracy = global_accuracy;
    results.mode_accuracies = mode_accuracies;
    results.precision = precision;
    results.recall = recall;
    results.f1_score = f1_score;
    
    save('phase_sign_evaluation.mat', 'results');
    disp('Evaluation results saved to phase_sign_evaluation.mat');
end
