function train_phase_sign_classifier(mmf_train, labels_train, mmf_val, labels_val, mmf_test, labels_test)
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

    % Process in batches to reduce memory usage
    fprintf('Generating predictions from absolute model for training set...\n');
    train_pred = predictInBatches(dlnet, mmf_train, 32);
    
    fprintf('Generating predictions from absolute model for validation set...\n');
    val_pred = predictInBatches(dlnet, mmf_val, 32);
    
    fprintf('Generating predictions from absolute model for test set...\n');
    test_pred = predictInBatches(dlnet, mmf_test, 32);
    

    train_signs_pred = sign(tanh(train_pred));
    val_signs_pred = sign(tanh(val_pred));
    test_signs_pred = sign(tanh(test_pred));


    % Add ones as first column to match canonical representation
    train_signs_pred = [ones(size(train_signs_pred, 1), 1) train_signs_pred];
    val_signs_pred = [ones(size(val_signs_pred, 1), 1) val_signs_pred];
    test_signs_pred = [ones(size(test_signs_pred, 1), 1) test_signs_pred];

    
    % Clear memory
    clear train_pred val_pred test_pred;

    train_labels = generateBinaryLabels(train_signs_pred, train_signs);
    val_labels = generateBinaryLabels(val_signs_pred, val_signs);
    test_labels = generateBinaryLabels(test_signs_pred, test_signs);
    
    % Clear more memory
    clear train_signs_pred val_signs_pred;
    
    % Create and train the global sign classifier with reduced memory footprint
    options = struct();
    options.miniBatchSize = 128; % Reduced from 128 to lower memory usage
    options.maxEpochs = 100;
    options.initialLearnRate = 1e-4;
    options.validationFrequency = 50;
    options.validationPatience = 20000;
    options.executionEnvironment = "gpu";
    options.plotProgress = "gui";
    
    % Train global sign classifier
    fprintf('Training global phase sign classifier...\n');
    global_classifier = trainGlobalSignClassifier(residuals_train, train_labels, ...
        residuals_val, val_labels, options);
    
    % Save the classifier
    save('phase_sign_classifier.mat', 'global_classifier', 'number_of_modes');
    disp('Global phase sign classifier saved to phase_sign_classifier.mat');
    
    % Evaluate classifier on test set
    % evaluateGlobalSignClassifier(global_classifier, residuals_test, test_labels, test_signs, test_signs_pred);
end

function predictions = predictInBatches(net, data, batchSize)
    % Process predictions in small batches to avoid memory issues
    numSamples = size(data, 4);
    outputSize = size(predict(net, data(:,:,:,1)), 2);
    predictions = zeros(numSamples, outputSize);
    
    for i = 1:ceil(numSamples/batchSize)
        startIdx = (i-1)*batchSize + 1;
        endIdx = min(i*batchSize, numSamples);
        batchData = data(:,:,:,startIdx:endIdx);
        
        % Move to GPU if available
        if canUseGPU
            batchData = gpuArray(batchData);
        end

        batchData = dlarray(batchData, 'SSCB');
        
        % Get predictions and move back to CPU
        batchPred = forward(net, batchData);
        predictions(startIdx:endIdx, :) = extract(batchPred)';
    end
end

function binary_labels = generateBinaryLabels(predicted_signs, actual_signs)
    num_samples = size(predicted_signs, 1);
    binary_labels = zeros(num_samples, 1);
    
    for i = 1:num_samples
        % Extract predicted and actual phase signs for the sample
        pred_signs = predicted_signs(i, :);
        true_signs = actual_signs(i, :);
        
        % Calculate how many signs match in original vs flipped version
        matches_original = sum(pred_signs == true_signs);
        matches_flipped = sum(-pred_signs == true_signs);
        
        % Assign binary label: 1 if flipped version has more matches
        if matches_flipped > matches_original
            binary_labels(i) = 1; % Flip
        else
            binary_labels(i) = 0; % Keep
        end
    end
end

function classifier = trainGlobalSignClassifier(X_train, y_train, X_val, y_val, options)
    % Create classifier network with single output
    inputSize = [size(X_train,1) size(X_train,2)];
    classifier = createGlobalSignClassifier(inputSize);
    %load('phase_sign_classifier.mat', 'global_classifier');
    %classifier = global_classifier;
    %classifier = createMLPNet(inputSize);
    
    % Convert labels to dlarray format - single output per sample
    y_train_dl = dlarray(y_train', 'CB'); % [1 x batch_size]
    y_val_dl = dlarray(y_val', 'CB');     % [1 x batch_size]
    
    % Initialize training monitor
    if strcmpi(options.plotProgress, "gui")
        metrics = ["Loss", "Accuracy", "ValidationLoss", "ValidationAccuracy"];
        
        monitor = trainingProgressMonitor(Metrics=metrics, ...
            Info=["Epoch", "LearningRate"], ...
            XLabel="Iteration");

        % Group plots
        groupSubPlot(monitor, "Loss", ["Loss", "ValidationLoss"]);
        groupSubPlot(monitor, "Accuracy", ["Accuracy", "ValidationAccuracy"]);
    end
    
    % Initialize state
    iteration = 0;
    epoch = 0;
    bestValAccuracy = 0;
    bestClassifier = classifier;
    patienceCounter = 0;
    averageGrad = [];
    averageSqGrad = [];
    stopTraining = false;
    
    % Create data indices
    numTrain = size(X_train, 4);
    trainInd = 1:numTrain;
    
    % Training loop
    while epoch < options.maxEpochs && patienceCounter < options.validationPatience && ~stopTraining
        epoch = epoch + 1;
        
        % Shuffle training data
        shuffleIdx = randperm(length(trainInd));
        trainInd = trainInd(shuffleIdx);
        
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
            
            % Calculate gradients
            [gradients, loss, accuracy] = dlfeval(@binarySignGradients, classifier, dlX, y);
            
            % Update network parameters
            [classifier, averageGrad, averageSqGrad] = adamupdate(classifier, gradients, ...
                averageGrad, averageSqGrad, iteration, options.initialLearnRate);
            
            % Log progress
            if strcmpi(options.plotProgress, "gui")
                recordMetrics(monitor, iteration, Loss=loss, Accuracy=accuracy);
                updateInfo(monitor, Epoch=epoch, LearningRate=options.initialLearnRate);
                
                % Check for stop button
                stopTraining = monitor.Stop;
                if stopTraining
                    break;
                end
            end
            
            % Validation check
            if mod(iteration, options.validationFrequency) == 0
                [valLoss, valAccuracy] = validateGlobalSignClassifier(classifier, X_val, y_val_dl, options);
                
                % Log validation metrics
                if strcmpi(options.plotProgress, "gui")
                    recordMetrics(monitor, iteration, ValidationLoss=valLoss, ValidationAccuracy=valAccuracy);
                end
                
                % Early stopping check
                if valAccuracy > bestValAccuracy
                    bestValAccuracy = valAccuracy;
                    bestClassifier = classifier;
                    patienceCounter = 0;
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
    % Creates a simpler network with a single binary output
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

    % Classification layers - now outputs numModes-1 values (one per mode)
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
    analyzeNetwork(classifier);
end

function y = applyChannelAttention(x, weights)
    % Reshape weights for broadcasting across spatial dimensions
    [h, w, c, n] = size(x);
    weights_reshaped = reshape(weights, [1, 1, c, n]);
    y = x .* weights_reshaped;
end

function net = createSqueezeNet(inputSize)
    net = dlnetwork;

    % Initial convolution with smaller kernel and fewer filters for 32x32 inputs
    tempNet = [
        imageInputLayer([inputSize 1], "Name", "data", "Normalization", "zscore")
        convolution2dLayer([3 3], 16, "Name", "conv1", "Stride", [1 1], "Padding", "same")
        leakyReluLayer(0.1, "Name", "relu_conv1")
        % No pooling here to preserve spatial information in small images
    ];
    net = addLayers(net, tempNet);

    % Fire module 1 - reduced size, appropriate for 32x32
    tempNet = [
        convolution2dLayer([1 1], 8, "Name", "fire1-squeeze1x1")
        leakyReluLayer(0.1, "Name", "fire1-relu_squeeze1x1")
    ];
    net = addLayers(net, tempNet);

    tempNet = [
        convolution2dLayer([1 1], 16, "Name", "fire1-expand1x1")
        leakyReluLayer(0.1, "Name", "fire1-relu_expand1x1")
    ];
    net = addLayers(net, tempNet);

    tempNet = [
        convolution2dLayer([3 3], 16, "Name", "fire1-expand3x3", "Padding", "same")
        leakyReluLayer(0.1, "Name", "fire1-relu_expand3x3")
    ];
    net = addLayers(net, tempNet);

    % Concatenate expanded layers with gentle pooling
    tempNet = [
        depthConcatenationLayer(2, "Name", "fire1-concat")
        maxPooling2dLayer([2 2], "Name", "pool1", "Stride", [2 2])  % Now 16x16
    ];
    net = addLayers(net, tempNet);

    % Fire module 2 - small but powerful
    tempNet = [
        convolution2dLayer([1 1], 16, "Name", "fire2-squeeze1x1")
        leakyReluLayer(0.1, "Name", "fire2-relu_squeeze1x1")
    ];
    net = addLayers(net, tempNet);

    tempNet = [
        convolution2dLayer([1 1], 32, "Name", "fire2-expand1x1")
        leakyReluLayer(0.1, "Name", "fire2-relu_expand1x1")
    ];
    net = addLayers(net, tempNet);

    tempNet = [
        convolution2dLayer([3 3], 32, "Name", "fire2-expand3x3", "Padding", "same")
        leakyReluLayer(0.1, "Name", "fire2-relu_expand3x3")
    ];
    net = addLayers(net, tempNet);

    % Concatenate expanded layers and pooling
    tempNet = [
        depthConcatenationLayer(2, "Name", "fire2-concat")
        maxPooling2dLayer([2 2], "Name", "pool2", "Stride", [2 2])  % Now 8x8
    ];
    net = addLayers(net, tempNet);

    % Fire module 3 - with residual connection for better learning
    tempNet = [
        convolution2dLayer([1 1], 16, "Name", "fire3-squeeze1x1")
        leakyReluLayer(0.1, "Name", "fire3-relu_squeeze1x1")
    ];
    net = addLayers(net, tempNet);

    tempNet = [
        convolution2dLayer([1 1], 64, "Name", "fire3-expand1x1")
        leakyReluLayer(0.1, "Name", "fire3-relu_expand1x1")
    ];
    net = addLayers(net, tempNet);

    tempNet = [
        convolution2dLayer([3 3], 64, "Name", "fire3-expand3x3", "Padding", "same")
        leakyReluLayer(0.1, "Name", "fire3-relu_expand3x3")
    ];
    net = addLayers(net, tempNet);

    % Concatenate with residual connection
    tempNet = [
        depthConcatenationLayer(2, "Name", "fire3-concat")
        % No pooling here to preserve remaining spatial information
    ];
    net = addLayers(net, tempNet);

    % Classification layers - designed for binary classification
    tempNet = [
        convolution2dLayer([1 1], 128, "Name", "conv10")
        reluLayer("Name", "relu_conv10")
        
        % Spatial attention mechanism
        convolution2dLayer([1 1], 1, "Name", "attention_map")
        sigmoidLayer("Name", "attention_sigmoid")
        multiplicationLayer(2, "Name", "attention_apply")
        
        globalAveragePooling2dLayer("Name", "pool10")
        dropoutLayer(0.5, "Name", "drop_final")
        
        % Class-balanced FC layers
        fullyConnectedLayer(64, "Name", "fc", "BiasLearnRateFactor", 2)
        leakyReluLayer(0.1, "Name", "leakyrelu") % Lower alpha value
        dropoutLayer(0.3, "Name", "dropout") % Reduced dropout
        
        fullyConnectedLayer(32, "Name", "fc_1", "BiasLearnRateFactor", 2)
        leakyReluLayer(0.1, "Name", "leakyrelu_1")
        
        fullyConnectedLayer(1, "Name", "fc_2", "BiasInitializer", "ones") % Initialize bias to slightly favor positive class
        sigmoidLayer("Name", "sigmoid")
    ];
    net = addLayers(net, tempNet);

    % Connect fire module paths
    net = connectLayers(net, "fire1-relu_squeeze1x1", "fire1-expand1x1");
    net = connectLayers(net, "fire1-relu_squeeze1x1", "fire1-expand3x3");
    net = connectLayers(net, "fire1-relu_expand1x1", "fire1-concat/in1");
    net = connectLayers(net, "fire1-relu_expand3x3", "fire1-concat/in2");
    
    net = connectLayers(net, "fire2-relu_squeeze1x1", "fire2-expand1x1");
    net = connectLayers(net, "fire2-relu_squeeze1x1", "fire2-expand3x3");
    net = connectLayers(net, "fire2-relu_expand1x1", "fire2-concat/in1");
    net = connectLayers(net, "fire2-relu_expand3x3", "fire2-concat/in2");
    
    net = connectLayers(net, "fire3-relu_squeeze1x1", "fire3-expand1x1");
    net = connectLayers(net, "fire3-relu_squeeze1x1", "fire3-expand3x3");
    net = connectLayers(net, "fire3-relu_expand1x1", "fire3-concat/in1");
    net = connectLayers(net, "fire3-relu_expand3x3", "fire3-concat/in2");

    net = connectLayers(net, "relu_conv1", "fire1-squeeze1x1");
    net = connectLayers(net, "pool1", "fire2-squeeze1x1");
    net = connectLayers(net, "pool2", "fire3-squeeze1x1");
    net = connectLayers(net, "fire3-concat", "conv10");
    
    % Connect attention mechanism
    %net = connectLayers(net, "relu_conv10", "attention_map");
    net = connectLayers(net, "relu_conv10", "attention_apply/in2");
    %net = connectLayers(net, "attention_sigmoid", "attention_apply/in2");
        
    analyzeNetwork(net);
    net = initialize(net);
    return;
end

function net = createMLPNet(inputSize)
    % Create a simple MLP network for binary classification
    layers = [
        imageInputLayer([inputSize 1],'Name','input')
        fullyConnectedLayer(512)
        leakyReluLayer(0.2)
        fullyConnectedLayer(256)
        leakyReluLayer(0.2)
        fullyConnectedLayer(128)
        leakyReluLayer(0.2)
        fullyConnectedLayer(64)
        leakyReluLayer(0.2)
        fullyConnectedLayer(1,'Name','fcEnd')
        sigmoidLayer('Name','sigmoid')
    ];
    
    net = dlnetwork(layers);
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

function [valLoss, valAccuracy] = validateGlobalSignClassifier(net, X_val, Y_val, options)
    % Validate global sign classifier
    numValidation = size(X_val, 4);
    batchSize = options.miniBatchSize;
    executionEnvironment = options.executionEnvironment;
    totalLoss = 0;
    correctPredictions = 0;
    totalPredictions = 0;
    
    for i = 1:ceil(numValidation/batchSize)
        startIdx = (i-1)*batchSize + 1;
        endIdx = min(i*batchSize, numValidation);
        
        % Extract batch
        X_batch = X_val(:,:,:,startIdx:endIdx);
        Y_batch = Y_val(:, startIdx:endIdx);
        
        % Convert to dlarray
        dlX_batch = dlarray(X_batch, 'SSCB');
        dlY_batch = dlarray(Y_batch, 'CB');
        
        if executionEnvironment == "gpu"
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
        correctPredictions = correctPredictions + sum(Y_binary == Y_true, 'all');
        totalPredictions = totalPredictions + numel(Y_binary);
    end
    
    valLoss = totalLoss / numValidation;
    valAccuracy = extractdata(correctPredictions) / totalPredictions;
end

function evaluateGlobalSignClassifier(classifier, X_test, global_test_labels, test_signs_true, test_signs_pred)
    numModes = size(test_signs_true, 2);
    numTest = size(X_test, 4);
    
    % Make predictions
    fprintf('Evaluating global sign classifier on test set...\n');
    
    % Process in batches to avoid memory issues
    batch_size = 64;
    all_predictions = zeros(1, numTest);
    
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
        
        predictions = predict(classifier, dlX_batch);
        all_predictions(startIdx:endIdx) = extractdata(predictions > 0.5);
    end

    all_predictions = extract(all_predictions);
    test_signs_pred = extract(test_signs_pred);
    test_signs_true = extract(test_signs_true);
    global_test_labels = extract(global_test_labels);

    size(test_signs_pred)
    size(test_signs_true)

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
        pred_signs = pred_signs.* (1 - 2 * flip_decision);
        
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
    results.all_true_labels = global_test_labels;
    results.global_accuracy = global_accuracy;
    results.mode_accuracies = mode_accuracies;
    results.precision = precision;
    results.recall = recall;
    results.f1_score = f1_score;
    
    save('phase_sign_evaluation.mat', 'results');
    disp('Evaluation results saved to phase_sign_evaluation.mat');
end

% filepath: c:\Users\nicks\MATLAB\Projects\MT2_prak\Versuch 3\train_phase_sign_classifier.m
% train_phase_sign_classifier.m - Trains a network to detect phase signs from residuals
% Modified to predict all mode signs with a single network

% function train_phase_sign_classifier()
%     % Check if residuals have been generated
%     if ~exist('residuals.mat', 'file')
%         error('Residuals not found. Please run generate_residuals.m first.');
%     end
    
%     % Load residuals
%     load('residuals.mat', 'residuals_train', 'residuals_val', 'residuals_test');
    
%     % Load dataset to get phase signs
%     load('mmf_dataset_enhanced.mat', 'labels_train', 'labels_val', 'labels_test');
    
%     % Get number of modes
%     number_of_modes = (size(labels_train, 2) + 1) / 2;
%     num_output_signs = number_of_modes - 1; % First mode is reference
    
%     % Extract phase signs from labels
%     train_signs = sign(labels_train(:, number_of_modes+1:end));
%     val_signs = sign(labels_val(:, number_of_modes+1:end));
%     test_signs = sign(labels_test(:, number_of_modes+1:end));
    
%     % Convert phase signs to 0/1 for binary classification
%     train_labels = (train_signs > 0);
%     val_labels = (val_signs > 0);
    
%     % Create and train the multi-output phase sign classifier
%     options = struct();
%     options.miniBatchSize = 512;
%     options.maxEpochs = 2000;
%     options.initialLearnRate = 1e-4;
%     options.validationFrequency = 50;
%     options.validationPatience = 200;
%     options.executionEnvironment = "gpu";
%     options.plotProgress = "gui";
    
%     % Train unified classifier for all modes
%     fprintf('Training unified classifier for all %d mode signs...\n', num_output_signs);
%     unified_classifier = trainUnifiedPhaseSignClassifier(residuals_train, train_labels, ...
%         residuals_val, val_labels, options, num_output_signs);
    
%     % Save the unified classifier
%     save('phase_sign_classifier.mat', 'unified_classifier', 'number_of_modes');
%     disp('Phase sign classifier saved to phase_sign_classifier.mat');
    
%     % Evaluate classifier on test set
%     evaluatePhaseSignClassifier(unified_classifier, residuals_test, test_signs);
% end

% function classifier = trainUnifiedPhaseSignClassifier(X_train, y_train, X_val, y_val, options, num_output_signs)
%     % Create classifier network with multiple outputs
%     inputSize = [size(X_train,1) size(X_train,2)];
%     classifier = createMultiOutputPhaseSignClassifier(inputSize, num_output_signs);
%     %load('phase_sign_classifier.mat', 'unified_classifier');
%     %classifier = unified_classifier;
    
%     % Convert labels to dlarray format - now has multiple outputs
%     y_train_dl = dlarray(y_train', 'CB'); % [num_outputs x batch_size]
%     y_val_dl = dlarray(y_val', 'CB');     % [num_outputs x batch_size]
    
%     % Initialize training monitor
%     if strcmpi(options.plotProgress, "gui")
%         metrics = ["Loss", "Accuracy", "ValidationLoss", "ValidationAccuracy"];
        
%         monitor = trainingProgressMonitor(Metrics=metrics, ...
%             Info=["Epoch", "LearningRate"], ...
%             XLabel="Iteration");

%         % Group plots
%         groupSubPlot(monitor, "Loss", ["Loss", "ValidationLoss"]);
%         groupSubPlot(monitor, "Accuracy", ["Accuracy", "ValidationAccuracy"]);
%     end
    
%     % Initialize state
%     iteration = 0;
%     epoch = 0;
%     bestValAccuracy = 0;
%     bestClassifier = classifier;
%     patienceCounter = 0;
%     averageGrad = [];
%     averageSqGrad = [];
%     stopTraining = false;
    
%     % Create data indices
%     numTrain = size(X_train, 4);
%     trainInd = 1:numTrain;
    
%     % Training loop
%     while epoch < options.maxEpochs && patienceCounter < options.validationPatience && ~stopTraining
%         epoch = epoch + 1;
        
%         % Shuffle training data
%         shuffleIdx = randperm(length(trainInd));
%         trainInd = trainInd(shuffleIdx);
        
%         % Loop over mini-batches
%         for i = 1:floor(numTrain/options.miniBatchSize)
%             iteration = iteration + 1;
            
%             % Get mini-batch
%             batchInd = trainInd((i-1)*options.miniBatchSize+1:min(i*options.miniBatchSize,numTrain));
%             X = X_train(:,:,:,batchInd);
%             y = y_train_dl(:,batchInd);
            
%             % Convert to dlarray
%             dlX = dlarray(X, 'SSCB');
            
%             % Move to GPU if needed
%             if (options.executionEnvironment == "auto" && canUseGPU) || options.executionEnvironment == "gpu"
%                 dlX = gpuArray(dlX);
%                 y = gpuArray(y);
%             end
            
%             % Calculate gradients
%             [gradients, loss, accuracy] = dlfeval(@multiOutputPhaseSignGradients, classifier, dlX, y);
            
%             % Update network parameters
%             [classifier, averageGrad, averageSqGrad] = adamupdate(classifier, gradients, ...
%                 averageGrad, averageSqGrad, iteration, options.initialLearnRate);
            
%             % Log progress
%             if strcmpi(options.plotProgress, "gui")
%                 recordMetrics(monitor, iteration, Loss=loss, Accuracy=accuracy);
%                 updateInfo(monitor, Epoch=epoch, LearningRate=options.initialLearnRate);
                
%                 % Check for stop button
%                 stopTraining = monitor.Stop;
%                 if stopTraining
%                     break;
%                 end
%             end
            
%             % Validation check
%             if mod(iteration, options.validationFrequency) == 0
%                 [valLoss, valAccuracy] = validateMultiOutputPhaseSign(classifier, X_val, y_val_dl, options);
                
%                 % Log validation metrics
%                 if strcmpi(options.plotProgress, "gui")
%                     recordMetrics(monitor, iteration, ValidationLoss=valLoss, ValidationAccuracy=valAccuracy);
%                 end
                
%                 % Early stopping check
%                 if valAccuracy > bestValAccuracy
%                     bestValAccuracy = valAccuracy;
%                     bestClassifier = classifier;
%                     patienceCounter = 0;
%                 else
%                     patienceCounter = patienceCounter + 1;
%                 end
%             end
%         end
        
%         if stopTraining
%             break;
%         end
%     end
    
%     % Use best network
%     classifier = bestClassifier;
    
%     % Print final accuracy
%     fprintf('Unified classifier - Best validation accuracy: %.2f%%\n', bestValAccuracy*100);
% end

% function classifier = createMultiOutputPhaseSignClassifier(inputSize, numOutputs)
%         % Creates a network that outputs signs for each mode independently
%         % Output will be a vector of size (numModes-1 x 1) with values in [0,1]
%         % representing the probability of each mode's phase being positive
        
%         % Create the base feature extraction layers
%         layers = [
%             % Input layer with normalization
%             imageInputLayer([inputSize 1], 'Name', 'input', 'Normalization', 'zscore')
            
%             % Feature extraction block 1
%             convolution2dLayer(7, 8, 'Padding', 'same', 'Name', 'conv1a')
%             leakyReluLayer(0.3, 'Name', 'relu1a')
%             convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv1b')
%             additionLayer(2, 'Name', 'add1')
%             leakyReluLayer(0.3, 'Name', 'relu1b')
%             maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
            
%             % Feature extraction block 2
%             convolution2dLayer(5, 16, 'Padding', 'same', 'Name', 'conv2a')
%             leakyReluLayer(0.3, 'Name', 'relu2a')
%             convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv2b')
%             additionLayer(2, 'Name', 'add2')
%             leakyReluLayer(0.3, 'Name', 'relu2b')
%             maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
            
%             % Multi-scale feature extraction
%             convolution2dLayer(3, 32, 'Padding', 'same', 'Dilation', [1 1], 'Name', 'conv3a')
%             leakyReluLayer(0.3, 'Name', 'relu3a')
            
%             convolution2dLayer(3, 32, 'Padding', 'same', 'Dilation', [2 2], 'Name', 'conv3b')
%             leakyReluLayer(0.3, 'Name', 'relu3b')
            
%             convolution2dLayer(3, 32, 'Padding', 'same', 'Dilation', [4 4], 'Name', 'conv3c')
%             leakyReluLayer(0.3, 'Name', 'relu3c')
            
%             additionLayer(3, 'Name', 'multi_scale_add')
%             maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
%         ];
        
%         % Create layer graph with skip connections
%         lgraph = layerGraph();
%         lgraph = addLayers(lgraph, layers);
        
%         % Skip connections
%         skipConv1 = convolution2dLayer(1, 8, 'Name', 'skip_conv1');
%         skipConv2 = convolution2dLayer(1, 16, 'Name', 'skip_conv2');
%         lgraph = addLayers(lgraph, skipConv1);
%         lgraph = addLayers(lgraph, skipConv2);
        
%         % Connect skip paths
%         lgraph = connectLayers(lgraph, 'input', 'skip_conv1');
%         lgraph = connectLayers(lgraph, 'pool1', 'skip_conv2');
%         lgraph = connectLayers(lgraph, 'skip_conv1', 'add1/in2');
%         lgraph = connectLayers(lgraph, 'skip_conv2', 'add2/in2');
        
%         % Connect multi-scale outputs
%         lgraph = connectLayers(lgraph, 'relu3b', 'multi_scale_add/in2');
%         lgraph = connectLayers(lgraph, 'relu3c', 'multi_scale_add/in3');
        
%         % Attention mechanism
%         attentionLayers = [
%             globalAveragePooling2dLayer('Name', 'gap')
%             fullyConnectedLayer(16, 'Name', 'att_fc1')
%             reluLayer('Name', 'att_relu')
%             fullyConnectedLayer(32, 'Name', 'att_fc2')
%             sigmoidLayer('Name', 'att_sigmoid')
%         ];
        
%         lgraph = addLayers(lgraph, attentionLayers);
%         lgraph = connectLayers(lgraph, 'pool3', 'gap');
        
%         multiplyLayer = functionLayer(@(x,weights) applyChannelAttention(x,weights), ...
%             'Name', 'att_multiply', ...
%             'NumInputs', 2, 'Formattable', true);
%         lgraph = addLayers(lgraph, multiplyLayer);
%         lgraph = connectLayers(lgraph, 'pool3', 'att_multiply/in1');
%         lgraph = connectLayers(lgraph, 'att_sigmoid', 'att_multiply/in2');
        
%         % Classification layers - now outputs numModes-1 values (one per mode)
%         classificationLayers = [
%             fullyConnectedLayer(256, 'Name', 'fc1')
%             leakyReluLayer(0.3, 'Name', 'relu_fc1')
%             dropoutLayer(0.5, 'Name', 'drop1')
            
%             fullyConnectedLayer(128, 'Name', 'fc2')
%             leakyReluLayer(0.3, 'Name', 'relu_fc2')
%             dropoutLayer(0.3, 'Name', 'drop2')
            
%             % Output layer - one value per mode (excluding reference mode)
%             fullyConnectedLayer(numOutputs, 'Name', 'fc_out')
%             sigmoidLayer('Name', 'output')
%         ];
        
%         lgraph = addLayers(lgraph, classificationLayers);
%         lgraph = connectLayers(lgraph, 'att_multiply', 'fc1');
        
%         classifier = dlnetwork(lgraph);
%         % analyzeNetwork(classifier);
%     end
    
%     function y = applyChannelAttention(x, weights)
%         % Reshape weights for broadcasting across spatial dimensions
%         [h, w, c, n] = size(x);
%         weights_reshaped = reshape(weights, [1, 1, c, n]);
%         y = x .* weights_reshaped;
%     end

% function [gradients, loss, accuracy] = multiOutputPhaseSignGradients(net, X, Y)
%     % Forward pass
%     y_pred = forward(net, X);
    
%     % Binary cross-entropy loss for multiple outputs
%     loss = -mean(Y.*log(y_pred + 1e-8) + (1-Y).*log(1-y_pred + 1e-8), 'all');
    
%     % Calculate accuracy across all outputs
%     y_binary = y_pred > 0.5;
%     y_true = Y > 0.5;
%     accuracy = mean(y_binary == y_true, 'all');
    
%     % Calculate gradients
%     gradients = dlgradient(loss, net.Learnables);
% end

% function [valLoss, valAccuracy] = validateMultiOutputPhaseSign(net, X_val, Y_val, options)
%     % Validate multi-output phase sign classifier
%     numValidation = size(X_val, 4);
%     batchSize = options.miniBatchSize;
%     executionEnvironment = options.executionEnvironment;
%     totalLoss = 0;
%     correctPredictions = 0;
%     totalPredictions = 0;
    
%     for i = 1:ceil(numValidation/batchSize)
%         startIdx = (i-1)*batchSize + 1;
%         endIdx = min(i*batchSize, numValidation);
        
%         % Extract batch
%         X_batch = X_val(:,:,:,startIdx:endIdx);
%         Y_batch = Y_val(:, startIdx:endIdx);
        
%         % Convert to dlarray
%         dlX_batch = dlarray(X_batch, 'SSCB');
%         dlY_batch = dlarray(Y_batch, 'CB');
        
%         if executionEnvironment == "gpu"
%             dlX_batch = gpuArray(dlX_batch);
%             dlY_batch = gpuArray(dlY_batch);
%         end
        
%         % Forward pass
%         Y_pred = predict(net, dlX_batch);
        
%         % Loss - multi-output binary cross-entropy
%         batchLoss = -mean(dlY_batch.*log(Y_pred + 1e-8) + (1-dlY_batch).*log(1-Y_pred + 1e-8), 'all');
%         totalLoss = totalLoss + batchLoss * size(X_batch, 4);
        
%         % Accuracy - count correct predictions across all modes
%         Y_binary = Y_pred > 0.5;
%         Y_true = dlY_batch > 0.5;
%         correctPredictions = correctPredictions + sum(Y_binary == Y_true, 'all');
%         totalPredictions = totalPredictions + numel(Y_binary);
%     end
    
%     valLoss = totalLoss / numValidation;
%     valAccuracy = correctPredictions / totalPredictions;
% end

% function evaluatePhaseSignClassifier(classifier, X_test, y_test_signs)
%     numModes = size(y_test_signs, 2);
%     numTest = size(X_test, 4);
    
%     % Convert signs to binary (0/1)
%     all_true_labels = y_test_signs > 0;
    
%     % Make predictions
%     fprintf('Evaluating unified classifier on test set...\n');
    
%     % Initialize prediction storage
%     all_predictions = zeros(numModes, numTest);
    
%     % Process in batches
%     batch_size = 64;
%     for i = 1:ceil(numTest/batch_size)
%         startIdx = (i-1)*batch_size + 1;
%         endIdx = min(i*batch_size, numTest);
        
%         % Get batch
%         X_batch = X_test(:,:,:,startIdx:endIdx);
        
%         % Predict all mode signs at once
%         dlX_batch = dlarray(X_batch, 'SSCB');
%         if canUseGPU
%             dlX_batch = gpuArray(dlX_batch);
%         end
        
%         predictions = predict(classifier, dlX_batch);
%         all_predictions(:, startIdx:endIdx) = extractdata(predictions > 0.5);
%     end
    
%     % Per-mode accuracy
%     mode_accuracies = zeros(numModes, 1);
%     for mode = 1:numModes
%         mode_accuracies(mode) = mean(all_predictions(mode, :) == all_true_labels(:, mode)');
%         fprintf('Mode %d sign accuracy: %.2f%%\n', mode+1, mode_accuracies(mode)*100);
%     end
    
%     % Overall accuracy
%     overall_accuracy = mean(all_predictions(:) == all_true_labels(:));
%     fprintf('Overall accuracy: %.2f%%\n', overall_accuracy*100);
    
%     % Create visualization
%     figure('Name', 'Unified Phase Sign Classification Results', 'Position', [100, 100, 1200, 800]);
    
%     % Plot per-mode accuracy
%     subplot(2, 2, 1);
%     bar(mode_accuracies * 100);
%     title('Phase Sign Detection Accuracy by Mode');
%     xlabel('Mode Number');
%     ylabel('Accuracy (%)');
%     ylim([0, 100]);
%     grid on;
    
%     % Plot confusion matrices for all modes
%     subplot(2, 2, 2);
%     confusion_data = zeros(numModes, 4); % [TP, FP, FN, TN] for each mode
    
%     for mode = 1:numModes
%         pred = all_predictions(mode, :)'; 
%         true_labels = all_true_labels(:, mode);
        
%         % Calculate confusion matrix values
%         TP = sum(pred == 1 & true_labels == 1);
%         FP = sum(pred == 1 & true_labels == 0);
%         FN = sum(pred == 0 & true_labels == 1);
%         TN = sum(pred == 0 & true_labels == 0);
        
%         confusion_data(mode, :) = [TP, FP, FN, TN];
%     end
    
%     % Convert to metrics
%     precision = confusion_data(:, 1) ./ (confusion_data(:, 1) + confusion_data(:, 2) + 1e-10);
%     recall = confusion_data(:, 1) ./ (confusion_data(:, 1) + confusion_data(:, 3) + 1e-10);
%     f1 = 2 * (precision .* recall) ./ (precision + recall + 1e-10);
    
%     bar([precision, recall, f1] * 100);
%     title('Precision, Recall and F1 Score by Mode');
%     xlabel('Mode Number');
%     ylabel('Value (%)');
%     legend({'Precision', 'Recall', 'F1 Score'});
%     ylim([0, 100]);
%     grid on;
    
%     % Plot combined confusion matrix across all modes
%     subplot(2, 2, 3);
%     total_TP = sum(confusion_data(:, 1));
%     total_FP = sum(confusion_data(:, 2));
%     total_FN = sum(confusion_data(:, 3));
%     total_TN = sum(confusion_data(:, 4));
%     combined_cm = [total_TN, total_FP; total_FN, total_TP];
%     combined_cm_norm = combined_cm ./ sum(combined_cm, 2);
    
%     imagesc(combined_cm_norm);
%     colormap('cool');
%     title('Combined Confusion Matrix (All Modes)');
%     xlabel('Predicted Class');
%     ylabel('True Class');
%     xticks([1 2]);
%     yticks([1 2]);
%     xticklabels({'Negative', 'Positive'});
%     yticklabels({'Negative', 'Positive'});
%     colorbar;
    
%     % Add text to confusion matrix
%     text(1, 1, sprintf('%.1f%%', combined_cm_norm(1,1)*100), 'HorizontalAlignment', 'center', 'Color', 'white');
%     text(2, 1, sprintf('%.1f%%', combined_cm_norm(1,2)*100), 'HorizontalAlignment', 'center', 'Color', 'white');
%     text(1, 2, sprintf('%.1f%%', combined_cm_norm(2,1)*100), 'HorizontalAlignment', 'center', 'Color', 'white');
%     text(2, 2, sprintf('%.1f%%', combined_cm_norm(2,2)*100), 'HorizontalAlignment', 'center', 'Color', 'white');
    
%     % Overall results
%     subplot(2, 2, 4);
%     text(0.5, 0.7, sprintf('Unified Phase Sign Detection\nOverall Accuracy: %.2f%%', overall_accuracy*100), ...
%         'FontSize', 18, 'HorizontalAlignment', 'center');
%     text(0.5, 0.3, sprintf('Average F1 Score: %.2f%%', mean(f1)*100), ...
%         'FontSize', 14, 'HorizontalAlignment', 'center');
%     axis off;
    
%     % Save results
%     results = struct();
%     results.all_predictions = all_predictions;
%     results.all_true_labels = all_true_labels;
%     results.mode_accuracies = mode_accuracies;
%     results.overall_accuracy = overall_accuracy;
%     results.precision = precision;
%     results.recall = recall;
%     results.f1 = f1;
    
%     save('phase_sign_evaluation.mat', 'results');
%     disp('Evaluation results saved to phase_sign_evaluation.mat');
% end

% filepath: c:\Users\nicks\MATLAB\Projects\MT2_prak\Versuch 3\train_phase_sign_classifier.m
% train_phase_sign_classifier.m - Trains a network to detect phase signs from residuals
% Modified to predict a single global phase sign decision (flip all or keep all)
