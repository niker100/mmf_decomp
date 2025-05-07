% Main script to run the entire MMF mode decomposition pipeline

% Clear workspace and command window
clear all;
% close all;
% clc;

% Load dataset
if ~exist('mmf_train', 'var')
    disp('Loading dataset...');
    load('mmf_dataset_multi_sign.mat');
end

% Step 0: Preprocess labels - enforce canonical sign representation by forcing the first sign to be positive
disp('========== STEP 0: PREPROCESSING LABELS ==========');
labels_train_canonical = labels_train;
labels_val_canonical = labels_val;
labels_test_canonical = labels_test;

numModes = (size(labels_train, 2) + 1) / 2; % Determine numModes from label size
phase_indices = (numModes + 1):(2 * numModes - 1);

fprintf('Processing dataset with %d modes (%d phases, %d phase signs to predict)\n', ...
    numModes, numModes-1, numModes-2);

% Canonicalize training data
for i = 1:size(labels_train, 1)
    % Ensure the first phase sign is positive
    if labels_train(i, numModes+1) < 0
        labels_train_canonical(i, phase_indices) = -labels_train(i, phase_indices);
    end
end

% Canonicalize validation data
for i = 1:size(labels_val, 1)
    % Ensure the first phase sign is positive
    if labels_val(i, numModes+1) < 0
        labels_val_canonical(i, phase_indices) = -labels_val(i, phase_indices);
    end
end

% Canonicalize test data
for i = 1:size(labels_test, 1)
    % Ensure the first phase sign is positive
    if labels_test(i, numModes+1) < 0
        labels_test_canonical(i, phase_indices) = -labels_test(i, phase_indices);
    end
end

fprintf('Data canonicalization complete. First phase sign forced to positive.\n');

% Step 0.5: Precompute BPMmatlab model and modes ONCE for all image generation
P = BPMmatlab.model;
P.name = 'pipeline_main';
P.useAllCPUs = true;
P.useGPU = true;
P.Lx_main = 50e-6;
P.Ly_main = 50e-6;
P.Nx_main = 64; % Use dataset image size
P.Ny_main = 64;
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
P = findModes(P, numModes, 'plotModes', false);

% Pass P to all mmf_build_image calls in downstream scripts for consistent mode basis

% Step 1a: Train model for absolute amplitudes and phases
disp('========== STEP 1a: TRAINING MODEL FOR MAGNITUDES PREDICTION ==========');
options = struct();
options.initialLearnRate = 1e-4;
options.maxEpochs = 100;
options.miniBatchSize = 128;
options.validationFrequency = 300;
options.validationPatience = 30;
options.executionEnvironment = "gpu";
options.modelType = "MLP";
options.plotProgress = "gui";
options.useReconLoss = false; % Disable reconstruction loss for amplitude training
options.showSignAccuracy = true;
options.evaluationFrequency = 1000;
options.adaptiveLossWeights = false;

dlnet_amplitude = train_absolute_model(mmf_train, labels_train_canonical, mmf_val, labels_val_canonical, options);
save('amplitude_model.mat', 'dlnet_amplitude', 'number_of_modes');

% Step 1b: Train dedicated model for phase prediction
disp('========== STEP 1b: TRAINING SPECIALIZED MODEL FOR PHASE PREDICTION ==========');

% Extract amplitude and phase values from canonical labels
amp_train = labels_train_canonical(:, 1:numModes);
phase_train = labels_train_canonical(:, numModes+1:end);
amp_val = labels_val_canonical(:, 1:numModes);
phase_val = labels_val_canonical(:, numModes+1:end);

% Standard settings for lower mode counts
options.initialLearnRate = 1e-4;
options.maxEpochs = 100;
options.miniBatchSize = 128;
options.validationFrequency = 50;
options.validationPatience = 40;
options.modelType = "PhaseSignMLP"; 
options.evaluationFrequency = 500;
options.correlationWeight = 0.6;
options.signLossWeight = 0.4;

% Train the phase model with appropriate options
dlnet_phase = train_phase_sign_model(mmf_train, amp_train, phase_train, mmf_val, amp_val, phase_val, options);
save('phase_sign_model.mat', 'dlnet_phase', 'number_of_modes');


% Step 2: Generate and visualize residuals (optional - for further analysis)
disp('========== STEP 2: GENERATING RESIDUALS ==========');
generate_residuals(mmf_train, labels_train, mmf_val, labels_val, mmf_test, labels_test);

% Step 3: Train phase sign classifier for determining original or complex conjugate
disp('========== STEP 3: TRAINING PHASE SIGN CLASSIFIER ==========');
train_phase_sign_classifier(mmf_train, labels_train, mmf_val, labels_val, mmf_test, labels_test);

% Step 4: Evaluate pipeline with the enhanced approach
disp('========== STEP 4: EVALUATING FULL PIPELINE ==========');
evaluate_full_pipeline();


% Nonlinear performance evaluation
disp('========== STEP 5: EVALUATING NONLINEAR PERFORMANCE ==========');
% evaluate_nonlinear_performance();


disp('Complete pipeline execution finished!');