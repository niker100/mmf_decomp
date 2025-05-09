% filepath: c:\Users\nicks\MATLAB\Projects\MT2_prak\mmf_decomp\run_mmf_pipeline.m
% run_mmf_pipeline.m - Main script to run the complete MMF decomposition pipeline
% This script orchestrates training and evaluation of all models in the pipeline
% 
% You can execute each section separately using the "Run Section" button in MATLAB
% or by placing your cursor in a section and pressing Ctrl+Enter

%% Initialize Pipeline and Configuration
tic; % Start timing the pipeline

% Pipeline configuration - modify these settings as needed
datasetFile = 'mmf_dataset_multi_sign.mat';
options = struct();
options.trainAmpModel = true;       % Set to false to skip amplitude model training
options.trainPhaseModel = true;     % Set to false to skip phase sign model training
options.trainGlobalClassifier = true; % Set to false to skip global classifier training
options.evaluate = true;            % Set to false to skip final evaluation
options.executionEnvironment = "gpu"; % Set to "cpu" if no GPU is available
options.batchSize = 64;             % Batch size for training
options.maxEpochs = 100;            % Maximum training epochs
options.initialLearnRate = 1e-4;    % Initial learning rate
options.plotProgress = "gui";       % Set to "none" to disable training plots
options.validationFrequency = 50;   % Validation frequency during training
options.validationPatience = 20;    % Patience for early stopping

% Get utility functions
utils = mmf_utils();

%% Load Dataset
try
    fprintf('Loading dataset from %s...\n', datasetFile);
    data = load(datasetFile);
    
    % Extract data fields
    mmf_train = data.mmf_train;
    mmf_val = data.mmf_val;
    mmf_test = data.mmf_test;
    labels_train = data.labels_train;
    labels_val = data.labels_val;
    labels_test = data.labels_test;
    number_of_modes = data.number_of_modes;
    
    fprintf('Dataset loaded successfully. Number of modes: %d\n', number_of_modes);
    fprintf('Training samples: %d, Validation samples: %d, Test samples: %d\n', ...
        size(mmf_train, 4), size(mmf_val, 4), size(mmf_test, 4));
catch ME
    error('Failed to load dataset: %s', ME.message);
end

%% Step 0: Canonicalize Phase Values
% Ensure first phase value is always positive to eliminate sign ambiguity
fprintf('\n=== Step 0: Canonicalizing phase values ===\n');

% Extract phase portions from labels
train_phases = labels_train(:, number_of_modes+1:end);
val_phases = labels_val(:, number_of_modes+1:end);
test_phases = labels_test(:, number_of_modes+1:end);

% Get first phase sign for each sample (reference phase)
train_first_phase_signs = sign(train_phases(:,1));
val_first_phase_signs = sign(val_phases(:,1));
test_first_phase_signs = sign(test_phases(:,1));

% Count samples with negative first phase
num_train_flipped = sum(train_first_phase_signs < 0);
num_val_flipped = sum(val_first_phase_signs < 0);
num_test_flipped = sum(test_first_phase_signs < 0);

fprintf('Found %d/%d training samples with negative first phase\n', num_train_flipped, size(train_phases,1));
fprintf('Found %d/%d validation samples with negative first phase\n', num_val_flipped, size(val_phases,1));
fprintf('Found %d/%d test samples with negative first phase\n', num_test_flipped, size(test_phases,1));

% Apply canonicalization: multiply all phases by the sign of the first phase
% This ensures the first phase is always positive
for i = 1:size(train_phases, 1)
    if train_first_phase_signs(i) < 0
        labels_train(i, number_of_modes+1:end) = -train_phases(i,:);
    end
end

for i = 1:size(val_phases, 1)
    if val_first_phase_signs(i) < 0
        labels_val(i, number_of_modes+1:end) = -val_phases(i,:);
    end
end

for i = 1:size(test_phases, 1)
    if test_first_phase_signs(i) < 0
        labels_test(i, number_of_modes+1:end) = -test_phases(i,:);
    end
end

fprintf('Phase canonicalization complete. All samples now have positive first phase.\n');

%% Validate Data Format
% Check dimensions of training data
if ndims(mmf_train) ~= 4 || size(mmf_train, 3) ~= 1
    error('mmf_train should have dimensions [height x width x 1 x samples]');
end

% Check dimensions of labels
expectedLabelCols = 2*number_of_modes - 1; % amplitudes + phases
if size(labels_train, 2) ~= expectedLabelCols
    error('labels_train should have dimensions [samples x %d]', expectedLabelCols);
end

% Check number of samples match
if size(mmf_train, 4) ~= size(labels_train, 1)
    error('Number of samples in mmf_train and labels_train do not match');
end

% Check if labels are properly formatted (amplitudes followed by phases)
amps = labels_train(:, 1:number_of_modes);
if any(amps(:) < 0)
    warning('Amplitudes contain negative values, which may indicate incorrect data formatting');
end

% Check if phases are in the expected range (-pi to pi)
phases = labels_train(:, number_of_modes+1:end);
if any(abs(phases(:)) > pi*1.1) % allow a little margin for numerical issues
    warning('Phases contain values outside the range [-pi,pi], which may indicate incorrect data formatting');
end

%% Create BPMmatlab Model
% Create BPMmatlab model with precomputed modes - will be passed to all components
inputSize = [size(mmf_train, 1), size(mmf_train, 2)];
fprintf('Creating BPMmatlab model with %d modes...\n', number_of_modes);
P = utils.getOrCreateModelWithModes(number_of_modes, inputSize(1), true);
fprintf('BPMmatlab model created with %d modes\n', number_of_modes);

% Configure training options
trainingOptions = struct();
trainingOptions.miniBatchSize = options.batchSize;
trainingOptions.maxEpochs = options.maxEpochs;
trainingOptions.initialLearnRate = options.initialLearnRate;
trainingOptions.executionEnvironment = options.executionEnvironment;
trainingOptions.plotProgress = options.plotProgress;
trainingOptions.validationFrequency = options.validationFrequency;
trainingOptions.validationPatience = options.validationPatience;
trainingOptions.modelType = 'MLP';
trainingOptions.adaptiveLossWeights = false;

%% Step 1: Train Amplitude and Phase Magnitude Model
if options.trainAmpModel
    fprintf('\n=== Step 1: Training amplitude and phase magnitude model ===\n');
    
    % Train the model with precomputed modes
    train_absolute_model(mmf_train, labels_train, ...
                         mmf_val, labels_val, ...
                         trainingOptions, P);
    
    fprintf('Amplitude and phase magnitude model training completed\n');
else
    fprintf('\n=== Step 1: Skipping amplitude and phase magnitude model training ===\n');
end

trainingOptions.modelType = 'PhaseSignResNet'; % Change model type for phase sign model

%% Step 2: Train Phase Sign Model
if options.trainPhaseModel
    fprintf('\n=== Step 2: Training phase sign model ===\n');
    
    % Extract training labels
    YTrain_amps = labels_train(:, 1:number_of_modes);
    YTrain_phases = labels_train(:, number_of_modes+1:end);
    YVal_amps = labels_val(:, 1:number_of_modes);
    YVal_phases = labels_val(:, number_of_modes+1:end);
    
    % Train the model with precomputed modes
    train_phase_sign_model(mmf_train, YTrain_amps, YTrain_phases, ...
                           mmf_val, YVal_amps, YVal_phases, ...
                           trainingOptions, P);
    
    fprintf('Phase sign model training completed\n');
else
    fprintf('\n=== Step 2: Skipping phase sign model training ===\n');
end

%% Step 3: Generate Residuals and Train Global Sign Classifier
if options.trainGlobalClassifier
    fprintf('\n=== Step 3: Generating residuals and training global classifier ===\n');
    
    % Generate residuals using precomputed modes
    fprintf('Generating residuals for training classifier...\n');
    generate_residuals(mmf_train, labels_train, mmf_val, labels_val, mmf_test, labels_test, P);
    
    % Train classifier
    fprintf('Training global phase sign classifier...\n');
    train_phase_sign_classifier(mmf_train, labels_train, mmf_val, labels_val, mmf_test, labels_test, trainingOptions, P);
    
    fprintf('Global phase sign classifier training completed\n');
else
    fprintf('\n=== Step 3: Skipping global sign classifier training ===\n');
end

%% Step 4: Evaluate Full Pipeline
if options.evaluate
    fprintf('\n=== Step 4: Evaluating full pipeline ===\n');
    
    % Create evaluation options
    evalOptions = struct();
    evalOptions.showPlots = true;
    evalOptions.batchSize = trainingOptions.miniBatchSize;
    evalOptions.executionEnvironment = trainingOptions.executionEnvironment;
    evalOptions.numSamplesToShow = 8;
    
    % Evaluate with precomputed modes
    fprintf('Running full pipeline evaluation...\n');
    evaluate_full_pipeline(mmf_test, labels_test, evalOptions, P);
    
    fprintf('Pipeline evaluation completed\n');
else
    fprintf('\n=== Step 4: Skipping pipeline evaluation ===\n');
end

%% Display Final Summary
% Report total execution time
elapsedTime = toc;
fprintf('\n=== Pipeline Execution Complete ===\n');
fprintf('Total execution time: %.2f seconds (%.2f minutes)\n', elapsedTime, elapsedTime/60);

% Display model file information
if exist('absolute_model.mat', 'file')
    info = dir('absolute_model.mat');
    fprintf('Amplitude model file size: %.2f MB\n', info.bytes/1024/1024);
end

if exist('phase_sign_model.mat', 'file')
    info = dir('phase_sign_model.mat');
    fprintf('Phase sign model file size: %.2f MB\n', info.bytes/1024/1024);
end

if exist('phase_sign_classifier.mat', 'file')
    info = dir('phase_sign_classifier.mat');
    fprintf('Global classifier file size: %.2f MB\n', info.bytes/1024/1024);
end

if exist('pipeline_evaluation.mat', 'file')
    info = dir('pipeline_evaluation.mat');
    fprintf('Evaluation results file size: %.2f MB\n', info.bytes/1024/1024);
end