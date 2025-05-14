# Multimode Fiber (MMF) Mode Decomposition Pipeline

This repository contains a pipeline for extracting mode weights from intensity-only images of multimode fiber (MMF) output patterns. The pipeline uses deep learning techniques to infer complex-valued modal coefficients from intensity patterns, without requiring phase information.

## Overview

The project addresses the challenge of retrieving modal weights (both amplitude and phase information) from intensity-only measurements of multimode fiber output patterns. This problem is inherently ill-posed due to the loss of phase information during measurement, but our approach leverages deep learning to recover this information.

## Features

- **Nonlinear Dataset Generation**: Uses the GMMNLSE Solver (by WiseLabAEP) to simulate realistic nonlinear propagation in multimode fibers
- **Reference-Less Mode Weight Extraction**: Extracts mode amplitudes and phases from intensity-only images without requiring phase reference
- **Multi-Stage Neural Network Pipeline**: A sequence of specialized models for amplitude estimation, phase sign prediction, and global sign disambiguation
- **Physics-Informed Neural Networks**: Incorporates physical constraints and domain knowledge into the neural network architecture

## Pipeline Components

1. **Dataset Generation**: Creates a synthetic dataset of MMF intensity patterns with known mode weights using nonlinear propagation simulations
2. **Amplitude and Absolute Phase Model**: Predicts mode amplitudes and absolute phase values from intensity patterns
3. **Phase Sign Model**: Predicts the signs of phase values for each mode relative to a reference mode
4. **Global Sign Classifier**: Resolves global sign ambiguity by analyzing residuals between predicted and actual patterns

## Installation Requirements

- MATLAB R2021b or later
- Deep Learning Toolbox
- Image Processing Toolbox
- Parallel Computing Toolbox (recommended for GPU acceleration)
- BPM-Matlab (included)
- GMMNLSE-Solver-FINAL (included, from WiseLabAEP)

## Usage

### Running the Complete Pipeline

```matlab
% Configuration
options = struct();
options.trainAmpModel = true;        % Train amplitude model
options.trainPhaseModel = true;      % Train phase sign model
options.trainGlobalClassifier = true; % Train global classifier
options.evaluate = true;             % Evaluate full pipeline
options.executionEnvironment = "gpu"; % Use GPU if available

% Run pipeline
run_mmf_pipeline
```

### Generating a Dataset

```matlab
% Dataset generation with varying nonlinearity
number_of_modes = 5;          % Number of modes to simulate
image_size = 128;             % Output image resolution
useNonLinear = true;          % Use nonlinear propagation
use_varying_nl = true;        % Use varying nonlinearity strength
nl_strength_min = 0.5;        % Minimum nonlinearity strength
nl_strength_max = 5.0;        % Maximum nonlinearity strength

% Run dataset generation
dataset_generation
```

### Evaluating a Trained Pipeline

```matlab
% Load dataset
load('mmf_dataset_multi_sign.mat');

% Evaluate pipeline
evaluate_full_pipeline(mmf_test, labels_test);
```

## Key Files

- `mmf_build_image.m` - Generates MMF output images using mode superposition and nonlinear propagation
- `mmf_utils.m` - Utility functions for working with BPM models and mode fields
- `dataset_generation.m` - Creates training, validation and test datasets
- `run_mmf_pipeline.m` - Main script to run the complete pipeline
- `train_absolute_model.m` - Trains model to predict amplitudes and absolute phases
- `train_phase_sign_model.m` - Trains model to predict phase signs
- `train_phase_sign_classifier.m` - Trains classifier for global sign disambiguation
- `evaluate_full_pipeline.m` - Evaluates the full pipeline performance
- `nonlinearity_influence.m` - Analysis script for studying nonlinear effects

## GMMNLSE Solver Integration

This project integrates the GMMNLSE Solver developed by WiseLabAEP for simulating nonlinear propagation in multimode fibers. The solver is used to generate realistic MMF output patterns with accurate mode coupling and nonlinear effects.

The GMMNLSE (Generalized Multi-Mode Nonlinear Schrödinger Equation) solver simulates:
- Self-phase modulation (SPM)
- Cross-phase modulation (XPM)
- Four-wave mixing (FWM)
- Raman scattering
- Self-steepening

## Model Architecture

The pipeline uses multiple specialized neural networks:

1. **Amplitude Model**: CNN-based architecture for predicting mode amplitudes and absolute phase values
2. **Phase Sign Model**: ResNet-based architecture for predicting relative phase signs
3. **Global Classifier**: CNN with attention mechanism for global sign disambiguation

## Performance Metrics

The pipeline is evaluated using several metrics:
- Amplitude prediction accuracy (correlation, MSE)
- Phase sign prediction accuracy
- Reconstruction fidelity (correlation between predicted and ground truth images)
- Physical plausibility measures (energy conservation, etc.)

## Acknowledgments

- GMMNLSE Solver by WiseLabAEP: https://github.com/WiseLabAEP/GMMNLSE-Solver-MATLAB
- BPM-Matlab for mode simulation and field propagation

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.