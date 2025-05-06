% filepath: c:\Users\nicks\MATLAB\Projects\MT2_prak\Versuch 3\mmf_build_image.m
function [Image_data, complex_field_out] = mmf_build_image(number_of_modes, image_size, number_of_data, complex_weights_vector, useNonLinear, nonlinear_strength, P_precomputed)
    % MMF_BUILD_IMAGE now uses BPM-Matlab's modeSuperposition for initial field
    % and GNLSE for propagation with GPU acceleration.
    if nargin < 6
        nonlinear_strength = 1.0; % Default nonlinear strength (s-parameter for GNLSE)
    end
    if isempty(which('BPMmatlab.model'))
        addpath(genpath(fullfile(pwd, 'BPM-Matlab')));
    end

    % Check for GPU availability
    persistent useGPU;
    if isempty(useGPU)
        useGPU = false;
    end
    if gpuDeviceCount > 0 && ~useGPU
        useGPU = true;
        fprintf('GPU acceleration enabled: %s\n', gpuDevice().Name);
    end

    if nargin < 7 || isempty(P_precomputed)
        % Set up BPMmatlab.model for initial field generation
        P = BPMmatlab.model;
        %% General and solver-related settings
        P.name = mfilename;
        P.useAllCPUs = true; % For BPM-Matlab parts
        P.useGPU = useGPU;   % Enable GPU for BPM-Matlab 

        %% Visualization parameters (less relevant if only using for initial field)
        updateFrequency = 5e4; 
        P.saveVideo = false; 
        P.plotZoom = 1;  
        P.plotEmax = 1; 

        %% Resolution-related parameters
        P.Lx_main = 25e-6;        % [m] x side length of main area
        P.Ly_main = 25e-6;        % [m] y side length of main area
        P.Nx_main = image_size;          % x resolution of main area
        P.Ny_main = image_size;          % y resolution of main area
        P.padfactor = 1.5;  
        P.dz_target = 1e-2; % [m] z step size to aim for

        %% Problem definition
        P.lambda = 1000e-9; % [m] Wavelength
        P.n_background = 1.45; % [] Background refractive index (cladding)
        P.n_0 = 1.46; % Core refractive index (used for RI profile)
        P.Lz = 10e-4; % [m] z propagation distance
        P.updates = P.Lz*updateFrequency; 
        P.figTitle = 'Segment 1';

        % Define refractive index profile (step-index fiber)
        core_radius = 15e-6;
        n_core = P.n_0; % Use P.n_0 for consistency
        n_clad = P.n_background; % Use P.n_background
        P = initializeRIfromFunction(P, @(X,Y,~,~) n_clad + (n_core-n_clad)*(X.^2+Y.^2 < core_radius^2));

        % Find modes for this fiber/grid
        P = findModes(P, number_of_modes, 'plotModes', false);
    else
        % Use precomputed P (with modes)
        P = P_precomputed;
    end

    % Ensure weights are on CPU before use
    if isa(complex_weights_vector, 'gpuArray')
        complex_weights_vector = gather(complex_weights_vector);
    end
    weights = complex_weights_vector(1, 1:number_of_modes);
    if size(weights,2) > 1
        weights = weights.';
    end

    % Create initial field as a superposition of the found modes
    P.E = modeSuperposition(P, 1:number_of_modes, weights);
    U_initial_raw = P.E.field; % Initial complex field for GNLSE

    % --- GNLSE Propagation with GPU Acceleration ---
    if useNonLinear
        % Define simulation parameters for GNLSE
        sim_nt = 256; % Simulation grid size (higher resolution for stability)
        
        % Upsample U_initial_raw to sim_nt x sim_nt if necessary
        if size(U_initial_raw, 1) ~= sim_nt || size(U_initial_raw, 2) ~= sim_nt
            U_initial_sim = imresize(U_initial_raw, [sim_nt, sim_nt], 'bicubic');
        else
            U_initial_sim = U_initial_raw;
        end

        % Move to GPU if available
        if useGPU
            U_initial_sim = gpuArray(U_initial_sim);
        end

        % Set up parameters for gnlse_propagate
        params = struct();
        params.T0 = 50; % fs, characteristic pulse width for nonlinear effects
        params.lam0 = P.lambda * 1e9; % nm, central wavelength from BPM model
        params.distance = 1; % normalized propagation distance (L_D units)
        params.N = nonlinear_strength; % Soliton order
        params.sbeta2 = -1; % sign of beta_2 (anomalous dispersion, adjust as needed)
        params.mshape = 0; % 0 for sech (not directly used as U_initial_sim is the input shape)
        params.delta3 = 0; % normalized beta3
        params.delta4 = 0; % normalized beta4
        params.chirp0 = 0; % input pulse chirp (U_initial_sim has its own phase)
        
        params.nt = sim_nt; % number of grid points for simulation
        params.Tmax = 50; % Dimensionless window size
        params.step_num = 10; % Increased number of z steps for stability
        params.zstep = 10; % number of output frames (not critical for final field)
        params.fR = 0.18; % Raman fraction
        params.fb = 0.21; % Raman parameter
        params.tol = 1e4; % Blowup threshold (can be adjusted)
        params.useGPU = useGPU; % Pass GPU flag to propagation function

        % Call gnlse_propagate for 2D propagation
        [Eout_sim, ~, ~] = gnlse_propagate(U_initial_sim, params);
        
        % Move result back to CPU if needed
        if useGPU
            Eout_sim = gather(Eout_sim);
        end
        
        % Downsample Eout_sim back to image_size x image_size if necessary
        if size(Eout_sim, 1) ~= image_size || size(Eout_sim, 2) ~= image_size
            Eout = imresize(Eout_sim, [image_size, image_size], 'bicubic');
        else
            Eout = Eout_sim;
        end
    else
        Eout = U_initial_raw; % Use the raw initial field if no non-linearity
    end
    % --- End GNLSE Propagation ---

    % Extract output field and intensity
    intensity = abs(Eout).^2; % Intensity of the output field
    max_val = max(intensity(:));
    if max_val == 0
        max_val = eps('double');
    end
    Image_data = intensity ./ max_val;
    Image_data = reshape(Image_data, [image_size, image_size, 1, 1]);
    complex_field_out = Eout;
end