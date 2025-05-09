function [Image_data, complex_field_out] = mmf_build_image(number_of_modes, image_size, number_of_data, complex_weights_vector, useNonLinear, nonlinear_strength, P_precomputed)
    % MMF_BUILD_IMAGE - Generates MMF output images using mode superposition and gnlse_propagate
    % Compatible with new gnlse_propagate and mmf_utils.
    % Inputs:
    %   number_of_modes: Number of modes to use
    %   image_size: Size of output image (NxN)
    %   number_of_data: Number of samples to generate
    %   complex_weights_vector: [number_of_data x number_of_modes] complex weights
    %   useNonLinear: true for nonlinear propagation, false for linear
    %   nonlinear_strength: N parameter for gnlse_propagate
    %   P_precomputed: (optional) precomputed model struct

    if nargin < 6 || isempty(nonlinear_strength)
        nonlinear_strength = 1.0;
    end

    % Use mmf_utils for model/mode generation
    utils = mmf_utils();

    % Check for GPU
    useGPU = (gpuDeviceCount > 0);

    % Get or create model with modes
    if nargin < 7 || isempty(P_precomputed)
        P = utils.getOrCreateModelWithModes(number_of_modes, image_size, true);
    else
        P = P_precomputed;
    end

    % Ensure weights are on CPU
    if isa(complex_weights_vector, 'gpuArray')
        complex_weights_vector = gather(complex_weights_vector);
    end

    % Output allocation
    Image_data = zeros(image_size, image_size, 1, number_of_data, 'like', complex_weights_vector);
    complex_field_out = zeros(image_size, image_size, number_of_data, 'like', complex_weights_vector);

    % Prepare propagation parameters (orient on nonlinearity_influence/tests)
    params = struct();
    params.T0 = 50;
    params.lam0 = P.lambda * 1e9;
    params.distance = 100; % Default, can be changed per sample if needed
    params.N = nonlinear_strength;
    params.sbeta2 = -0.1;
    params.nt = image_size;
    params.Tmax = 50;
    params.step_num = 100;
    params.zstep = 1;
    params.fR = 0.18;
    params.fb = 0.21;
    params.tol = 1e4;
    params.n_clad = P.n_background;
    params.n_core = P.n_0;
    params.core_radius = 25e-6;
    params.useGPU = useGPU;
    params.X = P.Nx_main;
    params.Y = P.Ny_main;
    params.nonlinear_in_cladding = false;

    for idx = 1:number_of_data
        weights = complex_weights_vector(idx, 1:number_of_modes);
        if size(weights,2) > 1
            weights = weights.';
        end
        % Generate initial field
        P.E = modeSuperposition(P, 1:number_of_modes, weights);
        U_initial = P.E.field;
        if size(U_initial, 1) ~= image_size || size(U_initial, 2) ~= image_size
            U_initial = imresize(U_initial, [image_size, image_size], 'bicubic');
        end
        if useGPU
            U_initial = gpuArray(U_initial);
        end

        % linear/nonlinear
        if useNonLinear
            params.N = nonlinear_strength;
            % Propagate
            [U_out, ~, ~] = gnlse_propagate(U_initial, params);
            if useGPU
                U_out = gather(U_out);
            end
        else
            U_out = U_initial;
        end        

        % Normalize intensity for output
        intensity = abs(U_out).^2;
        max_val = max(intensity(:));
        if max_val == 0
            max_val = eps('double');
        end
        Image_data(:,:,:,idx) = intensity ./ max_val;
        complex_field_out(:,:,idx) = U_out;
    end

    % Reshape Image_data to [image_size, image_size, 1, number_of_data]
    Image_data = reshape(Image_data, [image_size, image_size, 1, number_of_data]);
end