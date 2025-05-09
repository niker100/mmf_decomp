function [U_out, tau, z] = gnlse_propagate(U0, params)
    %GNLSE_PROPAGATE Propagates optical field through multimode fiber using GNLSE
    %   [U_out, tau, z] = gnlse_propagate(U0, params) propagates initial field U0
    %   through multimode fiber using Generalized Nonlinear Schrödinger Equation
    %   with Split-Step Fourier Method. Compatible with mmf_build_image functions.
    %
    %   INPUTS:
    %       U0: Initial field (2D spatial profile)
    %       params: Structure with propagation parameters including:
    %           - core_radius: Fiber core radius [m]
    %           - n_core: Core refractive index
    %           - n_clad: Cladding refractive index
    %
    %   OUTPUTS:
    %       U_out: Propagated field
    %       tau: Time vector
    %       z: Distance vector
    
    % Set defaults for missing params
    if ~isfield(params, 'chirp0'), params.chirp0 = 0; end
    if ~isfield(params, 'delta3'), params.delta3 = 0; end
    if ~isfield(params, 'delta4'), params.delta4 = 0; end
    if ~isfield(params, 'fR'), params.fR = 0.245; end
    if ~isfield(params, 'fb'), params.fb = 0.21; end
    if ~isfield(params, 'tol'), params.tol = 1e3; end
    if ~isfield(params, 'useGPU'), params.useGPU = false; end
    if ~isfield(params, 'n2'), params.n2 = 2.6e-20; end % Nonlinear index [m²/W]
    if ~isfield(params, 'adaptive_step'), params.adaptive_step = false; end
    if ~isfield(params, 'error_threshold'), params.error_threshold = 1e-6; end
    if ~isfield(params, 'save_steps'), params.save_steps = 1; end % Save every Nth step
    
    % Core/cladding defaults
    if ~isfield(params, 'core_radius'), error('Must specify core_radius'); end
    if ~isfield(params, 'n_core'), params.n_core = 1.46; end
    if ~isfield(params, 'n_clad'), params.n_clad = 1.45; end
    if ~isfield(params, 'nonlinear_in_cladding'), params.nonlinear_in_cladding = false; end
    
    % Constants
    c = 299792458; % Speed of light [m/s]
    
    % Extract parameters
    T0 = params.T0; % [ps]
    lam0 = params.lam0; % [nm]
    distance = params.distance; % [m]
    N = params.N; % Nonlinear parameter scaling
    sbeta2 = params.sbeta2; % [ps²/m]
    delta3 = params.delta3; % [ps³/m]
    delta4 = params.delta4; % [ps⁴/m]
    nt = params.nt; % Time grid points
    Tmax = params.Tmax; % [ps]
    step_num = params.step_num;
    zstep = params.zstep;
    fR = params.fR; % Raman fraction
    fb = params.fb; % Boson peak fraction
    tol = params.tol; % Divergence tolerance
    useGPU = params.useGPU;
    
    % Fiber parameters
    core_radius = params.core_radius; % [m]
    n_core = params.n_core;
    n_clad = params.n_clad;
    
    % Calculate spatial dimensions from input field
    [nx, ny] = size(U0);
    
    % Create or extract coordinate grid
    if ~isfield(params, 'X') || ~isfield(params, 'Y')
        % Default to grid spanning +/- 3 times the core radius
        x_span = 3 * core_radius;
        x = linspace(-x_span, x_span, nx);
        y = linspace(-x_span, x_span, ny);
        [X, Y] = meshgrid(x, y);
    else
        X = params.X;
        Y = params.Y;
    end
    
    % Calculate radial coordinates for core/cladding distinction
    R2 = X.^2 + Y.^2;
    core_mask = (R2 <= core_radius^2);
    
    % Create refractive index profile
    n_profile = n_clad * ones(size(R2));
    n_profile(core_mask) = n_core;
    
    % Calculate effective V-parameter
    V = (2*pi/lam0*1e-9) * core_radius * sqrt(n_core^2 - n_clad^2);
    
    % Create nonlinearity profile (typically stronger in core)
    gamma_profile = zeros(size(R2), 'like', U0);
    gamma_profile(core_mask) = 1.0;  % Full nonlinearity in core
    
    % Add reduced nonlinearity in cladding if specified
    if isfield(params, 'nonlinear_in_cladding') && params.nonlinear_in_cladding
        if isfield(params, 'cladding_nonlinearity_ratio')
            cladding_ratio = params.cladding_nonlinearity_ratio;
        else
            cladding_ratio = 0.1; % Default: 10% of core nonlinearity
        end
        gamma_profile(~core_mask) = cladding_ratio;
    end
    
    % Derived parameters
    omega0 = 2*pi*c/(lam0*1e-9); % Angular frequency [rad/s]
    s = lam0/(2*pi*3e2*T0); % Self-steepening parameter
    dz = distance/step_num; % Step size [m]
    
    % Check if input is already on GPU
    isInputGPU = isa(U0, 'gpuArray');
    
    % Time and frequency grid setup
    dtau = (2*Tmax)/nt; % Time step [ps]
    tau = (-nt/2:nt/2-1)*dtau; % Time array [ps]
    omega = fftshift((pi/Tmax)*(-nt/2:nt/2-1)); % Angular frequency array [rad/ps]
    
    % Move data to GPU if needed
    if useGPU && ~isInputGPU
        U0 = gpuArray(U0);
        omega = gpuArray(omega);
        tau = gpuArray(tau);
        gamma_profile = gpuArray(gamma_profile);
        core_mask = gpuArray(core_mask);
    end
    
    % Pre-compute meshgrid only once (for spatial-spectral operations)
    [O1, O2] = meshgrid(omega, omega);
    
    % Calculate spatially-dependent dispersion
    % For simplicity, we use the same dispersion for both regions in this version
    % For a more accurate model, use different dispersion in core and cladding
    
    % Raman response function calculation
    tau1 = 12.2/T0; % Adjusted Raman timescale [normalized]
    tau2 = 32/T0;   % Adjusted Raman timescale [normalized]
    tau_b = 96/T0;  % Boson peak timescale [normalized]
    h_tau = tau;
    
    if useGPU && ~isInputGPU
        h_tau = gpuArray(h_tau);
    end
    
    % Compute Raman response function h(t)
    h = (1-fb)*(tau1^2 + tau2^2)/(tau1*tau2^2)*exp(-h_tau/tau2).*sin(h_tau/tau1)...
        + fb*((2*tau_b-h_tau)/tau_b^2).*exp(-h_tau/tau_b);
    h(h_tau < 0) = 0; % Causality condition
    h = h/sum(h*dtau); % Normalize
    
    % Pre-compute FFT of Raman response for efficient convolution
    H_omega = fft(ifftshift(h)); % FFT for temporal convolution
    
    % Dispersive phase operator (for linear step)
    % Can implement spatially varying dispersion by creating a more complex operator
    dispersion = exp(1i*dz*(0.5*sbeta2*(O1.^2+O2.^2) + delta3/6*(O1.^3+O2.^3) + delta4/24*(O1.^4+O2.^4)));
    
    % Nonlinear coefficient
    gamma_nl = N^2; % Combined nonlinear parameter
    
    % Prepare for propagation
    U = U0; % Initial field
    
    % Pre-allocate arrays for speed
    P = zeros(nx, ny, 'like', U); % Power
    R = zeros(nx, ny, 'like', U); % Raman response term
    dPdt = zeros(nx, ny, 'like', U); % Time derivative of power
    dUdt = zeros(nx, ny, 'like', U); % Time derivative of field
    dRdt = zeros(nx, ny, 'like', U); % Time derivative of Raman term
    
    % Create output arrays
    save_indices = 1:params.save_steps:step_num+1;
    num_saves = length(save_indices);
    
    if nargout > 2
        z = linspace(0, distance, num_saves);
    end
    
    % Pre-compute FFT plans for efficient computation
    if useGPU
        fftw('planner', 'patient');
    end
    
    % Initialize FFT of field
    U_fft = fft2(U);
    
    % Track field at specified z positions
    U_out = U;
    save_idx = 1;
    
    % Main propagation loop
    for n = 1:step_num
        % --------------- LINEAR STEP (FIRST HALF) ---------------
        U_fft = U_fft .* dispersion; % Apply dispersion in frequency domain
        U = ifft2(U_fft);            % Transform back to spatial domain
        
        % --------------- NONLINEAR STEP ---------------
        % Calculate field intensity
        P = abs(U).^2;
        
        % Apply spatially-dependent nonlinearity profile
        P_effective = P .* gamma_profile;
        
        % Raman contribution via efficient FFT-based convolution
        P_fft = fft(P_effective);
        R = real(ifft(P_fft .* H_omega)); % Convolution: P ⊗ h(t)
        
        % Calculate time derivatives for self-steepening
        dPdt = spatial_time_derivative(P_effective, dtau);
        dUdt = spatial_time_derivative(U, dtau);
        dRdt = spatial_time_derivative(R, dtau);
        
        % Calculate nonlinear terms (Kerr + Raman + self-steepening)
        kerr_term = (1-fR)*P_effective;
        raman_term = fR*R;
        
        % Self-steepening terms
        ss_kerr = 1i*s*(1-fR)*(dPdt + conj(U).*dUdt);
        ss_raman = 1i*s*fR*(dRdt + (conj(U).*R)./(P_effective+eps).*dUdt);
        
        % Combined nonlinear phase
        nonlinear_phase = (kerr_term + raman_term + ss_kerr + ss_raman);
        
        % Apply nonlinear phase shift
        U = U .* exp(1i * gamma_nl * dz * nonlinear_phase);
        
        % --------------- LINEAR STEP (SECOND HALF) ---------------
        U_fft = fft2(U);              % Transform to frequency domain
        U_fft = U_fft .* dispersion;  % Apply dispersion again
        U = ifft2(U_fft);             % Transform back to spatial domain
        
        % Optional: Apply spatially varying loss (more loss in cladding)
        if isfield(params, 'apply_loss') && params.apply_loss
            if isfield(params, 'loss_dB_per_m')
                % Convert dB/m to amplitude loss coefficient
                alpha_power = params.loss_dB_per_m / (10*log10(exp(1)));
                alpha_amplitude = alpha_power/2;  % Power loss to amplitude loss
                
                % Create loss profile (higher in cladding)
                loss_profile = ones(size(core_mask), 'like', U);
                loss_profile(core_mask) = alpha_amplitude;
                loss_profile(~core_mask) = alpha_amplitude * 5; % 5x more loss in cladding
                
                % Apply loss
                U = U .* exp(-loss_profile * dz);
            end
        end
        
        % Check for numerical blowup periodically
        if mod(n, max(1, floor(step_num/10))) == 0
            max_amp = max(abs(U(:)));
            if max_amp > tol || any(isnan(U(:)))
                warning('Numerical instability detected at step %d of %d. Max amplitude: %g', n, step_num, max_amp);
                break;
            end
        end
        
        % Save output at specified steps
        if ismember(n+1, save_indices)
            save_idx = save_idx + 1;
            U_out = U;
        end
    end
    
    % Ensure output is in same format as input
    if ~isInputGPU && useGPU
        U_out = gather(U_out);
        tau = gather(tau);
        if nargout > 2
            z = gather(z);
        end
    end
end

function d = spatial_time_derivative(u, dt)
    % Efficient calculation of temporal derivative for 2D spatial fields
    % using spectral method with periodic boundary conditions
    
    % Compute spectral derivative using FFT for better accuracy
    [nx, ny] = size(u);
    u_fft = fft(u);
    k = 1i*[0:nx/2-1, 0, -nx/2+1:-1]; % Frequency components
    d = real(ifft(u_fft .* k(:) * (2*pi/(nx*dt)))); % Spectral differentiation
end