function [U_out, tau, z] = gnlse_propagate(U0, params)
    %GNLSE_PROPAGATE Propagates optical field through multimode fiber using GNLSE
    %   [U_out, tau, z] = gnlse_propagate(U0, params) propagates initial field U0
    %   through multimode fiber using Generalized Nonlinear Schrödinger Equation
    %   with Split-Step Fourier Method. Compatible with mmf_build_image functions.
    %
    %   INPUTS:
    %       U0: Initial field - 2D (x,y) or 3D (x,y,t) array
    %       params: Structure with propagation parameters including:
    %           - use_time: Boolean to enable temporal effects
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
    params.ss_active = true;
    
    % Check dimensions of input field
    inputDims = ndims(U0);
    if params.use_time
        if inputDims == 2
            error('When use_time=true, input field U0 must be 3D (x,y,t)');
        end
        [nx, ny, nt] = size(U0);
        use3D = true;
    else
        [nx, ny] = size(U0);
        nt = 1;
        use3D = false;
    end
    
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

    % Setup time and frequency grids
    if use3D
        Tmax = params.Tmax; % [ps] 
        dtau = (2*Tmax)/nt; % Time step [ps]
        tau = (-nt/2:nt/2-1)*dtau; % Time array [ps]
        omega = 2*pi*fftshift((-nt/2:nt/2-1)/(nt*dtau)); % Angular frequency [rad/ps]
    else
        Tmax = 1; % Not used
        dtau = 1; % Not used
        tau = 0; % Not used
        omega = 0; % Not used
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
    gamma_profile = zeros(size(R2), 'like', real(U0));
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
    
    % Move data to GPU if needed
    if useGPU && ~isInputGPU
        U0 = gpuArray(U0);
        omega = gpuArray(omega);
        tau = gpuArray(tau);
        gamma_profile = gpuArray(gamma_profile);
        core_mask = gpuArray(core_mask);
    end
    
    % Pre-compute spatial frequency grid
    dkx = 2*pi/(nx*(X(1,2)-X(1,1)));
    dky = 2*pi/(ny*(Y(2,1)-Y(1,1)));
    kx = fftshift((-nx/2:nx/2-1)*dkx);
    ky = fftshift((-ny/2:ny/2-1)*dky);
    [Kx, Ky] = meshgrid(kx, ky);
    K2 = Kx.^2 + Ky.^2;
    
    % Pre-compute spatial dispersion operator
    spatial_dispersion = exp(1i*dz*0.5*K2);
    
    % Pre-compute temporal dispersion operator if using 3D
    if use3D
        % Dispersion terms - We expand to second order by default, higher orders optional
        beta2_term = 0.5 * sbeta2 * omega.^2;
        beta3_term = delta3/6 * omega.^3;
        beta4_term = delta4/24 * omega.^4;
        
        % Complete dispersion operator
        temporal_dispersion = exp(1i*dz*(beta2_term + beta3_term + beta4_term));
    end
    
    % Calculate Raman response for temporal effects
    if use3D
        % Raman response function calculation (standard model from fiber optics literature)
        tau1 = 12.2/T0; % Adjusted Raman timescale [normalized]
        tau2 = 32/T0;   % Adjusted Raman timescale [normalized]
        tau_b = 96/T0;  % Boson peak timescale [normalized]
        h_tau = tau;
        
        if useGPU
            h_tau = gpuArray(h_tau);
        end
        
        % Compute Raman response function h(t)
        h = (1-fb)*(tau1^2 + tau2^2)/(tau1*tau2^2)*exp(-h_tau/tau2).*sin(h_tau/tau1)...
            + fb*((2*tau_b-h_tau)/tau_b^2).*exp(-h_tau/tau_b);
        h(h_tau < 0) = 0; % Causality condition
        h = h/sum(h*dtau); % Normalize
        
        % Pre-compute FFT of Raman response for efficient convolution
        H_omega = fft(ifftshift(h));
    end
    
    % Nonlinear coefficient
    gamma_nl = N^2; % Combined nonlinear parameter
    
    % Prepare for propagation
    U = U0; % Initial field
    
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
    
    % Track field at specified z positions
    U_out = U;
    save_idx = 1;
    
    % Main propagation loop
    for n = 1:step_num
        % SPLIT-STEP PROPAGATION BASED ON SPATIAL/TEMPORAL MODE
        if use3D
            % ------------- 3D SPATIOTEMPORAL PROPAGATION -------------
            
            % ------------- LINEAR STEP (FIRST HALF) -------------
            % 1. Apply spatial dispersion for each time slice
            for t_idx = 1:nt
                U_slice = U(:,:,t_idx);
                % FFT in spatial domain
                U_slice_fft = fft2(U_slice);
                % Apply spatial dispersion
                U_slice_fft = U_slice_fft .* spatial_dispersion;
                % IFFT back to spatial domain
                U(:,:,t_idx) = ifft2(U_slice_fft);
            end
            
            % 2. Apply temporal dispersion for each spatial point
            for i = 1:nx
                for j = 1:ny
                    % Extract temporal profile at this spatial point
                    temp_slice = squeeze(U(i,j,:));
                    % FFT in temporal domain
                    temp_slice_fft = fft(temp_slice);
                    % Apply temporal dispersion
                    temp_slice_fft = temp_slice_fft .* temporal_dispersion(:);
                    % IFFT back to time domain
                    U(i,j,:) = ifft(temp_slice_fft);
                end
            end
            
            % ------------- NONLINEAR STEP -------------
            % Calculate field intensity (|U|²)
            P = abs(U).^2;
            
            % Apply spatially-dependent nonlinearity profile
            % Expand gamma_profile to 3D by replicating along time dimension
            gamma3D = repmat(gamma_profile, [1, 1, nt]);
            P_effective = P .* gamma3D;
            
            % Apply Kerr nonlinearity (instantaneous)
            kerr_term = (1-fR) * P_effective;
            
            % Apply Raman effect (delayed response)
            % Loop over spatial points for Raman convolution
            R = zeros(size(U), 'like', U);
            for i = 1:nx
                for j = 1:ny
                    % Extract P at this spatial point
                    P_point = squeeze(P_effective(i,j,:));
                    % Raman convolution via FFT
                    R_fft = fft(P_point) .* H_omega(:);
                    % Back to time domain
                    R(i,j,:) = real(ifft(R_fft));
                end
            end
            raman_term = fR * R;
            
            % Self-steepening effect
            if params.ss_active
                % Calculate time derivatives
                dPdt = zeros(size(P_effective), 'like', P_effective);
                dUdt = zeros(size(U), 'like', U);
                
                % Compute time derivatives using spectral method
                for i = 1:nx
                    for j = 1:ny
                        % Time derivative of |U|²
                        dPdt(i,j,:) = time_derivative(squeeze(P_effective(i,j,:)), dtau);
                        % Time derivative of U
                        dUdt(i,j,:) = time_derivative(squeeze(U(i,j,:)), dtau);
                    end
                end
                
                % Self-steepening terms
                ss_term = 1i*s*((1-fR)*dPdt + fR*time_derivative(R, dtau));
            else
                ss_term = 0;
            end
            
            % Combined nonlinear term
            nonlinear_term = kerr_term + raman_term + ss_term;
            
            % Apply nonlinear phase shift
            U = U .* exp(1i * gamma_nl * dz * nonlinear_term);
            
            % ------------- LINEAR STEP (SECOND HALF) -------------
            % 1. Apply temporal dispersion for each spatial point
            for i = 1:nx
                for j = 1:ny
                    % Extract temporal profile at this spatial point
                    temp_slice = squeeze(U(i,j,:));
                    % FFT in temporal domain
                    temp_slice_fft = fft(temp_slice);
                    % Apply temporal dispersion
                    temp_slice_fft = temp_slice_fft .* temporal_dispersion(:);
                    % IFFT back to time domain
                    U(i,j,:) = ifft(temp_slice_fft);
                end
            end
            
            % 2. Apply spatial dispersion for each time slice
            for t_idx = 1:nt
                U_slice = U(:,:,t_idx);
                % FFT in spatial domain
                U_slice_fft = fft2(U_slice);
                % Apply spatial dispersion
                U_slice_fft = U_slice_fft .* spatial_dispersion;
                % IFFT back to spatial domain
                U(:,:,t_idx) = ifft2(U_slice_fft);
            end
        else
            % ------------- 2D SPATIAL-ONLY PROPAGATION -------------
            % LINEAR STEP (FIRST HALF)
            U_fft = fft2(U);             % Transform to frequency domain
            U_fft = U_fft .* spatial_dispersion; % Apply dispersion
            U = ifft2(U_fft);            % Transform back to spatial domain
            
            % NONLINEAR STEP
            P = abs(U).^2;               % Power
            P_effective = P .* gamma_profile; % Apply nonlinearity profile
            
            % Apply nonlinear phase shift (Kerr only for 2D case)
            U = U .* exp(1i * gamma_nl * dz * P_effective);
            
            % LINEAR STEP (SECOND HALF)
            U_fft = fft2(U);             % Transform to frequency domain
            U_fft = U_fft .* spatial_dispersion; % Apply dispersion
            U = ifft2(U_fft);            % Transform back to spatial domain
        end
        
        % Optional: Apply loss if specified
        if isfield(params, 'apply_loss') && params.apply_loss
            if isfield(params, 'loss_dB_per_m')
                % Convert dB/m to amplitude loss coefficient
                alpha_power = params.loss_dB_per_m / (10*log10(exp(1)));
                alpha_amplitude = alpha_power/2;  % Power loss to amplitude loss
                
                % Create loss profile (higher in cladding)
                loss_profile = ones(size(core_mask), 'like', real(U));
                loss_profile(core_mask) = alpha_amplitude;
                loss_profile(~core_mask) = alpha_amplitude * 5; % 5x more loss in cladding
                
                % Apply loss
                if use3D
                    % Expand loss_profile to 3D
                    loss3D = repmat(loss_profile, [1, 1, nt]);
                    U = U .* exp(-loss3D * dz);
                else
                    U = U .* exp(-loss_profile * dz);
                end
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

function d = time_derivative(u, dt)
    % Calculate time derivative using spectral method (more accurate than finite differences)
    n = length(u);
    u_fft = fft(u);
    k = 2*pi*1i*[0:n/2-1, 0, -n/2+1:-1]/n/dt;
    d = real(ifft(u_fft .* k(:)));
end