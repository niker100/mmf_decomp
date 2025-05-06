function [U_out, tau, z] = gnlse_propagate(U0, params)
    %GNLS_PROPagate: Optimized GNLSE propagation for 2D fields using split-step Fourier method
    
    % Set defaults for missing params
    if ~isfield(params, 'chirp0'), params.chirp0 = 0; end
    if ~isfield(params, 'delta3'), params.delta3 = 0; end
    if ~isfield(params, 'delta4'), params.delta4 = 0; end
    if ~isfield(params, 'fR'), params.fR = 0.245; end
    if ~isfield(params, 'fb'), params.fb = 0.21; end
    if ~isfield(params, 'tol'), params.tol = 1e3; end
    if ~isfield(params, 'useGPU'), params.useGPU = false; end
    
    T0 = params.T0;
    lam0 = params.lam0;
    s = lam0/(2*pi*3e2*T0);  % self-steepening
    
    distance = params.distance;
    N = params.N;
    sbeta2 = params.sbeta2;
    delta3 = params.delta3;
    delta4 = params.delta4;
    nt = params.nt;
    Tmax = params.Tmax;
    step_num = params.step_num;
    zstep = params.zstep;
    fR = params.fR;
    fb = params.fb;
    tol = params.tol;
    useGPU = params.useGPU;
    
    % Check if input is already on GPU
    isInputGPU = isa(U0, 'gpuArray');
    
    % We'll focus on 2D case only for optimization
    dtau = (2*Tmax)/nt;
    tau = (-nt/2:nt/2-1)*dtau;
    omega = fftshift((pi/Tmax)*(-nt/2:nt/2-1));
    
    % Move to GPU if needed
    if useGPU && ~isInputGPU
        U0 = gpuArray(U0);
        omega = gpuArray(omega);
    end
    
    % Pre-compute meshgrid only once
    [O1, O2] = meshgrid(omega, omega);
    
    % Raman response function - optimized calculation
    tau1=12.2/T0; tau2=32/T0; tau_b=96/T0;
    h_tau = tau;
    if useGPU && ~isInputGPU
        h_tau = gpuArray(h_tau);
    end
    
    % Compute Raman response
    h = (1-fb)*(tau1^2 + tau2^2)/(tau1*tau2^2)*exp(-h_tau/tau2).*sin(h_tau/tau1)...
            +fb*((2*tau_b-h_tau)/tau_b^2).*exp(-h_tau/tau_b);
    h(1:nt/2) = 0;  % causality
    
    % PRE-COMPUTE FFT of Raman response for faster convolution
    F_h = fft(h);
    
    % Dispersive phase operator (pre-compute)
    dispersion = exp(1i*distance/step_num*(0.5*sbeta2*(O1.^2+O2.^2) + delta3*(O1.^3+O2.^3) + delta4*(O1.^4+O2.^4)));
    
    % Pre-compute constant term for nonlinear step
    hhz = 1i*N^2*distance/step_num;
    
    % OPTIMIZATION: Pre-allocate arrays for speed
    U = U0;
    P = zeros(size(U), 'like', U);
    convl = zeros(size(U), 'like', U);
    sst1 = zeros(size(U), 'like', U);
    sst2 = zeros(size(U), 'like', U);
    sst3 = zeros(size(U), 'like', U);
    
    % OPTIMIZATION: Pre-compute FFT plans if on GPU
    if useGPU
        fftw('planner', 'patient');  % Spend time finding optimal FFT algorithm
    end
    
    % OPTIMIZATION: Use batch FFT when possible
    U_fft = fft2(U);
    
    % OPTIMIZATION: Check for blowup less frequently to save time
    check_interval = max(1, floor(step_num/10));
    
    % Main split-step loop
    for n = 1:step_num
        % Linear step (Fourier domain)
        U_fft = U_fft .* dispersion;  % Apply dispersion in frequency domain
        U = ifft2(U_fft);  % Back to spatial domain
        
        % Calculate intensity
        P = abs(U).^2;
        
        % MAJOR OPTIMIZATION: Vectorized Raman convolution using FFT
        % This replaces the slow loop in original code
        F_P = fft2(P);  % FFT of intensity
        convl = ifft2(F_P .* F_h);  % Convolution via FFT
        
        % Calculate derivatives for self-steepening
        sst1 = fast_deriv(P, dtau);  % Optimized derivative
        sst2 = fast_deriv(U, dtau);  
        
        % Calculate nonlinear terms
        sst = 1i*(1-fR)*s*(sst1 + conj(U).*sst2);
        sst3 = fast_deriv(convl, dtau);
        sstnew = 1i*s*fR*(sst3 + (conj(U).*convl)./(P+eps).*sst2);
        
        % Apply nonlinear phase shift
        U = U .* exp(((1-fR)*P + sst + fR*convl + sstnew)*hhz);
        
        % Update FFT for next iteration
        U_fft = fft2(U);
        
        % Check for blowup only periodically
        if mod(n, check_interval) == 0
            
            maxU = max(abs(U(:)));  % Keep on GPU, no gather
            if maxU > tol || any(isnan(U(:)))
                warning('Blowup detected at step %d, stopping propagation.', n);
                break;
            end

        end
    end
    
    % Ensure output is in same format as input
    if ~isInputGPU && useGPU
        U_out = gather(U);
    else
        U_out = U;
    end
    
    z = linspace(0, distance, params.zstep+1);
end

function d = fast_deriv(u, dt)
    % Faster derivative calculation using vectorized operations
    % Uses central difference with periodic boundary conditions
    
    % Shift matrices for central difference
    u_p = circshift(u, [-1, 0]);  % shift up
    u_m = circshift(u, [1, 0]);   % shift down
    
    % Calculate derivative in one vectorized operation
    d = (u_p - u_m)/(2*dt);
end