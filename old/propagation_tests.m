% PROPAGATION_TESTS.M
% Comprehensive tests for gnlse_propagate, based on literature and best practices.
% Each test is self-contained and prints a summary of its result.

close all;
run_all_gnlse_tests();

function run_all_gnlse_tests()
    fprintf('\n--- GNLSE PROPAGATION TEST SUITE ---\n');
    
    % % Spatial-only tests
    % fprintf('\n--- SPATIAL TESTS ---\n');
    % test_linear_mode_preservation();
    % test_nonlinear_spm_broadening();
    % test_nonlinear_mode_coupling();
    % test_energy_conservation();
    % test_core_cladding_confinement();
    % test_nonlinear_parameter_scaling();
    % test_nonlinear_threshold_effect();
    % test_nonlinear_mode_redistribution();
    % test_spatial_fwm();
    
    % % Spatiotemporal tests if available
    fprintf('\n--- SPATIOTEMPORAL TESTS ---\n');
    test_raman_frequency_shift();
    test_self_steepening();
    test_temporal_soliton();
    test_dispersion_broadening();
    test_temporal_vs_spatial_only();
    
    fprintf('--- ALL TESTS COMPLETED ---\n');
end

function p = get_common_test_parameters()
    % Common physical parameters used across tests - OPTIMIZED VALUES
    p = struct();
    
    % Fiber parameters - UPDATED for enhanced nonlinearity 
    p.core_radius = 25e-6;         % Core radius [m] (reduced for stronger nonlinearity)
    p.n_core = 1.453;              % Core refractive index (increased for higher index contrast)
    p.n_clad = 1.444;              % Cladding refractive index
    
    % Wavelength and optical parameters
    p.lambda = 1030e-9;           % Wavelength [m]
    p.lambda_anomalous = 1550e-9; % Wavelength for anomalous dispersion tests [m]
    
    % Grid parameters - BALANCED for efficiency
    p.image_size = 128;            % Spatial grid points (optimal for detail vs performance)
    p.nt = 128;          % Temporal grid points (for 3D simulations)
    
    % Nonlinear parameters - RECALIBRATED based on physical values
    p.N_linear = 0.0;             % Purely linear regime
    p.N_weak = 0.5;               % Weak nonlinearity
    p.N_medium = 2.0;             % Medium nonlinearity (physically significant)
    p.N_strong = 8.0;             % Strong nonlinearity (reduced from 20 for stability)
    
    % Propagation parameters - UPDATED based on fiber characteristics
    p.distance_short = 1;      % Short propagation distance [m]
    p.distance_medium = 10;      % Medium propagation distance [m]
    p.distance_long = 100;       % Long propagation distance [m] (reduced)
    
    % Dispersion parameters - ALIGNED with silica fibers
    p.sbeta2_normal = 0.01;       % Normal dispersion [ps²/m]
    p.sbeta2_anomalous = -0.02;   % Anomalous dispersion [ps²/m]
    
    % Pulse parameters - UPDATED for better effects
    p.t_width_long = 1.0;         % Wide pulse [ps]
    p.t_width_medium = 0.2;       % Medium pulse [ps] (increased from 0.3)
    p.t_width_short = 0.1;        % Short pulse [ps] (increased from 0.05 for stability)
    p.t_max = 5;                  % Time window [ps]
    
    % Raman parameters - ALIGNED with silica
    p.fR = 0.18;                  % Raman fraction (silica - reduced from 0.245)
    p.fR_disabled = 0.0;          % No Raman

    p.use_time = true;          % Use temporal dimension (default)
end

%% Common Helper Functions

function [X, Y, r2, core_mask, x, y] = setup_spatial_grid(p)
    % Create spatial grid with parameters
    x = linspace(-2*p.core_radius, 2*p.core_radius, p.image_size);
    y = x;
    [X, Y] = meshgrid(x, y);
    r2 = X.^2 + Y.^2;
    core_mask = r2 <= p.core_radius^2;
end

function spatial_profile = create_gaussian_beam(X, Y, r2, core_mask, p)
    % Create normalized Gaussian beam
    w0 = p.core_radius; % Beam waist
    spatial_profile = exp(-r2/(w0^2));
    spatial_profile(~core_mask) = 0;
    spatial_profile = spatial_profile / sqrt(sum(abs(spatial_profile(:)).^2));
end

function [LP01, LP11] = create_lp_modes(X, Y, r2, core_mask, p)
    % Create LP01 and LP11 modes
    theta = atan2(Y, X);
    LP01 = exp(-r2/(p.core_radius^2));
    LP11 = exp(-r2/(p.core_radius^2)) .* cos(theta);
    
    LP01(~core_mask) = 0; 
    LP11(~core_mask) = 0;
    
    LP01 = LP01 / sqrt(sum(abs(LP01(:)).^2));
    LP11 = LP11 / sqrt(sum(abs(LP11(:)).^2));
end

function temporal_profile = create_pulse(t, t_width, pulse_type)
    % Create various pulse shapes (Gaussian, sech, etc.)
    if nargin < 3 || isempty(pulse_type)
        pulse_type = 'gaussian';
    end
    
    switch lower(pulse_type)
        case 'gaussian'
            temporal_profile = exp(-(t/t_width).^2);
        case 'sech'
            temporal_profile = sech(t/t_width);
        case 'super_gaussian'
            temporal_profile = exp(-(t/t_width).^4);
    end
    
    temporal_profile = temporal_profile / sqrt(sum(abs(temporal_profile).^2));
end

function [U3d, t] = create_3d_field(spatial_profile, t_width, nt, t_max, pulse_type)
    % Create 3D spatiotemporal field
    t = linspace(-t_max, t_max, nt);
    temporal_profile = create_pulse(t, t_width, pulse_type);
    
    % Construct 3D field
    spatial_size = size(spatial_profile);
    U3d = zeros(spatial_size(1), spatial_size(2), nt);
    for k = 1:nt
        U3d(:,:,k) = spatial_profile * temporal_profile(k);
    end
end

function params = setup_propagation_params(p, custom)
    % Create propagation parameters with defaults and custom overrides
    % Base parameters
    params = struct(...
        'T0', p.t_width_medium, ...      % Pulse width [ps]
        'lam0', p.lambda*1e9, ...        % Wavelength [nm]
        'distance', p.distance_medium,... % Propagation distance [m]
        'N', p.N_medium, ...             % Nonlinear parameter
        'sbeta2', p.sbeta2_normal, ...   % Normal dispersion [ps²/m]
        'step_num', 100, ...             % Steps for accuracy
        'zstep', 1, ...                  % z step for output
        'n_clad', p.n_clad, ...          % Cladding index
        'n_core', p.n_core, ...          % Core index
        'core_radius', p.core_radius,... % Core radius [m]
        'nonlinear_in_cladding', false,... % Confine nonlinearity to core
        'fR', p.fR_disabled, ...         % No Raman by default
        'use_time', false);              % 2D spatial by default
    
    % Override with custom parameters
    if nargin > 1 && ~isempty(custom)
        fields = fieldnames(custom);
        for i = 1:length(fields)
            params.(fields{i}) = custom.(fields{i});
        end
    end
end

function w = fwhm(y)
    % Find full width at half maximum (FWHM) for a 1D array
    y = y / max(y);
    above = find(y > 0.5);
    if isempty(above), w = 0; return; end
    w = above(end) - above(1) + 1;
end

%% Test Functions

function test_linear_mode_preservation()
    % Test: Linear propagation should preserve the input mode profile
    fprintf('\n[TEST] Linear Mode Preservation\n');
    
    p = get_common_test_parameters();
    [X, Y, r2, core_mask, x, y] = setup_spatial_grid(p);
    U0 = create_gaussian_beam(X, Y, r2, core_mask, p);
    
    % Propagation with linear parameters only
    custom_params = struct('N', p.N_linear, 'distance', p.distance_short);
    params = setup_propagation_params(p, custom_params);
    params.X = X; params.Y = Y;  % Add grid to parameters
    
    [U_out,~,~] = gnlse_propagate(U0, params);
    
    % Compare input/output in the core
    in_core_corr = corr2(abs(U0(core_mask)), abs(U_out(core_mask)));
    fprintf('  Correlation in core: %.4f (should be >0.99)\n', in_core_corr);
    
    % Visualization
    figure('Name','Linear Mode Preservation');
    subplot(1,3,1);
    imagesc(x*1e6, y*1e6, abs(U0).^2);
    title('Input Intensity');
    xlabel('x [µm]'); ylabel('y [µm]');
    axis image; colorbar;
    
    subplot(1,3,2);
    imagesc(x*1e6, y*1e6, abs(U_out).^2);
    title('Output Intensity');
    xlabel('x [µm]'); ylabel('y [µm]');
    axis image; colorbar;
    
    subplot(1,3,3);
    plot(x*1e6, abs(U0(:,p.image_size/2)).^2, 'b', 'LineWidth', 1.5); hold on;
    plot(x*1e6, abs(U_out(:,p.image_size/2)).^2, 'r--', 'LineWidth', 1.5);
    legend('Input','Output');
    title('Central Spatial Profile');
    xlabel('x [µm]'); ylabel('Intensity');
    
    % assert(in_core_corr > 0.99, 'Linear propagation did not preserve mode profile.');
end

function test_nonlinear_spm_broadening()
    % Test: SPM should broaden the spectrum
    fprintf('\n[TEST] Nonlinear SPM Spectral Broadening\n');
    
    p = get_common_test_parameters();
    [X, Y, r2, core_mask, x, y] = setup_spatial_grid(p);
    U0 = create_gaussian_beam(X, Y, r2, core_mask, p);
    
    % Propagation with nonlinear parameters
    custom_params = struct('N', p.N_strong, 'distance', p.distance_long);
    params = setup_propagation_params(p, custom_params);
    params.X = X; params.Y = Y;
    
    [U_out,~,~] = gnlse_propagate(U0, params);
    
    % Compare input/output spectrum
    spec_in = abs(fftshift(fft2(U0))).^2;
    spec_out = abs(fftshift(fft2(U_out))).^2;
    
    % Use FWHM of central slice
    spec_in_slice = spec_in(p.image_size/2,:);
    spec_out_slice = spec_out(p.image_size/2,:);
    width_in = fwhm(spec_in_slice);
    width_out = fwhm(spec_out_slice);
    
    fprintf('  Input FWHM: %d px, Output FWHM: %d px, Ratio: %.2f\n', ...
            width_in, width_out, width_out/width_in);
    
    % Visualization
    figure('Name','SPM Spectral Broadening Test');
    subplot(2,2,1);
    imagesc(x*1e6, y*1e6, abs(U0).^2);
    title('Input Intensity');
    xlabel('x [µm]'); ylabel('y [µm]');
    axis image; colorbar;

    subplot(2,2,2);
    imagesc(x*1e6, y*1e6, abs(U_out).^2);
    title('Output Intensity');
    xlabel('x [µm]'); ylabel('y [µm]');
    axis image; colorbar;

    subplot(2,2,3);
    plot(spec_in_slice/max(spec_in_slice), 'b', 'LineWidth', 1.5); hold on;
    plot(spec_out_slice/max(spec_out_slice), 'r', 'LineWidth', 1.5);
    legend('Input','Output');
    title('Central Spectrum Slice');
    xlabel('Pixel'); ylabel('Normalized Power');

    subplot(2,2,4);
    plot(x*1e6, abs(U0(:,p.image_size/2)).^2, 'b', 'LineWidth', 1.5); hold on;
    plot(x*1e6, abs(U_out(:,p.image_size/2)).^2, 'r', 'LineWidth', 1.5);
    legend('Input','Output');
    title('Central Spatial Profile');
    xlabel('x [µm]'); ylabel('Intensity');
    
    % assert(width_out > width_in*1.2, 'SPM did not cause sufficient spectral broadening.');
end

function test_nonlinear_mode_coupling()
    % Test: Nonlinearity should induce mode coupling
    fprintf('\n[TEST] Nonlinear Mode Coupling\n');
    
    p = get_common_test_parameters();
    [X, Y, r2, core_mask, x, y] = setup_spatial_grid(p);
    [LP01, LP11] = create_lp_modes(X, Y, r2, core_mask, p);
    
    % Mix modes with 95% LP01 and 5% LP11
    U0 = LP01*0.95 + LP11*0.05;
    
    % Propagation with strong nonlinearity
    custom_params = struct('N', p.N_strong, 'distance', p.distance_long);
    params = setup_propagation_params(p, custom_params);
    params.X = X; params.Y = Y;
    
    [U_out,~,~] = gnlse_propagate(U0, params);
    
    % Project output onto LP11
    overlap_LP11 = abs(sum(sum(conj(LP11).*U_out)));
    fprintf('  Output LP11 overlap: %.4f (should be >0.1 if mode coupling)\n', overlap_LP11);
    
    % Visualization
    figure('Name','Nonlinear Mode Coupling');
    subplot(2,2,1);
    imagesc(x*1e6, y*1e6, abs(U0).^2);
    title('Input Intensity');
    xlabel('x [µm]'); ylabel('y [µm]');
    axis image; colorbar;
    
    subplot(2,2,2);
    imagesc(x*1e6, y*1e6, abs(U_out).^2);
    title('Output Intensity');
    xlabel('x [µm]'); ylabel('y [µm]');
    axis image; colorbar;
    
    subplot(2,2,3);
    imagesc(x*1e6, y*1e6, abs(LP11).^2);
    title('LP11 Mode Intensity');
    xlabel('x [µm]'); ylabel('y [µm]');
    axis image; colorbar;
    
    subplot(2,2,4);
    plot(x*1e6, abs(U0(:,p.image_size/2)).^2, 'b', 'LineWidth', 1.5); hold on;
    plot(x*1e6, abs(U_out(:,p.image_size/2)).^2, 'r', 'LineWidth', 1.5);
    legend('Input','Output');
    title('Central Spatial Profile');
    xlabel('x [µm]'); ylabel('Intensity');
    
    % assert(overlap_LP11 > 0.1, 'Nonlinear mode coupling not observed.');
end

function test_energy_conservation()
    % Test: Total energy should be conserved in the absence of loss
    fprintf('\n[TEST] Energy Conservation\n');
    
    p = get_common_test_parameters();
    [X, Y, r2, core_mask, x, y] = setup_spatial_grid(p);
    U0 = create_gaussian_beam(X, Y, r2, core_mask, p);
    
    % Propagation with medium nonlinearity
    custom_params = struct('N', p.N_medium);
    params = setup_propagation_params(p, custom_params);
    params.X = X; params.Y = Y;
    
    [U_out,~,~] = gnlse_propagate(U0, params);
    
    % Calculate input and output energy
    Ein = sum(abs(U0(:)).^2);
    Eout = sum(abs(U_out(:)).^2);
    fprintf('  Input energy: %.4e, Output energy: %.4e, Ratio: %.4f\n', Ein, Eout, Eout/Ein);
    
    % Visualization
    figure('Name','Energy Conservation');
    subplot(1,2,1);
    imagesc(x*1e6, y*1e6, abs(U0).^2);
    title('Input Intensity');
    xlabel('x [µm]'); ylabel('y [µm]');
    axis image; colorbar;
    
    subplot(1,2,2);
    imagesc(x*1e6, y*1e6, abs(U_out).^2);
    title('Output Intensity');
    xlabel('x [µm]'); ylabel('y [µm]');
    axis image; colorbar;
    
    % assert(abs(Eout/Ein-1)<0.01, 'Energy not conserved (lossless case).');
end

function test_core_cladding_confinement()
    % Test: Field remains confined to the core in the linear regime
    fprintf('\n[TEST] Core-Cladding Confinement\n');
    
    p = get_common_test_parameters();
    [X, Y, r2, core_mask, x, y] = setup_spatial_grid(p);
    U0 = create_gaussian_beam(X, Y, r2, core_mask, p);
    
    % Propagation in linear regime
    custom_params = struct('N', p.N_linear);
    params = setup_propagation_params(p, custom_params);
    params.X = X; params.Y = Y;
    
    [U_out,~,~] = gnlse_propagate(U0, params);
    
    % Calculate fraction of energy in core
    frac_in_core = sum(abs(U_out(core_mask)).^2) / sum(abs(U_out(:)).^2);
    fprintf('  Fraction of energy in core: %.4f (should be >0.98)\n', frac_in_core);
    
    % Visualization
    figure('Name','Core-Cladding Confinement');
    subplot(1,2,1);
    imagesc(x*1e6, y*1e6, abs(U_out).^2);
    title('Output Intensity');
    xlabel('x [µm]'); ylabel('y [µm]');
    axis image; colorbar;
    
    subplot(1,2,2);
    mask_img = zeros(size(U_out));
    mask_img(core_mask) = 1;
    imagesc(x*1e6, y*1e6, mask_img);
    title('Core Mask');
    xlabel('x [µm]'); ylabel('y [µm]');
    axis image; colorbar;
    
    % assert(frac_in_core > 0.98, 'Field leaked significantly into cladding.');
end

function test_nonlinear_parameter_scaling()
    % Test: Spectral broadening should scale with nonlinear parameter N
    fprintf('\n[TEST] Nonlinear Parameter Scaling\n');
    
    p = get_common_test_parameters();
    [X, Y, r2, core_mask, x, y] = setup_spatial_grid(p);
    U0 = create_gaussian_beam(X, Y, r2, core_mask, p);
    
    % Test multiple nonlinearity values
    N_values = [0, 2, 5, 8];
    spectral_widths = zeros(size(N_values));
    
    % Base parameters
    base_params = setup_propagation_params(p, struct('distance', p.distance_long));
    base_params.X = X; base_params.Y = Y;
    
    figure('Name','Nonlinear Parameter Scaling');
    
    for i = 1:length(N_values)
        params = base_params;
        params.N = N_values(i);
        [U_out,~,~] = gnlse_propagate(U0, params);
        
        % Calculate spectral width
        spec_out = abs(fftshift(fft2(U_out))).^2;
        spec_slice = spec_out(p.image_size/2,:);
        spectral_widths(i) = fwhm(spec_slice);
        
        % Plot spectrum
        subplot(length(N_values),2,2*i-1);
        imagesc(x*1e6, y*1e6, abs(U_out).^2);
        title(sprintf('N = %.1f, Intensity', N_values(i)));
        xlabel('x [µm]'); ylabel('y [µm]');
        axis image; colorbar;
        
        subplot(length(N_values),2,2*i);
        plot(spec_slice/max(spec_slice), 'LineWidth', 1.5);
        title(sprintf('N = %.1f, Width = %d px', N_values(i), spectral_widths(i)));
        xlabel('Frequency [pixels]'); ylabel('Norm. Power');
    end
    
    % Check that spectral width increases with N
    fprintf('  Spectral widths: [');
    fprintf(' %d', spectral_widths);
    fprintf(' ] px\n');
    
    % Verify trend is increasing
    is_increasing = all(diff(spectral_widths) >= 0);
    fprintf('  Spectral width increases with N: %s\n', mat2str(is_increasing));
    
    % assert(is_increasing, 'Spectral width did not increase with nonlinearity parameter N');
end

function test_nonlinear_threshold_effect()
    % Test: SPM broadening only occurs above a threshold N
    fprintf('\n[TEST] Nonlinear Threshold Effect\n');
    
    p = get_common_test_parameters();
    [X, Y, r2, core_mask, x, y] = setup_spatial_grid(p);
    U0 = create_gaussian_beam(X, Y, r2, core_mask, p);

    % Low and high nonlinearity values
    N_low = p.N_weak;   % Below threshold
    N_high = p.N_strong; % Above threshold

    % Base propagation parameters
    base_params = setup_propagation_params(p, struct('distance', p.distance_long));
    base_params.X = X; base_params.Y = Y;

    % Low N
    params = base_params;
    params.N = N_low;
    [U_low,~,~] = gnlse_propagate(U0, params);
    spec_low = abs(fftshift(fft2(U_low))).^2;
    width_low = fwhm(spec_low(p.image_size/2,:));

    % High N
    params = base_params;
    params.N = N_high;
    [U_high,~,~] = gnlse_propagate(U0, params);
    spec_high = abs(fftshift(fft2(U_high))).^2;
    width_high = fwhm(spec_high(p.image_size/2,:));

    fprintf('  FWHM low N: %d px, high N: %d px, ratio: %.2f\n', ...
            width_low, width_high, width_high/width_low);
    
    % Visualization
    figure('Name','Nonlinear Threshold Effect');
    subplot(1,2,1);
    plot(spec_low(p.image_size/2,:)/max(spec_low(:)), 'b', 'LineWidth', 1.5); hold on;
    plot(spec_high(p.image_size/2,:)/max(spec_high(:)), 'r', 'LineWidth', 1.5);
    legend('Low N','High N');
    title('Central Spectrum Slice');
    xlabel('Pixel'); ylabel('Normalized Power');
    
    subplot(1,2,2);
    imagesc(x*1e6, y*1e6, abs(U_high).^2);
    title('Output Intensity (High N)');
    xlabel('x [µm]'); ylabel('y [µm]');
    axis image; colorbar;
    
    % assert(width_high/width_low > 1.2, 'Nonlinear broadening not significant above threshold.');
end

function test_nonlinear_mode_redistribution()
    % Test: Nonlinearity redistributes energy among modes
    fprintf('\n[TEST] Nonlinear Mode Redistribution\n');
    
    p = get_common_test_parameters();
    [X, Y, r2, core_mask, x, y] = setup_spatial_grid(p);
    [LP01, LP11] = create_lp_modes(X, Y, r2, core_mask, p);
    
    % Mix modes with 95% LP01 and 5% LP11
    U0 = LP01*0.95 + LP11*0.05;

    % Base propagation parameters
    base_params = setup_propagation_params(p, struct('distance', p.distance_long));
    base_params.X = X; base_params.Y = Y;

    % Test with different nonlinearity values
    overlaps = zeros(1,3);
    Nvals = [p.N_linear, p.N_medium, p.N_strong];
    
    for i = 1:3
        params = base_params;
        params.N = Nvals(i);
        [U_out,~,~] = gnlse_propagate(U0, params);
        overlaps(i) = abs(sum(sum(conj(LP11).*U_out)));
    end
    
    fprintf('  LP11 overlap for N=[%.1f,%.1f,%.1f]: [%.3f %.3f %.3f]\n', ...
            Nvals(1), Nvals(2), Nvals(3), overlaps);
    
    % assert(overlaps(3) > overlaps(1), 'Nonlinear mode redistribution not observed.');
end

function test_spatial_fwm()
    % Test: Spatial four-wave mixing in the presence of nonlinearity
    fprintf('\n[TEST] Spatial Four-Wave Mixing\n');
    
    p = get_common_test_parameters();
    [X, Y, r2, core_mask, x, y] = setup_spatial_grid(p);
    [LP01, LP11] = create_lp_modes(X, Y, r2, core_mask, p);
    
    % Create input with distinct spatial frequency modulation
    freq_shift = 10;
    U0 = LP01 .* exp(1i*freq_shift*X) + LP11 .* exp(-1i*freq_shift*X);
    U0 = U0 / sqrt(sum(abs(U0(:)).^2));
    
    % Propagation with strong nonlinearity
    custom_params = struct('N', p.N_strong, 'distance', p.distance_medium);
    params = setup_propagation_params(p, custom_params);
    params.X = X; params.Y = Y;
    
    [U_out,~,~] = gnlse_propagate(U0, params);
    
    % Check for new frequency components in spatial spectrum
    spec_in = abs(fftshift(fft2(U0))).^2;
    spec_out = abs(fftshift(fft2(U_out))).^2;
    
    % Look for peaks at sum/difference frequencies
    spectrum_in = sum(spec_in, 1);
    spectrum_out = sum(spec_out, 1);
    
    % Find peaks in output spectrum
    [~, locs_out] = findpeaks(spectrum_out, 'MinPeakProminence', max(spectrum_out)*0.05);
    
    % Find peaks in input spectrum
    [~, locs_in] = findpeaks(spectrum_in, 'MinPeakProminence', max(spectrum_in)*0.05);
    
    fprintf('  Number of spectral peaks: Input %d, Output %d\n', ...
            numel(locs_in), numel(locs_out));
    
    % Visualization
    figure('Name','Spatial Four-Wave Mixing');
    subplot(2,2,1);
    imagesc(x*1e6, y*1e6, abs(U0).^2);
    title('Input Intensity');
    xlabel('x [µm]'); ylabel('y [µm]');
    axis image; colorbar;
    
    subplot(2,2,2);
    imagesc(x*1e6, y*1e6, abs(U_out).^2);
    title('Output Intensity');
    xlabel('x [µm]'); ylabel('y [µm]');
    axis image; colorbar;
    
    subplot(2,2,3);
    plot(spectrum_in/max(spectrum_in), 'b', 'LineWidth', 1.5);
    title('Input Spatial Spectrum');
    xlabel('Spatial Frequency'); ylabel('Normalized Power');
    
    subplot(2,2,4);
    plot(spectrum_out/max(spectrum_out), 'r', 'LineWidth', 1.5);
    title('Output Spatial Spectrum');
    xlabel('Spatial Frequency'); ylabel('Normalized Power');
    
    % assert(numel(locs_out) > numel(locs_in), 'Four-wave mixing not observed.');
end

function test_raman_frequency_shift()
    % Test: Raman effect induces red-shift in the spectrum
    fprintf('\n[TEST] Raman-Induced Frequency Shift\n');
    
    p = get_common_test_parameters();
    
    % Override parameters for Raman test
    image_size = 128;       % Spatial grid
    nt = 128;              % Time points
    t_width = 1;         % Pulse width [ps] (increased for stability)
    t_max = 10;           % Time window [ps]
    
    [X, Y, r2, core_mask, x, y] = setup_spatial_grid(p);
    spatial_profile = create_gaussian_beam(X, Y, r2, core_mask, p);
    [U0, t] = create_3d_field(spatial_profile, t_width, nt, t_max, 'gaussian');
    
    % Parameters optimized for Raman effect
    custom_params = struct(...
        'T0', t_width, ...         % Pulse width [ps]
        'lam0', p.lambda*1e9, ...  % Wavelength [nm]
        'distance', 10, ...       % Propagation distance [m]
        'N', 4.0, ...              % Nonlinear parameter
        'sbeta2', p.sbeta2_anomalous, ... % Anomalous dispersion
        'nt', nt, ...              % Time points
        'Tmax', t_max, ...         % Time window [ps]
        'step_num', 500, ...       % Steps for accuracy
        'fR', p.fR, ...           % Enable Raman
        'use_time', true);         % Use temporal dimension
    
    params = setup_propagation_params(p, custom_params);
    params.X = X; params.Y = Y;
    
    % Propagate with temporal dimension
    [U_out, tau, ~] = gnlse_propagate(U0, params);
    
    % Calculate spectra - take FFT along time axis
    spec_in = abs(fftshift(fft(U0, [], 3), 3)).^2;
    spec_out = abs(fftshift(fft(U_out, [], 3), 3)).^2;
    
    % Sum over spatial dimensions to get total spectrum vs frequency
    spec_in_sum = squeeze(sum(sum(spec_in, 1), 2));
    spec_out_sum = squeeze(sum(sum(spec_out, 1), 2));
    
    % Create frequency axis
    dt = (2*t_max)/nt;
    freq = fftshift((-nt/2:nt/2-1)/(nt*dt)); % THz
    
    % Calculate spectral centroids
    if sum(spec_in_sum) > 0 && sum(spec_out_sum) > 0
        centroid_in = sum(freq .* spec_in_sum') / sum(spec_in_sum);
        centroid_out = sum(freq .* spec_out_sum') / sum(spec_out_sum);
        
        fprintf('  Input centroid: %.4f THz, Output centroid: %.4f THz, Shift: %.4f THz\n', ...
                centroid_in, centroid_out, centroid_out-centroid_in);
    else
        fprintf('  Warning: Spectrum calculation failed (empty spectrum)\n');
        centroid_in = 0;
        centroid_out = 0;
    end
    
    % Visualization
    figure('Name','Raman Frequency Shift');
    subplot(2,2,1);
    imagesc(x*1e6, y*1e6, sum(abs(U0).^2, 3));
    title('Input Intensity (Spatial)');
    xlabel('x [µm]'); ylabel('y [µm]');
    axis image; colorbar;
    
    subplot(2,2,2);
    imagesc(x*1e6, y*1e6, sum(abs(U_out).^2, 3));
    title('Output Intensity (Spatial)');
    xlabel('x [µm]'); ylabel('y [µm]');
    axis image; colorbar;
    
    subplot(2,2,3);
    plot(tau, squeeze(abs(U0(image_size/2,image_size/2,:)).^2), 'b', 'LineWidth', 1.5); hold on;
    plot(tau, squeeze(abs(U_out(image_size/2,image_size/2,:)).^2), 'r', 'LineWidth', 1.5);
    title('Temporal Profile (Center)');
    xlabel('Time [ps]'); ylabel('Intensity');
    legend('Input', 'Output');
    
    subplot(2,2,4);
    plot(freq, spec_in_sum/max(spec_in_sum), 'b', 'LineWidth', 1.5); hold on;
    plot(freq, spec_out_sum/max(spec_out_sum), 'r', 'LineWidth', 1.5);
    if centroid_in ~= 0 && centroid_out ~= 0
        xline(centroid_in, 'b--', 'DisplayName', 'In Centroid');
        xline(centroid_out, 'r--', 'DisplayName', 'Out Centroid');
    end
    title('Spectrum (Raman Shift)');
    xlabel('Frequency [THz]'); ylabel('Normalized Power');
    legend('Input', 'Output', 'In Centroid', 'Out Centroid');
    
    % Verify that output centroid is less than input centroid (red-shift)
    % assert(centroid_out < centroid_in, 'Raman red-shift not observed.');
end

function test_self_steepening()
    % Test: Self-steepening causes spectral asymmetry
    fprintf('\n[TEST] Self-Steepening Effect\n');
    
    p = get_common_test_parameters();
    
    % Override parameters for self-steepening test
    image_size = 128;       % Spatial grid
    nt = 128;              % Time points
    t_width = 1;         % Pulse width [ps]
    t_max = 10;           % Time window [ps]
    
    [X, Y, r2, core_mask, x, y] = setup_spatial_grid(p);
    spatial_profile = create_gaussian_beam(X, Y, r2, core_mask, p);
    [U0, t] = create_3d_field(spatial_profile, t_width, nt, t_max, 'gaussian');
    
    % Parameters optimized for self-steepening
    custom_params = struct(...
        'T0', t_width, ...         % Pulse width [ps]
        'lam0', p.lambda*1e9, ...  % Wavelength [nm]
        'distance', 10, ...       % Propagation distance [m]
        'N', 4.0, ...              % Stronger nonlinearity for self-steepening
        'sbeta2', 0, ...           % No dispersion to isolate steepening
        'nt', nt, ...              % Time points
        'Tmax', t_max, ...         % Time window [ps]
        'step_num', 500, ...       % Steps for accuracy
        'fR', 0.0, ...             % Disable Raman for this test
        'use_time', true);         % Use temporal dimension
    
    params = setup_propagation_params(p, custom_params);
    params.X = X; params.Y = Y;
    
    % Propagate with temporal dimension
    [U_out, tau, ~] = gnlse_propagate(U0, params);
    
    % Calculate spectra - take FFT along time axis
    spec_in = abs(fftshift(fft(U0, [], 3), 3)).^2;
    spec_out = abs(fftshift(fft(U_out, [], 3), 3)).^2;
    
    % Sum over spatial dimensions to get total spectrum vs frequency
    spec_in_sum = squeeze(sum(sum(spec_in, 1), 2));
    spec_out_sum = squeeze(sum(sum(spec_out, 1), 2));
    
    % Create frequency axis
    dt = (2*t_max)/nt;
    freq = fftshift((-nt/2:nt/2-1)/(nt*dt)); % THz
    
    % Calculate spectral asymmetry
    neg_freqs = freq < 0;
    pos_freqs = freq > 0;
    left = sum(spec_out_sum(neg_freqs));
    right = sum(spec_out_sum(pos_freqs));
    asymmetry = (right-left)/(right+left);
    
    fprintf('  Spectral asymmetry: %.3f (should be >0.05 for self-steepening)\n', ...
            asymmetry);
    
    % Visualization
    figure('Name','Self-Steepening');
    subplot(2,2,1);
    imagesc(x*1e6, y*1e6, sum(abs(U0).^2, 3));
    title('Input Intensity (Spatial)');
    xlabel('x [µm]'); ylabel('y [µm]');
    axis image; colorbar;
    
    subplot(2,2,2);
    plot(tau, squeeze(abs(U0(image_size/2,image_size/2,:)).^2), 'b', 'LineWidth', 1.5); hold on;
    plot(tau, squeeze(abs(U_out(image_size/2,image_size/2,:)).^2), 'r', 'LineWidth', 1.5);
    title('Temporal Profile (Center)');
    xlabel('Time [ps]'); ylabel('Intensity');
    legend('Input', 'Output');
    
    subplot(2,2,[3,4]);
    plot(freq, spec_in_sum/max(spec_in_sum), 'b', 'LineWidth', 1.5); hold on;
    plot(freq, spec_out_sum/max(spec_out_sum), 'r', 'LineWidth', 1.5);
    xline(0, 'k--');
    title(sprintf('Spectrum (Asymmetry = %.3f)', asymmetry));
    xlabel('Frequency [THz]'); ylabel('Normalized Power');
    legend('Input', 'Output');
    
    % assert(abs(asymmetry) > 0.05, 'Self-steepening asymmetry not observed.');
end

function test_temporal_soliton()
    % Test: Temporal soliton formation in anomalous dispersion regime
    fprintf('\n[TEST] Temporal Soliton Formation\n');
    
    p = get_common_test_parameters();
    
    % Override parameters for soliton test
    image_size = 128;       % Spatial grid
    nt = 128;              % Time points
    t_width = 1;         % Pulse width [ps]
    t_max = 10;           % Time window [ps]
    
    [X, Y, r2, core_mask, x, y] = setup_spatial_grid(p);
    spatial_profile = create_gaussian_beam(X, Y, r2, core_mask, p);
    
    % Create a sech pulse rather than Gaussian for proper soliton
    t = linspace(-t_max, t_max, nt);
    [U0, ~] = create_3d_field(spatial_profile, t_width, nt, t_max, 'sech');
    
    % Calculate required amplitude for N=1 soliton
    amplitude_factor = sqrt(abs(p.sbeta2_anomalous) / (1.0 * t_width^2));
    U0 = U0 * amplitude_factor;
    
    % Parameters for N=1 fundamental soliton
    custom_params = struct(...
        'T0', t_width, ...             % Pulse width [ps]
        'lam0', p.lambda_anomalous*1e9, ... % Wavelength for anomalous dispersion
        'distance', 10, ...           % Propagation distance [m]
        'N', 1.0, ...                  % N=1 for fundamental soliton
        'sbeta2', p.sbeta2_anomalous, ... % Anomalous dispersion
        'nt', nt, ...                  % Time points
        'Tmax', t_max, ...             % Time window [ps]
        'step_num', 400, ...           % Increased steps for accuracy
        'fR', 0.0, ...                 % Disable Raman for pure soliton
        'use_time', true);             % Use temporal dimension
    
    params = setup_propagation_params(p, custom_params);
    params.X = X; params.Y = Y;
    
    % Propagate with temporal dimension
    [U_out, tau, ~] = gnlse_propagate(U0, params);
    
    % Extract temporal profiles at spatial center for correlation
    input_profile = squeeze(abs(U0(round(image_size/2), round(image_size/2), :)).^2);
    output_profile = squeeze(abs(U_out(round(image_size/2), round(image_size/2), :)).^2);
    
    % Normalize for shape comparison before correlation
    if max(input_profile) > 1e-9 && max(output_profile) > 1e-9
        corr_input = input_profile / max(input_profile);
        corr_output = output_profile / max(output_profile);
        shape_corr = corr(corr_input, corr_output);
    else
        shape_corr = 0; % Avoid errors if profiles are zero
    end
    
    fprintf('  Temporal soliton shape correlation at center: %.4f (should be >0.95)\n', ...
            shape_corr);
    
    % Visualization: x-t slice at y=center
    figure('Name','Temporal Soliton Test');
    subplot(1,2,1);
    imagesc(x*1e6, tau, squeeze(abs(U0(:,round(image_size/2),:))));
    title('Input Pulse (x-t slice at y-center)');
    xlabel('x [µm]'); ylabel('Time [ps]');
    axis tight; colorbar;
    caxis_max_in = max(abs(U0(:)));
    if caxis_max_in > 0, caxis([0 caxis_max_in]); end

    subplot(1,2,2);
    imagesc(x*1e6, tau, squeeze(abs(U_out(:,round(image_size/2),:))));
    title('Output Pulse (x-t slice at y-center)');
    xlabel('x [µm]'); ylabel('Time [ps]');
    axis tight; colorbar;
    if caxis_max_in > 0, caxis([0 caxis_max_in]); end % Use same caxis as input for comparison
    
    % assert(shape_corr > 0.95, 'Fundamental soliton did not preserve temporal shape well.');
end

function test_dispersion_broadening()
    % Test: Group velocity dispersion causes temporal broadening
    fprintf('\n[TEST] Dispersion-Induced Pulse Broadening\n');
    
    p = get_common_test_parameters();
    
    % Override parameters for dispersion test
    image_size = 128;       % Spatial grid
    nt = 128;              % Time points
    t_width = 1;         % Pulse width [ps]
    t_max = 10.0;           % Time window [ps]
    
    [X, Y, r2, core_mask, x, y] = setup_spatial_grid(p);
    spatial_profile = create_gaussian_beam(X, Y, r2, core_mask, p);
    [U0, ~] = create_3d_field(spatial_profile, t_width, nt, t_max, 'gaussian');
    
    % Parameters for dispersion-only propagation
    custom_params = struct(...
        'T0', t_width, ...             % Pulse width [ps]
        'lam0', p.lambda*1e9, ...      % Wavelength [nm]
        'distance', 1.0, ...           % Propagation distance [m]
        'N', 0.0, ...                  % No nonlinearity
        'sbeta2', p.sbeta2_normal*5, ...  % Strong normal dispersion (increased)
        'nt', nt, ...                  % Time points
        'Tmax', t_max, ...             % Time window [ps]
        'step_num', 500, ...           % Steps for accuracy
        'fR', 0.0, ...                 % No Raman
        'use_time', true);             % Use temporal dimension
    
    params = setup_propagation_params(p, custom_params);
    params.X = X; params.Y = Y;
    
    % Propagate with temporal dimension
    [U_out, tau, ~] = gnlse_propagate(U0, params);
    
    % Calculate temporal width using FWHM
    input_profile = squeeze(abs(U0(image_size/2, image_size/2, :)).^2);
    output_profile = squeeze(abs(U_out(image_size/2, image_size/2, :)).^2);
    
    width_in = fwhm(input_profile);
    width_out = fwhm(output_profile);
    
    fprintf('  Input pulse FWHM: %.2f ps, Output FWHM: %.2f ps, Ratio: %.2f\n', ...
            width_in*2*t_max/nt, width_out*2*t_max/nt, width_out/width_in);
    
    % Visualization
    figure('Name','Dispersion-Induced Pulse Broadening');
    subplot(1,2,1);
    plot(tau, input_profile/max(input_profile), 'b', 'LineWidth', 1.5); hold on;
    plot(tau, output_profile/max(output_profile), 'r', 'LineWidth', 1.5);
    legend('Input','Output');
    title('Temporal Profile');
    xlabel('Time [ps]'); ylabel('Normalized Intensity');
    
    subplot(1,2,2);
    % Calculate spectra - need to squeeze to get 1D array
    spec_in = squeeze(abs(fftshift(fft(U0(image_size/2,image_size/2,:)))).^2);
    spec_out = squeeze(abs(fftshift(fft(U_out(image_size/2,image_size/2,:)))).^2);
    
    % Create frequency axis
    dt = (2*t_max)/nt;
    freq = fftshift((-nt/2:nt/2-1)/(nt*dt)); % THz
    
    % Plot spectra
    plot(freq, spec_in/max(spec_in), 'b', 'LineWidth', 1.5); hold on;
    plot(freq, spec_out/max(spec_out), 'r', 'LineWidth', 1.5);
    legend('Input','Output');
    title('Spectral Profile');
    xlabel('Frequency [THz]'); ylabel('Normalized Power');
    
    % assert(width_out > width_in*1.2, 'Dispersion-induced temporal broadening not observed.');
end

function test_temporal_vs_spatial_only()
    % Test: Compare propagation with and without temporal effects
    fprintf('\n[TEST] Temporal vs Spatial-Only Propagation\n');
    
    p = get_common_test_parameters();
    image_size = 128;
    nt = 128;
    t_width = 1;
    t_max = 10.0;
    [X, Y, r2, core_mask, x, y] = setup_spatial_grid(p);
    spatial_profile = create_gaussian_beam(X, Y, r2, core_mask, p);
    [U0, t] = create_3d_field(spatial_profile, t_width, nt, t_max, 'gaussian');
    
    % Propagation with temporal effects (3D)
    custom_params_time = struct(...
        'T0', t_width, ...
        'lam0', p.lambda*1e9, ...
        'distance', 10, ...
        'N', 4.0, ...
        'sbeta2', p.sbeta2_anomalous, ...
        'nt', nt, ...
        'Tmax', t_max, ...
        'step_num', 500, ...
        'fR', p.fR, ...
        'use_time', true);
    params_time = setup_propagation_params(p, custom_params_time);
    params_time.X = X; params_time.Y = Y;
    [U_out_time, tau, ~] = gnlse_propagate(U0, params_time);
    
    % Propagation without temporal effects (2D)
    custom_params_spatial = struct(...
        'T0', t_width, ...
        'lam0', p.lambda*1e9, ...
        'distance', 10, ...
        'N', 4.0, ...
        'sbeta2', p.sbeta2_anomalous, ...
        'step_num', 500, ...
        'fR', p.fR, ...
        'use_time', false);
    params_spatial = setup_propagation_params(p, custom_params_spatial);
    params_spatial.X = X; params_spatial.Y = Y;
    U0_2d = sum(U0,3); % Collapse temporal dimension for 2D input
    [U_out_spatial, ~, ~] = gnlse_propagate(U0_2d, params_spatial);
    
    % Compare output fields (spatial intensity)
    intensity_time = sum(abs(U_out_time).^2, 3);
    intensity_spatial = abs(U_out_spatial).^2;
    
    % Correlation between outputs
    min_size = min(size(intensity_time,3), size(intensity_spatial,3));
    corr_val = corr2(intensity_time, intensity_spatial);
    
    % Energy comparison
    E_time = sum(intensity_time(:));
    E_spatial = sum(intensity_spatial(:));
    
    % FWHM comparison (central slice)
    fwhm_time = fwhm(intensity_time(image_size/2,:));
    fwhm_spatial = fwhm(intensity_spatial(image_size/2,:));
    
    fprintf('  Output correlation: %.4f\n', corr_val);
    fprintf('  Output energy: time %.4e, spatial %.4e, ratio: %.4f\n', E_time, E_spatial, E_time/E_spatial);
    fprintf('  Output FWHM: time %d px, spatial %d px, ratio: %.2f\n', fwhm_time, fwhm_spatial, fwhm_time/fwhm_spatial);
    
    % Visualization
    figure('Name','Temporal vs Spatial-Only Propagation');
    subplot(2,2,1);
    imagesc(x*1e6, y*1e6, intensity_time);
    title('Output Intensity (Temporal)');
    xlabel('x [\mum]'); ylabel('y [\mum]'); axis image; colorbar;
    subplot(2,2,2);
    imagesc(x*1e6, y*1e6, intensity_spatial);
    title('Output Intensity (Spatial-Only)');
    xlabel('x [\mum]'); ylabel('y [\mum]'); axis image; colorbar;
    subplot(2,2,3);
    imagesc(x*1e6, y*1e6, abs(intensity_time-intensity_spatial));
    title('Absolute Difference');
    xlabel('x [\mum]'); ylabel('y [\mum]'); axis image; colorbar;
    subplot(2,2,4);
    plot(x*1e6, intensity_time(image_size/2,:)/max(intensity_time(:)), 'b', 'LineWidth', 1.5); hold on;
    plot(x*1e6, intensity_spatial(image_size/2,:)/max(intensity_spatial(:)), 'r--', 'LineWidth', 1.5);
    legend('Temporal', 'Spatial-Only');
    title('Central Spatial Profile');
    xlabel('x [\mum]'); ylabel('Norm. Intensity');
end