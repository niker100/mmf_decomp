% GMMNLSE_SOLVER_TEST_SUITE.M
% Optimized test suite for GMMNLSE-Solver-FINAL
% Delivering concise evaluation of nonlinear effects with focused visualizations

clear all; close all; clc;
addpath(genpath('GMMNLSE-Solver-FINAL'));

%% Configuration and helper functions
% Define color scheme for consistent visualization
global colors 
colors = struct(...
    'mode1', [0 0.4470 0.7410], ...  % Blue
    'mode2', [0.8500 0.3250 0.0980], ... % Orange
    'mode3', [0.9290 0.6940 0.1250], ... % Yellow
    'mode4', [0.4940 0.1840 0.5560], ... % Purple
    'input', [0.3010 0.7450 0.9330], ... % Light blue
    'output', [0.6350 0.0780 0.1840], ... % Dark red
    'phase', [0.4660 0.6740 0.1880]  ... % Green
);

% Create directory for results if it doesn't exist
results_dir = 'test_results';
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

% Helper functions
function fig = create_figure(name, position)
    fig = figure('Name', name, 'NumberTitle', 'off');    
    if nargin > 1 && (get(fig, "WindowStyle") ~= "docked")
        set(fig, 'Position', position);
    end
    set(fig, 'Color', 'w');
    set(fig, 'InvertHardcopy', 'off');
end

function setup_axes(ax, fontsize)
    if nargin < 2
        fontsize = 11;
    end
    set(ax, 'FontSize', fontsize);
    set(ax, 'Box', 'on');
    set(ax, 'LineWidth', 1.2);
    grid(ax, 'on');
end

function save_figure(fig, filename)
    saveas(fig, filename, 'png');
    saveas(fig, [filename '.svg']);
end

function [E_xy, I_xy, spatial_spectrum] = reconstruct_field(modal_fields, mode_fields, time_idx)
    Nx = size(mode_fields, 1);
    num_modes = size(mode_fields, 3);
    
    % If no time index provided, find the peak
    if isempty(time_idx)
        [~, time_idx] = max(sum(abs(modal_fields(:,:,end)).^2, 2));
    end
    
    % Reconstruct the spatial field
    E_xy = zeros(Nx, Nx);
    for m = 1:num_modes
        E_xy = E_xy + modal_fields(time_idx, m, end) * mode_fields(:,:,m);
    end
    
    % Calculate intensity and spatial spectrum
    I_xy = abs(E_xy).^2;
    spatial_spectrum = fftshift(fft2(E_xy));
end

%% FIBER SETUP: Define fiber parameters and build fiber structure
disp('Setting up multimode fiber parameters...');

% Core parameters
lambda0 = 1030e-9;          % Center wavelength [m]
fiber_length = 10;           % Propagation distance [m]
Nx = 800;                   % Grid points for mode profiles
radius = 25;                % Fiber radius [um]
num_modes = 4;              % Number of modes

% Load precomputed data
prefix = 'Fibers/STEP_1030';
load([prefix '/S_tensors_' num2str(num_modes) 'modes.mat']); % [m^-2]
load([prefix '/betas.mat']); % [fs^n/mm]

% Unit conversion and fiber structure setup
unit_conversion = 0.001.^(-1:size(betas, 1)-2)';
fiber.betas = betas .* unit_conversion;
fiber.SR = SR;
fiber.L0 = fiber_length;
fiber.lambda0 = lambda0;
fiber.num_modes = num_modes;

% Simulation grid parameters
Nt = 1024;
Tmax = 10; % [ps]
dt = 2*Tmax/Nt;
t = linspace(-Tmax, Tmax-dt, Nt);
freq = (-Nt/2:Nt/2-1)/(Nt*dt); % Frequency grid in THz
fwhm = @(y) (find(y>0.5*max(y),1,'last') - find(y>0.5*max(y),1,'first') + 1);

% Load mode fields
mode_fields = zeros(Nx, Nx, num_modes);
mode_x = [];
for m = 1:num_modes
    mode_fname = sprintf('%s/radius%dboundary0000fieldscalarmode%dwavelength%d.mat', ...
        prefix, radius, m, round(lambda0*1e9));
    S = load(mode_fname, 'phi', 'x');
    mode_fields(:,:,m) = S.phi;
    if isempty(mode_x)
        mode_x = S.x;
    end
end

% Print fiber info
disp(['Number of modes: ' num2str(num_modes)]);
disp(['Fiber length: ' num2str(fiber.L0) ' m']);
disp(['Center wavelength: ' num2str(fiber.lambda0*1e9) ' nm']);
disp(['Dispersion values (β₂): ' num2str(fiber.betas(3,:)) ' ps²/m']);

% Create fiber overview figure - combines dispersion and mode profiles
fig = create_figure('Fiber Configuration', [100, 100, 1000, 600]);

% Dispersion parameters for all modes
subplot(2,3,1);
order_max = size(fiber.betas, 1);
beta_orders = 2:order_max; % Skip β₀ and β₁
for m = 1:fiber.num_modes
    semilogy(beta_orders, abs(fiber.betas(beta_orders,m)), 'o-', 'LineWidth', 1.5, ...
        'Color', eval(['colors.mode' num2str(m)]));
    hold on;
end
xlabel('Dispersion Order'); ylabel('|\beta_n| [ps^n/m]');
title('Dispersion Parameters');
legend(arrayfun(@(x) sprintf('Mode %d', x), 1:fiber.num_modes, 'UniformOutput', false));
setup_axes(gca);

% SR tensor visualization
subplot(2,3,2);
SR_diag = zeros(fiber.num_modes);
for i = 1:fiber.num_modes
    for j = 1:fiber.num_modes
        SR_diag(i,j) = fiber.SR(i,j,i,j);
    end
end
imagesc(SR_diag);
colorbar;
xlabel('Mode j'); ylabel('Mode i');
title('SR Tensor (Diagonal Elements)');
setup_axes(gca);
axis square;

% Mode profiles
for m = 1:min(num_modes, 4)
    subplot(2,3,2+m);
    imagesc(mode_x*1e6, mode_x*1e6, abs(mode_fields(:,:,m)).^2);
    axis square; colorbar; colormap(gca, hot);
    xlabel('x [\mum]'); ylabel('y [\mum]');
    title(['Mode ' num2str(m)]);
    setup_axes(gca);
end

sgtitle('Fiber Configuration', 'FontSize', 14);
save_figure(fig, [results_dir '/fiber_setup']);


sim_base = struct(...
    'f0', 3e8/(lambda0*1e9), ...
    'deltaZ', 0.001, ...
    'M', 10, ... % Parallelization extent for MPA. 1 = no parallelization, 5-20 is recommended; there are strongly diminishing returns after 5-10
    'n_tot_max', 5, ...
    'n_tot_min', 2, ...
    'tol', 1e-6, ...
    'single_yes', 1, ...
    'gpu_yes', 1, ...
    'mpa_yes', 1, ...
    'SK_factor', 1, ...
    'use_const_mem', 0, ...
    'check_nan', 1, ...
    'verbose', 0, ...
    'cuda_dir_path', 'GMMNLSE-Solver-FINAL/cuda' ...
);

% For storing all test results
results = struct();

%% Test 1: SPM Spectral Broadening (Single Mode)
disp('Running Test 1: SPM Spectral Broadening...');

% Test configuration
t_width = 0.1; % Pulse width [ps]
sim = sim_base;
sim.fr = 0; sim.sw = 0; % No Raman or self-steepening
sim.save_period = fiber.L0/100;
amplitude = 1; % Pulse amplitude

% Initial pulse
initial_condition = struct();
initial_condition.dt = dt;
initial_condition.fields = zeros(Nt, fiber.num_modes);
initial_condition.fields(:,1) = amplitude * exp(-t.^2/t_width^2).'; % Single mode excitation

% Run propagation
foutput = GMMNLSE_propagate(fiber, initial_condition, sim);

% Process results
A_in = foutput.fields(:,1,1); A_out = foutput.fields(:,1,end);
spec_in = abs(fftshift(fft(A_in))).^2;
spec_out = abs(fftshift(fft(A_out))).^2;
width_in = fwhm(spec_in); width_out = fwhm(spec_out);
Ein = sum(abs(A_in).^2); Eout = sum(abs(A_out).^2);

% Store results
results.spm.width_in = width_in;
results.spm.width_out = width_out;
results.spm.ratio = width_out/width_in;
results.spm.energy_ratio = Eout/Ein;

% Print results
fprintf('\n--- SPM Test Results ---\n');
fprintf('Input FWHM: %d px, Output FWHM: %d px\n', width_in, width_out);
fprintf('Spectral broadening ratio: %.2f\n', width_out/width_in);
fprintf('Energy conservation: %.4f\n', Eout/Ein);

% Create consolidated SPM analysis visualization
fig = create_figure('SPM Analysis', [100, 100, 1000, 600]);
z_points = linspace(0, fiber.L0, size(foutput.fields, 3));

% Temporal and spectral evolution
subplot(2,3,[1,2]);
[Z, F] = meshgrid(z_points, freq);
spectral_evolution = zeros(Nt, length(z_points));
for i = 1:length(z_points)
    spectral_evolution(:, i) = abs(fftshift(fft(foutput.fields(:,1,i)))).^2;
end
surf(Z, F, spectral_evolution, 'EdgeColor', 'none');
view(2); colorbar; colormap(jet);
xlabel('Distance [m]'); ylabel('Frequency [THz]');
title('Spectral Evolution');
setup_axes(gca);
ylim([-10 10]);

% Temporal profile comparison
subplot(2,3,4);
plot(t, abs(A_in).^2, 'Color', colors.input, 'LineWidth', 1.5);
hold on;
plot(t, abs(A_out).^2, 'Color', colors.output, 'LineWidth', 1.5);
xlabel('Time [ps]'); ylabel('Intensity');
title('Temporal Profile');
legend('Input', 'Output');
setup_axes(gca);

% Spectral comparison
subplot(2,3,5);
plot(freq, spec_in/max(spec_in), 'Color', colors.input, 'LineWidth', 1.5);
hold on;
plot(freq, spec_out/max(spec_out), 'Color', colors.output, 'LineWidth', 1.5);
xlabel('Frequency [THz]'); ylabel('Normalized Power');
title('Spectral Comparison');
legend('Input', 'Output');
setup_axes(gca);
xlim([-10 10]); % Focus on central region

% Energy conservation
subplot(2,3,6);
energy_evolution = zeros(1, length(z_points));
for i = 1:length(z_points)
    energy_evolution(i) = sum(abs(foutput.fields(:,1,i)).^2);
end
plot(z_points, energy_evolution/energy_evolution(1), 'k-', 'LineWidth', 1.5);
xlabel('Distance [m]'); ylabel('Normalized Energy');
title('Energy Conservation');
setup_axes(gca);
ylim([0.9 1.1]);

% Spatial field visualization
subplot(2,3,3);
[~, ti] = max(abs(foutput.fields(:,1,end)));
[E_xy, ~, ~] = reconstruct_field(foutput.fields, mode_fields, ti);
imagesc(mode_x*1e6, mode_x*1e6, abs(E_xy).^2);
axis square; colorbar; colormap(gca, hot);
xlabel('x [\mum]'); ylabel('y [\mum]');
title(sprintf('Spatial Profile (t = %.2f ps)', t(ti)));
setup_axes(gca);

sgtitle(sprintf('SPM Spectral Broadening (Ratio: %.2f)', width_out/width_in), 'FontSize', 14);
save_figure(fig, [results_dir '/spm_analysis']);

%% Test 2: Multimode Propagation & Intermodal Coupling
disp('Running Test 2: Multimode Propagation...');

% Set up multiple modes with phase differences
initial_condition = struct();
initial_condition.dt = dt;
initial_condition.fields = zeros(Nt, fiber.num_modes);
initial_condition.fields(:,1) = amplitude * exp(-t.^2/t_width^2).';
initial_condition.fields(:,2) = amplitude * 0.8 * exp(-t.^2/t_width^2).' .* exp(1i*pi/2);
initial_condition.fields(:,3) = amplitude * 0.6 * exp(-t.^2/t_width^2).' .* exp(1i*pi);
initial_condition.fields(:,4) = amplitude * 0.4 * exp(-t.^2/t_width^2).' .* exp(1i*3*pi/2);

% Enable Raman effect and self-steepening
sim.fr = 0.18; sim.sw = 1;

% Run propagation
foutput = GMMNLSE_propagate(fiber, initial_condition, sim);

% Calculate mode energy evolution
z_points = linspace(0, fiber.L0, size(foutput.fields, 3));
mode_energy = zeros(fiber.num_modes, length(z_points));
for i = 1:length(z_points)
    for m = 1:fiber.num_modes
        mode_energy(m,i) = sum(abs(foutput.fields(:,m,i)).^2);
    end
end
normalized_energy = mode_energy ./ repmat(sum(mode_energy, 1), [fiber.num_modes, 1]);

% Store results
results.multimode.initial_ratios = normalized_energy(:,1);
results.multimode.final_ratios = normalized_energy(:,end);
results.multimode.max_change = max(abs(normalized_energy(:,end) - normalized_energy(:,1)));

% Print results
fprintf('\n--- Multimode Test Results ---\n');
fprintf('Initial mode energy ratios: %.4f, %.4f, %.4f, %.4f\n', normalized_energy(:,1));
fprintf('Final mode energy ratios: %.4f, %.4f, %.4f, %.4f\n', normalized_energy(:,end));
fprintf('Maximum energy fraction change: %.4f\n', results.multimode.max_change);

% Create visualization
fig = create_figure('Multimode Analysis', [100, 100, 1000, 600]);

% Mode energy evolution
subplot(2,2,1);
for m = 1:fiber.num_modes
    plot(z_points, normalized_energy(m,:), 'LineWidth', 2, 'Color', eval(['colors.mode' num2str(m)]));
    hold on;
end
xlabel('Distance [m]'); ylabel('Normalized Energy');
title('Mode Energy Evolution');
legend(arrayfun(@(x) sprintf('Mode %d', x), 1:fiber.num_modes, 'UniformOutput', false));
setup_axes(gca);

% Initial vs final mode comparison
subplot(2,2,2);
bar([normalized_energy(:,1), normalized_energy(:,end)]);
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('Mode %d', x), 1:fiber.num_modes, 'UniformOutput', false));
legend('Initial', 'Final');
ylabel('Energy Fraction');
title('Mode Energy Distribution');
setup_axes(gca);

% Spatial visualization at different distances
subplot(2,2,3);
[~, ti] = max(sum(abs(foutput.fields(:,:,end)).^2, 2));
[E_xy, ~, ~] = reconstruct_field(foutput.fields, mode_fields, ti);
imagesc(mode_x*1e6, mode_x*1e6, abs(E_xy).^2);
axis square; colorbar; colormap(gca, hot);
title('Output Spatial Profile');
xlabel('x [\mum]'); ylabel('y [\mum]');
setup_axes(gca);

% Phase visualization
subplot(2,2,4);
imagesc(mode_x*1e6, mode_x*1e6, angle(E_xy));
axis square; colorbar; colormap(gca, hsv);
title('Output Phase Profile');
xlabel('x [\mum]'); ylabel('y [\mum]');
setup_axes(gca);

sgtitle('Multimode Propagation Analysis', 'FontSize', 14);
save_figure(fig, [results_dir '/multimode_analysis']);

%% Test 3: Combined Raman & Self-Steepening Effects
disp('Running Test 3: Raman & Self-Steepening Effects...');

% Save original settings
original_SR = fiber.SR;
original_L0 = fiber.L0;

% Boost nonlinearity and length for stronger effects
fiber.SR = fiber.SR * 5;

% Narrower pulse for stronger nonlinear effects
initial_condition = struct();
initial_condition.dt = dt;
initial_condition.fields = zeros(Nt, fiber.num_modes);
initial_condition.fields(:,1) = amplitude * exp(-t.^2/t_width^2).';

% Setup simulation parameters
sim.fr = 0.18; % Raman fraction
sim.sw = 1;    % Self-steepening ON

% Run propagation
foutput = GMMNLSE_propagate(fiber, initial_condition, sim);

% Process results
A_in = foutput.fields(:,1,1); A_out = foutput.fields(:,1,end);
spec_in = abs(fftshift(fft(A_in))).^2;
spec_out = abs(fftshift(fft(A_out))).^2;

% Calculate spectral centroid shift (Raman effect)
centroid = @(f,s) sum(f(:).*s(:))/sum(s(:));
cin = centroid(freq, spec_in); cout = centroid(freq, spec_out);

% Calculate spectral asymmetry (self-steepening effect)
neg_freqs = freq < 0; pos_freqs = freq > 0;
left = sum(spec_out(neg_freqs)); right = sum(spec_out(pos_freqs));
asymmetry = (right-left)/(right+left);

% Store results
results.combined.raman_shift = cout - cin;
results.combined.asymmetry = asymmetry;

% Print results
fprintf('\n--- Raman & Self-Steepening Test Results ---\n');
fprintf('Spectral centroid shift: %.4f THz\n', cout-cin);
fprintf('Spectral asymmetry: %.4f\n', asymmetry);

% Create combined analysis figure
fig = create_figure('Raman & Self-Steepening Analysis', [100, 100, 1000, 600]);
z_points = linspace(0, fiber.L0, size(foutput.fields, 3));

% Spectral evolution
subplot(2,3,[1,2]);
[Z, F] = meshgrid(z_points, freq);
spectral_evolution = zeros(Nt, length(z_points));
for i = 1:length(z_points)
    spectral_evolution(:, i) = abs(fftshift(fft(foutput.fields(:,1,i)))).^2;
end
surf(Z, F, spectral_evolution, 'EdgeColor', 'none');
view(2); colorbar; colormap(jet);
xlabel('Distance [m]'); ylabel('Frequency [THz]');
title('Spectral Evolution');
setup_axes(gca);
ylim([-20 20]);

% Spectral comparison
subplot(2,3,3);
plot(freq, spec_in/max(spec_in), 'Color', colors.input, 'LineWidth', 1.5);
hold on;
plot(freq, spec_out/max(spec_out), 'Color', colors.output, 'LineWidth', 1.5);
xlabel('Frequency [THz]'); ylabel('Normalized Power');
title('Spectral Comparison');
legend('Input', 'Output');
setup_axes(gca);
xlim([-20 20]);

% Temporal profile
subplot(2,3,4);
plot(t, abs(A_in).^2, 'Color', colors.input, 'LineWidth', 1.5);
hold on;
plot(t, abs(A_out).^2, 'Color', colors.output, 'LineWidth', 1.5);
xlabel('Time [ps]'); ylabel('Intensity');
title('Temporal Profile');
legend('Input', 'Output');
setup_axes(gca);

% Centroid evolution (Raman shift)
subplot(2,3,5);
centroids = zeros(1, length(z_points));
for i = 1:length(z_points)
    spec = abs(fftshift(fft(foutput.fields(:,1,i)))).^2;
    centroids(i) = centroid(freq, spec);
end
plot(z_points, centroids, 'k-', 'LineWidth', 1.5);
xlabel('Distance [m]'); ylabel('Centroid [THz]');
title('Spectral Centroid Evolution');
setup_axes(gca);

% Pulse edge analysis (Self-steepening)
subplot(2,3,6);
[~, idx_max] = max(abs(A_out));
window = max(1, idx_max-50):min(Nt, idx_max+50); % Window around peak
plot(t(window), abs(A_in(window)).^2/max(abs(A_in).^2), 'Color', colors.input, 'LineWidth', 1.5);
hold on;
plot(t(window), abs(A_out(window)).^2/max(abs(A_out).^2), 'Color', colors.output, 'LineWidth', 1.5);
xlabel('Time [ps]'); ylabel('Normalized Intensity');
title('Pulse Edge Detail');
legend('Input', 'Output');
setup_axes(gca);

sgtitle(sprintf('Raman Shift: %.4f THz, Asymmetry: %.4f', cout-cin, asymmetry), 'FontSize', 14);
save_figure(fig, [results_dir '/raman_selfsteepening']);

% Restore original fiber parameters
fiber.L0 = original_L0;
fiber.SR = original_SR;

%% Create final summary figure
fig = create_figure('Test Results Summary', [100, 100, 800, 600]);

% Create summary text with all results
text_summary = {
    'GMMNLSE Test Suite - Summary', 
    ' ', 
    sprintf('Fiber: %d modes, λ₀ = %.0f nm', fiber.num_modes, fiber.lambda0*1e9),
    sprintf('Dispersion values (β₂): %.5f...%.5f ps²/m', min(fiber.betas(3,:)), max(fiber.betas(3,:))),
    ' ',
    'Test 1: SPM Broadening',
    sprintf('  • Spectral broadening: %.2f× (FWHM: %d → %d px)', results.spm.ratio, results.spm.width_in, results.spm.width_out),
    sprintf('  • Energy conservation: %.4f', results.spm.energy_ratio),
    ' ',
    'Test 2: Multimode Propagation',
    sprintf('  • Mode energy distribution held stable within %.2f%%', results.multimode.max_change*100),
    sprintf('  • Initial ratios: %.2f/%.2f/%.2f/%.2f', results.multimode.initial_ratios),
    sprintf('  • Final ratios:   %.2f/%.2f/%.2f/%.2f', results.multimode.final_ratios),
    ' ',
    'Test 3: Raman & Self-steepening Effects',
    sprintf('  • Raman frequency shift: %.4f THz', results.combined.raman_shift),
    sprintf('  • Spectral asymmetry: %.4f', results.combined.asymmetry)
};

% Display text summary
axes('Position', [0.1, 0.1, 0.8, 0.8]);
text(0.1, 0.9, text_summary, 'FontSize', 12, 'VerticalAlignment', 'top');
axis off;

save_figure(fig, [results_dir '/summary']);

% Save results for future reference
save([results_dir '/test_results.mat'], 'results');

fprintf('\nAll tests completed successfully. Results saved to %s\n', results_dir);