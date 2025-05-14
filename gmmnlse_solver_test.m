%% GMMNLSE_SOLVER_TEST.M
% Simple test script to demonstrate how to generate an x-y intensity image
% for a 4-mode fiber propagation using the GMMNLSE solver with custom complex mode weights
% This script can be used with a mode prediction pipeline to visualize the predicted field.

clear all; close all; clc;
addpath(genpath('GMMNLSE-Solver-FINAL'));

% Create directory for results if it doesn't exist
results_dir = 'test_results';
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

%% FIBER SETUP: Define fiber parameters and build fiber structure
disp('Setting up multimode fiber parameters...');

% Core parameters
lambda0 = 1030e-9;          % Center wavelength [m]
fiber_length = 1;           % Propagation distance [m]
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

%% Reconstruct field helper functions
% Single time point reconstruction
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

% Time-integrated reconstruction (simulates camera detection)
function [I_xy_integrated] = reconstruct_field_integrated(modal_fields, mode_fields, time_window)
    Nx = size(mode_fields, 1);
    num_modes = size(mode_fields, 3);
    Nt = size(modal_fields, 1);
    
    % If no time window specified, use entire time range
    if isempty(time_window)
        time_window = 1:Nt;
    end
    
    % Initialize integrated intensity
    I_xy_integrated = zeros(Nx, Nx);
    
    % Loop through all time points and accumulate intensity
    for t_idx = time_window
        % Reconstruct field at this time point
        E_xy = zeros(Nx, Nx);
        for m = 1:num_modes
            E_xy = E_xy + modal_fields(t_idx, m, end) * mode_fields(:,:,m);
        end
        
        % Add intensity contribution from this time point
        I_xy_integrated = I_xy_integrated + abs(E_xy).^2;
    end
    
    % Normalize by number of time points (optional)
    I_xy_integrated = I_xy_integrated / length(time_window);
end

%% Simulation setup
sim = struct(...
    'f0', 3e8/(lambda0*1e9), ...
    'deltaZ', 1e-3, ...
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



% Enable Raman effect and self-steepening
sim.fr = 0.18; sim.sw = 1;
sim.save_period = fiber.L0/10; % Save 10 points along propagation

%% EXAMPLE 1: Propagation with custom complex mode weights

% Define pulse parameters
t_width = 0.05; % Pulse width [ps]
amplitude = 10; % Pulse amplitude

% Custom complex mode weights
% Format: [|A1|*exp(i*φ1), |A2|*exp(i*φ2), ...]
mode_weights = [1, 0.7*exp(1i*pi/3), 0.4*exp(-1i*pi/4), 0.2*exp(1i*pi/2)];
disp('Custom mode weights:');
for m = 1:num_modes
    fprintf('Mode %d: Amplitude = %.2f, Phase = %.2f rad\n', ...
        m, abs(mode_weights(m)), angle(mode_weights(m)));
end

% Initialize field with custom mode weights
initial_condition = struct();
initial_condition.dt = dt;
initial_condition.fields = zeros(Nt, fiber.num_modes);

% Apply the custom weights to a Gaussian pulse
for m = 1:num_modes
    initial_condition.fields(:,m) = mode_weights(m) * amplitude * exp(-t.^2/t_width^2).';
end

% Visualize the input field at the peak of the pulse
[~, peak_idx] = max(abs(initial_condition.fields(:,1)));
field_input = zeros(size(mode_fields,1), size(mode_fields,2));
for m = 1:num_modes
    field_input = field_input + mode_weights(m) * mode_fields(:,:,m);
end
I_input = abs(field_input).^2;

% Plot the input intensity
figure('Name','Input Field Intensity','Position',[200 200 800 400]);

% Left plot: Input field intensity
subplot(1,2,1);
imagesc(mode_x*1e6, mode_x*1e6, I_input);
axis image; colorbar; colormap(hot);
xlabel('x [\mum]'); ylabel('y [\mum]');
title('Input Field: X-Y Intensity from Custom Mode Weights');
set(gca,'FontSize',12);

% Right plot: Mode weights visualization
subplot(1,2,2);
bar(1:num_modes, abs(mode_weights).^2);
hold on;
yyaxis right;
plot(1:num_modes, angle(mode_weights), 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Mode Number');
yyaxis left; ylabel('Power Distribution');
yyaxis right; ylabel('Phase [rad]');
title('Custom Mode Weights');
grid on;
legend('Power', 'Phase');
set(gca,'FontSize',12);

saveas(gcf, [results_dir '/input_field_intensity.png']);

%% Run propagation with GMMNLSE solver
disp('Running propagation with custom mode weights...');
tic;
foutput = GMMNLSE_propagate(fiber, initial_condition, sim);
toc;

%% Analyze output field
disp('Analyzing output field...');

% Extract modal fields at the output
modal_fields_output = foutput.fields(:,:,end);

% Find time with maximum intensity
[~, peak_time] = max(sum(abs(modal_fields_output).^2, 2));

% Calculate output mode distribution
output_mode_powers = zeros(1, num_modes);
for m = 1:num_modes
    output_mode_powers(m) = sum(abs(modal_fields_output(:,m)).^2);
end
output_mode_powers = output_mode_powers / sum(output_mode_powers);

% Calculate output mode phases at peak time
output_mode_phases = angle(modal_fields_output(peak_time,:));

% Reconstruct spatial field at output - peak method
[E_output, I_output, spatial_spectrum] = reconstruct_field(foutput.fields, mode_fields, peak_time);

% Reconstruct spatial field - time-integrated method (camera-like)
% Define a window around the pulse (can adjust this as needed)
pulse_width = sum(abs(modal_fields_output(:,1)).^2 > 0.05*max(abs(modal_fields_output(:,1)).^2));
time_window = max(1, peak_time-pulse_width):min(length(t), peak_time+pulse_width);
I_output_integrated = reconstruct_field_integrated(foutput.fields, mode_fields, time_window);

% Create visualization figure
figure('Name', 'Propagation Results', 'Position', [100 100 1200 800]);

% Plot mode power evolution
subplot(2,3,1);
z_points = linspace(0, fiber.L0, size(foutput.fields, 3));
mode_energy = zeros(fiber.num_modes, length(z_points));
for i = 1:length(z_points)
    for m = 1:fiber.num_modes
        mode_energy(m,i) = sum(abs(foutput.fields(:,m,i)).^2);
    end
end
normalized_energy = mode_energy ./ repmat(sum(mode_energy, 1), [fiber.num_modes, 1]);

% Plot energy evolution with nice colors
colors = {'b', 'r', 'g', 'm', 'c', 'y', 'k'};
for m = 1:fiber.num_modes
    plot(z_points, normalized_energy(m,:), 'LineWidth', 2, 'Color', colors{mod(m-1, length(colors))+1});
    hold on;
end
xlabel('Distance [m]'); ylabel('Normalized Energy');
title('Mode Energy Evolution');
legend(arrayfun(@(x) sprintf('Mode %d', x), 1:fiber.num_modes, 'UniformOutput', false));
grid on;
set(gca,'FontSize',12);

% Compare input vs output mode distribution
subplot(2,2,2);
bar([abs(mode_weights).^2./sum(abs(mode_weights).^2); output_mode_powers]');
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('Mode %d', x), 1:fiber.num_modes, 'UniformOutput', false));
legend('Input', 'Output');
ylabel('Power Fraction');
title('Mode Power Distribution');
grid on;
set(gca,'FontSize',12);

% Output spatial profile - Peak method
subplot(2,3,3);
imagesc(mode_x*1e6, mode_x*1e6, I_output);
axis image; colorbar; colormap(hot);
xlabel('x [\mum]'); ylabel('y [\mum]');
title(sprintf('Peak Intensity (t = %.2f ps)', t(peak_time)));
set(gca,'FontSize',12);

% Output spatial profile - Time-integrated (camera-like)
subplot(2,3,4);
imagesc(mode_x*1e6, mode_x*1e6, I_output_integrated);
axis image; colorbar; colormap(hot);
xlabel('x [\mum]'); ylabel('y [\mum]');
title('Time-Integrated Intensity (Camera-like)');
set(gca,'FontSize',12);

% Output phase profile
subplot(2,3,5);
imagesc(mode_x*1e6, mode_x*1e6, angle(E_output));
axis image; colorbar; colormap(hsv);
xlabel('x [\mum]'); ylabel('y [\mum]');
title('Output Phase Profile (at peak time)');
set(gca,'FontSize',12);

% Difference between peak and time-integrated
subplot(2,3,6);
% Normalize both for fair comparison
I_peak_norm = I_output / max(I_output(:));
I_int_norm = I_output_integrated / max(I_output_integrated(:));
diff_image = I_peak_norm - I_int_norm;
imagesc(mode_x*1e6, mode_x*1e6, diff_image);
axis image; colorbar; colormap(jet);
xlabel('x [\mum]'); ylabel('y [\mum]');
title('Difference: Peak vs. Integrated');
set(gca,'FontSize',12);

saveas(gcf, [results_dir '/output_field_analysis.png']);

% Print summary
fprintf('\n--- Propagation Results ---\n');
fprintf('Input power fractions: ');
fprintf('%.3f ', abs(mode_weights).^2./sum(abs(mode_weights).^2));
fprintf('\nOutput power fractions: ');
fprintf('%.3f ', output_mode_powers);
fprintf('\n\n');

%% EXAMPLE 2: Function to visualize field for any custom weights

% Define function to quickly visualize any set of mode weights
function visualize_custom_weights(mode_weights, mode_fields, mode_x)
    % Normalize weights
    mode_weights = mode_weights(:).' / norm(mode_weights);
    
    % Reconstruct field
    field = zeros(size(mode_fields,1), size(mode_fields,2));
    for m = 1:length(mode_weights)
        if m <= size(mode_fields,3)
            field = field + mode_weights(m) * mode_fields(:,:,m);
        end
    end
    intensity = abs(field).^2;
    
    % Create visualization
    figure('Name','Custom Mode Weight Visualization','Position',[200 200 800 400]);
    
    % Left plot: Field intensity
    subplot(1,2,1);
    imagesc(mode_x*1e6, mode_x*1e6, intensity);
    axis image; colorbar; colormap(hot);
    xlabel('x [\mum]'); ylabel('y [\mum]');
    title('X-Y Intensity from Custom Mode Weights');
    
    % Right plot: Mode weights visualization
    subplot(1,2,2);
    bar(1:length(mode_weights), abs(mode_weights).^2);
    hold on;
    yyaxis right;
    plot(1:length(mode_weights), angle(mode_weights), 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('Mode Number');
    yyaxis left; ylabel('Power Distribution');
    yyaxis right; ylabel('Phase [rad]');
    title('Custom Mode Weights');
    grid on;
    legend('Power', 'Phase');
    
    % Optional: Return figure handle
end

% Define a more advanced visualization function with both peak and integrated options
function visualize_with_integration(modal_fields, mode_fields, mode_x, t)
    % Calculate peak time
    [~, peak_time] = max(sum(abs(modal_fields(:,:,end)).^2, 2));
    
    % Peak-based reconstruction
    [E_peak, I_peak, ~] = reconstruct_field(modal_fields, mode_fields, peak_time);
    
    % Time-integrated reconstruction
    pulse_width = sum(abs(modal_fields(:,1,end)).^2 > 0.05*max(abs(modal_fields(:,1,end)).^2));
    time_window = max(1, peak_time-pulse_width):min(length(t), peak_time+pulse_width);
    I_integrated = reconstruct_field_integrated(modal_fields, mode_fields, time_window);
    
    % Create figure
    figure('Name','Peak vs. Integrated Intensity Comparison','Position',[200 200 1000 400]);
    
    % Plot peak intensity
    subplot(1,3,1);
    imagesc(mode_x*1e6, mode_x*1e6, I_peak);
    axis image; colorbar; colormap(hot);
    xlabel('x [\mum]'); ylabel('y [\mum]');
    title(sprintf('Peak Intensity (t = %.2f ps)', t(peak_time)));
    
    % Plot time-integrated intensity
    subplot(1,3,2);
    imagesc(mode_x*1e6, mode_x*1e6, I_integrated);
    axis image; colorbar; colormap(hot);
    xlabel('x [\mum]'); ylabel('y [\mum]');
    title('Time-Integrated Intensity (Camera-like)');
    
    % Plot difference
    subplot(1,3,3);
    I_peak_norm = I_peak / max(I_peak(:));
    I_int_norm = I_integrated / max(I_integrated(:));
    diff_image = I_peak_norm - I_int_norm;
    imagesc(mode_x*1e6, mode_x*1e6, diff_image);
    axis image; colorbar; colormap(jet);
    xlabel('x [\mum]'); ylabel('y [\mum]');
    title('Difference (Peak - Integrated)');
    
    % Optional: Return figure handle
end

% Example usage - try different weight combinations
disp('Visualizing different mode combinations:');

% Example 1: Equal amplitude with varying phases
test_weights1 = [1, exp(1i*pi/2), exp(1i*pi), exp(1i*3*pi/2)];
visualize_custom_weights(test_weights1, mode_fields, mode_x);
saveas(gcf, [results_dir '/example1_equal_amplitudes.png']);

% Example 2: Amplitude decreases with mode order, constant phase
test_weights2 = [1, 0.6, 0.3, 0.1];
visualize_custom_weights(test_weights2, mode_fields, mode_x);
saveas(gcf, [results_dir '/example2_decreasing_amplitudes.png']);

% Example 3: Custom combination
test_weights3 = [0.5*exp(1i*pi/4), 1, 0.2*exp(1i*pi), 0.7*exp(-1i*pi/3)];
visualize_custom_weights(test_weights3, mode_fields, mode_x);
saveas(gcf, [results_dir '/example3_custom_combination.png']);

% Example 4: Demonstrate time-integrated visualization with propagation results
disp('Demonstrating peak vs. time-integrated intensity comparison...');
visualize_with_integration(foutput.fields, mode_fields, mode_x, t);
saveas(gcf, [results_dir '/peak_vs_integrated_comparison.png']);

%% Create an example to deliberately show the difference between peak and integrated intensity
disp('Creating example with significant peak vs. integrated intensity differences...');

% Create a test case with mode beating to highlight differences
% Start with two modes with nearly equal power but different group velocities
test_initial = struct();
test_initial.dt = dt;
test_initial.fields = zeros(Nt, fiber.num_modes);
test_initial.fields(:,1) = amplitude * exp(-t.^2/t_width^2).' * 0.7;
test_initial.fields(:,2) = amplitude * exp(-t.^2/t_width^2).' * 0.7 * exp(1i*pi/2);

% Enhance GVD difference between modes for more dramatic effect
temp_fiber = fiber;
temp_fiber.betas(3,1) = fiber.betas(3,1) * 1.5;  % Increase GVD for first mode
temp_fiber.betas(3,2) = fiber.betas(3,2) * 0.5;  % Decrease GVD for second mode
temp_fiber.L0 = 2;  % Longer fiber for more dramatic effects

% Run short propagation
disp('Running test propagation for peak vs. integrated comparison...');
temp_sim = sim;
temp_sim.save_period = temp_fiber.L0/5;
temp_output = GMMNLSE_propagate(temp_fiber, test_initial, temp_sim);

% Create visualization highlighting the differences
disp('Visualizing peak vs. time-integrated differences...');
visualize_with_integration(temp_output.fields, mode_fields, mode_x, t);
saveas(gcf, [results_dir '/significant_peak_vs_integrated_difference.png']);

% Add a temporal visualization to explain why there's a difference
figure('Name', 'Temporal Evolution Explaining Integration Effects', 'Position', [200 200 1000 400]);
subplot(1,2,1);
imagesc(t, 1:temp_fiber.num_modes, abs(squeeze(temp_output.fields(:,1:temp_fiber.num_modes,end)))'.^2);
xlabel('Time [ps]'); ylabel('Mode'); 
title('Output Temporal Intensity by Mode');
colorbar; colormap(jet);

subplot(1,2,2);
plot(t, sum(abs(temp_output.fields(:,:,1)).^2, 2), 'b-', 'LineWidth', 1.5);
hold on;
plot(t, sum(abs(temp_output.fields(:,:,end)).^2, 2), 'r-', 'LineWidth', 1.5);
xlabel('Time [ps]'); ylabel('Total Intensity');
title('Input vs Output Total Intensity');
legend('Input', 'Output');
grid on;
saveas(gcf, [results_dir '/temporal_evolution_explanation.png']);

disp('Script completed successfully. Results saved to test_results directory.');