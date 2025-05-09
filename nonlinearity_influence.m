% nonlinearity_influence.m - Analyze linear vs nonlinear propagation using new gnlse_propagate
% This script compares linear and nonlinear propagation for different mode counts and distances.

%% Configuration
clear;
clc;
close all;

% Mode counts to compare
mode_counts = [3, 7, 16, 32];
image_size = 128; % Match test size for consistency
nl_strength = 2.0; % Nonlinear parameter N (as in test)
distances = [10, 50, 100];
distance_labels = {'Short (5 units)', 'Medium (10 units)', 'Long (20 units)'};

% Utility for mode generation (assumed available)
utils = mmf_utils();

% GPU check
useGPU = (gpuDeviceCount > 0);
if useGPU
    fprintf('GPU acceleration enabled: %s\n', gpuDevice().Name);
else
    fprintf('GPU not available. Using CPU.\n');
end

%% Main analysis loop
fprintf('Generating comparison data...\n');
fig = figure('Name', 'Linear vs Nonlinear Propagation', 'Position', [50 50 1200 800]);

for d_idx = 1:length(distances)
    distance = distances(d_idx);
    fig_d = figure('Name', sprintf('Propagation Distance: %s', distance_labels{d_idx}), ...
        'Position', [50 50 1200 800]);
    for m_idx = 1:length(mode_counts)
        num_modes = mode_counts(m_idx);
        fprintf('Processing %d modes at distance %d...\n', num_modes, distance);

        % Get model and random weights
        P = utils.getOrCreateModelWithModes(num_modes, image_size, true);
        weights = randn(1, num_modes) + 1i*randn(1, num_modes);
        weights = weights / norm(weights);

        % Generate initial field
        P.E = modeSuperposition(P, 1:num_modes, weights');
        U_initial = P.E.field;
        if size(U_initial, 1) ~= image_size
            U_initial = imresize(U_initial, [image_size, image_size], 'bicubic');
        end

        % Prepare params for gnlse_propagate (match test structure)
        params = struct();
        params.T0 = 50;
        params.lam0 = P.lambda * 1e9;
        params.distance = distance;
        params.N = nl_strength;
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

        % Linear propagation (N=0)
        linear_intensity = abs(U_initial).^2;

        % Nonlinear propagation (N=nl_strength)
        params.N = nl_strength;
        [U_nl, ~, ~] = gnlse_propagate(U_initial, params);
        nl_intensity = abs(U_nl).^2;

        % Normalize for display
        linear_intensity_norm = linear_intensity / max(linear_intensity(:));
        nl_intensity_norm = nl_intensity / max(nl_intensity(:));
        diff_intensity_norm = abs(nl_intensity_norm - linear_intensity_norm);

        % Visualization for this distance/mode count
        figure(fig_d);
        subplot(length(mode_counts), 3, (m_idx-1)*3 + 1);
        imagesc(linear_intensity_norm); axis image off; colormap(jet);
        title(sprintf('%d Modes - Linear', num_modes));

        subplot(length(mode_counts), 3, (m_idx-1)*3 + 2);
        imagesc(nl_intensity_norm); axis image off;
        title(sprintf('%d Modes - Nonlinear (N=%.1f)', num_modes, nl_strength));

        subplot(length(mode_counts), 3, (m_idx-1)*3 + 3);
        imagesc(diff_intensity_norm); axis image off; colormap(jet);
        title(sprintf('%d Modes - Difference', num_modes));

        % Add correlation and peak diff
        corr_val = corr2(linear_intensity_norm, nl_intensity_norm);
        peak_diff = max(diff_intensity_norm(:));
        text(5, 15, sprintf('Corr: %.3f', corr_val), 'Color', 'white', ...
            'FontWeight', 'bold', 'FontSize', 8);
        text(5, image_size-5, sprintf('Max diff: %.1f%%', peak_diff*100), ...
            'Color', 'white', 'FontWeight', 'bold', 'FontSize', 8, ...
            'VerticalAlignment', 'bottom');

        % Main figure (middle distance only)
        if d_idx == 2
            figure(fig);
            subplot(2, length(mode_counts), m_idx);
            comp_img = [linear_intensity_norm, nl_intensity_norm];
            imagesc(comp_img); axis image off;
            title(sprintf('%d Modes (Corr: %.3f)', num_modes, corr_val));
            ylabel('Linear | Nonlinear');
            subplot(2, length(mode_counts), m_idx + length(mode_counts));
            plot(linear_intensity_norm(:,round(image_size/2)), 'b', 'LineWidth', 1.5); hold on;
            plot(nl_intensity_norm(:,round(image_size/2)), 'r', 'LineWidth', 1.5);
            legend('Linear','Nonlinear');
            title('Central Profile');
            hold off;
        end
    end
    figure(fig_d);
    sgtitle(sprintf('Linear vs Nonlinear Propagation (%s)', distance_labels{d_idx}), ...
        'FontSize', 16, 'FontWeight', 'bold');
    saveas(fig_d, sprintf('linear_vs_nonlinear_distance_%d.png', d_idx));
end

figure(fig);
sgtitle('Linear vs Nonlinear Propagation Comparison', 'FontSize', 16, 'FontWeight', 'bold');
saveas(fig, 'linear_vs_nonlinear_comparison.png');

fprintf('Analysis complete. Results saved as images.\n');