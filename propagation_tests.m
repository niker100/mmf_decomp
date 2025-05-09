% PROPAGATION_TESTS.M
% Comprehensive tests for gnlse_propagate, based on literature and best practices.
% Each test is self-contained and prints a summary of its result.

run_all_gnlse_tests();

function run_all_gnlse_tests()
    fprintf('\n--- GNLSE PROPAGATION TEST SUITE ---\n');
    test_linear_mode_preservation();
    test_nonlinear_spm_broadening();
    test_nonlinear_mode_coupling();
    test_energy_conservation();
    test_core_cladding_confinement();
    fprintf('--- ALL TESTS COMPLETED ---\n');
end

function test_linear_mode_preservation()
    % Test: Linear propagation should preserve the input mode profile (Agrawal, Ch.2)
    fprintf('\n[TEST] Linear Mode Preservation\n');
    core_radius = 25e-6;
    n_core = 1.46; n_clad = 1.45; lambda = 1030e-9;
    image_size = 128;
    x = linspace(-2*core_radius,2*core_radius,image_size);
    y = x;
    [X,Y] = meshgrid(x,y);
    r2 = X.^2 + Y.^2;
    core_mask = r2 <= core_radius^2;
    % Use a Gaussian as a proxy for the LP01 mode
    w0 = core_radius/1.5;
    U0 = exp(-r2/(w0^2));
    U0(~core_mask) = 0;
    U0 = U0 / sqrt(sum(abs(U0(:)).^2));
    params = struct('T0',50,'lam0',lambda*1e9,'distance',1,'N',0.0,...
        'sbeta2',-0.1,'nt',image_size,'Tmax',50,'step_num',50,'zstep',1,...
        'n_clad',n_clad,'n_core',n_core,'core_radius',core_radius,...
        'X',X,'Y',Y,'nonlinear_in_cladding',false);
    [U_out,~,~] = gnlse_propagate(U0,params);
    % Compare input/output in the core
    in_core_corr = corr2(abs(U0(core_mask)),abs(U_out(core_mask)));
    fprintf('  Correlation in core: %.4f (should be >0.99)\n', in_core_corr);
    assert(in_core_corr > 0.99, 'Linear propagation did not preserve mode profile.');
    % Visualization
    figure('Name','Linear Mode Preservation');
    subplot(1,3,1);
    imagesc(x*1e6, y*1e6, abs(U0).^2);
    title('Input Intensity');
    xlabel('x [\mum]'); ylabel('y [\mum]');
    axis image; colorbar;
    subplot(1,3,2);
    imagesc(x*1e6, y*1e6, abs(U_out).^2);
    title('Output Intensity');
    xlabel('x [\mum]'); ylabel('y [\mum]');
    axis image; colorbar;
    subplot(1,3,3);
    plot(x*1e6, abs(U0(:,image_size/2)).^2, 'b', 'LineWidth', 1.5); hold on;
    plot(x*1e6, abs(U_out(:,image_size/2)).^2, 'r--', 'LineWidth', 1.5);
    legend('Input','Output');
    title('Central Spatial Profile');
    xlabel('x [\mum]'); ylabel('Intensity');
end

function test_nonlinear_spm_broadening()
    % Test: SPM should broaden the spectrum (Agrawal, Ch.4)
    fprintf('\n[TEST] Nonlinear SPM Spectral Broadening\n');
    core_radius = 25e-6;
    n_core = 1.46; n_clad = 1.45; lambda = 1030e-9;
    image_size = 128;
    x = linspace(-2*core_radius,2*core_radius,image_size);
    y = x;
    [X,Y] = meshgrid(x,y);
    r2 = X.^2 + Y.^2;
    core_mask = r2 <= core_radius^2;
    w0 = core_radius/1.5;
    U0 = exp(-r2/(w0^2));
    U0(~core_mask) = 0;
    U0 = U0 / sqrt(sum(abs(U0(:)).^2));
    params = struct('T0',50,'lam0',lambda*1e9,'distance',10,'N',20.0,...
        'sbeta2',-0.1,'nt',image_size,'Tmax',50,'step_num',100,'zstep',1,...
        'n_clad',n_clad,'n_core',n_core,'core_radius',core_radius,...
        'X',X,'Y',Y,'nonlinear_in_cladding',false);
    [U_out,~,~] = gnlse_propagate(U0,params);
    % Compare input/output spectrum
    spec_in = abs(fftshift(fft2(U0))).^2;
    spec_out = abs(fftshift(fft2(U_out))).^2;
    % Use FWHM of central slice
    spec_in_slice = spec_in(image_size/2,:);
    spec_out_slice = spec_out(image_size/2,:);
    width_in = fwhm(spec_in_slice);
    width_out = fwhm(spec_out_slice);
    fprintf('  Input FWHM: %d px, Output FWHM: %d px, Ratio: %.2f\n', width_in, width_out, width_out/width_in);

    % Visualization
    figure('Name','SPM Spectral Broadening Test');
    subplot(2,2,1);
    imagesc(x*1e6, y*1e6, abs(U0).^2);
    title('Input Intensity');
    xlabel('x [\mum]'); ylabel('y [\mum]');
    axis image; colorbar;

    subplot(2,2,2);
    imagesc(x*1e6, y*1e6, abs(U_out).^2);
    title('Output Intensity');
    xlabel('x [\mum]'); ylabel('y [\mum]');
    axis image; colorbar;

    subplot(2,2,3);
    plot(spec_in_slice/max(spec_in_slice), 'b', 'LineWidth', 1.5); hold on;
    plot(spec_out_slice/max(spec_out_slice), 'r', 'LineWidth', 1.5);
    legend('Input','Output');
    title('Central Spectrum Slice');
    xlabel('Pixel'); ylabel('Normalized Power');

    subplot(2,2,4);
    plot(x*1e6, abs(U0(:,image_size/2)).^2, 'b', 'LineWidth', 1.5); hold on;
    plot(x*1e6, abs(U_out(:,image_size/2)).^2, 'r', 'LineWidth', 1.5);
    legend('Input','Output');
    title('Central Spatial Profile');
    xlabel('x [\mum]'); ylabel('Intensity');
end

function test_nonlinear_mode_coupling()
    % Test: Nonlinearity should induce mode coupling (Wright et al., Nat. Photonics 2015)
    fprintf('\n[TEST] Nonlinear Mode Coupling\n');
    core_radius = 25e-6;
    n_core = 1.46; n_clad = 1.45; lambda = 1030e-9;
    image_size = 128;
    x = linspace(-2*core_radius,2*core_radius,image_size);
    y = x;
    [X,Y] = meshgrid(x,y);
    r2 = X.^2 + Y.^2;
    theta = atan2(Y,X);
    core_mask = r2 <= core_radius^2;
    % LP01 and LP11 approximations
    LP01 = exp(-r2/(core_radius^2));
    LP11 = exp(-r2/(core_radius^2)) .* cos(theta);
    LP01(~core_mask) = 0; LP11(~core_mask) = 0;
    LP01 = LP01 / sqrt(sum(abs(LP01(:)).^2));
    LP11 = LP11 / sqrt(sum(abs(LP11(:)).^2));
    U0 = LP01*0.95 + LP11*0.05;
    params = struct('T0',50,'lam0',lambda*1e9,'distance',10,'N',20.0,...
        'sbeta2',-0.1,'nt',image_size,'Tmax',50,'step_num',100,'zstep',1,...
        'n_clad',n_clad,'n_core',n_core,'core_radius',core_radius,...
        'X',X,'Y',Y,'nonlinear_in_cladding',false);
    [U_out,~,~] = gnlse_propagate(U0,params);
    % Project output onto LP11
    overlap_LP11 = abs(sum(sum(conj(LP11).*U_out)));
    fprintf('  Output LP11 overlap: %.4f (should be >0.1 if mode coupling)\n', overlap_LP11);
    assert(overlap_LP11 > 0.1, 'Nonlinear mode coupling not observed.');
    % Visualization
    figure('Name','Nonlinear Mode Coupling');
    subplot(2,2,1);
    imagesc(x*1e6, y*1e6, abs(U0).^2);
    title('Input Intensity');
    xlabel('x [\mum]'); ylabel('y [\mum]');
    axis image; colorbar;
    subplot(2,2,2);
    imagesc(x*1e6, y*1e6, abs(U_out).^2);
    title('Output Intensity');
    xlabel('x [\mum]'); ylabel('y [\mum]');
    axis image; colorbar;
    subplot(2,2,3);
    imagesc(x*1e6, y*1e6, abs(LP11).^2);
    title('LP11 Mode Intensity');
    xlabel('x [\mum]'); ylabel('y [\mum]');
    axis image; colorbar;
    subplot(2,2,4);
    plot(x*1e6, abs(U0(:,image_size/2)).^2, 'b', 'LineWidth', 1.5); hold on;
    plot(x*1e6, abs(U_out(:,image_size/2)).^2, 'r', 'LineWidth', 1.5);
    legend('Input','Output');
    title('Central Spatial Profile');
    xlabel('x [\mum]'); ylabel('Intensity');
end

function test_energy_conservation()
    % Test: Total energy should be conserved in the absence of loss (Agrawal, Ch.2)
    fprintf('\n[TEST] Energy Conservation\n');
    core_radius = 25e-6;
    n_core = 1.46; n_clad = 1.45; lambda = 1030e-9;
    image_size = 128;
    x = linspace(-2*core_radius,2*core_radius,image_size);
    y = x;
    [X,Y] = meshgrid(x,y);
    r2 = X.^2 + Y.^2;
    core_mask = r2 <= core_radius^2;
    w0 = core_radius/1.5;
    U0 = exp(-r2/(w0^2));
    U0(~core_mask) = 0;
    U0 = U0 / sqrt(sum(abs(U0(:)).^2));
    params = struct('T0',50,'lam0',lambda*1e9,'distance',5,'N',2.0,...
        'sbeta2',-0.1,'nt',image_size,'Tmax',50,'step_num',100,'zstep',1,...
        'n_clad',n_clad,'n_core',n_core,'core_radius',core_radius,...
        'X',X,'Y',Y,'nonlinear_in_cladding',false);
    [U_out,~,~] = gnlse_propagate(U0,params);
    Ein = sum(abs(U0(:)).^2);
    Eout = sum(abs(U_out(:)).^2);
    fprintf('  Input energy: %.4e, Output energy: %.4e, Ratio: %.4f\n', Ein, Eout, Eout/Ein);
    % Visualization
    figure('Name','Energy Conservation');
    subplot(1,2,1);
    imagesc(x*1e6, y*1e6, abs(U0).^2);
    title('Input Intensity');
    xlabel('x [\mum]'); ylabel('y [\mum]');
    axis image; colorbar;
    subplot(1,2,2);
    imagesc(x*1e6, y*1e6, abs(U_out).^2);
    title('Output Intensity');
    xlabel('x [\mum]'); ylabel('y [\mum]');
    axis image; colorbar;
    % Plot energy bar chart
    figure('Name','Total Energy Comparison');
    bar([1 2], [Ein Eout]);
    set(gca, 'XTickLabel', {'Input','Output'});
    ylabel('Total Energy');
    title('Energy Conservation');
    assert(abs(Eout/Ein-1)<0.01, 'Energy not conserved (lossless case).');
end

function test_core_cladding_confinement()
    % Test: Field remains confined to the core in the linear regime (Okamoto, Ch.3)
    fprintf('\n[TEST] Core-Cladding Confinement\n');
    core_radius = 25e-6;
    n_core = 1.46; n_clad = 1.45; lambda = 1030e-9;
    image_size = 128;
    x = linspace(-2*core_radius,2*core_radius,image_size);
    y = x;
    [X,Y] = meshgrid(x,y);
    r2 = X.^2 + Y.^2;
    core_mask = r2 <= core_radius^2;
    w0 = core_radius/1.5;
    U0 = exp(-r2/(w0^2));
    U0(~core_mask) = 0;
    U0 = U0 / sqrt(sum(abs(U0(:)).^2));
    params = struct('T0',50,'lam0',lambda*1e9,'distance',5,'N',0.0,...
        'sbeta2',-0.1,'nt',image_size,'Tmax',50,'step_num',100,'zstep',1,...
        'n_clad',n_clad,'n_core',n_core,'core_radius',core_radius,...
        'X',X,'Y',Y,'nonlinear_in_cladding',false);
    [U_out,~,~] = gnlse_propagate(U0,params);
    frac_in_core = sum(abs(U_out(core_mask)).^2) / sum(abs(U_out(:)).^2);
    fprintf('  Fraction of energy in core: %.4f (should be >0.98)\n', frac_in_core);
    % Visualization
    figure('Name','Core-Cladding Confinement');
    subplot(1,2,1);
    imagesc(x*1e6, y*1e6, abs(U_out).^2);
    title('Output Intensity');
    xlabel('x [\mum]'); ylabel('y [\mum]');
    axis image; colorbar;
    subplot(1,2,2);
    mask_img = zeros(size(U_out));
    mask_img(core_mask) = 1;
    imagesc(x*1e6, y*1e6, mask_img);
    title('Core Mask');
    xlabel('x [\mum]'); ylabel('y [\mum]');
    axis image; colorbar;
    assert(frac_in_core > 0.98, 'Field leaked significantly into cladding.');
end
function w = fwhm(y)
    % Find full width at half maximum (FWHM) for a 1D array
    y = y / max(y);
    above = find(y > 0.5);
    if isempty(above), w = 0; return; end
    w = above(end) - above(1) + 1;
end