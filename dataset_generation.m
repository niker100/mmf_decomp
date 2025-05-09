%% Generation of dataset with comprehensive phase sign augmentation
% pseudo random mode combination with multiple phase sign combinations
clear all

%%  set parameters 
% Add variable for max supported modes
max_supported_modes = 500;  % Set this to your desired maximum

% Change mode selection to allow higher counts
fprintf('Select number of modes (3-%d): ', max_supported_modes);
number_of_modes = input('');
if isempty(number_of_modes) || number_of_modes < 1
    number_of_modes = 5;  % Default to 5 if invalid
elseif number_of_modes > max_supported_modes
    number_of_modes = max_supported_modes;
    fprintf('Limited to maximum of %d modes\n', max_supported_modes);
end

% Determine if we should generate all possible sign combinations for visualization
if number_of_modes <= 7
    generate_all_signs = true;
    fprintf('Generating all possible sign combinations for %d modes (2^%d = %d combinations)\n', ...
            number_of_modes, number_of_modes-1, 2^(number_of_modes-1));
else
    generate_all_signs = false;
end

% Improved parameter settings
base_samples = 1000;  % Base number of unique amplitude/phase combinations

% Dynamic number of variants based on mode count
if generate_all_signs
    % For small mode counts, create all sign combinations (2^(modes-1))
    % First mode always positive as reference
    num_variants = 2^(number_of_modes-1);
else
    % For larger mode counts, use a fixed number of variants
    num_variants = 1;  % Increased from 4 to 8 sign variants
end

image_size = 64;      % Increased resolution for better visualization
total_samples = base_samples * num_variants;

% Non-linearity strength parameters
useNonLinear = true;
use_varying_nl = true;  % Use varying non-linearity strength
nl_strength_min = 20;  % Minimum non-linearity strength
nl_strength_max = 20;  % Maximum non-linearity strength
batch_size = 512;       % Increased batch size for efficiency

%% Precompute BPMmatlab model and modes ONCE for all samples
P = mmf_utils().getOrCreateModelWithModes(number_of_modes, image_size, true);

%% Generation of complex mode weights and label vector

% 1. Generate base amplitude and phase values
base_rho = rand(base_samples, number_of_modes);
base_rho = base_rho ./ sqrt(sum(base_rho.^2, 2));  % Normalize

% Generate base phases between -π and π
base_phi = (rand(base_samples, number_of_modes) - 0.5) * 2*pi;

% Calculate phase differences relative to first mode
base_phi_diff = base_phi - base_phi(:, ones(1, number_of_modes));
base_phi_diff = mod(base_phi_diff + pi, 2*pi) - pi;  % Wrap to [-π,π]

% 2. Create multiple sign variants per base sample
rho = zeros(total_samples, number_of_modes);
mode_weights = zeros(total_samples, number_of_modes);
phase_signs = cell(num_variants, 1);

% Define sign pattern variants (mode 1 always stays the same as reference)
if generate_all_signs
    % Generate all possible sign combinations for the remaining modes
    % Mode 1 is always positive as reference
    for v = 1:num_variants
        % Convert variant index to binary representation for sign pattern
        % v-1 is used to start from all positive (0000...)
        bin_pattern = dec2bin(v-1, number_of_modes-1) - '0';
        
        % Convert binary pattern (0,1) to signs (1,-1)
        signs = 1 - 2*bin_pattern;
        
        % Add the first mode's sign (always positive)
        phase_signs{v} = [1, signs];
    end
else
    % For larger mode counts, use predefined sign patterns
    % Original: all positive
    phase_signs{1} = ones(1, number_of_modes);
    
    % All others flipped
    phase_signs{2} = [1, -ones(1, number_of_modes-1)];
    
    % Even indices flipped
    phase_signs{3} = ones(1, number_of_modes);
    phase_signs{3}(2:2:end) = -1;
    
    % Odd indices (except first) flipped
    phase_signs{4} = ones(1, number_of_modes);
    phase_signs{4}(3:2:end) = -1;
    
    % First half flipped (except first mode)
    phase_signs{5} = ones(1, number_of_modes);
    half_point = ceil(number_of_modes/2);
    phase_signs{5}(2:half_point) = -1;
    
    % Second half flipped
    phase_signs{6} = ones(1, number_of_modes);
    phase_signs{6}(half_point+1:end) = -1;
    
    % Alternating groups of two
    phase_signs{7} = ones(1, number_of_modes);
    for i = 2:4:number_of_modes
        if i+1 <= number_of_modes
            phase_signs{7}(i:i+1) = -1;
        else
            phase_signs{7}(i) = -1;
        end
    end
    
    % Random pattern - but fixed across all samples
    phase_signs{8} = ones(1, number_of_modes);
    rng(42); % For reproducibility
    phase_signs{8}(2:end) = sign(randn(1, number_of_modes-1));
end

% Generate variants with different sign combinations
for v = 1:num_variants
    sample_indices = ((v-1)*base_samples+1):(v*base_samples);
    
    % Apply the sign pattern to phase differences
    variant_phi_diff = base_phi_diff .* phase_signs{v}(ones(base_samples, 1), :);
    
    % Store amplitude (same for all variants of a base sample)
    rho(sample_indices, :) = base_rho;
    
    % Calculate complex weights with the sign variant
    mode_weights(sample_indices, :) = base_rho .* exp(1i * variant_phi_diff);
    
    % Normalized phase for labels (map [-π,π] to [-1,1])
    variant_phi_norm = variant_phi_diff / pi;
    
    % Combine amplitude and phase into label vector
    if v == 1
        % Initialize labels array
        labels = zeros(total_samples, number_of_modes + (number_of_modes-1));
    end
    
    % Store amplitude and phase differences
    labels(sample_indices, 1:number_of_modes) = base_rho;
    labels(sample_indices, number_of_modes+1:end) = variant_phi_norm(:, 2:end);
end

% 3. Split data into training, validation and test sets
% Ensure all variants of a base sample stay in the same split
split_ratio = [0.8 0.1 0.1];
base_indices = 1:base_samples;
split_idx = cumsum([0 split_ratio*base_samples]);

% Generate indices for each split
train_base = base_indices(1:split_idx(2));
val_base = base_indices(split_idx(2)+1:split_idx(3));
test_base = base_indices(split_idx(3)+1:end);

% Expand indices to include all variants of each base sample
train_indices = [];
val_indices = [];
test_indices = [];

for v = 1:num_variants
    train_indices = [train_indices, (v-1)*base_samples + train_base];
    val_indices = [val_indices, (v-1)*base_samples + val_base];
    test_indices = [test_indices, (v-1)*base_samples + test_base];
end

% Create training, validation, and test sets
mode_weights_train = mode_weights(train_indices, :);
mode_weights_val = mode_weights(val_indices, :);
mode_weights_test = mode_weights(test_indices, :);
labels_train = labels(train_indices, :);
labels_val = labels(val_indices, :);
labels_test = labels(test_indices, :);

% Transfer to GPU for faster computation
mode_weights_train = gpuArray(mode_weights_train);
mode_weights_val = gpuArray(mode_weights_val);
mode_weights_test = gpuArray(mode_weights_test);

% Initialize metadata tracking
variant_indices = struct();
variant_indices.train = cell(num_variants, 1);
variant_indices.val = cell(num_variants, 1);
variant_indices.test = cell(num_variants, 1);

for v = 1:num_variants
    variant_indices.train{v} = find(ismember(train_indices, (v-1)*base_samples + (1:base_samples)));
    variant_indices.val{v} = find(ismember(val_indices, (v-1)*base_samples + (1:base_samples)));
    variant_indices.test{v} = find(ismember(test_indices, (v-1)*base_samples + (1:base_samples)));
end

%% Process data with varying non-linearity strength
datasets = struct('train', struct('data', mode_weights_train, 'size', size(mode_weights_train, 1)), ...
                 'val', struct('data', mode_weights_val, 'size', size(mode_weights_val, 1)), ...
                 'test', struct('data', mode_weights_test, 'size', size(mode_weights_test, 1)));

% Initialize output images and nl strength tracking
mmf_train = zeros(image_size, image_size, 1, datasets.train.size, 'gpuArray');
mmf_val = zeros(image_size, image_size, 1, datasets.val.size, 'gpuArray');
mmf_test = zeros(image_size, image_size, 1, datasets.test.size, 'gpuArray');

nl_strengths = struct();
nl_strengths.train = zeros(datasets.train.size, 1);
nl_strengths.val = zeros(datasets.val.size, 1);
nl_strengths.test = zeros(datasets.test.size, 1);

split_names = {'train', 'val', 'test'};

if use_varying_nl
    for s = 1:length(split_names)
        split = split_names{s};
        split_size = datasets.(split).size;
        batch_strengths = logspace(log10(nl_strength_min), log10(nl_strength_max), split_size);
        for idx = 1:split_size
            fprintf('Processing %s sample %d/%d with NL strength %.2e\n', split, idx, split_size, batch_strengths(idx));
            sample_weights = datasets.(split).data(idx, :);
            [img, ~] = mmf_build_image(number_of_modes, image_size, 1, sample_weights, useNonLinear, batch_strengths(idx), P);
            img = extract(img);
            if strcmp(split, 'train')
                mmf_train(:,:,:,idx) = img;
                nl_strengths.train(idx) = batch_strengths(idx);
            elseif strcmp(split, 'val')
                mmf_val(:,:,:,idx) = img;
                nl_strengths.val(idx) = batch_strengths(idx);
            else
                mmf_test(:,:,:,idx) = img;
                nl_strengths.test(idx) = batch_strengths(idx);
            end
        end
    end
else
    for s = 1:length(split_names)
        split = split_names{s};
        split_size = datasets.(split).size;
        for idx = 1:split_size
            sample_weights = datasets.(split).data(idx, :);
            [img, ~] = mmf_build_image(number_of_modes, image_size, 1, sample_weights, useNonLinear, 1.0, P);
            img = extract(img);
            if strcmp(split, 'train')
                mmf_train(:,:,:,idx) = img;
            elseif strcmp(split, 'val')
                mmf_val(:,:,:,idx) = img;
            else
                mmf_test(:,:,:,idx) = img;
            end
        end
    end
end

%% Save dataset with enhanced metadata
if use_varying_nl
    % Enhanced metadata
    nl_metadata = struct('use_varying_nl', use_varying_nl, ...
                        'nl_strength_min', nl_strength_min, ...
                        'nl_strength_max', nl_strength_max, ...
                        'train_nl_strengths', nl_strengths.train, ...
                        'val_nl_strengths', nl_strengths.val, ...
                        'test_nl_strengths', nl_strengths.test, ...
                        'num_variants', num_variants, ...
                        'phase_signs', {phase_signs}, ...
                        'variant_indices', variant_indices);
                     
    save('mmf_dataset_multi_sign.mat', 'mmf_train', 'mmf_val', 'mmf_test', ...
         'labels_train', 'labels_val', 'labels_test', ...
         'nl_metadata', 'number_of_modes', 'phase_signs', '-v7.3');
else
    save('mmf_dataset_multi_sign.mat', 'mmf_train', 'mmf_val', 'mmf_test', ...
         'labels_train', 'labels_val', 'labels_test', ...
         'phase_signs', 'variant_indices', 'number_of_modes', '-v7.3');
end

%% Create enhanced phase sign pattern visualization with expanded variants
fprintf('Creating phase sign pattern visualization with expanded variants...\n');

% Expand to more sign patterns for better analysis
num_expanded_variants = num_variants;  % Set to 32 sign variants

% Create expanded sign patterns (first mode always positive as reference)
expanded_phase_signs = cell(num_expanded_variants, 1);

if number_of_modes > 10
    % For larger mode counts, create systematic and random patterns
    
    % 1-8: Keep original patterns
    for v = 1:min(8, num_variants)
        expanded_phase_signs{v} = phase_signs{v};
    end
    
    % 9: First third flipped
    expanded_phase_signs{9} = ones(1, number_of_modes);
    third_point = ceil(number_of_modes/3);
    expanded_phase_signs{9}(2:third_point) = -1;
    
    % 10: Second third flipped
    expanded_phase_signs{10} = ones(1, number_of_modes);
    expanded_phase_signs{10}(third_point+1:2*third_point) = -1;
    
    % 11: Last third flipped
    expanded_phase_signs{11} = ones(1, number_of_modes);
    expanded_phase_signs{11}(2*third_point+1:end) = -1;
    
    % 12-16: Alternating groups of different sizes
    for g = 1:5
        expanded_phase_signs{11+g} = ones(1, number_of_modes);
        for i = 2:2*g:number_of_modes
            end_idx = min(i+g-1, number_of_modes);
            expanded_phase_signs{11+g}(i:end_idx) = -1;
        end
    end
    
    % 17-32: Pseudorandom patterns with different seeds
    for v = 17:num_expanded_variants
        rng(v);  % Different seed for each pattern
        expanded_phase_signs{v} = ones(1, number_of_modes);
        expanded_phase_signs{v}(2:end) = sign(randn(1, number_of_modes-1));
    end
else
    % For smaller mode counts, create all possible combinations
    for v = 1:num_expanded_variants
        if v <= num_variants
            expanded_phase_signs{v} = phase_signs{v};
        else
            % Create additional pseudorandom patterns
            rng(v);
            expanded_phase_signs{v} = ones(1, number_of_modes);
            expanded_phase_signs{v}(2:end) = sign(randn(1, number_of_modes-1));
        end
    end
end

% Select a representative sample from test set
sample_idx = 1;  % Using first base sample
base_weights = extract(mode_weights_test(sample_idx,:));  % Ensure it's on CPU

% Create pattern names for all expanded variants
expanded_pattern_names = cell(num_expanded_variants, 1);
for v = 1:num_expanded_variants
    if number_of_modes <= 10
        % Show full pattern for small mode counts
        pattern_str = sprintf('Pattern %d: [', v);
        for m = 1:number_of_modes
            if m > 1
                pattern_str = [pattern_str ' '];
            end
            pattern_str = [pattern_str sprintf('%2d', expanded_phase_signs{v}(m))];
        end
        expanded_pattern_names{v} = [pattern_str ']'];
    else
        % Abbreviated format for large mode counts
        neg_count = sum(expanded_phase_signs{v} < 0);
        pattern_str = sprintf('Pattern %d: %d neg signs', v, neg_count);
        expanded_pattern_names{v} = pattern_str;
    end
end

% Create matrix to store all images for correlation analysis
all_images = zeros(image_size, image_size, num_expanded_variants);

% Generate images for all variants
for v = 1:num_expanded_variants
    % Apply sign pattern to phases
    amps = abs(base_weights);
    phases = angle(base_weights);
    phases = phases .* expanded_phase_signs{v};
    modified_weights = amps .* exp(1i * phases);
    
    % Generate image
    [img, ~] = mmf_build_image(number_of_modes, image_size, 1, modified_weights, true, 1e0, P);
    img = extract(img);
    all_images(:,:,v) = squeeze(img);
end

%% Create optimized visualization of phase sign patterns
% Determine optimal grid layout based on number of variants
rows = floor(sqrt(num_expanded_variants));
cols = ceil(num_expanded_variants/rows);

% Create multi-page figure if needed
num_plots_per_page = 20;
num_pages = ceil(num_expanded_variants/num_plots_per_page);

for page = 1:num_pages
    % Calculate indices for this page
    start_idx = (page-1)*num_plots_per_page + 1;
    end_idx = min(page*num_plots_per_page, num_expanded_variants);
    count = end_idx - start_idx + 1;
    
    % Create new figure
    fig = figure('Name', sprintf('Phase Sign Patterns (Page %d/%d)', page, num_pages), ...
                'Position', [100 100 1200 900]);
    
    % Calculate grid size for this page
    page_rows = floor(sqrt(count));
    page_cols = ceil(count/page_rows);
    
    % Display patterns for this page
    for i = 1:count
        v = start_idx + i - 1;
        subplot(page_rows, page_cols, i);
        imagesc(all_images(:,:,v));
        axis image off;
        title(expanded_pattern_names{v}, 'Interpreter', 'none', 'FontSize', 8);
    end
    
    % Add overall title
    sgtitle(sprintf('Phase Sign Pattern Effects (Patterns %d-%d)', start_idx, end_idx), 'FontSize', 14);
    
    % Save figure
    saveas(fig, sprintf('phase_sign_patterns_page%d.png', page));
    fprintf('Saved phase sign pattern page %d/%d\n', page, num_pages);
end

%% Enhanced correlation analysis with optimized visualization
fprintf('\nPerforming enhanced correlation analysis...\n');

% Calculate correlation matrix between all pattern pairs
correlation_matrix = zeros(num_expanded_variants);
for i = 1:num_expanded_variants
    for j = 1:num_expanded_variants
        correlation_matrix(i,j) = corr2(all_images(:,:,i), all_images(:,:,j));
    end
end

% Create better visualization of correlation matrix
figure('Name', 'Pattern Correlation Matrix', 'Position', [100 100 1000 900]);

% Plot correlation matrix with improved visualization
imagesc(correlation_matrix);
colormap(jet);
h = colorbar;
ylabel(h, 'Correlation');
title('Correlation Between Different Phase Sign Patterns', 'FontSize', 14);

% Add grid for better readability with many patterns
hold on;
for i = 1.5:1:num_expanded_variants
    plot([0.5, num_expanded_variants+0.5], [i, i], 'k-', 'LineWidth', 0.2);
    plot([i, i], [0.5, num_expanded_variants+0.5], 'k-', 'LineWidth', 0.2);
end
hold off;

% Add axis labels
xlabel('Pattern Index');
ylabel('Pattern Index');

% Add ticks but limit for readability
if num_expanded_variants <= 16
    % Show all ticks for smaller numbers
    xticks(1:num_expanded_variants);
    yticks(1:num_expanded_variants);
else
    % Show fewer ticks for larger numbers
    tick_interval = 4;
    xticks(1:tick_interval:num_expanded_variants);
    yticks(1:tick_interval:num_expanded_variants);
end

% Save correlation matrix
saveas(gcf, 'phase_sign_correlations_expanded.png');
fprintf('Pattern correlation visualization saved\n');


%% Analyze correlation distribution
fprintf('\nAnalyzing correlation distribution...\n');

% Extract upper triangle of correlation matrix (excluding diagonal)
upper_corr = correlation_matrix(triu(true(size(correlation_matrix)), 1));

% Create histogram of correlations
figure('Name', 'Correlation Distribution', 'Position', [100 100 800 600]);
histogram(upper_corr, 20, 'Normalization', 'probability');
title('Distribution of Pattern Correlations', 'FontSize', 14);
xlabel('Correlation Value');
ylabel('Probability');
grid on;

% Add statistics
mean_corr = mean(upper_corr);
median_corr = median(upper_corr);
min_corr = min(upper_corr);
max_corr = max(upper_corr);

text(0.05, 0.95, sprintf('Mean: %.4f\nMedian: %.4f\nMin: %.4f\nMax: %.4f', ...
    mean_corr, median_corr, min_corr, max_corr), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', ...
    'BackgroundColor', [1 1 1 0.7], 'EdgeColor', 'k');

% Save correlation distribution
saveas(gcf, 'phase_sign_correlation_distribution.png');
fprintf('Correlation distribution analysis saved\n');

