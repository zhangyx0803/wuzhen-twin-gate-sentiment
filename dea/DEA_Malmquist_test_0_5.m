%% DEA-Malmquist analysis of the tourism resource conversion efficiency of Wuzhen East Gate and West Gate
% Version: 0.5
% Extended malmquist_index so that all four distance functions are computed under both VRS and CRS.
% New outputs: pure technical efficiency change (PEC) and scale efficiency change (SEC).
% Improved how the output data are displayed
% Updated the main loop to store PEC_matrix and SEC_matrix.
% Added a per-region (DMU) summary table that takes the geometric mean of each indicator. Output columns: efficiency change, technical progress, pure technical efficiency, scale efficiency, total factor productivity.
% Added a per-period (year range) summary table that takes the geometric mean over all DMUs. Reports every period plus the overall average.
% Adjusted: dea_distance now supports both index-based and vector-based calls (through varargin).
% Note: this code is written for efficiency analysis of two decision-making units (DMUs) over multiple time points
%       MATLAB Optimization Toolbox and Statistics and Machine Learning Toolbox must be installed before use

clc; clear; close all;

%% 1. Select the Excel file
[file, path] = uigetfile('*.xlsx', '请选择包含数据的 Excel 文件');
if file == 0
    error('未选择文件，程序退出。');
end
fullpath = fullfile(path, file);
fprintf('已读取文件：%s\n\n', fullpath);

% Read the whole worksheet (the first row is assumed to hold the column names)
dataTable = readtable(fullpath, 'VariableNamingRule', 'preserve');
colNames = dataTable.Properties.VariableNames;
fprintf('数据表中的列名如下：\n');
for i = 1:length(colNames)
    fprintf('  %2d. %s\n', i, colNames{i});
end

%% 2. User input: assign a purpose to each column
fprintf('\n请根据上面的列号，指定以下信息：\n');

% 2.1 DMU name column
dmuColIdx = input('请输入 DMU 名称所在的列号：');
dmuRaw = dataTable{:, dmuColIdx};
dmuNames = string(dmuRaw);
uniqueDMU = unique(dmuNames);
fprintf('检测到 %d 个不同的 DMU：%s\n', length(uniqueDMU), strjoin(uniqueDMU, ', '));

% 2.2 Year column
yearColIdx = input('请输入 年份 所在的列号：');
yearsRaw = dataTable{:, yearColIdx};
if isnumeric(yearsRaw)
    years = yearsRaw;
else
    years = str2double(string(yearsRaw));   % Convert the text to numbers
end
uniqueYears = unique(years);
uniqueYears = sort(uniqueYears);
fprintf('检测到年份：%s\n', mat2str(uniqueYears));

% 2.3 Input indicator columns (multiple selections allowed)
inputCols = input('请输入投入指标所在的列号（向量形式，如 [3,4]）：');
if isempty(inputCols)
    error('至少需要选择一个投入指标');
end
nInput = length(inputCols);
fprintf('选择了 %d 个投入指标：%s\n', nInput, strjoin(colNames(inputCols), ', '));

% 2.4 Output indicator columns (multiple selections allowed)
outputCols = input('请输入产出指标所在的列号（向量形式，如 [5,6]）：');
if isempty(outputCols)
    error('至少需要选择一个产出指标');
end
nOutput = length(outputCols);
fprintf('选择了 %d 个产出指标：%s\n', nOutput, strjoin(colNames(outputCols), ', '));

% 2.5 Model parameters (optional)
fprintf('\n');
orientation = input('模型导向（input / output，默认 input）：', 's');
if isempty(orientation), orientation = 'input'; end
RTS = input('规模报酬假设（crs / vrs，默认 vrs）：', 's');
if isempty(RTS), RTS = 'vrs'; end

%% 3. Reshape the data into panel format (year × DMU × indicator)
T = length(uniqueYears);      % Number of time points
K = length(uniqueDMU);        % Number of DMUs

% Preallocate the matrices: X_panel(T, K, nInput), Y_panel(T, K, nOutput)
X_panel = NaN(T, K, nInput);
Y_panel = NaN(T, K, nOutput);

% Map years and DMUs to indices
yearMap = containers.Map(uniqueYears, 1:T);
dmuMap = containers.Map(uniqueDMU, 1:K);

% Loop over every row of the data table
for row = 1:height(dataTable)
    dmuName = dmuNames(row);
    yr = years(row);
    % Check the row is valid
    if ~yearMap.isKey(yr)
        warning('第 %d 行：年份 %d 不在唯一年份列表中，已跳过', row, yr);
        continue;
    end
    if ~dmuMap.isKey(dmuName)
        warning('第 %d 行：DMU 名称 %s 不在唯一 DMU 列表中，已跳过', row, dmuName);
        continue;
    end
    t = yearMap(yr);
    k = dmuMap(dmuName);
    
    % Extract the input values
    for i = 1:nInput
        X_panel(t, k, i) = dataTable{row, inputCols(i)};
    end
    % Extract the output values
    for o = 1:nOutput
        Y_panel(t, k, o) = dataTable{row, outputCols(o)};
    end
end

% Check for missing data
if any(isnan(X_panel(:))) || any(isnan(Y_panel(:)))
    warning('数据中存在缺失值（NaN）。请确保每个 DMU 在每个年份都有完整数据。');
    % Simple handling: drop DMU-year combinations that contain NaN (only a warning is issued here, nothing is deleted automatically)
end

%% 4. Call the Malmquist engine (uses the two functions defined below)
% Initialise the result matrices
MI_matrix = zeros(K, T-1);
EC_matrix = zeros(K, T-1);
TC_matrix = zeros(K, T-1);
PEC_matrix = zeros(K, T-1);
SEC_matrix = zeros(K, T-1);

% Compute the Malmquist index year by year
for t = 1:T-1
    % Extract the input and output matrices of periods t and t+1 (shape: K × number of indicators)
    X_t = squeeze(X_panel(t, :, :));   % K × nInput
    Y_t = squeeze(Y_panel(t, :, :));   % K × nOutput
    X_t1 = squeeze(X_panel(t+1, :, :));
    Y_t1 = squeeze(Y_panel(t+1, :, :));
    
    % Compute the Malmquist index
    [MI, EC, TC, PEC, SEC] = malmquist_index(X_t, Y_t, X_t1, Y_t1, orientation);
    MI_matrix(:, t) = MI;
    EC_matrix(:, t) = EC;
    TC_matrix(:, t) = TC;
    PEC_matrix(:, t) = PEC;
    SEC_matrix(:, t) = SEC;
end

%% 5. Print the result tables
% Build the year labels
periods = cell(1, T-1);
for i = 1:T-1
    periods{i} = sprintf('%d→%d', uniqueYears(i), uniqueYears(i+1));
end

%% ==================== Collect all tables into a tabbed window ====================
% Create a main window
fig_main = figure('Name', 'Malmquist-DEA 分析结果', 'NumberTitle', 'off', ...
    'Position', [100 100 1100 650]);

% Create the tab group
tabgroup = uitabgroup(fig_main, 'Position', [0.02 0.02 0.96 0.96]);

% Helper function: add a table to a tab (shared position and size)
add_table_to_tab = @(parent, data, row_names, col_names, title_str) ...
    uitable(parent, 'Data', data, 'ColumnName', col_names, 'RowName', row_names, ...
    'Position', [20 20 1040 570], 'FontSize', 11);

% ----- 1. Malmquist index (MI) tab -----
tab_MI = uitab(tabgroup, 'Title', 'Malmquist指数 (MI)');
add_table_to_tab(tab_MI, MI_matrix, uniqueDMU, periods, 'MI');

% ----- 2. Technical efficiency change (EC) tab -----
tab_EC = uitab(tabgroup, 'Title', '技术效率变化 (EC)');
add_table_to_tab(tab_EC, EC_matrix, uniqueDMU, periods, 'EC');

% ----- 3. Technical progress change (TC) tab -----
tab_TC = uitab(tabgroup, 'Title', '技术进步变化 (TC)');
add_table_to_tab(tab_TC, TC_matrix, uniqueDMU, periods, 'TC');

% ----- 4. Pure technical efficiency change (PEC) tab (if available) -----
if exist('PEC_matrix', 'var') && ~isempty(PEC_matrix)
    tab_PEC = uitab(tabgroup, 'Title', '纯技术效率变化 (PEC)');
    add_table_to_tab(tab_PEC, PEC_matrix, uniqueDMU, periods, 'PEC');
end

% ----- 5. Scale efficiency change (SEC) tab (if available) -----
if exist('SEC_matrix', 'var') && ~isempty(SEC_matrix)
    tab_SEC = uitab(tabgroup, 'Title', '规模效率变化 (SEC)');
    add_table_to_tab(tab_SEC, SEC_matrix, uniqueDMU, periods, 'SEC');
end

% ----- 6. Per-region summary tab -----
tab_region = uitab(tabgroup, 'Title', '按地区汇总');

% Define the geometric mean function
geo_mean = @(x) exp(mean(log(x), 2));

% Geometric means per region
EC_geo = geo_mean(EC_matrix);
TC_geo = geo_mean(TC_matrix);
PEC_geo = geo_mean(PEC_matrix);
SEC_geo = geo_mean(SEC_matrix);
MI_geo = geo_mean(MI_matrix);
% Prepare the data (including the mean row)

data_region = [EC_geo, TC_geo, PEC_geo, SEC_geo, MI_geo];
data_region = [data_region; mean(EC_geo), mean(TC_geo), mean(PEC_geo), mean(SEC_geo), mean(MI_geo)];
row_names_region = [uniqueDMU; {'平均'}];
col_names_region = {'效率变动', '技术进步', '纯技术效率', '规模效率', '全要素生产率'};
uitable(tab_region, 'Data', data_region, 'ColumnName', col_names_region, 'RowName', row_names_region, ...
    'Position', [20 20 1040 570], 'FontSize', 11);

% ----- 7. Per-period summary tab -----
tab_time = uitab(tabgroup, 'Title', '按时间汇总');
% Geometric mean over all DMUs for each period
T_period = length(periods);
data_time = zeros(T_period, 5);
for t = 1:T_period
% Define the scalar geometric mean (averages all elements of the vector)
geo_scalar = @(x) exp(mean(log(x)));

for t = 1:T_period
    data_time(t, :) = [geo_scalar(EC_matrix(:,t)), geo_scalar(TC_matrix(:,t)), ...
                       geo_scalar(PEC_matrix(:,t)), geo_scalar(SEC_matrix(:,t)), ...
                       geo_scalar(MI_matrix(:,t))];
end
end
% Append the overall mean row
data_time = [data_time; mean(EC_geo), mean(TC_geo), mean(PEC_geo), mean(SEC_geo), mean(MI_geo)];
row_names_time = [periods, {'2016-2020'}];
col_names_time = {'效率变动', '技术进步', '纯技术效率', '规模效率', '全要素生产率'};
uitable(tab_time, 'Data', data_time, 'ColumnName', col_names_time, 'RowName', row_names_time, ...
    'Position', [20 20 1040 570], 'FontSize', 11);

%% 6. Plotting (multi-chart analysis)
year_ticks = uniqueYears(2:end);      % End-of-period years (used as the x-axis of the trend chart)
period_labels = year_ticks;           % The bar chart uses the years directly as labels (e.g. 2019, 2020, ...)

% 6.1 Basic trend chart
figure('Name', 'Malmquist 指数变化趋势', 'NumberTitle', 'off');
plot(year_ticks, MI_matrix', 'o-', 'LineWidth', 2);
xlabel('年份', 'FontSize', 12);
ylabel('Malmquist 指数', 'FontSize', 12);
title('全要素生产率变化趋势', 'FontSize', 14);
legend(uniqueDMU, 'Location', 'best');
grid on;
% Force the x-axis settings
ax1 = gca;
ax1.XTick = year_ticks;
ax1.XTickLabel = year_ticks;
ax1.XLim = [min(year_ticks)-0.5, max(year_ticks)+0.5]; %padding

% 6.2 Grouped bar chart
figure('Name', '分组柱状图 - MI 对比', 'NumberTitle', 'off');
bar(MI_matrix');
% Use single years as the x-axis labels
set(gca, 'XTickLabel', period_labels);
% Add left and right padding (half a bar width on each side)
n_periods = length(period_labels);
xlim([0.5, n_periods + 0.5]);
xlabel('时期（期末年份）', 'FontSize', 12);
ylabel('Malmquist 指数', 'FontSize', 12);
title('各景区 Malmquist 指数对比', 'FontSize', 14);
legend(uniqueDMU, 'Location', 'best');
grid on;

% 6.3 Subplots: decomposition trends of MI, EC and TC
% Define the year ticks
year_ticks = uniqueYears(2:end);

% Give every DMU a fixed colour (from the lines colormap)
num_dmu = length(uniqueDMU);
color_order = lines(num_dmu);  % Each row is an RGB colour vector

% Create the figure window
figure('Name', '分解指标趋势', 'NumberTitle', 'off');

% ================== Subplot 1: total factor productivity (MI) ==================
subplot(3,1,1);
hold on;
for i = 1:num_dmu
    plot(year_ticks, MI_matrix(i, :), 'o-', 'LineWidth', 1.5, ...
        'Color', color_order(i, :), 'DisplayName', uniqueDMU{i});
end
hold off;
ylabel('MI');
title('全要素生产率变化');
legend('Location', 'best', 'FontSize', 9);
grid on;
xticks(year_ticks);
xticklabels(year_ticks);
xlim([min(year_ticks)-0.5, max(year_ticks)+0.5]);

% ================== Subplot 2: technical efficiency change (EC) ==================
subplot(3,1,2);
hold on;
for i = 1:num_dmu
    plot(year_ticks, EC_matrix(i, :), 's-', 'LineWidth', 1.5, ...
        'Color', color_order(i, :), 'DisplayName', uniqueDMU{i});
end
hold off;
ylabel('EC');
title('技术效率变化');
legend('Location', 'best', 'FontSize', 9);
grid on;
xticks(year_ticks);
xticklabels(year_ticks);
xlim([min(year_ticks)-0.5, max(year_ticks)+0.5]);

% ================== Subplot 3: technical progress change (TC) ==================
subplot(3,1,3);
hold on;
for i = 1:num_dmu
    plot(year_ticks, TC_matrix(i, :), 'd-', 'LineWidth', 1.5, ...
        'Color', color_order(i, :), 'DisplayName', uniqueDMU{i});
end
hold off;
ylabel('TC');
title('技术进步变化');
legend('Location', 'best', 'FontSize', 9);
grid on;
xticks(year_ticks);
xticklabels(year_ticks);
xlim([min(year_ticks)-0.5, max(year_ticks)+0.5]);
xlabel('年份', 'FontSize', 12);

% 6.4 Heat map (the Statistics Toolbox is not needed, imagesc is a base function)
figure('Name', 'MI 热力图', 'NumberTitle', 'off');
imagesc(MI_matrix);          % Note: the transpose was removed, the raw matrix is used directly
colorbar;
colormap(jet);               % parula or turbo would be a more sound choice
% Set the axis ticks and labels
set(gca, 'XTick', 1:length(period_labels), 'XTickLabel', period_labels);
set(gca, 'YTick', 1:length(uniqueDMU), 'YTickLabel', uniqueDMU);
xlabel('时期（期末年份）', 'FontSize', 12);
ylabel('景区', 'FontSize', 12);
title('Malmquist 指数热图', 'FontSize', 14);

% 6.5 Composition of MI (MI = EC*TC)
mean_EC = mean(EC_matrix, 2);
mean_TC = mean(TC_matrix, 2);
figure('Name', 'MI 分解构成', 'NumberTitle', 'off');
bar([mean_EC, mean_TC], 'stacked');
set(gca, 'XTickLabel', uniqueDMU);
xlabel('景区', 'FontSize', 12);
ylabel('平均指数', 'FontSize', 12);
title('全要素生产率变化来源分解', 'FontSize', 14);
legend({'技术效率变化 (EC)', '技术进步变化 (TC)'}, 'Location', 'best');
grid on;

fprintf('\n=== 计算完成 ===\n');

%% =========================================================================
% The core functions for the DEA distance function and the Malmquist index follow
% =========================================================================

function distance = dea_distance(X, Y, varargin)
    % Compute the distance function (input- or output-oriented)
    % Usage 1: distance = dea_distance(X, Y, idx_dmu, orientation, RTS)
    %       evaluate DMU number idx_dmu of the reference set X, Y
    % Usage 2: distance = dea_distance(X, Y, x0, y0, orientation, RTS)
    %       pass the input vector x0 (m×1) and output vector y0 (s×1) of the evaluated DMU directly
    
    [n, m] = size(X);   % n DMUs, m inputs
    [~, s] = size(Y);   % s outputs
    
    % Parse the input arguments
    if nargin == 5  % Usage 1: idx_dmu is passed in
        idx_dmu = varargin{1};
        orientation = varargin{2};
        RTS = varargin{3};
        x0 = X(idx_dmu, :)';
        y0 = Y(idx_dmu, :)';
    elseif nargin == 6  % Usage 2: the vectors x0 and y0 are passed in
        x0 = varargin{1}(:);
        y0 = varargin{2}(:);
        orientation = varargin{3};
        RTS = varargin{4};
    else
        error('参数数目不正确，请提供 idx_dmu 或 (x0, y0)');
    end
    
    % Decision variables: [theta; lambda_1; ...; lambda_n], n+1 in total
    n_vars = n + 1;
    lb = [0; zeros(n, 1)];
    
    if strcmp(orientation, 'input')
        % Input-oriented: minimise theta
        f = [1; zeros(n, 1)];
        
        % Input constraint: sum_j lambda_j * x_j <= theta * x0
        A_in = zeros(m, n_vars);
        for i = 1:m
            A_in(i, 1) = -x0(i);
            A_in(i, 2:end) = X(:, i)';
        end
        b_in = zeros(m, 1);
        
        % Output constraint: sum_j lambda_j * y_j >= y0
        A_out = zeros(s, n_vars);
        for r = 1:s
            A_out(r, 1) = 0;
            A_out(r, 2:end) = -Y(:, r)';
        end
        b_out = -y0;
        
        A = [A_in; A_out];
        b = [b_in; b_out];
        
    else % Output-oriented
        f = [-1; zeros(n, 1)];
        % Input constraint: sum_j lambda_j * x_j <= x0
        A_in = zeros(m, n_vars);
        for i = 1:m
            A_in(i, 1) = 0;
            A_in(i, 2:end) = X(:, i)';
        end
        b_in = x0;
        % Output constraint: sum_j lambda_j * y_j >= phi * y0
        A_out = zeros(s, n_vars);
        for r = 1:s
            A_out(r, 1) = y0(r);
            A_out(r, 2:end) = -Y(:, r)';
        end
        b_out = zeros(s, 1);
        
        A = [A_in; A_out];
        b = [b_in; b_out];
    end
    
    % Returns-to-scale assumption
    if strcmp(RTS, 'vrs')
        Aeq = [0, ones(1, n)];
        beq = 1;
    else
        Aeq = [];
        beq = [];
    end
    
    options = optimoptions('linprog', 'Display', 'off', 'Algorithm', 'dual-simplex');
    [sol, ~, exitflag] = linprog(f, A, b, Aeq, beq, lb, [], options);
    
    if exitflag > 0
        if strcmp(orientation, 'input')
            distance = sol(1);
        else
            distance = 1 / sol(1);
        end
    else
        distance = NaN;
        warning('线性规划求解失败，exitflag = %d', exitflag);
    end
end

function [MI, EC, TC, PEC, SEC] = malmquist_index(X_t, Y_t, X_t1, Y_t1, orientation)
    % Compute the Malmquist index and its decomposition (including PEC and SEC)
    % Inputs: X_t, Y_t (period t), X_t1, Y_t1 (period t+1), orientation ('input' or 'output')
    % Outputs: MI, EC, TC, PEC, SEC
    n = size(X_t, 1);
    
    % Distances under VRS
    Dv_t_t = zeros(n,1); Dv_t_t1 = zeros(n,1);
    Dv_t1_t = zeros(n,1); Dv_t1_t1 = zeros(n,1);
    % Distances under CRS
    Dc_t_t = zeros(n,1); Dc_t_t1 = zeros(n,1);
    Dc_t1_t = zeros(n,1); Dc_t1_t1 = zeros(n,1);
    
    for i = 1:n
        % VRS distances
        Dv_t_t(i) = dea_distance(X_t, Y_t, i, orientation, 'vrs');
        Dv_t_t1(i) = dea_distance(X_t, Y_t, X_t1(i,:)', Y_t1(i,:)', orientation, 'vrs');
        Dv_t1_t(i) = dea_distance(X_t1, Y_t1, X_t(i,:)', Y_t(i,:)', orientation, 'vrs');
        Dv_t1_t1(i) = dea_distance(X_t1, Y_t1, i, orientation, 'vrs');
        
        % CRS distances
        Dc_t_t(i) = dea_distance(X_t, Y_t, i, orientation, 'crs');
        Dc_t_t1(i) = dea_distance(X_t, Y_t, X_t1(i,:)', Y_t1(i,:)', orientation, 'crs');
        Dc_t1_t(i) = dea_distance(X_t1, Y_t1, X_t(i,:)', Y_t(i,:)', orientation, 'crs');
        Dc_t1_t1(i) = dea_distance(X_t1, Y_t1, i, orientation, 'crs');
    end
    
    % Pure technical efficiency change (PEC)
    PEC = Dv_t1_t1 ./ Dv_t_t;
    % Scale efficiency change (SEC)
    SEC = (Dc_t1_t1 ./ Dv_t1_t1) ./ (Dc_t_t ./ Dv_t_t);
    % Technical efficiency change (EC) = PEC * SEC
    EC = PEC .* SEC;
    % Technical progress (TC) based on CRS
    TC = sqrt( (Dc_t_t ./ Dc_t1_t) .* (Dc_t_t1 ./ Dc_t1_t1) );
    % Total factor productivity change (MI)
    MI = EC .* TC;
end