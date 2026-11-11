%% DEA-Malmquist analysis of the tourism resource conversion efficiency of Wuzhen East Gate and West Gate
% Version: 0.2
% Updated the data import module so that data can be read from an xlsx workbook
% Note: this code is written for efficiency analysis of two decision-making units (DMUs) over multiple time points
%       MATLAB Optimization Toolbox must be installed before use

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

% Compute the Malmquist index year by year
for t = 1:T-1
    % Extract the input and output matrices of periods t and t+1 (shape: K × number of indicators)
    X_t = squeeze(X_panel(t, :, :));   % K × nInput
    Y_t = squeeze(Y_panel(t, :, :));   % K × nOutput
    X_t1 = squeeze(X_panel(t+1, :, :));
    Y_t1 = squeeze(Y_panel(t+1, :, :));
    
    % Compute the Malmquist index
    [MI, EC, TC] = malmquist_index(X_t, Y_t, X_t1, Y_t1, orientation, RTS);
    MI_matrix(:, t) = MI;
    EC_matrix(:, t) = EC;
    TC_matrix(:, t) = TC;
end

%% 5. Print the result tables
% Build the year labels
periods = cell(1, T-1);
for i = 1:T-1
    periods{i} = sprintf('%d→%d', uniqueYears(i), uniqueYears(i+1));
end

fprintf('\n==================== Malmquist 指数 (MI) ====================\n');
fprintf('%-12s', 'DMU');
for p = 1:length(periods)
    fprintf('%-16s', periods{p});
end
fprintf('\n');
for k = 1:K
    fprintf('%-12s', uniqueDMU(k));
    for t = 1:T-1
        fprintf('%-16.4f', MI_matrix(k, t));
    end
    fprintf('\n');
end

fprintf('\n==================== 技术效率变化 (EC) ====================\n');
fprintf('%-12s', 'DMU');
for p = 1:length(periods)
    fprintf('%-16s', periods{p});
end
fprintf('\n');
for k = 1:K
    fprintf('%-12s', uniqueDMU(k));
    for t = 1:T-1
        fprintf('%-16.4f', EC_matrix(k, t));
    end
    fprintf('\n');
end

fprintf('\n==================== 技术进步变化 (TC) ====================\n');
fprintf('%-12s', 'DMU');
for p = 1:length(periods)
    fprintf('%-16s', periods{p});
end
fprintf('\n');
for k = 1:K
    fprintf('%-12s', uniqueDMU(k));
    for t = 1:T-1
        fprintf('%-16.4f', TC_matrix(k, t));
    end
    fprintf('\n');
end

fprintf('\n结果解读：\n');
fprintf('  MI > 1 表示全要素生产率提升；EC > 1 表示技术效率改善；TC > 1 表示技术进步。\n');
fprintf('  MI = EC × TC\n');

%% 6. Plot the trend chart
figure('Name', 'Malmquist 指数变化趋势', 'NumberTitle', 'off');
plot(uniqueYears(2:end), MI_matrix', 'o-', 'LineWidth', 2);
xlabel('年份（期末）', 'FontSize', 12);
ylabel('Malmquist 指数', 'FontSize', 12);
title('全要素生产率变化趋势', 'FontSize', 14);
legend(uniqueDMU, 'Location', 'best');
grid on;

fprintf('\n=== 计算完成 ===\n');

%% =========================================================================
% The core functions for the DEA distance function and the Malmquist index follow
% =========================================================================

function distance = dea_distance(X, Y, idx_dmu, orientation, RTS)
    % Distance function of a single DMU against a given reference set (input- or output-oriented)
    % Inputs:
    %   X - n×m matrix, n DMUs, m input indicators
    %   Y - n×s matrix, n DMUs, s output indicators
    %   idx_dmu - index of the DMU being evaluated
    %   orientation - 'input' or 'output'
    %   RTS - 'crs' or 'vrs'
    % Outputs:
    %   distance - value of the distance function (efficiency score)
    
    [n, m] = size(X);
    [~, s] = size(Y);
    x0 = X(idx_dmu, :)';   % m×1
    y0 = Y(idx_dmu, :)';   % s×1
    
    n_vars = n + 1;        % Decision variables: [theta; lambda_1...lambda_n]
    lb = [0; zeros(n, 1)]; % Variable lower bounds
    
    if strcmp(orientation, 'input')
        % Input-oriented: minimise theta
        f = [1; zeros(n, 1)];
        
        % Input constraint: sum_j lambda_j * x_ji <= theta * x0_i
        % i.e. -x0_i * theta + sum_j lambda_j * x_ji <= 0
        A_in = zeros(m, n_vars);
        for i = 1:m
            A_in(i, 1) = -x0(i);
            A_in(i, 2:end) = X(:, i)';
        end
        b_in = zeros(m, 1);
        
        % Output constraint: sum_j lambda_j * y_jr >= y0_r
        % i.e. -sum_j lambda_j * y_jr <= -y0_r
        A_out = zeros(s, n_vars);
        for r = 1:s
            A_out(r, 1) = 0;
            A_out(r, 2:end) = -Y(:, r)';
        end
        b_out = -y0;
        
        A = [A_in; A_out];
        b = [b_in; b_out];
        
    else % Output-oriented
        % Output-oriented: maximise phi, which is equivalent to minimising -phi
        f = [-1; zeros(n, 1)];
        
        % Input constraint: sum_j lambda_j * x_ji <= x0_i
        A_in = zeros(m, n_vars);
        for i = 1:m
            A_in(i, 1) = 0;
            A_in(i, 2:end) = X(:, i)';
        end
        b_in = x0;
        
        % Output constraint: sum_j lambda_j * y_jr >= phi * y0_r
        % i.e. phi*y0_r - sum_j lambda_j*y_jr <= 0
        A_out = zeros(s, n_vars);
        for r = 1:s
            A_out(r, 1) = y0(r);
            A_out(r, 2:end) = -Y(:, r)';
        end
        b_out = zeros(s, 1);
        
        A = [A_in; A_out];
        b = [b_in; b_out];
    end
    
    % Returns to scale: VRS needs the additional sum(lambda) = 1
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
        warning('第 %d 个 DMU 的线性规划求解失败，exitflag = %d', idx_dmu, exitflag);
    end
end

function [MI, EC, TC] = malmquist_index(X_t, Y_t, X_t1, Y_t1, orientation, RTS)
    % Compute the Malmquist index and its decomposition
    % Inputs:
    %   X_t, Y_t   - input and output matrices of period t (one DMU per row)
    %   X_t1, Y_t1 - input and output matrices of period t+1
    %   orientation, RTS - as in dea_distance
    % Outputs:
    %   MI - Malmquist index (total factor productivity change)
    %   EC - technical efficiency change
    %   TC - technical progress change
    
    n = size(X_t, 1);   % Number of DMUs
    D_t_t = zeros(n, 1);
    D_t_t1 = zeros(n, 1);
    D_t1_t = zeros(n, 1);
    D_t1_t1 = zeros(n, 1);
    
    for i = 1:n
        D_t_t(i) = dea_distance(X_t, Y_t, i, orientation, RTS);
        D_t_t1(i) = dea_distance(X_t, Y_t, i, orientation, RTS);
        D_t1_t(i) = dea_distance(X_t1, Y_t1, i, orientation, RTS);
        D_t1_t1(i) = dea_distance(X_t1, Y_t1, i, orientation, RTS);
    end
    
    EC = D_t1_t1 ./ D_t_t;
    TC = sqrt( (D_t_t ./ D_t1_t) .* (D_t_t1 ./ D_t1_t1) );
    MI = EC .* TC;
end