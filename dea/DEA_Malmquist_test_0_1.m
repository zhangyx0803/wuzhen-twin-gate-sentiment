%% DEA-Malmquist analysis of the tourism resource conversion efficiency of Wuzhen East Gate and West Gate
% Version: 0.1
% First test program! Here we go!
% Note: this code is written for efficiency analysis of two decision-making units (DMUs) over multiple time points
%       MATLAB Optimization Toolbox must be installed before use

clc;        % Clear the command window
clear;      % Clear the workspace variables
close all;  % Close all open figure windows

%% ==================== Section the user has to edit ====================
% 1. Adapt the input/output indicator names and data matrices below to your actual data
% 2. The input/output orientation and the returns-to-scale assumption (RTS) can be changed to suit the study

%% Data input example (2018-2020, two scenic areas)
% Note: the actual data must be entered into the matrices below
% Rows are years (T years) and columns are decision-making units (DMUs); there are 2 DMUs here

% Input indicators (the number of indicators can be added to or reduced as needed; two are used here as an example)
% Indicator definitions: number of employees (persons), operating cost (10k CNY)
% Format: each indicator is a T×2 matrix, T being the number of time points and 2 the number of scenic areas (East Gate, West Gate)
X1 = [120 150;   % 2018: East Gate 120 persons, West Gate 150 persons
      125 155;   % 2019: East Gate 125 persons, West Gate 155 persons
      130 160];  % 2020: East Gate 130 persons, West Gate 160 persons

X2 = [2000 2500;  % 2018: East Gate 2000 (10k CNY), West Gate 2500 (10k CNY)
      2100 2600;  % 2019: East Gate 2100 (10k CNY), West Gate 2600 (10k CNY)
      2200 2700]; % 2020: East Gate 2200 (10k CNY), West Gate 2700 (10k CNY)

% Output indicators (the number of indicators can be added to or reduced as needed; two are used here as an example)
% Indicator definitions: visitor volume (10k visits), operating revenue (million CNY)
Y1 = [80 100;    % 2018: East Gate 80 (10k visits), West Gate 100 (10k visits)
      85 110;    % 2019: East Gate 85 (10k visits), West Gate 110 (10k visits)
      90 120];   % 2020: East Gate 90 (10k visits), West Gate 120 (10k visits)

Y2 = [50 70;     % 2018: East Gate 50 (million CNY), West Gate 70 (million CNY)
      55 75;     % 2019: East Gate 55 (million CNY), West Gate 75 (million CNY)
      60 80];    % 2020: East Gate 60 (million CNY), West Gate 80 (million CNY)

% ==================== Parameter settings ====================
% Model parameters (the user may change them as the study requires)
orientation = 'input';   % 'input' = input-oriented, 'output' = output-oriented
RTS = 'vrs';             % 'crs' = constant returns to scale, 'vrs' = variable returns to scale

%% Build the panel data matrices
% Merge the multi-year data into panel format in chronological order
X_panel = cat(3, X1, X2);  % Merge the input indicators into a 3-D array
Y_panel = cat(3, Y1, Y2);  % Merge the output indicators into a 3-D array
% X_panel(i, j, k): i = year, j = scenic area (1 East Gate, 2 West Gate), k = input indicator
% Y_panel(i, j, k): i = year, j = scenic area (1 East Gate, 2 West Gate), k = output indicator

%% ==================== Core algorithm functions ====================

%% Define the DEA distance function (solved by linear programming)
function distance = dea_distance(X, Y, idx_dmu, orientation, RTS)
    % X: input matrix (n×m), n DMUs, m input indicators
    % Y: output matrix (n×s), n DMUs, s output indicators
    % idx_dmu: index of the DMU currently being evaluated
    % orientation: 'input' or 'output'
    % RTS: 'crs' or 'vrs'
    % distance: efficiency score (theta when input-oriented, 1/phi when output-oriented)
    
    [n, m] = size(X);
    [~, s] = size(Y);
    
    % Input and output vectors of the current DMU
    x0 = X(idx_dmu, :)';   % m×1
    y0 = Y(idx_dmu, :)';   % s×1
    
    % Decision variables: [theta; lambda_1; ...; lambda_n], n+1 in total
    f = [1; zeros(n, 1)];   % Input-oriented: minimise theta
    % Variable lower bounds: theta >= 0, lambda_j >= 0
    lb = [0; zeros(n, 1)];
    
    % Build the constraints according to the orientation
    if strcmp(orientation, 'input')
        % ----- Input constraint: sum_j lambda_j * x_j <= theta * x0  -----
        % For every input i: -x0(i) * theta + sum_j lambda_j * X(j,i) <= 0
        % Matrix form: A_ineq = [ -x0, X ]   (m rows, n+1 columns)
        A_ineq = [ -x0, X ];      % X is the raw n×m data - does it have to be transposed? Note: X comes in as n×m, one DMU per row and one input per column
                                   % Here, however, each row has to correspond to one input and each column to one DMU, so X' should be used.
                                   % Check carefully: X is n×m and row i is the input vector of DMU i (1×m).
                                   % When building the constraints, the values of input k over all DMUs form an n-dimensional column vector, i.e. X(:,k).
                                   % So the correct A_ineq should use [ -x0(k), X(:,k)' ] as one row. Assembled:
        % Fix: rebuild A_ineq as m×(n+1)
        A_ineq = zeros(m, n+1);
        for i = 1:m
            A_ineq(i, 1) = -x0(i);                % coefficient of theta
            A_ineq(i, 2:end) = X(:, i)';          % lambda coefficients (input i over all DMUs)
        end
        b_ineq = zeros(m, 1);
        
        % ----- Output constraint: sum_j lambda_j * y_j >= y0  -----
        % Rewritten as <= : -sum_j lambda_j * y_j <= -y0
        % For every output r: 0*theta + sum_j lambda_j * (-Y(j,r)) <= -y0(r)
        A_ineq2 = zeros(s, n+1);
        for r = 1:s
            A_ineq2(r, 1) = 0;                    % theta coefficient is 0
            A_ineq2(r, 2:end) = -Y(:, r)';        % lambda coefficients are -y_jr
        end
        b_ineq2 = -y0;
        
        % Concatenate the inequality constraints
        A = [A_ineq; A_ineq2];
        b = [b_ineq; b_ineq2];
        
    else  % Output-oriented
        % Output-oriented: max phi, which is equivalent to min -phi
        f = [-1; zeros(n, 1)];
        % Constraint: sum_j lambda_j * x_j <= x0
        A_ineq = zeros(m, n+1);
        for i = 1:m
            A_ineq(i, 1) = 0;                     % theta (phi) coefficient is 0
            A_ineq(i, 2:end) = X(:, i)';          % lambda coefficients
        end
        b_ineq = x0;
        
        % Output constraint: sum_j lambda_j * y_j >= phi * y0
        % Rewritten as: -sum_j lambda_j * y_j + phi * y0 <= 0
        A_ineq2 = zeros(s, n+1);
        for r = 1:s
            A_ineq2(r, 1) = y0(r);                % phi coefficient is the positive value y0(r)
            A_ineq2(r, 2:end) = -Y(:, r)';        % lambda coefficients are -y_jr
        end
        b_ineq2 = zeros(s, 1);
        
        A = [A_ineq; A_ineq2];
        b = [b_ineq; b_ineq2];
    end
    
    % Returns to scale: VRS needs the additional sum(lambda) = 1
    if strcmp(RTS, 'vrs')
        Aeq = [0, ones(1, n)];   % theta coefficient is 0, all lambda coefficients are 1
        beq = 1;
    else  % CRS: no equality constraint
        Aeq = [];
        beq = [];
    end
    
    % Solve the linear program
    options = optimoptions('linprog', 'Display', 'off', 'Algorithm', 'dual-simplex');
    [sol, ~, exitflag] = linprog(f, A, b, Aeq, beq, lb, [], options);
    
    if exitflag > 0
        if strcmp(orientation, 'input')
            distance = sol(1);      % theta
        else
            distance = 1 / sol(1);  % Output-oriented: efficiency score = 1/phi
        end
    else
        distance = NaN;
        warning('第%d个DMU的线性规划求解失败，exitflag=%d', idx_dmu, exitflag);
    end
end

%% Compute the Malmquist index and its decomposition
function [MI, EC, TC] = malmquist_index(X_t, Y_t, X_t1, Y_t1, orientation, RTS)
    % X_t, Y_t: input and output matrices of period t
    % X_t1, Y_t1: input and output matrices of period t+1
    % orientation: 'input' or 'output'
    % RTS: 'crs' or 'vrs'
    % MI: Malmquist index (total factor productivity change)
    % EC: technical efficiency change (Effch)
    % TC: technical progress change (Techch)
    
    [n, ~] = size(X_t);  % Number of DMUs
    
    % Initialise the result vectors
    D_t_t = zeros(n, 1);      % D^t(x^t, y^t)
    D_t_t1 = zeros(n, 1);     % D^t(x^{t+1}, y^{t+1})
    D_t1_t = zeros(n, 1);     % D^{t+1}(x^t, y^t)
    D_t1_t1 = zeros(n, 1);    % D^{t+1}(x^{t+1}, y^{t+1})
    
    % Compute the individual distance functions
    for i = 1:n
        % Distance of the period-t DMUs under period-t technology
        D_t_t(i) = dea_distance(X_t, Y_t, i, orientation, RTS);
        % Distance of the period-(t+1) DMUs under period-t technology
        D_t_t1(i) = dea_distance(X_t, Y_t, i, orientation, RTS);
        % Distance of the period-t DMUs under period-(t+1) technology
        D_t1_t(i) = dea_distance(X_t1, Y_t1, i, orientation, RTS);
        % Distance of the period-(t+1) DMUs under period-(t+1) technology
        D_t1_t1(i) = dea_distance(X_t1, Y_t1, i, orientation, RTS);
    end
    
    % Compute the Malmquist index and its decomposition
    % Technical efficiency change EC = D^{t+1}(x^{t+1}, y^{t+1}) / D^t(x^t, y^t)
    EC = D_t1_t1 ./ D_t_t;
    
    % Technical progress change TC = sqrt( (D^t(x^t, y^t)/D^{t+1}(x^t, y^t)) * (D^t(x^{t+1}, y^{t+1})/D^{t+1}(x^{t+1}, y^{t+1})) )
    TC = sqrt( (D_t_t ./ D_t1_t) .* (D_t_t1 ./ D_t1_t1) );
    
    % Malmquist index MI = EC * TC
    MI = EC .* TC;
end

%% ==================== Main computation ====================
% Get the number of time points and the number of DMUs
[T, n_dmu, ~] = size(X_panel);  % T: number of years, n_dmu: number of scenic areas (2)
n_input = size(X_panel, 3);     % Number of input indicators
n_output = size(Y_panel, 3);     % Number of output indicators

% Display basic information about the data
fprintf('\n=== 乌镇景区Malmquist-DEA分析 ===\n');
fprintf('时间点数: %d\n', T);
fprintf('景区数量: %d (1-东栅, 2-西栅)\n', n_dmu);
fprintf('投入指标数: %d\n', n_input);
fprintf('产出指标数: %d\n', n_output);
fprintf('模型导向: %s\n', orientation);
fprintf('规模报酬假设: %s\n', RTS);
fprintf('================================\n\n');

% Initialise the matrices that store the Malmquist index of each period
% Each row is a DMU and each column is the index from period t to period t+1
MI_matrix = zeros(n_dmu, T-1);
EC_matrix = zeros(n_dmu, T-1);
TC_matrix = zeros(n_dmu, T-1);

% Compute the Malmquist index year by year
for t = 1:T-1
    % Extract the input and output data of periods t and t+1
    X_t = squeeze(X_panel(t, :, :));      % Input matrix of period t
    Y_t = squeeze(Y_panel(t, :, :));      % Output matrix of period t
    X_t1 = squeeze(X_panel(t+1, :, :));   % Input matrix of period t+1
    Y_t1 = squeeze(Y_panel(t+1, :, :));   % Output matrix of period t+1
    
    % Compute the Malmquist index of the current period
    [MI, EC, TC] = malmquist_index(X_t, Y_t, X_t1, Y_t1, orientation, RTS);
    
    % Store the results
    MI_matrix(:, t) = MI;
    EC_matrix(:, t) = EC;
    TC_matrix(:, t) = TC;
end

%% ==================== Result output and visualisation ====================

%% Display the result tables
% Build the scenic-area names and the year labels
dmu_names = {'东栅', '西栅'};
periods = cell(1, T-1);
for i = 1:T-1
    periods{i} = sprintf('%d→%d', 2017+i, 2018+i);  % The user has to adjust this to the actual years
end

% Print the result tables
fprintf('\n=== Malmquist指数计算结果 ===\n');
fprintf('%-10s', '景区');
for p = 1:length(periods)
    fprintf('%-15s', periods{p});
end
fprintf('\n');

for i = 1:n_dmu
    fprintf('%-10s', dmu_names{i});
    for t = 1:T-1
        fprintf('%-15.4f', MI_matrix(i, t));
    end
    fprintf('\n');
end

% Print the efficiency change and technical progress decomposition
fprintf('\n=== 技术效率变化(EC) ===\n');
for i = 1:n_dmu
    fprintf('%-10s', dmu_names{i});
    for t = 1:T-1
        fprintf('%-15.4f', EC_matrix(i, t));
    end
    fprintf('\n');
end

fprintf('\n=== 技术进步变化(TC) ===\n');
for i = 1:n_dmu
    fprintf('%-10s', dmu_names{i});
    for t = 1:T-1
        fprintf('%-15.4f', TC_matrix(i, t));
    end
    fprintf('\n');
end

%% Hints for interpreting the results
fprintf('\n=== 结果解读说明 ===\n');
fprintf('Malmquist指数(MI) > 1 表示全要素生产率提升\n');
fprintf('技术效率变化(EC) > 1 表示效率改善\n');
fprintf('技术进步变化(TC) > 1 表示技术进步\n');
fprintf('MI = EC × TC\n');

%% Visualisation: plot the trend of the Malmquist index
figure('Name', '乌镇景区Malmquist指数变化趋势');
colors = {'b-o', 'r-s'};  % East Gate as a blue circle line, West Gate as a red square line
for i = 1:n_dmu
    plot(2:T, MI_matrix(i, :), colors{i}, 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
end
grid on;
xlabel('年份', 'FontSize', 12);
ylabel('Malmquist指数', 'FontSize', 12);
title('乌镇景区全要素生产率变化趋势', 'FontSize', 14);
legend(dmu_names, 'Location', 'best');
hold off;

% Plot the EC versus TC comparison chart
figure('Name', 'Malmquist指数分解对比（2020年）');
% Show the decomposition using the last period as an example
last_period = T-1;
bar_data = [EC_matrix(:, last_period)'; TC_matrix(:, last_period)'];
bar(bar_data);
set(gca, 'XTickLabel', dmu_names);
xlabel('景区', 'FontSize', 12);
ylabel('指数值', 'FontSize', 12);
title(sprintf('Malmquist指数分解对比（%s）', periods{last_period}), 'FontSize', 14);
legend({'技术效率变化(EC)', '技术进步变化(TC)'}, 'Location', 'best');
grid on;

fprintf('\n=== 计算完成 ===\n');