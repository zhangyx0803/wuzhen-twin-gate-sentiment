%% DEA-Malmquist efficiency analysis tool v0.9
% Supports multi-DMU multi-period efficiency analysis and computes MI, EC, TC, PEC, SEC and other indicators. Adds super-efficiency DEA, an assessment of validity and slack variables
function dealmquist_analysis()
    clc; clear; close all;
    
    [dataTable, config, uniqueDMU, uniqueYears, X_panel, Y_panel] = load_and_configure_data();
    validate_data(X_panel, Y_panel);
    [MI_matrix, EC_matrix, TC_matrix, PEC_matrix, SEC_matrix, dea_results] = calculate_malmquist(X_panel, Y_panel, config.orientation, config.dea_type);
    create_result_tables(MI_matrix, EC_matrix, TC_matrix, PEC_matrix, SEC_matrix, dea_results, uniqueDMU, uniqueYears);
    create_visualizations(MI_matrix, EC_matrix, TC_matrix, PEC_matrix, SEC_matrix, uniqueDMU, uniqueYears);
    
    fprintf('\n=== 分析完成 ===\n');
end

%% Data loading and configuration functions
function [dataTable, config, uniqueDMU, uniqueYears, X_panel, Y_panel] = load_and_configure_data()
    % Select the Excel file
    [file, path] = uigetfile('*.xlsx', '请选择包含数据的 Excel 文件');
    if file == 0
        error('未选择文件，程序退出。');
    end
    fullpath = fullfile(path, file);
    fprintf('已读取文件：%s\n\n', fullpath);
    
    % Read the data into a table
    dataTable = readtable(fullpath, 'VariableNamingRule', 'preserve');
    colNames = dataTable.Properties.VariableNames;
    fprintf('检测到列名：%s\n', strjoin(colNames, ', '));
    
    % User input: configure column roles and model parameters in a GUI
    config = gui_column_config(colNames);
    
    % Extract the data from the selected columns
    dmuRaw = dataTable{:, config.dmuColIdx};
    dmuNames = string(dmuRaw);
    uniqueDMU = unique(dmuNames);
    fprintf('检测到 %d 个不同的 DMU：%s\n', length(uniqueDMU), strjoin(uniqueDMU, ', '));
    
    yearsRaw = dataTable{:, config.yearColIdx};
    if isnumeric(yearsRaw)
        years = yearsRaw;
    else
        years = str2double(string(yearsRaw));
    end
    uniqueYears = unique(years);
    uniqueYears = sort(uniqueYears);
    fprintf('检测到年份：%s\n', mat2str(uniqueYears));
    
    fprintf('投入指标列：%s\n', strjoin(colNames(config.inputColIdx), ', '));
    fprintf('产出指标列：%s\n', strjoin(colNames(config.outputColIdx), ', '));
    fprintf('模型导向：%s\n', config.orientation);
    fprintf('规模报酬假设：%s\n', config.RTS);
    
    [X_panel, Y_panel] = restructure_data(dataTable, dmuNames, years, uniqueDMU, uniqueYears, ...
        config.inputColIdx, config.outputColIdx);
end

function [X_panel, Y_panel] = restructure_data(dataTable, dmuNames, years, uniqueDMU, uniqueYears, inputColIdx, outputColIdx)
    T = length(uniqueYears);
    K = length(uniqueDMU);
    nInput = length(inputColIdx);
    nOutput = length(outputColIdx);
    
    X_panel = NaN(T, K, nInput);
    Y_panel = NaN(T, K, nOutput);
    
    yearMap = containers.Map(uniqueYears, 1:T);
    dmuMap = containers.Map(uniqueDMU, 1:K);
    
    for row = 1:height(dataTable)
        dmuName = dmuNames(row);
        yr = years(row);
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
        
        for i = 1:nInput
            X_panel(t, k, i) = dataTable{row, inputColIdx(i)};
        end
        for o = 1:nOutput
            Y_panel(t, k, o) = dataTable{row, outputColIdx(o)};
        end
    end
end

function validate_data(X_panel, Y_panel)
    fprintf('\n=== 数据质量检查 ===\n');
    
    if any(isnan(X_panel(:))) || any(isnan(Y_panel(:)))
        warning('数据中存在缺失值（NaN）。请确保每个 DMU 在每个年份都有完整数据。');
        num_nan_x = sum(isnan(X_panel(:)));
        num_nan_y = sum(isnan(Y_panel(:)));
        fprintf('投入数据缺失值数量: %d\n', num_nan_x);
        fprintf('产出数据缺失值数量: %d\n', num_nan_y);
    else
        fprintf('✓ 数据无缺失值\n');
    end
    
    if any(X_panel(:) < 0)
        error('投入数据中存在负值，DEA分析无法进行。请检查数据。');
    else
        fprintf('✓ 投入数据无负值\n');
    end
    
    if any(Y_panel(:) < 0)
        error('产出数据中存在负值，DEA分析无法进行。请检查数据。');
    else
        fprintf('✓ 产出数据无负值\n');
    end
    
    if any(X_panel(:) == 0)
        warning('投入数据中存在零值，可能会影响DEA分析结果。');
        num_zero_x = sum(X_panel(:) == 0);
        fprintf('投入数据零值数量: %d\n', num_zero_x);
    else
        fprintf('✓ 投入数据无零值\n');
    end
    
    if any(Y_panel(:) == 0)
        warning('产出数据中存在零值，可能会影响DEA分析结果。');
        num_zero_y = sum(Y_panel(:) == 0);
        fprintf('产出数据零值数量: %d\n', num_zero_y);
    else
        fprintf('✓ 产出数据无零值\n');
    end
    
    % Check the data range
    min_x = min(X_panel(:));
    max_x = max(X_panel(:));
    min_y = min(Y_panel(:));
    max_y = max(Y_panel(:));
    fprintf('投入数据范围: %.2f - %.2f\n', min_x, max_x);
    fprintf('产出数据范围: %.2f - %.2f\n', min_y, max_y);
    
    fprintf('=== 数据质量检查完成 ===\n\n');
end

function [MI_matrix, EC_matrix, TC_matrix, PEC_matrix, SEC_matrix, dea_results] = calculate_malmquist(X_panel, Y_panel, orientation, dea_type)
    T = size(X_panel, 1);
    K = size(X_panel, 2);
    m = size(X_panel, 3);
    s = size(Y_panel, 3);
    
    MI_matrix = zeros(K, T-1);
    EC_matrix = zeros(K, T-1);
    TC_matrix = zeros(K, T-1);
    PEC_matrix = zeros(K, T-1);
    SEC_matrix = zeros(K, T-1);
    
    dea_results = struct();
    dea_results.efficiency = cell(T, 1);
    dea_results.status = cell(T, 1);
    dea_results.slack_x = cell(T, 1);
    dea_results.slack_y = cell(T, 1);
    
    for t = 1:T-1
        X_t = squeeze(X_panel(t, :, :));
        Y_t = squeeze(Y_panel(t, :, :));
        X_t1 = squeeze(X_panel(t+1, :, :));
        Y_t1 = squeeze(Y_panel(t+1, :, :));
        
        [MI, EC, TC, PEC, SEC, ~] = malmquist_index(X_t, Y_t, X_t1, Y_t1, orientation, dea_type);
        MI_matrix(:, t) = MI;
        EC_matrix(:, t) = EC;
        TC_matrix(:, t) = TC;
        PEC_matrix(:, t) = PEC;
        SEC_matrix(:, t) = SEC;
    end
    
    for t = 1:T
        X_t = squeeze(X_panel(t, :, :));
        Y_t = squeeze(Y_panel(t, :, :));
        dea_results.efficiency{t} = zeros(K, 1);
        dea_results.status{t} = cell(K, 1);
        dea_results.slack_x{t} = zeros(K, m);
        dea_results.slack_y{t} = zeros(K, s);
        
        for i = 1:K
            [dea_results.efficiency{t}(i), sx, sy, dea_results.status{t}{i}] = ...
                dea_distance(X_t, Y_t, i, orientation, 'vrs', dea_type);
            if ~isempty(sx) && ~isnan(sx(1))
                dea_results.slack_x{t}(i, :) = sx';
                dea_results.slack_y{t}(i, :) = sy';
            end
        end
    end
end

function create_result_tables(MI_matrix, EC_matrix, TC_matrix, PEC_matrix, SEC_matrix, dea_results, uniqueDMU, uniqueYears)
    T = length(uniqueYears);
    periods = cell(1, T-1);
    for i = 1:T-1
        periods{i} = sprintf('%d→%d', uniqueYears(i), uniqueYears(i+1));
    end
    create_result_window(MI_matrix, EC_matrix, TC_matrix, PEC_matrix, SEC_matrix, dea_results, uniqueDMU, periods, uniqueYears);
end

function create_visualizations(MI_matrix, EC_matrix, TC_matrix, PEC_matrix, SEC_matrix, uniqueDMU, uniqueYears)
    T = length(uniqueYears);
    K = length(uniqueDMU);
    year_ticks = uniqueYears(2:end);
    period_labels = year_ticks;
    color_order = lines(K);
    create_chart_window(MI_matrix, EC_matrix, TC_matrix, PEC_matrix, SEC_matrix, ...
        uniqueDMU, year_ticks, period_labels, color_order);
end

function theme = getSystemTheme()
    try
        if ispc
            regValue = winqueryreg('HKEY_CURRENT_USER', ...
                'Software\Microsoft\Windows\CurrentVersion\Themes\Personalize', 'AppsUseLightTheme');
            if regValue == 0
                theme = 'dark';
            else
                theme = 'light';
            end
        else
            theme = 'light';
        end
    catch
        theme = 'light';
    end
end

% Run the main function
dealmquist_analysis();

%% Result-window creation functions
function create_result_window(MI_matrix, EC_matrix, TC_matrix, PEC_matrix, SEC_matrix, dea_results, uniqueDMU, periods, uniqueYears)
    fig_main = figure('Name', 'Malmquist-DEA 分析结果', 'NumberTitle', 'off', ...
        'Position', [100 100 1100 650]);

    tabgroup = uitabgroup(fig_main, 'Position', [0.02 0.02 0.96 0.96]);

    add_table_to_tab = @(parent, data, row_names, col_names, title_str) ...
        uitable(parent, 'Data', data, 'ColumnName', col_names, 'RowName', row_names, ...
        'Position', [20 20 1040 535], 'FontSize', 11);

    tab_MI = uitab(tabgroup, 'Title', 'Malmquist指数 (MI)');
    add_table_to_tab(tab_MI, MI_matrix, uniqueDMU, periods, 'MI');

    tab_EC = uitab(tabgroup, 'Title', '技术效率变化 (EC)');
    add_table_to_tab(tab_EC, EC_matrix, uniqueDMU, periods, 'EC');

    tab_TC = uitab(tabgroup, 'Title', '技术进步变化 (TC)');
    add_table_to_tab(tab_TC, TC_matrix, uniqueDMU, periods, 'TC');

    tab_PEC = [];
    if exist('PEC_matrix', 'var') && ~isempty(PEC_matrix)
        tab_PEC = uitab(tabgroup, 'Title', '纯技术效率变化 (PEC)');
        add_table_to_tab(tab_PEC, PEC_matrix, uniqueDMU, periods, 'PEC');
    end

    tab_SEC = [];
    if exist('SEC_matrix', 'var') && ~isempty(SEC_matrix)
        tab_SEC = uitab(tabgroup, 'Title', '规模效率变化 (SEC)');
        add_table_to_tab(tab_SEC, SEC_matrix, uniqueDMU, periods, 'SEC');
    end

    if exist('dea_results', 'var') && ~isempty(dea_results)
        for t = 1:length(uniqueYears)
            year = uniqueYears(t);
            K = length(uniqueDMU);
            efficiency_data = cell(K, 2);
            for i = 1:K
                efficiency_data{i, 1} = dea_results.efficiency{t}(i);
                efficiency_data{i, 2} = dea_results.status{t}{i};
            end
            slack_x = dea_results.slack_x{t};
            slack_y = dea_results.slack_y{t};
            
            tab_dea = uitab(tabgroup, 'Title', sprintf('DEA效率 %d', year));
            uitable(tab_dea, 'Data', efficiency_data, ...
                'ColumnName', {'效率值', '有效性状态'}, ...
                'RowName', uniqueDMU, ...
                'Position', [20 20 1040 535], 'FontSize', 11);
            
            if ~isempty(slack_x) && size(slack_x, 2) > 0
                tab_slack_x = uitab(tabgroup, 'Title', sprintf('投入松弛 %d', year));
                uitable(tab_slack_x, 'Data', slack_x, ...
                    'ColumnName', arrayfun(@(i) sprintf('投入%d', i), 1:size(slack_x, 2), 'UniformOutput', false), ...
                    'RowName', uniqueDMU, ...
                    'Position', [20 20 1040 535], 'FontSize', 11);
            end
            
            if ~isempty(slack_y) && size(slack_y, 2) > 0
                tab_slack_y = uitab(tabgroup, 'Title', sprintf('产出松弛 %d', year));
                uitable(tab_slack_y, 'Data', slack_y, ...
                    'ColumnName', arrayfun(@(i) sprintf('产出%d', i), 1:size(slack_y, 2), 'UniformOutput', false), ...
                    'RowName', uniqueDMU, ...
                    'Position', [20 20 1040 535], 'FontSize', 11);
            end
        end
    end

    % Per-region summary tab
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
        'Position', [20 20 1040 535], 'FontSize', 11);

    % Per-period summary tab
    tab_time = uitab(tabgroup, 'Title', '按时间汇总');
    % Geometric mean over all DMUs for each period
    T_period = length(periods);
    data_time = zeros(T_period, 5);
    geo_scalar = @(x) exp(mean(log(x)));   % Geometric mean function (for a vector)

    for t = 1:T_period
        data_time(t, :) = [geo_scalar(EC_matrix(:,t)), geo_scalar(TC_matrix(:,t)), ...
                           geo_scalar(PEC_matrix(:,t)), geo_scalar(SEC_matrix(:,t)), ...
                           geo_scalar(MI_matrix(:,t))];
    end
    % Append the overall mean row
    data_time = [data_time; mean(EC_geo), mean(TC_geo), mean(PEC_geo), mean(SEC_geo), mean(MI_geo)];
    row_names_time = [periods, {sprintf('%d-%d', min(uniqueYears), max(uniqueYears))}];
    col_names_time = {'效率变动', '技术进步', '纯技术效率', '规模效率', '全要素生产率'};
    uitable(tab_time, 'Data', data_time, 'ColumnName', col_names_time, 'RowName', row_names_time, ...
        'Position', [20 20 1040 535], 'FontSize', 11);

    % ----- Add an export button to every data-table tab -----
    add_export_buttons(tab_MI, tab_EC, tab_TC, tab_PEC, tab_SEC, tab_region, tab_time, ...
        MI_matrix, EC_matrix, TC_matrix, PEC_matrix, SEC_matrix, data_region, data_time, ...
        uniqueDMU, periods, row_names_region, col_names_region, row_names_time, col_names_time);
end

%% Export-button helper
function add_export_buttons(tab_MI, tab_EC, tab_TC, tab_PEC, tab_SEC, tab_region, tab_time, ...
    MI_matrix, EC_matrix, TC_matrix, PEC_matrix, SEC_matrix, data_region, data_time, ...
    uniqueDMU, periods, row_names_region, col_names_region, row_names_time, col_names_time)
    % Export button callback
    function exportTableToExcel(src, evt)
        data = src.UserData;
        [file, path] = uiputfile('*.xlsx', '保存Excel文件');
        if file ~= 0
            fullpath = fullfile(path, file);
            
            % Get the data
            table_data = data.table_data;
            row_names = data.row_names;
            col_names = data.col_names;
            sheet_name = data.sheet_name;
            
            % Determine the data size
            if iscell(table_data)
                num_rows = size(table_data, 1);
                num_cols = size(table_data, 2);
            else
                [num_rows, num_cols] = size(table_data);
            end
            
            % Build the complete table data
            full_data = cell(num_rows + 1, num_cols + 1);
            
            % First row: scenic area + column names
            full_data{1, 1} = '景区';
            for c = 1:num_cols
                if iscell(col_names) && numel(col_names) >= c
                    full_data{1, c+1} = col_names{c};
                elseif ischar(col_names) && size(col_names, 1) >= c
                    full_data{1, c+1} = col_names(c,:);
                elseif isstring(col_names) && numel(col_names) >= c
                    full_data{1, c+1} = char(col_names(c));
                else
                    full_data{1, c+1} = ['列' num2str(c)];
                end
            end
            
            % Data rows: row name + data
            for r = 1:num_rows
                if iscell(row_names) && numel(row_names) >= r
                    full_data{r+1, 1} = row_names{r};
                elseif ischar(row_names) && size(row_names, 1) >= r
                    full_data{r+1, 1} = row_names(r,:);
                elseif isstring(row_names) && numel(row_names) >= r
                    full_data{r+1, 1} = char(row_names(r));
                else
                    full_data{r+1, 1} = ['行' num2str(r)];
                end
                for c = 1:num_cols
                    if iscell(table_data)
                        full_data{r+1, c+1} = table_data{r, c};
                    else
                        full_data{r+1, c+1} = table_data(r, c);
                    end
                end
            end
            xlswrite(fullpath, full_data, sheet_name);
            msgbox('Excel文件导出成功！', '导出成功');
        end
    end

    % Add an export button to every data-table tab (in the upper right corner so it stands out)
    btn_MI = uicontrol('Style', 'pushbutton', 'String', '📊 导出Excel', ...
        'Parent', tab_MI, 'Position', [900 575 140 35], 'FontSize', 11, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.8 0.9 1], 'Callback', @exportTableToExcel);
    btn_MI.UserData = struct('table_data', MI_matrix, 'row_names', uniqueDMU, 'col_names', periods, 'sheet_name', 'MI');

    btn_EC = uicontrol('Style', 'pushbutton', 'String', '📊 导出Excel', ...
        'Parent', tab_EC, 'Position', [900 575 140 35], 'FontSize', 11, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.8 0.9 1], 'Callback', @exportTableToExcel);
    btn_EC.UserData = struct('table_data', EC_matrix, 'row_names', uniqueDMU, 'col_names', periods, 'sheet_name', 'EC');

    btn_TC = uicontrol('Style', 'pushbutton', 'String', '📊 导出Excel', ...
        'Parent', tab_TC, 'Position', [900 575 140 35], 'FontSize', 11, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.8 0.9 1], 'Callback', @exportTableToExcel);
    btn_TC.UserData = struct('table_data', TC_matrix, 'row_names', uniqueDMU, 'col_names', periods, 'sheet_name', 'TC');

    if ~isempty(tab_PEC)
        btn_PEC = uicontrol('Style', 'pushbutton', 'String', '📊 导出Excel', ...
            'Parent', tab_PEC, 'Position', [900 575 140 35], 'FontSize', 11, 'FontWeight', 'bold', ...
            'BackgroundColor', [0.8 0.9 1], 'Callback', @exportTableToExcel);
        btn_PEC.UserData = struct('table_data', PEC_matrix, 'row_names', uniqueDMU, 'col_names', periods, 'sheet_name', 'PEC');
    end

    if ~isempty(tab_SEC)
        btn_SEC = uicontrol('Style', 'pushbutton', 'String', '📊 导出Excel', ...
            'Parent', tab_SEC, 'Position', [900 575 140 35], 'FontSize', 11, 'FontWeight', 'bold', ...
            'BackgroundColor', [0.8 0.9 1], 'Callback', @exportTableToExcel);
        btn_SEC.UserData = struct('table_data', SEC_matrix, 'row_names', uniqueDMU, 'col_names', periods, 'sheet_name', 'SEC');
    end

    btn_region = uicontrol('Style', 'pushbutton', 'String', '📊 导出Excel', ...
        'Parent', tab_region, 'Position', [900 575 140 35], 'FontSize', 11, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.8 0.9 1], 'Callback', @exportTableToExcel);
    btn_region.UserData = struct('table_data', data_region, 'row_names', row_names_region, 'col_names', col_names_region, 'sheet_name', '按地区汇总');

    btn_time = uicontrol('Style', 'pushbutton', 'String', '📊 导出Excel', ...
        'Parent', tab_time, 'Position', [900 575 140 35], 'FontSize', 11, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.8 0.9 1], 'Callback', @exportTableToExcel);
    btn_time.UserData = struct('table_data', data_time, 'row_names', row_names_time, 'col_names', col_names_time, 'sheet_name', '按时间汇总');
end

%% Chart-window creation functions
function create_chart_window(MI_matrix, EC_matrix, TC_matrix, PEC_matrix, SEC_matrix, ...
    uniqueDMU, year_ticks, period_labels, color_order)
    K = length(uniqueDMU);
    num_dmu = K;
    
    % Create the figure window with several tabs
    fig_charts = figure('Name', 'DEA-Malmquist 分析图表', 'NumberTitle', 'off', 'Position', [50 50 1200 800]);
    chart_tabgroup = uitabgroup(fig_charts);

    % Image export callback
    function exportChartToPNG(src, evt)
        data = src.UserData;
        fig = data.fig;
        
        [file, path] = uiputfile('*.png', '保存图片文件');
        if file ~= 0
            fullpath = fullfile(path, file);
            exportapp(fig, fullpath);
            msgbox('图片导出成功！', '导出成功');
        end
    end

    % Tab 1: total factor productivity trend
    tab1 = uitab(chart_tabgroup, 'Title', '全要素生产率趋势');
    btn_chart1 = uicontrol('Style', 'pushbutton', 'String', '📈 导出图片', ...
        'Parent', tab1, 'Position', [1030 730 140 35], 'FontSize', 11, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.9 1 0.9], 'Callback', @exportChartToPNG);
    btn_chart1.UserData = struct('fig', fig_charts);
    ax1 = axes(tab1, 'Position', [0.1 0.12 0.8 0.73]);
    hold(ax1, 'on');
    for i = 1:K
        plot(ax1, year_ticks, MI_matrix(i, :), 'o-', 'LineWidth', 2, 'Color', color_order(i, :), 'DisplayName', uniqueDMU{i});
    end
    hold(ax1, 'off');
    xlabel(ax1, '年份', 'FontSize', 10, 'FontWeight', 'bold');
    ylabel(ax1, 'Malmquist 指数', 'FontSize', 10, 'FontWeight', 'bold');
    title(ax1, '全要素生产率趋势', 'FontSize', 11, 'FontWeight', 'bold');
    grid(ax1, 'on');
    ax1.XTick = year_ticks;
    ax1.XLim = [min(year_ticks)-0.5, max(year_ticks)+0.5];
    legend(ax1, 'Location', 'best', 'FontSize', 8);

    % Tab 2: technical efficiency change
    tab2 = uitab(chart_tabgroup, 'Title', '技术效率变化');
    btn_chart2 = uicontrol('Style', 'pushbutton', 'String', '📈 导出图片', ...
        'Parent', tab2, 'Position', [1030 730 140 35], 'FontSize', 11, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.9 1 0.9], 'Callback', @exportChartToPNG);
    btn_chart2.UserData = struct('fig', fig_charts);
    ax2 = axes(tab2, 'Position', [0.1 0.12 0.8 0.73]);
    hold(ax2, 'on');
    for i = 1:num_dmu
        plot(ax2, year_ticks, EC_matrix(i, :), 's-', 'LineWidth', 1.5, 'Color', color_order(i, :), 'DisplayName', uniqueDMU{i}, 'MarkerSize', 6);
    end
    hold(ax2, 'off');
    xlabel(ax2, '年份', 'FontSize', 10, 'FontWeight', 'bold');
    ylabel(ax2, 'EC', 'FontSize', 10, 'FontWeight', 'bold');
    title(ax2, '技术效率变化', 'FontSize', 11, 'FontWeight', 'bold');
    grid(ax2, 'on');
    ax2.XTick = year_ticks;
    ax2.XLim = [min(year_ticks)-0.5, max(year_ticks)+0.5];
    legend(ax2, 'Location', 'best', 'FontSize', 8);

    % Tab 3: technical progress change
    tab3 = uitab(chart_tabgroup, 'Title', '技术进步变化');
    btn_chart3 = uicontrol('Style', 'pushbutton', 'String', '📈 导出图片', ...
        'Parent', tab3, 'Position', [1030 730 140 35], 'FontSize', 11, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.9 1 0.9], 'Callback', @exportChartToPNG);
    btn_chart3.UserData = struct('fig', fig_charts);
    ax3 = axes(tab3, 'Position', [0.1 0.12 0.8 0.73]);
    hold(ax3, 'on');
    for i = 1:num_dmu
        plot(ax3, year_ticks, TC_matrix(i, :), 'd-', 'LineWidth', 1.5, 'Color', color_order(i, :), 'DisplayName', uniqueDMU{i}, 'MarkerSize', 6);
    end
    hold(ax3, 'off');
    xlabel(ax3, '年份', 'FontSize', 10, 'FontWeight', 'bold');
    ylabel(ax3, 'TC', 'FontSize', 10, 'FontWeight', 'bold');
    title(ax3, '技术进步变化', 'FontSize', 11, 'FontWeight', 'bold');
    grid(ax3, 'on');
    ax3.XTick = year_ticks;
    ax3.XLim = [min(year_ticks)-0.5, max(year_ticks)+0.5];
    legend(ax3, 'Location', 'best', 'FontSize', 8);

    % Tab 4: MI comparison across scenic areas
    tab4 = uitab(chart_tabgroup, 'Title', '各景区MI对比');
    btn_chart4 = uicontrol('Style', 'pushbutton', 'String', '📈 导出图片', ...
        'Parent', tab4, 'Position', [1030 730 140 35], 'FontSize', 11, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.9 1 0.9], 'Callback', @exportChartToPNG);
    btn_chart4.UserData = struct('fig', fig_charts);
    ax4 = axes(tab4, 'Position', [0.1 0.12 0.8 0.73]);
    b = bar(ax4, MI_matrix', 'FaceColor', 'flat');
    for i = 1:K
        b(i).FaceColor = color_order(i, :);
    end
    ax4.XTickLabel = period_labels;
    xlabel(ax4, '时期', 'FontSize', 10, 'FontWeight', 'bold');
    ylabel(ax4, 'Malmquist 指数', 'FontSize', 10, 'FontWeight', 'bold');
    title(ax4, '各景区 MI 对比', 'FontSize', 11, 'FontWeight', 'bold');
    grid(ax4, 'on');
    legend(ax4, uniqueDMU, 'Location', 'best', 'FontSize', 8);

    % Tab 5: MI heat map
    tab5 = uitab(chart_tabgroup, 'Title', 'MI热图');
    btn_chart5 = uicontrol('Style', 'pushbutton', 'String', '📈 导出图片', ...
        'Parent', tab5, 'Position', [1030 730 140 35], 'FontSize', 11, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.9 1 0.9], 'Callback', @exportChartToPNG);
    btn_chart5.UserData = struct('fig', fig_charts);
    ax5 = axes(tab5, 'Position', [0.15 0.12 0.7 0.73]);
    imagesc(ax5, MI_matrix);
    colormap(ax5, turbo);
    ax5.XTick = 1:length(period_labels);
    ax5.XTickLabel = period_labels;
    ax5.YTick = 1:length(uniqueDMU);
    ax5.YTickLabel = uniqueDMU;
    xlabel(ax5, '时期', 'FontSize', 10, 'FontWeight', 'bold');
    ylabel(ax5, '景区', 'FontSize', 10, 'FontWeight', 'bold');
    title(ax5, 'MI 热图', 'FontSize', 11, 'FontWeight', 'bold');
    colorbar(ax5);

    % Tab 6: decomposition of the sources of MI
    tab6 = uitab(chart_tabgroup, 'Title', 'MI来源分解');
    btn_chart6 = uicontrol('Style', 'pushbutton', 'String', '📈 导出图片', ...
        'Parent', tab6, 'Position', [1030 730 140 35], 'FontSize', 11, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.9 1 0.9], 'Callback', @exportChartToPNG);
    btn_chart6.UserData = struct('fig', fig_charts);
    ax6 = axes(tab6, 'Position', [0.15 0.12 0.7 0.73]);
    mean_EC = mean(EC_matrix, 2);
    mean_TC = mean(TC_matrix, 2);
    bar(ax6, [mean_EC, mean_TC], 'stacked');
    ax6.XTickLabel = uniqueDMU;
    xlabel(ax6, '景区', 'FontSize', 10, 'FontWeight', 'bold');
    ylabel(ax6, '平均指数', 'FontSize', 10, 'FontWeight', 'bold');
    title(ax6, 'MI 来源分解', 'FontSize', 11, 'FontWeight', 'bold');
    legend(ax6, {'EC', 'TC'}, 'Location', 'best', 'FontSize', 9);
    grid(ax6, 'on');

    % Tab 7: overall efficiency performance (radar chart)
    if K <= 10
        tab7 = uitab(chart_tabgroup, 'Title', '综合效率表现');
        btn_chart7 = uicontrol('Style', 'pushbutton', 'String', '📈 导出图片', ...
            'Parent', tab7, 'Position', [1030 730 140 35], 'FontSize', 11, 'FontWeight', 'bold', ...
            'BackgroundColor', [0.9 1 0.9], 'Callback', @exportChartToPNG);
        btn_chart7.UserData = struct('fig', fig_charts);
        ax7 = polaraxes(tab7, 'Position', [0.15 0.07 0.7 0.78]);
        mean_MI = mean(MI_matrix, 2);
        mean_EC = mean(EC_matrix, 2);
        mean_TC = mean(TC_matrix, 2);
        mean_PEC = mean(PEC_matrix, 2);
        mean_SEC = mean(SEC_matrix, 2);
        radar_data = [mean_MI, mean_EC, mean_TC, mean_PEC, mean_SEC];
        categories = {'全要素生产率', '技术效率', '技术进步', '纯技术效率', '规模效率'};
        
        hold(ax7, 'on');
        for i = 1:K
            theta = linspace(0, 2*pi, 6);
            r = [radar_data(i,:), radar_data(i,1)];
            polarplot(ax7, theta, r, 'o-', 'LineWidth', 2, 'Color', color_order(i, :), 'DisplayName', uniqueDMU{i});
        end
        hold(ax7, 'off');
        title(ax7, '综合效率表现', 'FontSize', 11, 'FontWeight', 'bold');
        ax7.RLim = [0 1.2];
        ax7.ThetaTick = 0:72:288;
        ax7.ThetaTickLabel = categories;
        ax7.ThetaDir = 'clockwise';
        ax7.ThetaZeroLocation = 'top';
        legend(ax7, 'Location', 'best', 'FontSize', 8);
    end

    % Tab 8: EC vs TC scatter plot
    tab8 = uitab(chart_tabgroup, 'Title', 'EC vs TC');
    btn_chart8 = uicontrol('Style', 'pushbutton', 'String', '📈 导出图片', ...
        'Parent', tab8, 'Position', [1030 730 140 35], 'FontSize', 11, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.9 1 0.9], 'Callback', @exportChartToPNG);
    btn_chart8.UserData = struct('fig', fig_charts);
    ax8 = axes(tab8, 'Position', [0.15 0.12 0.7 0.73]);
    hold(ax8, 'on');
    for i = 1:K
        scatter(ax8, EC_matrix(i,:), TC_matrix(i,:), 80, color_order(i,:), 'filled', 'DisplayName', uniqueDMU{i});
    end
    hold(ax8, 'off');
    xlabel(ax8, '技术效率 (EC)', 'FontSize', 10, 'FontWeight', 'bold');
    ylabel(ax8, '技术进步 (TC)', 'FontSize', 10, 'FontWeight', 'bold');
    title(ax8, 'EC vs TC', 'FontSize', 11, 'FontWeight', 'bold');
    grid(ax8, 'on');
    xline(ax8, 1, 'k--', 'LineWidth', 1.5);
    yline(ax8, 1, 'k--', 'LineWidth', 1.5);
    legend(ax8, 'Location', 'best', 'FontSize', 8);

    % Tab 9: distribution of the efficiency indicators
    tab9 = uitab(chart_tabgroup, 'Title', '效率指标分布');
    btn_chart9 = uicontrol('Style', 'pushbutton', 'String', '📈 导出图片', ...
        'Parent', tab9, 'Position', [1030 730 140 35], 'FontSize', 11, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.9 1 0.9], 'Callback', @exportChartToPNG);
    btn_chart9.UserData = struct('fig', fig_charts);
    ax9 = axes(tab9, 'Position', [0.15 0.12 0.7 0.73]);
    box_data = [MI_matrix(:), EC_matrix(:), TC_matrix(:), PEC_matrix(:), SEC_matrix(:)];
    boxplot(ax9, box_data, 'Labels', {'MI', 'EC', 'TC', 'PEC', 'SEC'});
    xlabel(ax9, '指标', 'FontSize', 10, 'FontWeight', 'bold');
    ylabel(ax9, '指数值', 'FontSize', 10, 'FontWeight', 'bold');
    title(ax9, '各效率指标分布情况', 'FontSize', 11, 'FontWeight', 'bold');
    grid(ax9, 'on');
end


%% Core functions
function [distance, slack_input, slack_output, status] = dea_distance(X, Y, varargin)
    
    [n, m] = size(X);
    [~, s] = size(Y);
    
    % Parse the input arguments
    if nargin == 6
        idx_dmu = varargin{1};
        orientation = varargin{2};
        RTS = varargin{3};
        dea_type = varargin{4};
        x0 = X(idx_dmu, :)';
        y0 = Y(idx_dmu, :)';
        is_idx_mode = true;
    elseif nargin == 7
        x0 = varargin{1}(:);
        y0 = varargin{2}(:);
        orientation = varargin{3};
        RTS = varargin{4};
        dea_type = varargin{5};
        is_idx_mode = false;
    else
        error('参数数目不正确，请提供 idx_dmu 或 (x0, y0)');
    end
    
    is_super = strcmp(dea_type, 'super');
    
    if is_super && is_idx_mode && n > 1
        X_ref = X;
        Y_ref = Y;
        X_ref(idx_dmu, :) = [];
        Y_ref(idx_dmu, :) = [];
        n_ref = n - 1;
    else
        X_ref = X;
        Y_ref = Y;
        n_ref = n;
    end
    
    % Decision variables
    n_vars = n_ref + 1;
    lb = [0; zeros(n_ref, 1)];
    
    if strcmp(orientation, 'input')     % Input-oriented
        
        f = [1; zeros(n_ref, 1)];
        
        % Input constraint
        A_in = zeros(m, n_vars);
        for i = 1:m
            A_in(i, 1) = -x0(i);
            A_in(i, 2:end) = X_ref(:, i)';
        end
        b_in = zeros(m, 1);
        
        % Output constraint
        A_out = zeros(s, n_vars);
        for r = 1:s
            A_out(r, 1) = 0;
            A_out(r, 2:end) = -Y_ref(:, r)';
        end
        b_out = -y0;
        
        A = [A_in; A_out];
        b = [b_in; b_out];
        
    else % Output-oriented
        f = [-1; zeros(n_ref, 1)];
        % Input constraint
        A_in = zeros(m, n_vars);
        for i = 1:m
            A_in(i, 1) = 0;
            A_in(i, 2:end) = X_ref(:, i)';
        end
        b_in = x0;
        % Output constraint
        A_out = zeros(s, n_vars);
        for r = 1:s
            A_out(r, 1) = y0(r);
            A_out(r, 2:end) = -Y_ref(:, r)';
        end
        b_out = zeros(s, 1);
        
        A = [A_in; A_out];
        b = [b_in; b_out];
    end
    
    % Returns-to-scale assumption
    if strcmp(RTS, 'vrs')
        Aeq = [0, ones(1, n_ref)];
        beq = 1;
    else
        Aeq = [];
        beq = [];
    end
    
    options = optimoptions('linprog', 'Display', 'off', 'Algorithm', 'dual-simplex');
    [sol, fval, exitflag, output, lambda] = linprog(f, A, b, Aeq, beq, lb, [], options);
    
    if exitflag > 0
        if strcmp(orientation, 'input')
            distance = sol(1);
            theta = sol(1);
            slack_input = theta * x0 - X_ref' * sol(2:end);
            slack_output = Y_ref' * sol(2:end) - y0;
        else
            distance = 1 / sol(1);
            phi = sol(1);
            slack_input = x0 - X_ref' * sol(2:end);
            slack_output = Y_ref' * sol(2:end) - phi * y0;
        end
        
        % Assess validity
        tol = 1e-6;
        if abs(distance - 1) < tol
            if all(slack_input(:) < tol) && all(slack_output(:) < tol)
                status = '强有效';
            else
                status = '弱有效';
            end
        else
            status = '无效';
        end
    else
        distance = NaN;
        slack_input = NaN(size(x0));
        slack_output = NaN(size(y0));
        status = '计算失败';
        warning('线性规划求解失败，exitflag = %d', exitflag);
    end
end

function [MI, EC, TC, PEC, SEC, efficiency_table] = malmquist_index(X_t, Y_t, X_t1, Y_t1, orientation, dea_type)
    n = size(X_t, 1);
    
    % VRS distances
    Dv_t_t = zeros(n,1); Dv_t_t1 = zeros(n,1);
    Dv_t1_t = zeros(n,1); Dv_t1_t1 = zeros(n,1);
    % CRS distances
    Dc_t_t = zeros(n,1); Dc_t_t1 = zeros(n,1);
    Dc_t1_t = zeros(n,1); Dc_t1_t1 = zeros(n,1);
    
    m = size(X_t, 2);
    s = size(Y_t, 2);
    
    status_t = cell(n, 1);
    status_t1 = cell(n, 1);
    slack_x_t = zeros(n, m);
    slack_x_t1 = zeros(n, m);
    slack_y_t = zeros(n, s);
    slack_y_t1 = zeros(n, s);
    
    for i = 1:n
        [Dv_t_t(i), sx, sy, status_t{i}] = dea_distance(X_t, Y_t, i, orientation, 'vrs', dea_type);
        Dv_t_t1(i) = dea_distance(X_t, Y_t, X_t1(i,:)', Y_t1(i,:)', orientation, 'vrs', dea_type);
        Dv_t1_t(i) = dea_distance(X_t1, Y_t1, X_t(i,:)', Y_t(i,:)', orientation, 'vrs', dea_type);
        [Dv_t1_t1(i), sx1, sy1, status_t1{i}] = dea_distance(X_t1, Y_t1, i, orientation, 'vrs', dea_type);
        
        Dc_t_t(i) = dea_distance(X_t, Y_t, i, orientation, 'crs', dea_type);
        Dc_t_t1(i) = dea_distance(X_t, Y_t, X_t1(i,:)', Y_t1(i,:)', orientation, 'crs', dea_type);
        Dc_t1_t(i) = dea_distance(X_t1, Y_t1, X_t(i,:)', Y_t(i,:)', orientation, 'crs', dea_type);
        Dc_t1_t1(i) = dea_distance(X_t1, Y_t1, i, orientation, 'crs', dea_type);
        
        if ~isempty(sx) && ~isnan(sx(1))
            slack_x_t(i, :) = sx';
            slack_y_t(i, :) = sy';
        end
        if ~isempty(sx1) && ~isnan(sx1(1))
            slack_x_t1(i, :) = sx1';
            slack_y_t1(i, :) = sy1';
        end
        
        if any(isnan([Dv_t_t(i), Dv_t_t1(i), Dv_t1_t(i), Dv_t1_t1(i)]))
            warning('DMU %d 的 VRS 距离计算失败，将使用 CRS 结果替代', i);
            Dv_t_t(i) = Dc_t_t(i);
            Dv_t_t1(i) = Dc_t_t1(i);
            Dv_t1_t(i) = Dc_t1_t(i);
            Dv_t1_t1(i) = Dc_t1_t1(i);
        end
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
    
    % Build the efficiency analysis table
    efficiency_table = struct();
    efficiency_table.status_t = status_t;
    efficiency_table.status_t1 = status_t1;
    efficiency_table.slack_x_t = slack_x_t;
    efficiency_table.slack_x_t1 = slack_x_t1;
    efficiency_table.slack_y_t = slack_y_t;
    efficiency_table.slack_y_t1 = slack_y_t1;
    efficiency_table.Dv_t_t = Dv_t_t;
    efficiency_table.Dv_t1_t1 = Dv_t1_t1;
end

%% GUI configuration functions
function config = gui_column_config(colNames)
    % Detect the system theme
    theme = getSystemTheme();
    
    % Choose the colour scheme according to the theme
    if strcmp(theme, 'dark')
        bgWindow = [0.15 0.15 0.15];
        fgText = [1 1 1];
        bgPopup = [0.30 0.30 0.30];
        fgPopup = [1 1 1];
        bgButton = [0.25 0.65 0.90];
        fgButton = [1 1 1];
        bgHeader = [0.20 0.20 0.20];
        bgAccent = [0.30 0.30 0.30];
        borderColor = [0.35 0.35 0.35];
        highlightColor = [0.30 0.50 0.70];
    else
        bgWindow = [0.96 0.96 0.96];
        fgText = [0 0 0];
        bgPopup = [1 1 1];
        fgPopup = [0 0 0];
        bgButton = [0.40 0.65 0.90];
        fgButton = [1 1 1];
        bgHeader = [0.80 0.85 0.90];
        bgAccent = [0.90 0.92 0.95];
        borderColor = [0.7 0.7 0.7];
        highlightColor = [0.50 0.70 0.90];
    end
    
    nCols = length(colNames);
    rowHeight = 45;          % Height of each row
    labelWidth1 = 70;        % Width of the field-number column
    labelWidth2 = 220;       % Width of the field-name column
    popupWidth = 180;        % Width of the drop-down
    leftMargin = 40;         % Left margin
    rightMargin = 40;        % Right margin
    topMargin = 100;         % Top margin (reserved for the title)
    bottomMargin = 120;      % Bottom margin
    
    % Compute the window size
    winWidth = leftMargin + labelWidth1 + 10 + labelWidth2 + 10 + popupWidth + rightMargin;
    winHeight = topMargin + nCols * rowHeight + 100 + bottomMargin;
    if winHeight < 650
        winHeight = 650;
    end
    if winWidth < 800
        winWidth = 800;
    end
    
    % Create the window
    fig = figure('Name', '定义数据字段角色', 'NumberTitle', 'off', ...
        'Position', [200, 100, winWidth, winHeight], 'Resize', 'off', 'MenuBar', 'none', ...
        'Color', bgWindow);
    
    % Title bar
    titleBg = uicontrol('Style', 'text', 'String', '', ...
        'Position', [0, winHeight-80, winWidth, 80], ...
        'BackgroundColor', bgHeader);
    
    % Title text
    uicontrol('Style', 'text', 'String', 'DEA-Malmquist 分析配置', ...
        'Position', [0, winHeight-80, winWidth, 80], ...
        'BackgroundColor', 'none', 'ForegroundColor', fgText, ...
        'FontSize', 18, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    
    % Separator line
    uicontrol('Style', 'text', 'String', '', ...
        'Position', [0, winHeight-81, winWidth, 2], ...
        'BackgroundColor', highlightColor);
    
    % Subtitle
    uicontrol('Style', 'text', 'String', '请为每个字段指定角色', ...
        'Position', [leftMargin, winHeight-110, 300, 25], ...
        'BackgroundColor', bgWindow, 'ForegroundColor', fgText, ...
        'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
    
    % Column-header area
    headerY = winHeight - 140;
    uicontrol('Style', 'text', 'String', '', ...
        'Position', [leftMargin-10, headerY-5, labelWidth1+labelWidth2+popupWidth+20, 40], ...
        'BackgroundColor', bgAccent);
    
    % Column-header text
    uicontrol('Style', 'text', 'String', '编号', ...
        'Position', [leftMargin, headerY, labelWidth1, 35], ...
        'BackgroundColor', 'none', 'ForegroundColor', fgText, ...
        'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'FontSize', 12);
    uicontrol('Style', 'text', 'String', '字段名称', ...
        'Position', [leftMargin+labelWidth1+5, headerY, labelWidth2, 35], ...
        'BackgroundColor', 'none', 'ForegroundColor', fgText, ...
        'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'FontSize', 12);
    uicontrol('Style', 'text', 'String', '角色', ...
        'Position', [leftMargin+labelWidth1+labelWidth2+10, headerY, popupWidth, 35], ...
        'BackgroundColor', 'none', 'ForegroundColor', fgText, ...
        'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'FontSize', 12);
    
    % Store the popupmenu handle and the current role of every field
    popupHandles = gobjects(nCols, 1);
    roles = repmat({'忽略'}, nCols, 1);
    
    % Create each row dynamically
    for i = 1:nCols
        yPos = headerY - i * rowHeight;
        
        % Alternating row background colour
        if mod(i, 2) == 1
            rowBg = bgWindow;
        else
            if strcmp(theme, 'dark')
                rowBg = [0.18 0.18 0.18];
            else
                rowBg = [0.92 0.94 0.96];
            end
        end
        
        % Background
        rowBgObj = uicontrol('Style', 'text', 'String', '', ...
            'Position', [leftMargin-10, yPos-5, labelWidth1+labelWidth2+popupWidth+20, rowHeight+10], ...
            'BackgroundColor', rowBg);
        
        % Field number
        uicontrol('Style', 'text', 'String', num2str(i), ...
            'Position', [leftMargin, yPos, labelWidth1, rowHeight], ...
            'BackgroundColor', 'none', 'ForegroundColor', fgText, ...
            'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'normal');
        
        % Field name
        uicontrol('Style', 'text', 'String', colNames{i}, ...
            'Position', [leftMargin+labelWidth1+5, yPos, labelWidth2, rowHeight], ...
            'BackgroundColor', 'none', 'ForegroundColor', fgText, ...
            'HorizontalAlignment', 'left', 'FontSize', 12, 'FontWeight', 'normal');
        
        % Role drop-down (the callback captures the value of the current index i)
        popup = uicontrol('Style', 'popupmenu', ...
            'String', {'忽略', 'DMU', '年份', '投入', '产出'}, ...
            'Position', [leftMargin+labelWidth1+labelWidth2+10, yPos, popupWidth, rowHeight], ...
            'BackgroundColor', bgPopup, 'ForegroundColor', fgPopup, ...
            'Value', 1, 'FontSize', 12, ...
            'Callback', @(src,~) updateRole(i, src));
        
        % Style the drop-down
        set(popup, 'FontWeight', 'normal');
        popupHandles(i) = popup;
    end
    
    % Add a separator line
    uicontrol('Style', 'text', 'String', '', ...
        'Position', [leftMargin-10, headerY-(nCols+1)*rowHeight, winWidth-2*leftMargin+20, 2], ...
        'BackgroundColor', highlightColor);
    
    % Orientation and returns-to-scale assumption
    paramY = headerY - (nCols+1) * rowHeight - 50;
    if paramY < 100
        paramY = 100;
    end
    
    % Parameter-panel background
    uicontrol('Style', 'text', 'String', '', ...
        'Position', [leftMargin-10, paramY-5, winWidth-2*leftMargin+20, 100], ...
        'BackgroundColor', bgAccent);
    
    % Parameter-panel title
    uicontrol('Style', 'text', 'String', '模型参数设置', ...
        'Position', [leftMargin, paramY+65, 200, 25], ...
        'BackgroundColor', 'none', 'ForegroundColor', fgText, ...
        'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
    
    uicontrol('Style', 'text', 'String', '模型导向：', ...
        'Position', [leftMargin, paramY, 100, 40], ...
        'BackgroundColor', 'none', 'ForegroundColor', fgText, ...
        'HorizontalAlignment', 'left', 'FontSize', 12, 'FontWeight', 'normal');
    orient_popup = uicontrol('Style', 'popupmenu', ...
        'String', {'投入导向 (input)', '产出导向 (output)'}, ...
        'Position', [leftMargin+110, paramY, 200, 40], ...
        'BackgroundColor', bgPopup, 'ForegroundColor', fgPopup, ...
        'Value', 1, 'FontSize', 12, 'FontWeight', 'normal');
    
    uicontrol('Style', 'text', 'String', '规模报酬假设：', ...
        'Position', [leftMargin+330, paramY, 120, 40], ...
        'BackgroundColor', 'none', 'ForegroundColor', fgText, ...
        'HorizontalAlignment', 'left', 'FontSize', 12, 'FontWeight', 'normal');
    rts_popup = uicontrol('Style', 'popupmenu', ...
        'String', {'规模报酬可变 (VRS)', '规模报酬不变 (CRS)'}, ...
        'Position', [leftMargin+460, paramY, 200, 40], ...
        'BackgroundColor', bgPopup, 'ForegroundColor', fgPopup, ...
        'Value', 1, 'FontSize', 12, 'FontWeight', 'normal');
    
    uicontrol('Style', 'text', 'String', 'DEA模型类型：', ...
        'Position', [leftMargin, paramY-50, 120, 40], ...
        'BackgroundColor', 'none', 'ForegroundColor', fgText, ...
        'HorizontalAlignment', 'left', 'FontSize', 12, 'FontWeight', 'normal');
    dea_type_popup = uicontrol('Style', 'popupmenu', ...
        'String', {'传统DEA', '超效率DEA'}, ...
        'Position', [leftMargin+110, paramY-50, 200, 40], ...
        'BackgroundColor', bgPopup, 'ForegroundColor', fgPopup, ...
        'Value', 1, 'FontSize', 12, 'FontWeight', 'normal');
    
    % Confirm button
    btnY = paramY - 100;
    if btnY < 40
        btnY = 40;
    end
    
    % Add the button
    btn = uicontrol('Style', 'pushbutton', 'String', '确认并继续', ...
        'Position', [winWidth/2 - 90, btnY, 180, 50], ...
        'BackgroundColor', bgButton, 'ForegroundColor', fgButton, ...
        'FontWeight', 'bold', 'FontSize', 14, ...
        'Callback', @(src,~) submit());
    
    % Add the button hover effect
    set(btn, 'ButtonDownFcn', @(src,~) set(src, 'BackgroundColor', highlightColor));
    set(btn, 'KeyPressFcn', @(src,~) set(src, 'BackgroundColor', bgButton));
    set(btn, 'CreateFcn', @(src,~) set(src, 'BackgroundColor', bgButton));
    
    % Update the stored role (nested function, but the index is passed as an argument, so no warning)
    function updateRole(idx, src)
        strList = get(src, 'String');
        val = get(src, 'Value');
        roles{idx} = strList{val};
    end
    
    uiwait(fig);
    
    function submit()
        % Read the final role from every popupmenu
        for i = 1:nCols
            strList = get(popupHandles(i), 'String');
            val = get(popupHandles(i), 'Value');
            roles{i} = strList{val};
        end
        
        dmu_idx = find(strcmp(roles, 'DMU'));
        year_idx = find(strcmp(roles, '年份'));
        input_idx = find(strcmp(roles, '投入'));
        output_idx = find(strcmp(roles, '产出'));
        
        if isempty(dmu_idx)
            errordlg('必须指定一个 DMU 列', '配置错误');
            return;
        end
        if isempty(year_idx)
            errordlg('必须指定一个 年份 列', '配置错误');
            return;
        end
        if length(input_idx) < 1
            errordlg('至少指定一个投入指标列', '配置错误');
            return;
        end
        if length(output_idx) < 1
            errordlg('至少指定一个产出指标列', '配置错误');
            return;
        end
        
        orient_val = get(orient_popup, 'Value');
        if orient_val == 1
            orientation = 'input';
        else
            orientation = 'output';
        end
        
        rts_val = get(rts_popup, 'Value');
        if rts_val == 1
            RTS = 'vrs';
        else
            RTS = 'crs';
        end
        
        dea_type_val = get(dea_type_popup, 'Value');
        if dea_type_val == 1
            dea_type = 'traditional';
        else
            dea_type = 'super';
        end
        
        config.dmuColIdx = dmu_idx(1);
        config.yearColIdx = year_idx(1);
        config.inputColIdx = input_idx;
        config.outputColIdx = output_idx;
        config.orientation = orientation;
        config.RTS = RTS;
        config.dea_type = dea_type;
        
        delete(fig);
    end
end
