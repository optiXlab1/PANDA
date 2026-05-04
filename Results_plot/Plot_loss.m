clc;
clear;
close all;

%% =========================================================
%  Loss comparison (4 methods)
%% =========================================================

%% ---------------------- 用户改这里 -----------------------
files = {
    'training_log_panda.csv'
    'training_log_acados.csv'
    'training_log_casadi.csv'
    'training_log_safepdp.csv'
    };

methodNames = {
    'PANDA'
    'acados'
    'CasADi'
    'SafePDP'
    };

saveFig = false;
figure_Name = 'loss_comparison_singlecol_half';
%% --------------------------------------------------------


%% ================== 统一风格（按之前约定） ==================
font_Name = 'Times New Roman';
figure_width = 5.0;      % 半栏宽度
figure_hight = 3.6;      % 小一些
figure_FontSize = 7;

axis_Width = 1.0;
line_Width = 1.1;
legendLW   = 0.6;

% ---- 配色 ----
base_panda    = [1.00 0.00 0.00];          % 红
base_acados   = [0.00 0.00 1.00];          % 蓝
base_casadi   = [0, 128, 0] / 256;         % 深绿色
base_safepdp  = [1.00 0.60 0.00];          % 橙

% ---- 稍微浅一点，但不要太淡 ----
alpha_line = 0.72;
color_panda   = blendWithWhite(base_panda,   alpha_line);
color_acados  = blendWithWhite(base_acados,  alpha_line);
color_casadi  = blendWithWhite(base_casadi,  alpha_line);
color_safepdp = blendWithWhite(base_safepdp, alpha_line);

set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','tex');

%% ================== 读取数据 ==================
nMethod = numel(files);
epochCell = cell(nMethod,1);
lossCell  = cell(nMethod,1);

for i = 1:nMethod
    T = readtable(files{i});
    epochCell{i} = pickNumericColumn(T, {'epoch'});
    lossCell{i}  = pickNumericColumn(T, {'loss'});
end

%% ================== 作图 ==================
fig = figure;
set(fig,'unit','centimeters','position',[5,5,figure_width,figure_hight]);
set(fig,'color','w');

ax = axes(fig);
hold(ax,'on');
box(ax,'on');

% 网格放底层
grid(ax,'on');
ax.Layer = 'bottom';
ax.GridLineStyle = '-';
ax.GridAlpha = 0.12;
ax.MinorGridAlpha = 0.08;
ax.XMinorGrid = 'off';
ax.YMinorGrid = 'off';

h = gobjects(nMethod,1);

% 绘图顺序：
% SafePDP -> acados -> PANDA -> CasADi
% 这样 CasADi 最后画，不容易被覆盖

% SafePDP
i = find(strcmp(methodNames,'SafePDP'));
h(i) = plot(epochCell{i}, lossCell{i}, '-', ...
    'LineWidth', line_Width, ...
    'Color', color_safepdp);

% acados
i = find(strcmp(methodNames,'acados'));
h(i) = plot(epochCell{i}, lossCell{i}, '-', ...
    'LineWidth', line_Width, ...
    'Color', color_acados);

% PANDA
i = find(strcmp(methodNames,'PANDA'));
h(i) = plot(epochCell{i}, lossCell{i}, '-', ...
    'LineWidth', line_Width + 0.15, ...
    'Color', color_panda);

% CasADi 最后画，虚线，深绿色
i = find(strcmp(methodNames,'CasADi'));
h(i) = plot(epochCell{i}, lossCell{i}, '--', ...
    'LineWidth', line_Width + 0.05, ...
    'Color', color_casadi);

set(ax, ...
    'FontName', font_Name, ...
    'FontSize', figure_FontSize, ...
    'LineWidth', axis_Width, ...
    'TickDir', 'in', ...
    'TickLength', [0.015 0.015], ...
    'XAxisLocation', 'bottom', ...
    'YAxisLocation', 'left');

xlabel('Epoch', ...
    'FontName', font_Name, ...
    'FontSize', figure_FontSize);

ylabel('Loss', ...
    'FontName', font_Name, ...
    'FontSize', figure_FontSize);

% 横轴范围
allEpoch = vertcat(epochCell{:});
xlim([min(allEpoch), max(allEpoch)]);

% 纵轴范围自动稍微放宽一点
allLoss = vertcat(lossCell{:});
ymin = min(allLoss);
ymax = max(allLoss);
yr = ymax - ymin;
if yr <= 0
    yr = 1;
end
ylim([ymin - 0.03*yr, ymax + 0.06*yr]);

%% ================== legend ==================
lgd = legend([h(1), h(2), h(3), h(4)], ...
    {'PANDA', 'acados', 'CasADi', 'SafePDP'}, ...
    'Location', 'northeast', ...
    'Box', 'on', ...
    'Interpreter', 'tex', ...
    'FontName', font_Name, ...
    'FontSize', 6.5);
lgd.LineWidth = legendLW;

%% ================== 导出 PDF 和 SVG ==================
set(fig,'PaperUnits','centimeters');
set(fig,'PaperPosition',[0,0,figure_width,figure_hight]);
set(fig,'PaperSize',[figure_width,figure_hight]);

print(fig, figure_Name, '-dpdf', '-painters');
print(fig, [figure_Name '.svg'], '-dsvg', '-painters');

if saveFig
    exportgraphics(fig, [figure_Name '.png'], 'Resolution', 300);
end

%% ====================== 局部函数 ==========================
function data = pickNumericColumn(T, candidates)
candidates = [candidates, {'loss_mean'}];  % 自动加入 'loss_mean'

vars = T.Properties.VariableNames;
idx = [];
for k = 1:numel(candidates)
    j = find(strcmp(vars, candidates{k}), 1);
    if ~isempty(j)
        idx = j;
        break;
    end
end
if isempty(idx)
    error('没找到列：%s', strjoin(candidates, ' / '));
end
data = T{:, idx};
end

function c = blendWithWhite(baseColor, alphaVal)
% 用与白色混合来模拟透明度效果
c = alphaVal * baseColor + (1 - alphaVal) * [1 1 1];
end