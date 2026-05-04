clc;
clear;
close all;

%% ================== 数据读取 ==================
% 读取 CSV 文件
data = readtable('PANOC_PANDA_summary.csv');

iter_panoc = data.PANOC_iters;
iter_panda = data.PANDA_iters;

%% ================== 绘图参数设置 ==================
font_Name = 'Times New Roman';   
figure_width = 10;               
figure_hight = 4;              
figure_FontSize = 8;             
line_Width = 1.0;                
axis_Width = 1.5;                
figure_Name = 'panoc_panda_iter_singlecol';

% 主配色：PANOC 蓝，PANDA 红
blueColor = [0 0 1];    % PANOC
redColor  = [1 0 0];    % PANDA

bgLeft  = [1.00 0.97 0.93];      
bgRight = [0.96 0.98 1.00];      
split_k = 25;                  

%% ================== 全局 LaTeX/字体设置 ==================
set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

%% ================== 数据（只取前50个 instant） ==================
n_show = min(50, length(iter_panoc));

iter_panoc = iter_panoc(1:n_show);
iter_panda = iter_panda(1:n_show);

k = 1:n_show;

%% ================== 设置图片大小 ==================
fig = figure;
set(fig,'unit','centimeters','position',[5,5,figure_width,figure_hight]);
set(fig,'color','w');
hold on;

%% ================== 纵轴范围 ==================
yl = [min([iter_panoc(:); iter_panda(:)]), ...
      max([iter_panoc(:); iter_panda(:)])];
dy = yl(2) - yl(1);
yl = [yl(1)-0.03*dy, yl(2)+0.08*dy];

%% ================== 背景分区 ==================
% patch([1 split_k split_k 1], ...
%       [yl(1) yl(1) yl(2) yl(2)], ...
%       bgLeft, 'EdgeColor','none', 'HandleVisibility','off');
% 
% patch([split_k length(k) length(k) split_k], ...
%       [yl(1) yl(1) yl(2) yl(2)], ...
%       bgRight, 'EdgeColor','none', 'HandleVisibility','off');
% 
% xline(split_k,'--','Color',[0.5 0.5 0.5],'LineWidth',0.8, ...
%       'HandleVisibility','off');

%% ================== 曲线 ==================
h1 = plot(k, iter_panoc, '-', ...
    'Color', blueColor, 'LineWidth', line_Width);

h2 = plot(k, iter_panda, '-', ...
    'Color', redColor, 'LineWidth', line_Width);

xlim([1,length(k)]);
ylim(yl);

xlabel('Time instant $k$', 'FontName', font_Name, 'FontSize', figure_FontSize);
ylabel('Iterations', 'FontName', font_Name, 'FontSize', figure_FontSize);

% text(0.26,0.92,'Enlargement triggered', ...
%     'Units','normalized', ...
%     'HorizontalAlignment','center', ...
%     'FontName',font_Name,'FontSize',7);
% 
% text(0.76,0.92,'No enlargement triggered', ...
%     'Units','normalized', ...
%     'HorizontalAlignment','center', ...
%     'FontName',font_Name,'FontSize',7);

legend([h1,h2], {'PANOC+','PANDA'}, ...
    'Location','northeast', ...
    'Box','off', ...
    'FontName',font_Name, ...
    'FontSize',figure_FontSize);

box on;
set(gca, ...
    'FontName',font_Name, ...
    'FontSize',figure_FontSize, ...
    'LineWidth',axis_Width, ...
    'TickDir','in', ...
    'Box','on', ...
    'Layer','top');

%% ================== 保存 PDF / SVG ==================
set(fig,'PaperUnits','centimeters');
set(fig,'PaperPosition',[0,0,figure_width,figure_hight]);
set(fig,'PaperSize',[figure_width,figure_hight]);

print(fig, figure_Name, '-dpdf', '-painters');
print(fig, figure_Name, '-dsvg', '-painters');