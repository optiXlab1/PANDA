clc;
clear;
close all;

T_panoc = readtable('PANOC_RES_0.csv');
T_panda = readtable('PANDA_RES_0.csv');

res_panoc = T_panoc{:,2};
res_panda = T_panda{:,2};

k_panoc = 1:length(res_panoc);
k_panda = 1:length(res_panda);

font_Name = 'Times New Roman';
figure_width = 10;        % cm
figure_hight = 4;       % cm
figure_FontSize = 8;
line_Width = 1.0;
axis_Width = 1.5;
figure_Name = 'panoc_panda_residual_singlecol';

blueColor = [0 0 1];      % PANOC
redColor  = [1 0 0];      % PANDA

set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

fig = figure;
set(fig,'unit','centimeters','position',[5,5,figure_width,figure_hight]);
set(fig,'color','w');

box on;

h1 = semilogy(k_panoc, res_panoc, '-', ...
    'Color', blueColor, ...
    'LineWidth', line_Width + 0.2, ...
    'DisplayName', 'PANOC');
hold on;

h2 = semilogy(k_panda, res_panda, '-', ...
    'Color', redColor, ...
    'LineWidth', line_Width + 0.2, ...
    'DisplayName', 'PANDA');

xlim([1, length(k_panoc)+10]);
set(gca, 'YScale', 'log');
ylim([1e-4, 1e2]);
yticks(10.^(-4:2));

xlabel('Iteration', ...
    'FontName', font_Name, ...
    'FontSize', figure_FontSize);

ylabel('Residual', ...
    'FontName', font_Name, ...
    'FontSize', figure_FontSize);

legend([h1,h2], {'PANOC+','PANDA'}, ...
    'Location','northeast', ...
    'Box','off', ...
    'FontName',font_Name, ...
    'FontSize',figure_FontSize);

set(gca, ...
    'FontName', font_Name, ...
    'FontSize', figure_FontSize, ...
    'LineWidth', axis_Width, ...
    'TickDir', 'in', ...
    'Box', 'on', ...
    'Layer', 'top');
set(gca,'YMinorTick','off');

set(fig,'PaperUnits','centimeters');
set(fig,'PaperPosition',[0,0,figure_width,figure_hight]);
set(fig,'PaperSize',[figure_width,figure_hight]);

print(fig, figure_Name, '-dpdf', '-painters');
print(fig, figure_Name, '-dsvg', '-painters');