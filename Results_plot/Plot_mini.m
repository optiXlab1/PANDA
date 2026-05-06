clc;
clear;
close all;

files = {
    'training_log_panda.csv'
    'training_log_acados.csv'
    };

methodNames = {
    'PANDA'
    'acados'
    };

figure_Name = 'timing_inset_acados_panda_clean';
saveFig = false;
%% --------------------------------------------------------

font_Name = 'Times New Roman';

figure_width = 4.2;      % cm
figure_hight = 2.0;      % cm
figure_FontSize = 6.5;

blueColor   = [0.05 0.05 0.95];   % Forward
orangeColor = [0.95 0.05 0.05];   % Backward

set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','tex');

nMethod = numel(files);

forwardMean  = zeros(nMethod,1);
backwardMean = zeros(nMethod,1);
totalMean    = zeros(nMethod,1);

for i = 1:nMethod
    T = readtable(files{i});

    fwd = pickNumericColumn(T, {'forward_mean_time','forward_time_mean_s'});
    bwd = pickNumericColumn(T, {'backward_mean_time','backward_time_mean_s'});

    forwardMean(i)  = mean(fwd, 'omitnan');
    backwardMean(i) = mean(bwd, 'omitnan');
    totalMean(i)    = forwardMean(i) + backwardMean(i);
end

plotOrder = [2 1];
methodNames_plot = methodNames(plotOrder);
forward_plot     = forwardMean(plotOrder);
backward_plot    = backwardMean(plotOrder);
total_plot       = totalMean(plotOrder);

fig = figure;
set(fig,'unit','centimeters','position',[5,5,figure_width,figure_hight]);
set(fig,'color','w');

ax = axes(fig);
hold(ax,'on');

Y = [2, 1];
barHeight = 0.36;

for i = 1:2
    y0 = Y(i) - barHeight/2;

    % forward
    rectangle('Position',[0, y0, forward_plot(i), barHeight], ...
        'FaceColor', blueColor, 'EdgeColor', 'none');

    % backward
    rectangle('Position',[forward_plot(i), y0, backward_plot(i), barHeight], ...
        'FaceColor', orangeColor, 'EdgeColor', 'none');
end

xmax = max(total_plot);
xlim([-0.0012, xmax*1.20]);
ylim([0.4, 2.6]);

bwdlen = [0.0003,0.0005];
for i = 1:2
    fwd = forward_plot(i);
    bwd = backward_plot(i);
    yc  = Y(i);

    y_text = yc + 0.5;

    text(fwd/2, y_text, sprintf('%.4f', fwd), ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','top', ...
        'FontName', font_Name, ...
        'FontSize', figure_FontSize, ...
        'Color', 'k');

    text(fwd + bwdlen(i), y_text, sprintf('%.4f', bwd), ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','top', ...
        'FontName', font_Name, ...
        'FontSize', figure_FontSize, ...
        'Color', 'k');
end

axis(ax, 'off');

set(fig,'PaperUnits','centimeters');
set(fig,'PaperPosition',[0,0,figure_width,figure_hight]);
set(fig,'PaperSize',[figure_width,figure_hight]);

print(fig, figure_Name, '-dpdf', '-painters');
print(fig, [figure_Name '.svg'], '-dsvg', '-painters');

if saveFig
    exportgraphics(fig, [figure_Name '.png'], 'Resolution', 300, 'BackgroundColor', 'none');
end

function data = pickNumericColumn(T, candidates)
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
        error('Not found column：%s', strjoin(candidates, ' / '));
    end
    data = T{:, idx};
    data = data(~isnan(data));
end