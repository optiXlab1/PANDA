clc;
clear;
close all;

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
figure_Name = 'timing_comparison_horizontal_singlecol';
%% --------------------------------------------------------


font_Name = 'Times New Roman';
figure_width = 10;
figure_hight = 4.0;
figure_FontSize = 8;

axis_Width = 1.2;
legendLW   = 0.6;

blueColor = [0.05 0.05 0.95];
redColor  = [0.95 0.05 0.05];

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

plotOrder = [4 3 2 1];
methodNames_plot = methodNames(plotOrder);
forward_plot     = forwardMean(plotOrder);
backward_plot    = backwardMean(plotOrder);
total_plot       = totalMean(plotOrder);

fig = figure;
set(fig,'unit','centimeters','position',[5,5,figure_width,figure_hight]);
set(fig,'color','w');

ax = axes(fig);
hold(ax,'on');
box(ax,'on');

Y = 1:nMethod;
barHeight = 0.72;

for i = 1:nMethod
    y0 = Y(i) - barHeight/2;

    % forward
    rectangle('Position',[0, y0, forward_plot(i), barHeight], ...
        'FaceColor', blueColor, 'EdgeColor', 'none');

    % backward
    rectangle('Position',[forward_plot(i), y0, backward_plot(i), barHeight], ...
        'FaceColor', redColor, 'EdgeColor', 'none');
end

set(ax, ...
    'YDir', 'reverse', ...
    'YTick', Y, ...
    'YTickLabel', methodNames_plot, ...
    'FontName', font_Name, ...
    'FontSize', figure_FontSize, ...
    'LineWidth', axis_Width, ...
    'TickDir', 'in', ...
    'TickLength', [0 0], ...
    'Layer', 'top');

xlabel('Mean time per sample (s)', ...
    'FontName', font_Name, ...
    'FontSize', figure_FontSize);

xmax = max(total_plot);
xlim([0, xmax*1.3]);

for i = 1:nMethod
    fwd = forward_plot(i);
    bwd = backward_plot(i);
    tot = total_plot(i);
    yc  = Y(i);

    if fwd > 0.015
        text(fwd/2, yc, sprintf('%.3fs', fwd), ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'FontName', font_Name, ...
            'FontSize', figure_FontSize, ...
            'Color', 'w');
    end

    if bwd > 0.015
        text(fwd + bwd/2, yc, sprintf('%.3fs', bwd), ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'FontName', font_Name, ...
            'FontSize', figure_FontSize, ...
            'Color', 'w');
    end

    text(tot + xmax*0.015, yc, sprintf('total: %.4fs', tot), ...
        'HorizontalAlignment','left', ...
        'VerticalAlignment','middle', ...
        'FontName', font_Name, ...
        'FontSize', figure_FontSize, ...
        'Color', [0.15 0.15 0.15]);
end

hLeg1 = patch(nan, nan, blueColor, 'EdgeColor', 'none');
hLeg2 = patch(nan, nan, redColor, 'EdgeColor', 'none');

lgd = legend([hLeg1, hLeg2], {'Forward', 'Backward'}, ...
    'Location', 'southeast', ...
    'Box', 'on', ...
    'Interpreter', 'tex', ...
    'FontName', font_Name, ...
    'FontSize', figure_FontSize);
lgd.LineWidth = legendLW;

set(fig,'PaperUnits','centimeters');
set(fig,'PaperPosition',[0,0,figure_width,figure_hight]);
set(fig,'PaperSize',[figure_width,figure_hight]);

print(fig, figure_Name, '-dpdf', '-painters');
print(fig, [figure_Name '.svg'], '-dsvg', '-painters');

if saveFig
    exportgraphics(fig, [figure_Name '.png'], 'Resolution', 300);
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