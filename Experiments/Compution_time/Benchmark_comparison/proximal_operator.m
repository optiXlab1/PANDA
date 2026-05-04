function varargout = proximal_operator(flag, x, upper_bound, lower_bound, gamma)
%PROXIMAL_OPERATOR
%   flag = 0 : prox
%       [proxg, g] = proximal_operator(0, x, ub, lb, gamma)
%
%   flag = 1 : dprox
%       dprox = proximal_operator(1, x, ub, lb, gamma)
%
% 完全参考 indBox_manual 的写法

    switch flag
        case 'box'
            proxg = min(upper_bound, max(lower_bound, x));
            g = 0;

            varargout{1} = proxg;
            varargout{2} = g;

        case 'box_grad'
            dprox = double((x > lower_bound) & (x < upper_bound));

            varargout{1} = dprox;

        otherwise
            error('Unknown flag');
    end
end
