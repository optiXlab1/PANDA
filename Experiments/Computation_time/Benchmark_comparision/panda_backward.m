function [L, dLdtheta, info] = panda_backward( ...
    u_star, theta, variable, gamma, ...
    Jprox_fun, hvp_fun, L_and_grad_fun, vjp_fun, opts)
% PANDA_BACKWARD
% Generic backward for PANOC/PANDA family.
%
% Inputs:
%   u_star         : forward solution
%   theta          : differentiated parameters
%   variable       : non-differentiated external variables
%   gamma          : forward stepsize at solution
%   Jprox_fun      : e.g. proximal_operator('box_grad', ...)
%   hvp_fun        : hvp_fun(v, u_star, theta, variable)
%   L_and_grad_fun : L_and_grad_fun(u_star, variable)
%   vjp_fun        : vjp_fun(v, u_star, theta, variable)

if nargin < 9 || isempty(opts), opts = struct(); end
if ~isfield(opts,'tol'),   opts.tol   = 1e-10; end
if ~isfield(opts,'maxit'), opts.maxit = 200;   end

u_star   = u_star(:);
theta    = theta(:);
variable = variable(:);

[L, B] = L_and_grad_fun(u_star, variable);
B = B(:);

J = Jprox_fun(u_star, gamma);
J = J(:);

Aidx = find(J == 0);
Fidx = find(J ~= 0);

n  = numel(u_star);
nf = numel(Fidx);

x = zeros(n,1);
x(Aidx) = B(Aidx);

if nf == 0
    v = -gamma * J .* x;
    dLdtheta = vjp_fun(v, u_star, theta, variable);
    dLdtheta = dLdtheta(:);

    info.flag     = 0;
    info.relres   = 0;
    info.iter     = 0;
    info.n        = n;
    info.n_free   = 0;
    info.n_active = numel(Aidx);
    return;
end

HFFop = @(xF) localHFFop(xF, Fidx, gamma, hvp_fun, u_star, theta, variable, n);
rhsF  = B(Fidx);

[xF, flag, relres, iter] = minres(HFFop, rhsF, opts.tol, opts.maxit);

x(Fidx) = xF;

v = -gamma * J .* x;
dLdtheta = vjp_fun(v, u_star, theta, variable);
dLdtheta = dLdtheta(:);

info.flag     = flag;
info.relres   = relres;
info.iter     = iter;
info.n        = n;
info.n_free   = nf;
info.n_active = numel(Aidx);
end

function yF = localHFFop(xF, Fidx, gamma, hvp_fun, u_star, theta, variable, n)
    x_full = zeros(n,1);
    x_full(Fidx) = xF;

    Hx = hvp_fun(x_full, u_star, theta, variable);
    Hx = Hx(:);

    yF = gamma * Hx(Fidx);
end