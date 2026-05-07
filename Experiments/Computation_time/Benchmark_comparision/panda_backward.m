function [L, dLdtheta, info] = panda_backward( ...
    u_star, theta, variable, gamma, ...
    Jprox_fun, hvp_fun, L_and_grad_fun, vjp_fun, opts)
% PANDA_BACKWARD
% Reduced active-set backward for PANOC/PANDA family.
%
% Assumption:
%   P is diagonal, given by J = diag(P).
%
% Procedure:
%   1. Identify free set F and active set A from J.
%   2. Build reduced operator H_FF.
%   3. Check whether H_FF is numerically symmetric.
%   4. If symmetric, solve H_FF * lambda_F = dL/d u_F by MINRES.
%      Otherwise, solve it by GMRES.
%   5. Compute dL/dtheta by VJP with direction v = -P^T lambda.
%
% Inputs:
%   u_star         : forward solution
%   theta          : differentiated parameters
%   variable       : non-differentiated external variables
%   gamma          : forward stepsize at solution
%   Jprox_fun      : returns diagonal of prox Jacobian, e.g., box mask
%   hvp_fun        : hvp_fun(v, u_star, theta, variable)
%   L_and_grad_fun : L_and_grad_fun(u_star, variable)
%   vjp_fun        : vjp_fun(v, u_star, theta, variable)
%   opts           : options
%
% opts fields:
%   opts.tol              : linear solver tolerance
%   opts.maxit            : maximum iterations
%   opts.restart          : GMRES restart length
%   opts.sym_tol          : symmetry check tolerance
%   opts.sym_num_tests    : number of random tests for symmetry
%   opts.force_solver     : 'auto', 'minres', or 'gmres'
%   opts.recover_active   : whether to recover lambda_A for info

%% -------------------- Options --------------------
if nargin < 9 || isempty(opts)
    opts = struct();
end

if ~isfield(opts, 'tol')
    opts.tol = 1e-10;
end

if ~isfield(opts, 'maxit')
    opts.maxit = 200;
end

if ~isfield(opts, 'restart')
    opts.restart = 40;
end

if ~isfield(opts, 'sym_tol')
    opts.sym_tol = 1e-8;
end

if ~isfield(opts, 'sym_num_tests')
    opts.sym_num_tests = 5;
end

if ~isfield(opts, 'force_solver')
    opts.force_solver = 'auto';
end

if ~isfield(opts, 'recover_active')
    opts.recover_active = true;
end

%% -------------------- Vector formatting --------------------
u_star   = u_star(:);
theta    = theta(:);
variable = variable(:);

n = numel(u_star);

%% -------------------- Outer loss and gradient --------------------
[L, B] = L_and_grad_fun(u_star, variable);
B = B(:);

if numel(B) ~= n
    error('L_and_grad_fun must return grad_L with the same length as u_star.');
end

%% -------------------- Prox Jacobian diagonal --------------------
J = Jprox_fun(u_star, gamma);
J = J(:);

if numel(J) ~= n
    error('Jprox_fun must return a vector with the same length as u_star.');
end

% For numerical robustness, classify by threshold.
% J near 1: free variable.
% J near 0: active variable.
Jtol = 1e-12;
Fidx = find(abs(J) > Jtol);
Aidx = find(abs(J) <= Jtol);

nf = numel(Fidx);
na = numel(Aidx);

lambda = zeros(n, 1);

%% -------------------- All variables active --------------------
if nf == 0
    % If P = 0, then v = -P^T lambda = 0 for theta-independent prox.
    v = zeros(n, 1);

    dLdtheta = vjp_fun(v, u_star, theta, variable);
    dLdtheta = dLdtheta(:);

    info.flag         = 0;
    info.relres       = 0;
    info.iter         = 0;
    info.iter_raw     = 0;
    info.resvec       = [];
    info.n            = n;
    info.n_free       = 0;
    info.n_active     = na;
    info.method       = 'all-active';
    info.is_symmetric = true;
    info.sym_error    = 0;
    info.gamma        = gamma;
    return;
end

%% -------------------- Reduced Hessian operator H_FF --------------------
HFFop = @(xF) localHFFop( ...
    xF, Fidx, hvp_fun, u_star, theta, variable, n);

rhsF = B(Fidx);

%% -------------------- Symmetry check --------------------
switch lower(opts.force_solver)
    case 'minres'
        is_symmetric = true;
        sym_error = 0;

    case 'gmres'
        is_symmetric = false;
        sym_error = NaN;

    case 'auto'
        [is_symmetric, sym_error] = localCheckSymmetry( ...
            HFFop, nf, opts.sym_num_tests, opts.sym_tol);

    otherwise
        error('opts.force_solver must be ''auto'', ''minres'', or ''gmres''.');
end

%% -------------------- Solve reduced system --------------------
% Reduced adjoint system:
%
%     H_FF * lambda_F = B_F.
%
% If H_FF is symmetric, use MINRES.
% Otherwise, use GMRES.

if is_symmetric
    [lambdaF, flag, relres, iter_raw, resvec] = minres( ...
        HFFop, rhsF, opts.tol, opts.maxit);

    iter_total = localIterToScalar(iter_raw, []);
    solver_used = 'reduced MINRES';

else
    lambdaF0 = zeros(nf, 1);

    if isempty(opts.restart)
        [lambdaF, flag, relres, iter_raw, resvec] = gmres( ...
            HFFop, rhsF, [], opts.tol, opts.maxit, [], [], lambdaF0);

        iter_total = localIterToScalar(iter_raw, []);
    else
        [lambdaF, flag, relres, iter_raw, resvec] = gmres( ...
            HFFop, rhsF, opts.restart, opts.tol, opts.maxit, [], [], lambdaF0);

        iter_total = localIterToScalar(iter_raw, opts.restart);
    end

    solver_used = 'reduced GMRES';
end

lambda(Fidx) = lambdaF(:);

%% -------------------- Optional active component recovery --------------------
% From the full adjoint block system for box constraints:
%
%     H_FF lambda_F = B_F,
%     H_AF lambda_F + (1/gamma) lambda_A = B_A.
%
% Hence:
%
%     lambda_A = gamma * (B_A - H_AF lambda_F).
%
% Note that lambda_A is NOT needed for theta-independent box constraints,
% because v = -P^T lambda and P_A = 0. It is recovered only for diagnostics.

if opts.recover_active && na > 0
    x_full = zeros(n, 1);
    x_full(Fidx) = lambdaF(:);

    Hx = hvp_fun(x_full, u_star, theta, variable);
    Hx = Hx(:);

    if numel(Hx) ~= n
        error('hvp_fun must return a vector with the same length as u_star.');
    end

    lambda(Aidx) = gamma * (B(Aidx) - Hx(Aidx));
end

%% -------------------- Gradient recovery --------------------
% For theta-independent prox/constraints, S_gamma = 0.
% Then:
%
%     dL/dtheta = - lambda^T P * d^2 ell / du dtheta.
%
% Therefore the VJP direction is:
%
%     v = -P^T lambda.
%
% Since P is diagonal here:
v = -J .* lambda;

dLdtheta = vjp_fun(v, u_star, theta, variable);
dLdtheta = dLdtheta(:);

%% -------------------- Info --------------------
info.flag         = flag;
info.relres       = relres;
info.iter         = iter_total;
info.iter_raw     = iter_raw;
info.resvec       = resvec;
info.n            = n;
info.n_free       = nf;
info.n_active     = na;
info.Fidx         = Fidx;
info.Aidx         = Aidx;
info.method       = solver_used;
info.is_symmetric = is_symmetric;
info.sym_error    = sym_error;
info.restart      = opts.restart;
info.maxit        = opts.maxit;
info.tol          = opts.tol;
info.gamma        = gamma;

end


%% ========================================================================
function yF = localHFFop(xF, Fidx, hvp_fun, u_star, theta, variable, n)
% LOCALHFFOP
%
% Matrix-free reduced Hessian operator:
%
%     yF = H_FF * xF.
%
% Construct a full-space vector whose active components are zero, evaluate
% HVP, and extract the free components.

xF = xF(:);

x_full = zeros(n, 1);
x_full(Fidx) = xF;

Hx = hvp_fun(x_full, u_star, theta, variable);
Hx = Hx(:);

if numel(Hx) ~= n
    error('hvp_fun must return a vector with length n.');
end

yF = Hx(Fidx);

end


%% ========================================================================
function [is_symmetric, sym_error] = localCheckSymmetry(Aop, n, num_tests, tol)
% LOCALCHECKSYMMETRY
%
% Random bilinear-form symmetry test:
%
%     a^T A b  ?=  b^T A a.
%
% This is a matrix-free numerical test. It does not prove symmetry, but is
% usually enough to decide whether MINRES is appropriate.

if n == 0
    is_symmetric = true;
    sym_error = 0;
    return;
end

sym_error = 0;

for i = 1:num_tests
    a = randn(n, 1);
    b = randn(n, 1);

    Aa = Aop(a);
    Ab = Aop(b);

    lhs = a' * Ab;
    rhs = b' * Aa;

    denom = max([1, abs(lhs), abs(rhs)]);
    err_i = abs(lhs - rhs) / denom;

    sym_error = max(sym_error, err_i);
end

is_symmetric = sym_error <= tol;

end


%% ========================================================================
function iter_total = localIterToScalar(iter_raw, restart)
% LOCALITERTOSCALAR
%
% MATLAB gmres may return:
%   iter_raw = scalar
%   iter_raw = [outer_iter, inner_iter]
%
% MATLAB minres usually returns scalar iter.
% Convert all cases to scalar total iteration number.

if isempty(iter_raw)
    iter_total = 0;
    return;
end

if numel(iter_raw) == 1
    iter_total = iter_raw;
    return;
end

if numel(iter_raw) == 2
    outer_iter = iter_raw(1);
    inner_iter = iter_raw(2);

    if isempty(restart)
        iter_total = inner_iter;
    else
        iter_total = max(outer_iter - 1, 0) * restart + inner_iter;
    end

    return;
end

iter_total = iter_raw(end);

end
