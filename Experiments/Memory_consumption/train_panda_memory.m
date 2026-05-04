%% ===================== PANDA Memory Test Only =====================
clear; clc; close all;
clear mex;
import casadi.*

%% ------------------- 参数设置 -------------------
nx = 3; 
nu = 2; 
N  = 80; 
L  = 0.5; 
dt = 0.1;

x_ref_default  = [3.5;1.5;0];

umin = -0.8; 
umax = 0.8;
lb = umin*ones(nu*N,1); 
ub = umax*ones(nu*N,1);

%% ------------------- Student parameters -------------------
student_pars.q          = [0.34; 0.91; 0.12];
student_pars.r          = [0.07; 0.02];
student_pars.w          = [5.6; 3.3; 0.4];
student_pars.eta_circle = 100.0;
student_pars.circle_r   = 0.5;

circle_c_default = [1.55; 1.45];

theta_val = [student_pars.q; ...
             student_pars.r; ...
             student_pars.w; ...
             student_pars.eta_circle; ...
             student_pars.circle_r];

%% ------------------- Imitation weights -------------------
wx = [10; 10; 2];
wu = [2; 2];

%% ------------------- Paths -------------------
mex_dir = fullfile(pwd,'mex');
if ~exist(mex_dir,'dir')
    mkdir(mex_dir);
end

addpath(mex_dir);
addpath(genpath('../../casadi-3.7.2-windows64-matlab2018b'));
addpath(genpath('../backward'));

%% ------------------- Load Demonstration -------------------
if ~exist('teacher_record.mat','file')
    error('teacher_record.mat not found. Please generate record first.');
end

load('teacher_record.mat');   % record

%% ------------------- Batch setup -------------------
batch_stride = 1;
batch_idx    = 1:batch_stride:numel(record);
batch_size   = numel(batch_idx);

fprintf('Total record length = %d\n', numel(record));
fprintf('Batch stride        = %d\n', batch_stride);
fprintf('Batch size          = %d\n', batch_size);

%% ------------------- Variable dimension -------------------
variable_dim = nx + nx + 2 + nu*N + nx*(N+1);

fprintf('Variable dimension = %d\n', variable_dim);

%% ------------------- CasADi symbolics -------------------
U_sym        = SX.sym('U', nu*N);
theta_sym    = SX.sym('theta', numel(theta_val));
variable_sym = SX.sym('variable', variable_dim);
v_sym        = SX.sym('v', nu*N);

%% ------------------- unpack theta -------------------
q          = theta_sym(1:3);
r          = theta_sym(4:5);
w          = theta_sym(6:8);
eta_circle = theta_sym(9);
circle_r   = theta_sym(10);

%% ------------------- unpack variable -------------------
offset = 0;

x0_sym = variable_sym(offset + (1:nx));
offset = offset + nx;

xref_sym = variable_sym(offset + (1:nx));
offset = offset + nx;

circle_c_sym = variable_sym(offset + (1:2));
offset = offset + 2;

U_tar_sym = reshape(variable_sym(offset + (1:nu*N)), nu, N);
offset = offset + nu*N;

X_tar_sym = reshape(variable_sym(offset + (1:nx*(N+1))), nx, N+1);

%% ------------------- Forward Rollout + Loss -------------------
xk   = x0_sym;
J    = SX(0);
Lval = SX(0);

for k = 1:N
    uk = U_sym((k-1)*nu+1:k*nu);

    dx = xk - xref_sym;

    % forward objective J
    J = J ...
        + q(1)*dx(1)^2 ...
        + q(2)*dx(2)^2 ...
        + q(3)*dx(3)^2 ...
        + r(1)*uk(1)^2 ...
        + r(2)*uk(2)^2;

    z = xk(1:2);
    h = 1 - ((z - circle_c_sym).' * (z - circle_c_sym)) / circle_r^2;
    psi = 0.5 * eta_circle * if_else(h > 0, h^2, 0);
    J = J + psi;

    % imitation loss Lval
    eu = uk - U_tar_sym(:,k);
    ex = xk - X_tar_sym(:,k);

    Lval = Lval ...
        + wu(1)*eu(1)^2 ...
        + wu(2)*eu(2)^2 ...
        + wx(1)*ex(1)^2 ...
        + wx(2)*ex(2)^2 ...
        + wx(3)*ex(3)^2;

    % Euler rollout
    xk = euler_step_symbolic(xk, uk, dt, L);
end

dxN = xk - xref_sym;

J = J ...
    + w(1)*dxN(1)^2 ...
    + w(2)*dxN(2)^2 ...
    + w(3)*dxN(3)^2;

% terminal imitation state loss
exN = xk - X_tar_sym(:,N+1);
Lval = Lval ...
    + wx(1)*exN(1)^2 ...
    + wx(2)*exN(2)^2 ...
    + wx(3)*exN(3)^2;

grad_J = gradient(J, U_sym);
grad_L = gradient(Lval, U_sym);

vjp_theta = jtimes(grad_J, theta_sym, v_sym, true);
Hv        = jtimes(grad_J, U_sym, v_sym, false);

%% ------------------- Generate and MEX -------------------
clear mex;

delete_if_exists(fullfile(mex_dir, ['tl_f_and_grad_u.' mexext]));
delete_if_exists(fullfile(mex_dir, ['tl_L_and_grad_u.' mexext]));
delete_if_exists(fullfile(mex_dir, ['tl_vjp_f_u_theta.' mexext]));
delete_if_exists(fullfile(mex_dir, ['tl_hvp_f_u.' mexext]));

delete_if_exists('tl_f_and_grad_u.c');
delete_if_exists('tl_L_and_grad_u.c');
delete_if_exists('tl_vjp_f_u_theta.c');
delete_if_exists('tl_hvp_f_u.c');

Function('tl_f_and_grad_u', ...
    {U_sym, theta_sym, variable_sym}, ...
    {J, grad_J}) ...
    .generate('tl_f_and_grad_u.c', struct('mex',true));

Function('tl_L_and_grad_u', ...
    {U_sym, variable_sym}, ...
    {Lval, grad_L}) ...
    .generate('tl_L_and_grad_u.c', struct('mex',true));

Function('tl_vjp_f_u_theta', ...
    {v_sym, U_sym, theta_sym, variable_sym}, ...
    {vjp_theta}) ...
    .generate('tl_vjp_f_u_theta.c', struct('mex',true));

Function('tl_hvp_f_u', ...
    {v_sym, U_sym, theta_sym, variable_sym}, ...
    {Hv}) ...
    .generate('tl_hvp_f_u.c', struct('mex',true));

mex('-outdir', mex_dir, 'tl_f_and_grad_u.c');
mex('-outdir', mex_dir, 'tl_L_and_grad_u.c');
mex('-outdir', mex_dir, 'tl_vjp_f_u_theta.c');
mex('-outdir', mex_dir, 'tl_hvp_f_u.c');

fprintf('All MEX files regenerated in: %s\n', mex_dir);

%% ------------------- PANOC/PANDA setup -------------------
problem.dimension = N*nu;
problem.constraint_type = 'costum';
problem.constraint = @(x,gamma) indBox_manual(x, ub, lb, gamma);

solver_params.tolerance       = 1e-4;
solver_params.buffer_size     = 10;
solver_params.max_iterations  = 1000;
solver_params.max_stable_iter = 80;

Jprox_u    = @(x,gamma) proximal_operator('box_grad', x, ub, lb, gamma);
oracle_L   = @(u,variable) tl_L_and_grad_u(u, variable);
oracle_vjp = @(v,u,theta,variable) tl_vjp_f_u_theta(v, u, theta, variable);
oracle_hvp = @(v,u,theta,variable) tl_hvp_f_u(v, u, theta, variable);

opts.tol   = 1e-4;
opts.maxit = 200;

%% ------------------- Training hyperparameters -------------------
epochs = 1;
lr     = 2e-4;

theta = theta_val;

u_guess_bank = zeros(problem.dimension, batch_size);

%% ------------------- PANOC init memory -------------------
mem_init_before = get_matlab_memory_mb();
fprintf('[panoc init] before init | mem = %.2f MB\n', mem_init_before);

panoc('init', problem, solver_params);

mem_init_after = get_matlab_memory_mb();
fprintf('[panoc init] after  init | mem = %.2f MB | delta = %+.2f MB\n', ...
    mem_init_after, mem_init_after - mem_init_before);

%% ------------------- Training loop: memory output only -------------------
for ep = 1:epochs
    grad_acc = zeros(size(theta));
    loss_acc = 0;

    fwd_time_sum = 0;
    bwd_time_sum = 0;
    fwd_iter_sum = 0;
    gamma_sum    = 0;
    back_iter_sum = 0;
    relres_sum = 0;
    back_iter_count = 0;
    relres_count = 0;

    for b = 1:batch_size
        idx = batch_idx(b);

        variable_b = pack_record_variable( ...
            record(idx), ...
            nx, nu, N, ...
            x_ref_default, ...
            circle_c_default);

        %% ----- forward -----
        oracle_fg = @(u) tl_f_and_grad_u(u, theta, variable_b);

        mem_fwd_before = get_matlab_memory_mb();
        fprintf('[ep=%03d, b=%04d] before forward  | mem = %.2f MB\n', ...
            ep, b, mem_fwd_before);

        t_fwd = tic;
        [u_star, iters, gamma] = solve_once(u_guess_bank(:,b), oracle_fg);
        fwd_time = toc(t_fwd);

        mem_fwd_after = get_matlab_memory_mb();
        fprintf('[ep=%03d, b=%04d] after  forward  | mem = %.2f MB | delta = %+.2f MB\n', ...
            ep, b, mem_fwd_after, mem_fwd_after - mem_fwd_before);

        %% ----- backward -----
        mem_bwd_before = get_matlab_memory_mb();
        fprintf('[ep=%03d, b=%04d] before backward | mem = %.2f MB\n', ...
            ep, b, mem_bwd_before);

        t_bwd = tic;
        [Lval, dLdtheta, info] = panda_backward( ...
            u_star, theta, variable_b, gamma, ...
            Jprox_u, oracle_hvp, oracle_L, oracle_vjp, opts);
        bwd_time = toc(t_bwd);

        mem_bwd_after = get_matlab_memory_mb();
        fprintf('[ep=%03d, b=%04d] after  backward | mem = %.2f MB | delta = %+.2f MB\n', ...
            ep, b, mem_bwd_after, mem_bwd_after - mem_bwd_before);

        %% ----- update only, no file recording -----
        Lval = double(full(Lval));
        dLdtheta = double(full(dLdtheta(:)));

        grad_acc = grad_acc + dLdtheta;
        loss_acc = loss_acc + Lval;

        fwd_time_sum = fwd_time_sum + fwd_time;
        bwd_time_sum = bwd_time_sum + bwd_time;
        fwd_iter_sum = fwd_iter_sum + iters;
        gamma_sum    = gamma_sum + gamma;

        if isfield(info,'iter')
            back_iter_sum = back_iter_sum + info.iter;
            back_iter_count = back_iter_count + 1;
        end

        if isfield(info,'relres')
            relres_sum = relres_sum + info.relres;
            relres_count = relres_count + 1;
        end

        u_guess_bank(:,b) = zeros(problem.dimension,1);
    end

    grad_acc = grad_acc / batch_size;
    loss_acc = loss_acc / batch_size;

    theta = theta - lr * grad_acc;
    theta = max(theta, 0.01);

    fwd_mean = fwd_time_sum / batch_size;
    bwd_mean = bwd_time_sum / batch_size;
    fwd_it_mean = fwd_iter_sum / batch_size;
    gamma_mean = gamma_sum / batch_size;

    if back_iter_count > 0
        back_it_mean = back_iter_sum / back_iter_count;
    else
        back_it_mean = NaN;
    end

    if relres_count > 0
        relres_mean = relres_sum / relres_count;
    else
        relres_mean = NaN;
    end

    fprintf(['ep=%03d | L=%.6e | fwd_mean=%.6fs | bwd_mean=%.6fs | ', ...
             'fwd_it=%.2f | back_it=%.2f | gamma=%.3e | relres=%.2e\n'], ...
             ep, ...
             loss_acc, ...
             fwd_mean, ...
             bwd_mean, ...
             fwd_it_mean, ...
             back_it_mean, ...
             gamma_mean, ...
             relres_mean);

    fprintf('  theta = [');
    fprintf(' %.4f', theta);
    fprintf(' ]\n');
end

%% ------------------- PANOC cleanup memory -------------------
mem_cleanup_before = get_matlab_memory_mb();
fprintf('[panoc cleanup] before cleanup | mem = %.2f MB\n', mem_cleanup_before);

panoc('cleanup');

mem_cleanup_after = get_matlab_memory_mb();
fprintf('[panoc cleanup] after  cleanup | mem = %.2f MB | delta = %+.2f MB\n', ...
    mem_cleanup_after, mem_cleanup_after - mem_cleanup_before);

clear mex;

fprintf('\nMemory-only test finished. No CSV/MAT/result files were written.\n');

%% ==================== Helper Functions ====================

function mem_mb = get_matlab_memory_mb()
    % Current MATLAB process memory usage in MB.
    if ispc
        m = memory;
        mem_mb = m.MemUsedMATLAB / 1024^2;
    else
        rt = java.lang.Runtime.getRuntime();
        mem_mb = double(rt.totalMemory() - rt.freeMemory()) / 1024^2;
    end
end

function variable = pack_record_variable(rec, nx, nu, N, xref_default, circle_c_default)

    if isfield(rec, 'x0')
        x0 = rec.x0(:);
    elseif isfield(rec, 'P')
        x0 = rec.P(1:nx);
    else
        error('record sample does not contain x0 or P.');
    end

    if isfield(rec, 'P')
        xref = rec.P(nx+1:2*nx);
    else
        xref = xref_default(:);
    end

    if isfield(rec, 'circle_c')
        circle_c = rec.circle_c(:);
    else
        circle_c = circle_c_default(:);
    end

    if isfield(rec, 'U_star')
        U_tar = rec.U_star;
        if isvector(U_tar)
            U_tar = reshape(U_tar(:), nu, N);
        end
    elseif isfield(rec, 'u_star')
        U_tar = reshape(rec.u_star(:), nu, N);
    else
        error('record sample does not contain U_star or u_star.');
    end

    if isfield(rec, 'X_star')
        X_tar = rec.X_star;
    else
        error('record sample does not contain X_star.');
    end

    if size(X_tar,2) ~= N+1
        error('X_star must be nx x (N+1). Current N=%d, but size(X_star,2)=%d.', ...
            N, size(X_tar,2));
    end

    variable = [ ...
        x0(:); ...
        xref(:); ...
        circle_c(:); ...
        U_tar(:); ...
        X_tar(:)];
end

function [u_sol, iters, gamma] = solve_once(u_init, oracle_fun)
    u_sol = u_init(:);
    [iters, gamma] = panoc('solve', u_sol, oracle_fun);
end

function xnext = euler_step_symbolic(x,u,dt,L)
    dx = dyn_symbolic(x,u,L);
    xnext = x + dt*dx;
end

function dx = dyn_symbolic(x,u,L)
    th = x(3); 
    ux = u(1); 
    uy = u(2);

    thdot = (uy*cos(th) - ux*sin(th)) / L;
    pxdot = ux + L*sin(th)*thdot;
    pydot = uy - L*cos(th)*thdot;

    dx = [pxdot; pydot; thdot];
end

function delete_if_exists(fname)
    if exist(fname, 'file')
        delete(fname);
    end
end