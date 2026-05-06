%% ===================== Full PANDA Training over All Samples =====================
clear; clc; close all;
clear mex;
import casadi.*

%% ------------------- Parameter setup -------------------
nx = 3; 
nu = 2; 
N  = 40; 
L  = 0.5; 
dt = 0.1;

x_init_default = [0;0;pi/5]; 
x_ref_default  = [3.5;1.5;0];

umin = -0.8; 
umax = 0.8;
lb = umin*ones(nu*N,1); 
ub = umax*ones(nu*N,1);

U_zero = zeros(nu*N,1);

%% ------------------- Student parameters -------------------
student_pars.q          = [0.34; 0.91; 0.12];
student_pars.r          = [0.07; 0.02];
student_pars.w          = [5.6; 3.3; 0.4];
student_pars.eta_circle = 100.0;
student_pars.circle_r   = 0.5;

circle_c_default = [1.55; 1.45];

% theta = [q; r; w; eta_circle; circle_r]
theta_val = [student_pars.q; ...
             student_pars.r; ...
             student_pars.w; ...
             student_pars.eta_circle; ...
             student_pars.circle_r];

theta_init = theta_val;

%% ------------------- Imitation weights -------------------
wx = [10; 10; 2];
wu = [2; 2];

%% ------------------- Paths -------------------
mex_dir = fullfile(pwd,'mex');
if ~exist(mex_dir,'dir')
    mkdir(mex_dir);
end

addpath(mex_dir);
addpath(genpath('../casadi-3.7.2-windows64-matlab2018b'));
addpath(genpath('../Benchmark_comparision'));

%% ------------------- Load demonstration -------------------
if ~exist('teacher_record.mat','file')
    error('Run generate_record.m first.');
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
% variable = [x0; xref; circle_c; U_tar(:); X_tar(:)]
% x0        : nx
% xref      : nx
% circle_c  : 2
% U_tar     : nu*N
% X_tar     : nx*(N+1)
variable_dim = nx + nx + 2 + nu*N + nx*(N+1);

fprintf('Variable dimension = %d\n', variable_dim);

%% ------------------- CasADi symbolics -------------------
U_sym        = SX.sym('U', nu*N);
theta_sym    = SX.sym('theta', numel(theta_val));
variable_sym = SX.sym('variable', variable_dim);
v_sym        = SX.sym('v', nu*N);

%% ------------------- Unpack theta -------------------
q          = theta_sym(1:3);
r          = theta_sym(4:5);
w          = theta_sym(6:8);
eta_circle = theta_sym(9);
circle_r   = theta_sym(10);

%% ------------------- Unpack variable -------------------
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

%% ------------------- Forward rollout + loss -------------------
xk   = x0_sym;
J    = SX(0);
Lval = SX(0);

for k = 1:N
    uk = U_sym((k-1)*nu+1:k*nu);

    dx = xk - xref_sym;

    % Forward objective J
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

    % Imitation loss Lval
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

% Terminal imitation state loss
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
% clear mex;
% 
% delete_if_exists(fullfile(mex_dir, ['tl_f_and_grad_u.' mexext]));
% delete_if_exists(fullfile(mex_dir, ['tl_L_and_grad_u.' mexext]));
% delete_if_exists(fullfile(mex_dir, ['tl_vjp_f_u_theta.' mexext]));
% delete_if_exists(fullfile(mex_dir, ['tl_hvp_f_u.' mexext]));
% 
% delete_if_exists('tl_f_and_grad_u.c');
% delete_if_exists('tl_L_and_grad_u.c');
% delete_if_exists('tl_vjp_f_u_theta.c');
% delete_if_exists('tl_hvp_f_u.c');
% 
% Function('tl_f_and_grad_u', ...
%     {U_sym, theta_sym, variable_sym}, ...
%     {J, grad_J}) ...
%     .generate('tl_f_and_grad_u.c', struct('mex',true));
% 
% Function('tl_L_and_grad_u', ...
%     {U_sym, variable_sym}, ...
%     {Lval, grad_L}) ...
%     .generate('tl_L_and_grad_u.c', struct('mex',true));
% 
% Function('tl_vjp_f_u_theta', ...
%     {v_sym, U_sym, theta_sym, variable_sym}, ...
%     {vjp_theta}) ...
%     .generate('tl_vjp_f_u_theta.c', struct('mex',true));
% 
% Function('tl_hvp_f_u', ...
%     {v_sym, U_sym, theta_sym, variable_sym}, ...
%     {Hv}) ...
%     .generate('tl_hvp_f_u.c', struct('mex',true));
% 
% mex('-outdir', mex_dir, 'tl_f_and_grad_u.c');
% mex('-outdir', mex_dir, 'tl_L_and_grad_u.c');
% mex('-outdir', mex_dir, 'tl_vjp_f_u_theta.c');
% mex('-outdir', mex_dir, 'tl_hvp_f_u.c');
% 
% fprintf('All MEX files regenerated in: %s\n', mex_dir);

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
epochs = 50;
lr     = 2e-4;

theta = theta_val;

loss_hist             = nan(epochs,1);
t_forward_mean_hist   = nan(epochs,1);
t_backward_mean_hist  = nan(epochs,1);
t_forward_total_hist  = nan(epochs,1);
t_backward_total_hist = nan(epochs,1);
back_iter_mean_hist   = nan(epochs,1);
relres_mean_hist      = nan(epochs,1);

u_guess_bank = zeros(problem.dimension, batch_size);

panda('init', problem, solver_params);

%% ------------------- Training loop over all samples -------------------
for ep = 1:epochs
    grad_acc = zeros(size(theta));
    loss_acc = 0;

    t_forward_samples  = nan(batch_size,1);
    t_backward_samples = nan(batch_size,1);
    back_iter_samples  = nan(batch_size,1);
    relres_samples     = nan(batch_size,1);
    fwd_iter_samples   = nan(batch_size,1);
    gamma_samples      = nan(batch_size,1);

    for b = 1:batch_size
        idx = batch_idx(b);

        variable_b = pack_record_variable( ...
            record(idx), ...
            nx, nu, N, ...
            x_ref_default, ...
            circle_c_default);

        %% ----- Forward -----
        oracle_fg = @(u) tl_f_and_grad_u(u, theta, variable_b);

        t_fwd = tic;
        [u_star, iters, gamma] = solve_once(u_guess_bank(:,b), oracle_fg);
        t_forward_samples(b) = toc(t_fwd);

        %% ----- Backward -----
        t_bwd = tic;
        [Lval, dLdtheta, info] = panda_backward( ...
            u_star, theta, variable_b, gamma, ...
            Jprox_u, oracle_hvp, oracle_L, oracle_vjp, opts);
        t_backward_samples(b) = toc(t_bwd);

        Lval = double(full(Lval));
        dLdtheta = double(full(dLdtheta(:)));

        grad_acc = grad_acc + dLdtheta;
        loss_acc = loss_acc + Lval;

        fwd_iter_samples(b) = iters;
        gamma_samples(b)    = gamma;

        if isfield(info,'iter')
            back_iter_samples(b) = info.iter;
        end

        if isfield(info,'relres')
            relres_samples(b) = info.relres;
        end

        u_guess_bank(:,b) = zeros(problem.dimension,1);
    end

    grad_acc = grad_acc / batch_size;
    loss_acc = loss_acc / batch_size;

    theta = theta - lr * grad_acc;
    
    theta = max(theta, 0.01);

    loss_hist(ep)             = loss_acc;
    t_forward_total_hist(ep)  = sum(t_forward_samples);
    t_backward_total_hist(ep) = sum(t_backward_samples);
    t_forward_mean_hist(ep)   = mean(t_forward_samples,  'omitnan');
    t_backward_mean_hist(ep)  = mean(t_backward_samples, 'omitnan');
    back_iter_mean_hist(ep)   = mean(back_iter_samples,  'omitnan');
    relres_mean_hist(ep)      = mean(relres_samples,     'omitnan');

    fprintf(['ep=%03d | L=%.6e | fwd_mean=%.6fs | bwd_mean=%.6fs | ', ...
             'fwd_it=%.2f | back_it=%.2f | gamma=%.3e | relres=%.2e\n'], ...
             ep, ...
             loss_hist(ep), ...
             t_forward_mean_hist(ep), ...
             t_backward_mean_hist(ep), ...
             mean(fwd_iter_samples,'omitnan'), ...
             back_iter_mean_hist(ep), ...
             mean(gamma_samples,'omitnan'), ...
             relres_mean_hist(ep));

    fprintf('  theta = [');
    fprintf(' %.4f', theta);
    fprintf(' ]\n');
end

disp(mean(t_forward_mean_hist));

panda('cleanup');

theta_trained = theta;

%% ------------------- Save training log -------------------
results = table( ...
    (1:epochs)', ...
    loss_hist, ...
    t_forward_total_hist, ...
    t_backward_total_hist, ...
    t_forward_mean_hist, ...
    t_backward_mean_hist, ...
    back_iter_mean_hist, ...
    relres_mean_hist, ...
    'VariableNames', { ...
    'epoch', ...
    'loss', ...
    'forward_total_time', ...
    'backward_total_time', ...
    'forward_mean_time', ...
    'backward_mean_time', ...
    'backward_mean_iter', ...
    'backward_mean_relres'});

writetable(results, 'training_log_panda.csv');

save('training_result.mat', ...
    'theta_init', ...
    'theta_trained', ...
    'student_pars', ...
    'loss_hist', ...
    't_forward_total_hist', ...
    't_backward_total_hist', ...
    't_forward_mean_hist', ...
    't_backward_mean_hist', ...
    'back_iter_mean_hist', ...
    'relres_mean_hist');

fprintf('\nSaved training_log.csv and training_result.mat\n');

%% ==================== Full closed-loop MPC trajectory comparison ====================
Tsim_eval = min(50, numel(record));   % Run 50 MPC time steps
% Tsim_eval = numel(record);          % Use this line to run the full record

% Fixed scenario information
variable_1 = pack_record_variable( ...
    record(1), ...
    nx, nu, N, ...
    x_ref_default, ...
    circle_c_default);

[x0_eval, xref_eval, circle_eval, ~, ~] = ...
    unpack_variable_numeric(variable_1, nx, nu, N);

%% ---------- Demonstration closed-loop trajectory ----------
X_demo_full = zeros(nx, Tsim_eval + 1);

for t = 1:Tsim_eval
    if isfield(record(t), 'x0')
        X_demo_full(:,t) = record(t).x0(:);
    elseif isfield(record(t), 'P')
        X_demo_full(:,t) = record(t).P(1:nx);
    else
        error('record(%d) does not contain x0 or P.', t);
    end
end

% Final demonstration state
if Tsim_eval < numel(record)
    if isfield(record(Tsim_eval+1), 'x0')
        X_demo_full(:,Tsim_eval+1) = record(Tsim_eval+1).x0(:);
    elseif isfield(record(Tsim_eval+1), 'P')
        X_demo_full(:,Tsim_eval+1) = record(Tsim_eval+1).P(1:nx);
    else
        X_demo_full(:,Tsim_eval+1) = X_demo_full(:,Tsim_eval);
    end
else
    if isfield(record(Tsim_eval), 'X_star')
        X_demo_full(:,Tsim_eval+1) = record(Tsim_eval).X_star(:,2);
    elseif isfield(record(Tsim_eval), 'u_star')
        u_last = record(Tsim_eval).u_star(:);
        Xtmp = rollout_numeric(X_demo_full(:,Tsim_eval), u_last, nx, nu, N, dt, L);
        X_demo_full(:,Tsim_eval+1) = Xtmp(:,2);
    else
        X_demo_full(:,Tsim_eval+1) = X_demo_full(:,Tsim_eval);
    end
end

%% ---------- Closed-loop rollout before / after training ----------
panda('init', problem, solver_params);

[X_before_full, U_before_full, iter_before_full, gamma_before_full] = ...
    rollout_closed_loop_full( ...
        theta_init, ...
        record, ...
        Tsim_eval, ...
        nx, nu, N, dt, L, ...
        x_ref_default, ...
        circle_c_default, ...
        problem.dimension);

[X_after_full, U_after_full, iter_after_full, gamma_after_full] = ...
    rollout_closed_loop_full( ...
        theta_trained, ...
        record, ...
        Tsim_eval, ...
        nx, nu, N, dt, L, ...
        x_ref_default, ...
        circle_c_default, ...
        problem.dimension);

panda('cleanup');

%% ---------- Radius information ----------
r_real   = 0.4;               % True obstacle radius
r_before = theta_init(10);    % Obstacle radius assumed by MPC before training
r_after  = theta_trained(10); % Obstacle radius assumed by MPC after training

%% ---------- Save data for plotting ----------
save('mpc_full_closed_loop_traj.mat', ...
    'Tsim_eval', ...
    'X_demo_full', ...
    'X_before_full', ...
    'X_after_full', ...
    'U_before_full', ...
    'U_after_full', ...
    'iter_before_full', ...
    'iter_after_full', ...
    'gamma_before_full', ...
    'gamma_after_full', ...
    'theta_init', ...
    'theta_trained', ...
    'x0_eval', ...
    'xref_eval', ...
    'circle_eval', ...
    'r_real', ...
    'r_before', ...
    'r_after');

fprintf('\nSaved full closed-loop MPC trajectories to mpc_full_closed_loop_traj.mat\n');

clear mex;

%% ==================== Helper Functions ====================

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
        error('X_star must be nx x (N+1).');
    end

    variable = [ ...
        x0(:); ...
        xref(:); ...
        circle_c(:); ...
        U_tar(:); ...
        X_tar(:)];
end

function [x0, xref, circle_c, U_tar, X_tar] = unpack_variable_numeric(variable, nx, nu, N)
    offset = 0;

    x0 = variable(offset + (1:nx));
    offset = offset + nx;

    xref = variable(offset + (1:nx));
    offset = offset + nx;

    circle_c = variable(offset + (1:2));
    offset = offset + 2;

    U_tar = reshape(variable(offset + (1:nu*N)), nu, N);
    offset = offset + nu*N;

    X_tar = reshape(variable(offset + (1:nx*(N+1))), nx, N+1);
end

function [u_sol, iters, gamma] = solve_once(u_init, oracle_fun)
    u_sol = u_init(:);
    [iters, gamma] = panda('solve', u_sol, oracle_fun);
end

function X = rollout_numeric(x0,U,nx,nu,N,dt,L)
    U = U(:);
    X = zeros(nx,N+1);
    X(:,1) = x0(:);
    xk = x0(:);

    for k = 1:N
        uk = U((k-1)*nu+1:k*nu);
        xk = euler_step_numeric(xk, uk, dt, L);
        X(:,k+1) = xk;
    end
end

function xnext = euler_step_numeric(x,u,dt,L)
    dx = dyn_numeric(x,u,L);
    xnext = x + dt*dx;
end

function dx = dyn_numeric(x,u,L)
    th = x(3); 
    ux = u(1); 
    uy = u(2);

    thdot = (uy*cos(th) - ux*sin(th)) / L;
    pxdot = ux + L*sin(th)*thdot;
    pydot = uy - L*cos(th)*thdot;

    dx = [pxdot; pydot; thdot];
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

function [X_cl, U_cl, iter_log, gamma_log] = rollout_closed_loop_full( ...
    theta_use, ...
    record, ...
    Tsim, ...
    nx, nu, N, dt, L, ...
    xref_default, ...
    circle_c_default, ...
    dim_u)

    % Initial state
    if isfield(record(1), 'x0')
        xcur = record(1).x0(:);
    elseif isfield(record(1), 'P')
        xcur = record(1).P(1:nx);
    else
        error('record(1) does not contain x0 or P.');
    end

    X_cl = zeros(nx, Tsim + 1);
    U_cl = zeros(nu, Tsim);

    iter_log  = nan(Tsim,1);
    gamma_log = nan(Tsim,1);

    X_cl(:,1) = xcur;

    % Warm start
    u_guess = zeros(dim_u,1);

    for t = 1:Tsim

        variable_t_old = pack_record_variable( ...
            record(t), ...
            nx, nu, N, ...
            xref_default, ...
            circle_c_default);

        [~, xref_t, circle_t, U_tar_t, X_tar_t] = ...
            unpack_variable_numeric(variable_t_old, nx, nu, N);

        variable_t = [ ...
            xcur(:); ...
            xref_t(:); ...
            circle_t(:); ...
            U_tar_t(:); ...
            X_tar_t(:)];

        oracle_fg = @(u) tl_f_and_grad_u(u, theta_use, variable_t);

        [u_star, iters, gamma] = solve_once(u_guess, oracle_fg);

        iter_log(t)  = iters;
        gamma_log(t) = gamma;

        u_apply = u_star(1:nu);
        U_cl(:,t) = u_apply;

        xcur = euler_step_numeric(xcur, u_apply, dt, L);

        X_cl(:,t+1) = xcur;

        u_guess = shift_control_sequence(u_star, nu, N);
    end
end

function u_shift = shift_control_sequence(u, nu, N)

    U = reshape(u(:), nu, N);

    U_shift = zeros(nu, N);
    U_shift(:,1:N-1) = U(:,2:N);
    U_shift(:,N)     = U(:,N);

    u_shift = U_shift(:);
end