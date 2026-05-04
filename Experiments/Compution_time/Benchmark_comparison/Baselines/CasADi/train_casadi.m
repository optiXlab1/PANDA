%% ===================== CasADi Baseline Training over All Samples =====================
clear; clc; close all;
clear mex;

addpath(genpath('../../casadi-3.7.2-windows64-matlab2018b'));
import casadi.*

%% =========================================================
% 0. problem settings
%% =========================================================
nx = 3;
nu = 2;
N  = 40;
L  = 0.5;
dt = 0.1;

x_init_default   = [0;0;pi/5];
x_ref_default    = [3.5;1.5;0];
circle_c_default = [1.55;1.45];

umin = -0.8;
umax =  0.8;
lb = umin * ones(nu*N,1);
ub = umax * ones(nu*N,1);

theta_dim = nx + nu + nx + 1 + 1;  % q(3), r(2), w(3), eta, radius

% variable = [x0; xref; circle_c; U_tar(:); X_tar(:)]
variable_dim = nx + nx + 2 + nu*N + nx*(N+1);

%% =========================================================
% 1. load teacher record directly
%% =========================================================
teacher_record_file = 'teacher_record.mat';

if ~exist(teacher_record_file, 'file')
    error('teacher_record.mat not found. Please generate target record first.');
end

load(teacher_record_file);  % should contain record

%% =========================================================
% 2. batch selection
%% =========================================================
batch_stride = 1;
batch_idx = 1:batch_stride:numel(record);
batch_size = numel(batch_idx);

fprintf('Total record length = %d\n', numel(record));
fprintf('Batch stride        = %d\n', batch_stride);
fprintf('Batch size          = %d\n', batch_size);
fprintf('Variable dimension  = %d\n', variable_dim);
fprintf('Theta dimension     = %d\n', theta_dim);

%% =========================================================
% 3. student initial parameters
%% =========================================================
student_pars.q          = [0.34; 0.91; 0.12];
student_pars.r          = [0.07; 0.02];
student_pars.w          = [5.6; 3.3; 0.4];
student_pars.eta_circle = 100.0;
student_pars.circle_r   = 0.50;

theta_init = encode_theta(student_pars);
theta      = theta_init;

pars_init = decode_theta(theta_init);

%% =========================================================
% 4. fixed imitation weights
%% =========================================================
wx = [10; 10; 2];
wu = [2; 2];

%% =========================================================
% 5. build forward/backward CasADi functions
%% =========================================================
opti_fwd = build_trailer_opti_direct( ...
    nx, nu, N, dt, L, lb, ub, theta_dim, variable_dim, wx, wu, 'ipopt');

Usolve = opti_fwd.opti.to_function( ...
    'Usolve', ...
    {opti_fwd.p_theta, opti_fwd.p_var, opti_fwd.u}, ...
    {opti_fwd.u});

opti_bwd = build_trailer_opti_direct( ...
    nx, nu, N, dt, L, lb, ub, theta_dim, variable_dim, wx, wu, 'sqpmethod');

Z = opti_bwd.opti.to_function( ...
    'Z', ...
    {opti_bwd.p_theta, opti_bwd.p_var, opti_bwd.u}, ...
    {opti_bwd.Lout});

th_mx  = MX.sym('th',  theta_dim, 1);
var_mx = MX.sym('var', variable_dim, 1);
u0_mx  = MX.sym('u0',  N*nu, 1);

L_sym = Z(th_mx, var_mx, u0_mx);
dL_dtheta = jacobian(L_sym, th_mx);

Z_sens = Function('Z_sens', {th_mx, var_mx, u0_mx}, {L_sym, dL_dtheta});

%% =========================================================
% 6. sanity check
%% =========================================================
idx_test = batch_idx(1);

variable_test = pack_record_variable_direct( ...
    record(idx_test), nx, nu, N, x_ref_default, circle_c_default);

u_init_test = zeros(N*nu,1);

nruns = 5;

u_star_test = full(Usolve(theta, variable_test, u_init_test));

tic;
u_tmp = full(Usolve(theta, variable_test, u_init_test)); %#ok<NASGU>
t_fwd_first = toc;

tic;
for i = 1:nruns
    u_tmp = full(Usolve(theta, variable_test, u_init_test)); %#ok<NASGU>
end
t_fwd_avg = toc / nruns;

tic;
[L_tmp, g_tmp] = Z_sens(theta, variable_test, u_star_test);
L_tmp = full(L_tmp); %#ok<NASGU>
g_tmp = full(g_tmp); %#ok<NASGU>
t_bwd_first = toc;

tic;
for i = 1:nruns
    [L_tmp, g_tmp] = Z_sens(theta, variable_test, u_star_test);
    L_tmp = full(L_tmp); %#ok<NASGU>
    g_tmp = full(g_tmp); %#ok<NASGU>
end
t_bwd_avg = toc / nruns;

fprintf('\n========== CasADi sanity check ==========\n');
fprintf('Forward first call   : %.6e s\n', t_fwd_first);
fprintf('Forward repeated avg : %.6e s\n', t_fwd_avg);
fprintf('Backward first call  : %.6e s\n', t_bwd_first);
fprintf('Backward repeated avg: %.6e s\n', t_bwd_avg);
fprintf('Backward / Forward   : %.6f\n', t_bwd_avg / t_fwd_avg);
fprintf('=========================================\n\n');

%% =========================================================
% 7. training
%% =========================================================
epochs = 50;
lr     = 2e-4;

loss_hist               = nan(epochs,1);
t_forward_total_hist    = nan(epochs,1);
t_forward_mean_hist     = nan(epochs,1);
t_backward_total_hist   = nan(epochs,1);
t_backward_mean_hist    = nan(epochs,1);

back_iter_mean_hist     = nan(epochs,1);
relres_mean_hist        = nan(epochs,1);

u_zero = zeros(N*nu,1);

for ep = 1:epochs
    grad_acc = zeros(theta_dim,1);
    loss_acc = 0;

    t_forward_samples  = nan(batch_size,1);
    t_backward_samples = nan(batch_size,1);

    for b = 1:batch_size
        idx = batch_idx(b);

        variable_b = pack_record_variable_direct( ...
            record(idx), nx, nu, N, x_ref_default, circle_c_default);

        u_init = u_zero;

        %% ----- forward -----
        t_fwd = tic;
        u_star = Usolve(theta, variable_b, u_init);
        t_forward_samples(b) = toc(t_fwd);

        u_star = full(u_star);

        %% ----- backward -----
        t_bwd = tic;
        [L_val, grad_val] = Z_sens(theta, variable_b, u_star);
        t_backward_samples(b) = toc(t_bwd);

        L_val    = full(L_val);
        grad_val = full(grad_val);

        grad_acc = grad_acc + grad_val(:);
        loss_acc = loss_acc + double(L_val);
    end

    grad_acc = grad_acc / batch_size;
    loss_acc = loss_acc / batch_size;

    theta = theta - lr * grad_acc;

    % non-negative projection
    theta = max(theta, 0.01);

    loss_hist(ep)              = loss_acc;
    t_forward_total_hist(ep)   = sum(t_forward_samples);
    t_forward_mean_hist(ep)    = mean(t_forward_samples, 'omitnan');
    t_backward_total_hist(ep)  = sum(t_backward_samples);
    t_backward_mean_hist(ep)   = mean(t_backward_samples, 'omitnan');

    fprintf(['ep=%03d | L=%.8e | fwd_total=%.4fs | fwd_mean=%.6fs | ', ...
             'bwd_total=%.4fs | bwd_mean=%.6fs\n'], ...
        ep, loss_hist(ep), ...
        t_forward_total_hist(ep), t_forward_mean_hist(ep), ...
        t_backward_total_hist(ep), t_backward_mean_hist(ep));

    pars_cur = decode_theta(theta);
    fprintf('  q = [%.4f %.4f %.4f]\n', pars_cur.q(1), pars_cur.q(2), pars_cur.q(3));
    fprintf('  r = [%.4f %.4f]\n', pars_cur.r(1), pars_cur.r(2));
    fprintf('  w = [%.4f %.4f %.4f]\n', pars_cur.w(1), pars_cur.w(2), pars_cur.w(3));
    fprintf('  eta = %.4f, radius = %.4f\n', pars_cur.eta_circle, pars_cur.circle_r);
end

theta_trained = theta;
pars_final = decode_theta(theta_trained);

%% =========================================================
% 8. closed-loop evaluation over full record
%% =========================================================
Tsim_eval = numel(record);

X_demo = build_demo_closed_loop(record, nx, N);

var1 = pack_record_variable_direct(record(1), nx, nu, N, x_ref_default, circle_c_default);
[x0_eval, xref_eval, circle_eval, ~, ~] = unpack_variable_numeric(var1, nx, nu, N);

[X_init_cl, U_init_cl] = rollout_closed_loop_casadi_direct( ...
    theta_init, record, Usolve, ...
    nx, nu, N, dt, L, x_ref_default, circle_c_default);

[X_trained_cl, U_trained_cl] = rollout_closed_loop_casadi_direct( ...
    theta_trained, record, Usolve, ...
    nx, nu, N, dt, L, x_ref_default, circle_c_default);

%% =========================================================
% 9. save CSV
%% =========================================================
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

writetable(results, 'training_log_casadi.csv');

%% =========================================================
% 10. save MAT
%% =========================================================
save('casadi_benchmark_result.mat', ...
    'record', ...
    'batch_idx', 'batch_size', 'batch_stride', ...
    'theta_init', 'theta_trained', ...
    'pars_init', 'pars_final', ...
    'loss_hist', ...
    't_forward_total_hist', 't_forward_mean_hist', ...
    't_backward_total_hist', 't_backward_mean_hist', ...
    'back_iter_mean_hist', 'relres_mean_hist', ...
    'X_demo', 'X_init_cl', 'X_trained_cl', ...
    'U_init_cl', 'U_trained_cl', ...
    't_fwd_first', 't_fwd_avg', 't_bwd_first', 't_bwd_avg');

%% =========================================================
% 11. print summary
%% =========================================================
disp('========== Initial decoded parameters ==========');
disp(pars_init);

disp('========== Final decoded parameters ==========');
disp(pars_final);

fprintf('\n========== CasADi benchmark summary ==========\n');
fprintf('Mean forward total time / epoch  : %.6f s\n', mean(t_forward_total_hist, 'omitnan'));
fprintf('Mean forward time / sample       : %.6f s\n', mean(t_forward_mean_hist,  'omitnan'));
fprintf('Mean backward total time / epoch : %.6f s\n', mean(t_backward_total_hist, 'omitnan'));
fprintf('Mean backward time / sample      : %.6f s\n', mean(t_backward_mean_hist,  'omitnan'));
fprintf('Saved CSV: training_log_casadi.csv\n');
fprintf('Saved MAT: casadi_benchmark_result.mat\n\n');

%% =========================================================
% 12. plot closed-loop trajectory comparison
%% =========================================================
figure('Color','w'); hold on; grid on; box on; axis equal;

th_plot = linspace(0,2*pi,400);

r_init  = pars_init.circle_r;
r_final = pars_final.circle_r;

% teacher radius only for visualization
if exist('teacher_pars','var') && isfield(teacher_pars,'circle_r')
    r_demo = teacher_pars.circle_r;
elseif exist('teacher_pars','var') && isfield(teacher_pars,'theta')
    r_demo = teacher_pars.theta(10);
else
    r_demo = r_init;
end

plot(circle_eval(1) + r_demo*cos(th_plot), ...
     circle_eval(2) + r_demo*sin(th_plot), ...
     'b--', 'LineWidth', 1.5);

plot(circle_eval(1) + r_init*cos(th_plot), ...
     circle_eval(2) + r_init*sin(th_plot), ...
     'k--', 'LineWidth', 1.5);

plot(circle_eval(1) + r_final*cos(th_plot), ...
     circle_eval(2) + r_final*sin(th_plot), ...
     'r--', 'LineWidth', 1.5);

plot(X_demo(1,:),       X_demo(2,:),       'b-',  'LineWidth', 2.0);
plot(X_init_cl(1,:),    X_init_cl(2,:),    'k-',  'LineWidth', 2.0);
plot(X_trained_cl(1,:), X_trained_cl(2,:), 'r-',  'LineWidth', 2.0);

plot(X_demo(1,1), X_demo(2,1), ...
     'ko', 'MarkerFaceColor','k', 'MarkerSize', 6);

plot(xref_eval(1), xref_eval(2), ...
     'mx', 'LineWidth', 2.0, 'MarkerSize', 10);

legend({ ...
    sprintf('demo radius ($r=%.2f$)', r_demo), ...
    sprintf('initial radius ($r=%.2f$)', r_init), ...
    sprintf('trained radius ($r=%.2f$)', r_final), ...
    'demonstration', ...
    'initial-parameter MPC', ...
    'trained-parameter MPC', ...
    'start', ...
    'goal'}, ...
    'Location', 'best');

xlabel('p_x');
ylabel('p_y');
title('CasADi closed-loop trajectory comparison over full record');

clear mex;

%% =========================================================
% local functions
%% =========================================================

function theta = encode_theta(pars)
    theta = [pars.q(:); pars.r(:); pars.w(:); pars.eta_circle; pars.circle_r];
end

function pars = decode_theta(theta)
    theta = theta(:);
    pars.q          = theta(1:3);
    pars.r          = theta(4:5);
    pars.w          = theta(6:8);
    pars.eta_circle = theta(9);
    pars.circle_r   = theta(10);
end

function S = build_trailer_opti_direct(nx, nu, N, dt, L, lb, ub, theta_dim, variable_dim, wx, wu, solver_name)
    import casadi.*

    opti = Opti();

    u = opti.variable(N*nu, 1);
    p_theta = opti.parameter(theta_dim, 1);
    p_var   = opti.parameter(variable_dim, 1);

    % unpack theta
    q          = p_theta(1:3);
    r          = p_theta(4:5);
    w          = p_theta(6:8);
    eta_circle = p_theta(9);
    circle_r   = p_theta(10);

    % unpack variable
    [x0, xref, circle_c, U_tar, X_tar] = unpack_variable_symbolic(p_var, nx, nu, N);

    x = x0;

    J = 0;
    Lout = 0;

    for k = 1:N
        uk = reshape(u((k-1)*nu+1:k*nu), nu, 1);

        %% inner objective
        dx = x - xref;
        J = J ...
            + q(1) * dx(1)^2 ...
            + q(2) * dx(2)^2 ...
            + q(3) * dx(3)^2 ...
            + r(1) * uk(1)^2 ...
            + r(2) * uk(2)^2;

        z = x(1:2);
        h = 1 - ((z - circle_c).' * (z - circle_c)) / circle_r^2;
        J = J + eta_circle * 0.5 * fmax(h, 0)^2;

        %% imitation control loss
        eu = uk - U_tar(:,k);
        ex = x - X_tar(:,k);

        Lout = Lout ...
            + wu(1) * eu(1)^2 ...
            + wu(2) * eu(2)^2 ...
            + wx(1) * ex(1)^2 ...
            + wx(2) * ex(2)^2 ...
            + wx(3) * ex(3)^2;

        %% dynamics
        x = euler_step_symbolic_local(x, uk, dt, L);
            
    end

    %% terminal inner objective
    dxN = x - xref;
    J = J ...
        + w(1) * dxN(1)^2 ...
        + w(2) * dxN(2)^2 ...
        + w(3) * dxN(3)^2;

    exN = x - X_tar(:,N+1);
    Lout = Lout ...
    + wx(1)*exN(1)^2 ...
    + wx(2)*exN(2)^2 ...
    + wx(3)*exN(3)^2;

    opti.minimize(J);
    opti.subject_to(lb <= u <= ub);

    if strcmp(solver_name, 'ipopt')
        opts = struct;
        opts.print_time = false;
        opts.ipopt.print_level = 0;
        opts.ipopt.sb = 'yes';
        opts.ipopt.max_iter = 1000;
        opts.ipopt.tol = 1e-6;
        opts.ipopt.acceptable_tol = 1e-5;
        opti.solver('ipopt', opts);

    elseif strcmp(solver_name, 'sqpmethod')
        opts = struct;
        opts.print_header    = false;
        opts.print_iteration = false;
        opts.print_status    = false;
        opts.print_time      = false;
        opts.record_time     = false;
        opts.qpsol           = 'qpoases';
        opts.qpsol_options   = struct;
        opts.qpsol_options.printLevel = 'none';
        opti.solver('sqpmethod', opts);

    else
        error('Unknown solver_name: %s', solver_name);
    end

    S.opti    = opti;
    S.u       = u;
    S.p_theta = p_theta;
    S.p_var   = p_var;
    S.Lout    = Lout;
end

function [x0, xref, circle_c, U_tar, X_tar] = unpack_variable_symbolic(variable, nx, nu, N)
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

function variable = pack_record_variable_direct(rec, nx, nu, N, xref_default, circle_c_default)

    % x0 从 record 读取，因为每个样本对应不同闭环时刻
    if isfield(rec, 'x0')
        x0 = rec.x0(:);
    elseif isfield(rec, 'P')
        x0 = rec.P(1:nx);
    elseif isfield(rec, 'variable')
        [x0, ~, ~, ~, ~] = unpack_variable_numeric(rec.variable, nx, nu, N);
    else
        error('record sample does not contain x0, P, or variable.');
    end

    xref = xref_default(:);

    circle_c = circle_c_default(:);
    
    if isfield(rec, 'U_star')
        U_tar = rec.U_star;
        if isvector(U_tar)
            U_tar = reshape(U_tar(:), nu, N);
        end
    elseif isfield(rec, 'u_star')
        U_tar = reshape(rec.u_star(:), nu, N);
    elseif isfield(rec, 'U_teacher')
        U_tar = rec.U_teacher;
        if isvector(U_tar)
            U_tar = reshape(U_tar(:), nu, N);
        end
    elseif isfield(rec, 'variable')
        [~, ~, ~, U_tar, ~] = unpack_variable_numeric(rec.variable, nx, nu, N);
    else
        error('record sample does not contain U_star, u_star, U_teacher, or variable.');
    end

    % X target 从 record 读取
    if isfield(rec, 'X_star')
        X_tar = rec.X_star;
    elseif isfield(rec, 'X_teacher')
        X_tar = rec.X_teacher;
    elseif isfield(rec, 'variable')
        [~, ~, ~, ~, X_tar] = unpack_variable_numeric(rec.variable, nx, nu, N);
    else
        error('record sample does not contain X_star, X_teacher, or variable.');
    end

    if size(X_tar,1) ~= nx
        X_tar = reshape(X_tar(:), nx, N+1);
    end

    if size(X_tar,2) ~= N+1
        error('X_tar must be nx x (N+1).');
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

function X_demo = build_demo_closed_loop(record, nx, N)
    Tsim = numel(record);
    X_demo = zeros(nx, Tsim+1);

    for t = 1:Tsim
        if isfield(record(t), 'x0')
            X_demo(:,t) = record(t).x0(:);
        elseif isfield(record(t), 'P')
            X_demo(:,t) = record(t).P(1:nx);
        elseif isfield(record(t), 'variable')
            [x0, ~, ~, ~, ~] = unpack_variable_numeric(record(t).variable, nx, 2, N);
            X_demo(:,t) = x0(:);
        else
            error('Cannot recover demonstration state at t=%d.', t);
        end
    end

    if isfield(record(end), 'X_star')
        X_demo(:,Tsim+1) = record(end).X_star(:,2);
    elseif isfield(record(end), 'X_teacher')
        X_demo(:,Tsim+1) = record(end).X_teacher(:,2);
    elseif isfield(record(end), 'variable')
        [~, ~, ~, ~, X_tar] = unpack_variable_numeric(record(end).variable, nx, 2, N);
        X_demo(:,Tsim+1) = X_tar(:,2);
    else
        error('Cannot recover final demonstration state.');
    end
end

function [X_cl, U_cl] = rollout_closed_loop_casadi_direct( ...
    theta_use, record, Usolve, nx, nu, N, dt, L, xref_default, circle_c_default)

    Tsim = numel(record);

    if isfield(record(1), 'x0')
        xcur = record(1).x0(:);
    elseif isfield(record(1), 'P')
        xcur = record(1).P(1:nx);
    elseif isfield(record(1), 'variable')
        [xcur, ~, ~, ~, ~] = unpack_variable_numeric(record(1).variable, nx, nu, N);
    else
        error('record(1) does not contain x0/P/variable.');
    end

    X_cl = zeros(nx, Tsim+1);
    U_cl = zeros(nu, Tsim);

    X_cl(:,1) = xcur;

    u_guess = zeros(N*nu,1);

    for t = 1:Tsim
        variable_t = pack_record_variable_with_x0( ...
            record(t), xcur, nx, nu, N, xref_default, circle_c_default);

        u_star = full(Usolve(theta_use, variable_t, u_guess));

        u_apply = u_star(1:nu);
        U_cl(:,t) = u_apply;

        xcur = euler_step_numeric(xcur, u_apply, dt, L);
        X_cl(:,t+1) = xcur;

        u_guess = shift_control(u_star, nu, N);
    end
end

function variable = pack_record_variable_with_x0(rec, x0_use, nx, nu, N, xref_default, circle_c_default)
    variable0 = pack_record_variable_direct(rec, nx, nu, N, xref_default, circle_c_default);
    [~, xref, circle_c, U_tar, X_tar] = unpack_variable_numeric(variable0, nx, nu, N);

    variable = [ ...
        x0_use(:); ...
        xref(:); ...
        circle_c(:); ...
        U_tar(:); ...
        X_tar(:)];
end

function u_shift = shift_control(u, nu, N)
    U = reshape(u(:), nu, N);
    U_shift = [U(:,2:end), U(:,end)];
    u_shift = U_shift(:);
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

function xnext = euler_step_symbolic_local(x,u,dt,L)
    dx = dyn_symbolic_local(x,u,L);
    xnext = x + dt*dx;
end

function dx = dyn_symbolic_local(x,u,L)
    th = x(3);
    ux = u(1);
    uy = u(2);

    thdot = (uy*cos(th) - ux*sin(th)) / L;
    pxdot = ux + L*sin(th)*thdot;
    pydot = uy - L*cos(th)*thdot;

    dx = [pxdot; pydot; thdot];
end