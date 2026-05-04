%% ===================== CasADi Memory Test Only =====================
clear; clc; close all;
clear mex;

addpath(genpath('../../casadi-3.7.2-windows64-matlab2018b'));
import casadi.*

%% =========================================================
% 0. problem settings
%% =========================================================
nx = 3;
nu = 2;
N  = 80;
L  = 0.5;
dt = 0.1;

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
% 7. training: memory output only
%% =========================================================
epochs = 1;
lr     = 2e-4;

u_zero = zeros(N*nu,1);

for ep = 1:epochs
    grad_acc = zeros(theta_dim,1);
    loss_acc = 0;

    for b = 1:batch_size
        idx = batch_idx(b);

        variable_b = pack_record_variable_direct( ...
            record(idx), nx, nu, N, x_ref_default, circle_c_default);

        u_init = u_zero;

        %% ----- forward memory -----
        mem_fwd_before = get_matlab_memory_mb();
        fprintf('[ep=%03d, b=%04d] before forward  | mem = %.2f MB\n', ...
            ep, b, mem_fwd_before);

        u_star = Usolve(theta, variable_b, u_init);

        mem_fwd_after = get_matlab_memory_mb();
        fprintf('[ep=%03d, b=%04d] after  forward  | mem = %.2f MB | delta = %+.2f MB\n', ...
            ep, b, mem_fwd_after, mem_fwd_after - mem_fwd_before);

        u_star = full(u_star);

        %% ----- backward memory -----
        mem_bwd_before = get_matlab_memory_mb();
        fprintf('[ep=%03d, b=%04d] before backward | mem = %.2f MB\n', ...
            ep, b, mem_bwd_before);

        [L_val, grad_val] = Z_sens(theta, variable_b, u_star);

        mem_bwd_after = get_matlab_memory_mb();
        fprintf('[ep=%03d, b=%04d] after  backward | mem = %.2f MB | delta = %+.2f MB\n', ...
            ep, b, mem_bwd_after, mem_bwd_after - mem_bwd_before);

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

    pars_cur = decode_theta(theta);

    fprintf('ep=%03d | L=%.8e\n', ep, loss_acc);
    fprintf('  q = [%.4f %.4f %.4f]\n', pars_cur.q(1), pars_cur.q(2), pars_cur.q(3));
    fprintf('  r = [%.4f %.4f]\n', pars_cur.r(1), pars_cur.r(2));
    fprintf('  w = [%.4f %.4f %.4f]\n', pars_cur.w(1), pars_cur.w(2), pars_cur.w(3));
    fprintf('  eta = %.4f, radius = %.4f\n', pars_cur.eta_circle, pars_cur.circle_r);
end

fprintf('\nCasADi memory-only test finished. No CSV/MAT/figure files were written.\n');

clear mex;

%% =========================================================
% local functions
%% =========================================================

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
        error('X_tar must be nx x (N+1). Current N=%d, but size(X_tar,2)=%d.', ...
            N, size(X_tar,2));
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