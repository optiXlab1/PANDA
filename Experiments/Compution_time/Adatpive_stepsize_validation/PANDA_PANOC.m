%% benchmark_panoc_panda_trailer_euler.m
% PANOC vs PANDA benchmark for trailer OCP
% - forward Euler discretization
% - no warm start
% - compare runtime
% - compare iteration number

clear; clc; close all;
clear mex;

%% ========================= Paths =========================
addpath(genpath('../Adaptive_stepsize_validation'));
addpath(genpath('../casadi-3.7.2-windows64-matlab2018b'));
import casadi.*

global TRAILER_PARAM_CURRENT

%% ====================== Problem setup ====================
probName = 'trailer_panoc_panda_euler';

nx = 3;                % x = [px; py; theta]
nu = 2;                % u = [ux; uy]
N  = 40;
L  = 0.5;
dt = 0.1;

Tsim = 50;

% initial / goal
x_init = [0.0; 0.0; pi/5];
x_ref  = [3.5; 1.5; 0.0];

% circle obstacle only
circle_c = [1.55; 1.45];
circle_r = 0.5;
circle_r_e = circle_r;

% input box constraints
umin = -0.8;
umax =  0.8;

lb = umin * ones(nu*N,1);
ub = umax * ones(nu*N,1);
U_zero = zeros(nu*N,1);

% weights
Q = diag([0.6, 0.6, 0.03]);
R = diag([0.02, 0.04]);
W = diag([4.0, 4.0, 0.8]);

eta_circle = 100;
eta_rect   = 0;

tol = 1e-4;

%% ========== Parameterized single-shooting objective ==========
% Parameters:
% P = [x0(3); xref(3)]
np = 6;

U = SX.sym('U', nu*N);
P = SX.sym('P', np);

x0_sym   = P(1:3);
xref_sym = P(4:6);

xk = x0_sym;
J  = 0;

for k = 1:N
    uk = U(2*k-1:2*k);

    % stage cost
    dx = xk - xref_sym;
    J = J + dx.'*Q*dx + uk.'*R*uk;

    % circle obstacle penalty only
    z = xk(1:2);
    J = J + obstacle_penalty_symbolic(z, ...
        circle_c, circle_r_e, eta_circle);

    % forward Euler dynamics
    xk = euler_step_symbolic(xk, uk, dt, L);
end

% terminal cost
dxN = xk - xref_sym;
J = J + dxN.'*W*dxN;

grad_J = gradient(J, U);
F = Function(probName, {U, P}, {J, grad_J});

%% ========== Generate PANOC/PANDA mex oracle if needed ==========
% mex_name = [probName '.' mexext];
% fprintf('[Build] Generating %s ...\n', mex_name);
% F.generate([probName '.c'], struct('mex', true));
% mex([probName '.c']);

%% ====================== PANOC setup ======================
problem.dimension       = N * nu;
problem.constraint_type = 'costum';
problem.constraint      = @(x,gamma) indBox_manual(x, ub, lb, gamma);

solver_params_panoc.tolerance       = tol;
solver_params_panoc.buffer_size     = 10;
solver_params_panoc.max_iterations  = 800;
solver_params_panoc.max_stable_iter = 0;      % PANOC

solver_params_panda.tolerance       = tol;
solver_params_panda.buffer_size     = 10;
solver_params_panda.max_iterations  = 800;
solver_params_panda.max_stable_iter = 80;     % PANDA

%% =================== Runtime recording arrays =============
runtime_panoc = nan(Tsim,1);
runtime_panda = nan(Tsim,1);

iter_panoc = nan(Tsim,1);
iter_panda = nan(Tsim,1);

gamma_panoc = nan(Tsim,1);
gamma_panda = nan(Tsim,1);

%% ======================== PANOC ==========================
fprintf('\n========== Rolling PANOC ==========\n');
[X_panoc, runtime_panoc, iter_panoc, gamma_panoc] = ...
    run_panoc_family_solver(x_init, x_ref, Tsim, dt, L, ...
    U_zero, problem, solver_params_panoc);

fprintf('\n========== Rolling PANDA ==========\n');
[X_panda, runtime_panda, iter_panda, gamma_panda] = ...
    run_panoc_family_solver(x_init, x_ref, Tsim, dt, L, ...
    U_zero, problem, solver_params_panda);

%% ========================== Plot 1 =========================
figure('Color','w');
plot(1:Tsim, runtime_panoc, 'LineWidth', 2.0); hold on;
plot(1:Tsim, runtime_panda, 'LineWidth', 2.0);

grid on; box on;
xlabel('Time instant $k$','Interpreter','latex');
ylabel('Runtime (s)','Interpreter','latex');
legend('PANOC','PANDA','Location','best');
title('Runtime vs. time instant','Interpreter','latex');

%% ========================== Plot 2 =========================
figure('Color','w');
plot(1:Tsim, iter_panoc, 'LineWidth', 2.0); hold on;
plot(1:Tsim, iter_panda, 'LineWidth', 2.0);

grid on; box on;
xlabel('Time instant $k$','Interpreter','latex');
ylabel('Iterations','Interpreter','latex');
legend('PANOC','PANDA','Location','best');
title('Iterations vs. time instant','Interpreter','latex');

%% ========================== Plot 3 =========================
figure('Color','w'); hold on;

% PANOC trajectory
plot(X_panoc(1,:), X_panoc(2,:), 'LineWidth', 2.0);

% PANDA trajectory
plot(X_panda(1,:), X_panda(2,:), 'LineWidth', 2.0);

% initial point
plot(x_init(1), x_init(2), 'ko', 'MarkerSize', 8, 'LineWidth', 1.5);

% target point
plot(x_ref(1), x_ref(2), 'kp', 'MarkerSize', 10, 'LineWidth', 1.5);

% circular obstacle
tt = linspace(0, 2*pi, 300);
xc = circle_c(1) + circle_r*cos(tt);
yc = circle_c(2) + circle_r*sin(tt);
plot(xc, yc, 'k--', 'LineWidth', 1.5);

grid on; box on; axis equal;
xlabel('$p_x$','Interpreter','latex');
ylabel('$p_y$','Interpreter','latex');
legend('PANOC','PANDA','Initial state','Target state','Obstacle', ...
       'Location','best');
title('Closed-loop trajectories','Interpreter','latex');

Tvec = (1:Tsim).';
tbl = table(Tvec, iter_panoc, gamma_panoc, iter_panda, gamma_panda, ...
    'VariableNames', {'Time', 'PANOC_iters', 'PANOC_gamma', 'PANDA_iters', 'PANDA_gamma'});
writetable(tbl, 'PANOC_PANDA_summary.csv');
fprintf('CSV saved: PANOC_PANDA_summary.csv\n');

clear mex;

%% ====================== Local functions ==================
function [f, g] = trailer_oracle_panoc(U)
    global TRAILER_PARAM_CURRENT
    [f, g] = trailer_panoc_panda_euler(U, TRAILER_PARAM_CURRENT);
    f = full(f);
    g = full(g);
end

function val = obstacle_penalty_symbolic(z, circle_c, circle_r, eta_circle)
    import casadi.*
    h_circle = 1 - ((z-circle_c).' * (z-circle_c)) / circle_r^2;
    psi_circle = 0.5 * fmax(h_circle,0)^2;
    val = eta_circle * psi_circle;
end

function xnext = euler_step_numeric(x, u, dt, L)
    xnext = x + dt * trailer_dyn_numeric(x, u, L);
end

function xnext = euler_step_symbolic(x, u, dt, L)
    xnext = x + dt * trailer_dyn_symbolic(x, u, L);
end

function dx = trailer_dyn_numeric(x, u, L)
    theta = x(3);
    ux = u(1);
    uy = u(2);

    thetadot = (uy*cos(theta) - ux*sin(theta))/L;
    pxdot = ux + L*sin(theta)*thetadot;
    pydot = uy - L*cos(theta)*thetadot;

    dx = [pxdot; pydot; thetadot];
end

function dx = trailer_dyn_symbolic(x, u, L)
    theta = x(3);
    ux = u(1);
    uy = u(2);

    thetadot = (uy*cos(theta) - ux*sin(theta))/L;
    pxdot = ux + L*sin(theta)*thetadot;
    pydot = uy - L*cos(theta)*thetadot;

    dx = [pxdot; pydot; thetadot];
end

function [X_hist, runtime, iters, gamma] = run_panoc_family_solver( ...
    x_init, x_ref, Tsim, dt, L, U_zero, problem, solver_params)

    global TRAILER_PARAM_CURRENT

    nx = length(x_init);
    X_hist = nan(nx, Tsim+1);
    X_hist(:,1) = x_init;

    runtime = nan(Tsim,1);
    iters   = nan(Tsim,1);
    gamma   = nan(Tsim,1);

    xcur = x_init;

    panda('init', problem, solver_params);

    for t = 1:Tsim
        %% ====================== Delete old debug csv ======================
        TRAILER_PARAM_CURRENT = [xcur; x_ref];

        % no warm start: always start from zero
        Uguess = U_zero;

        tic;
        [iters(t),gamma(t)] = panda('solve', Uguess, @trailer_oracle_panoc);
        runtime(t) = toc;

        Usol = Uguess;
        u_apply = Usol(1:2);

        xcur = euler_step_numeric(xcur, u_apply, dt, L);
        X_hist(:,t+1) = xcur;
    end

    panda('cleanup');
end
