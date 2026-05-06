%% ==================== Main script ====================
clear; clc; close all;
import casadi.*

%% ------------------- Parameter settings -------------------
nx = 3; nu = 2; N = 40; L = 0.5; dt = 0.1; Tsim = 50;
x_init = [0;0;pi/5]; x_ref = [3.5;1.5;0];

umin = -0.8; umax = 0.8;
lb = umin*ones(nu*N,1); ub = umax*ones(nu*N,1);
U_zero = zeros(nu*N,1);

% Tunable parameters
Q_val = [0.6;0.6;0.03];
R_val = [0.02;0.04];
W_val = [4;4;0.8];
eta_circle_val = 100;
circle_r_val = 0.7;
circle_c = [1.55;1.45];

% Construct the unified theta vector
theta_val = [Q_val; R_val; W_val; eta_circle_val; circle_r_val];

%% ------------------- Path settings -------------------
addpath(genpath('../compution_time'));
addpath(genpath('../../casadi-3.7.2-windows64-matlab2018b'));
if ~exist('mex','dir'); mkdir('mex'); end
addpath('./mex');

%% ------------------- Generate CasADi functions -------------------
mex_file = fullfile('mex',['tl_f_and_grad_u_theta.' mexext]);
if true
    clear mex;

    U = SX.sym('U', nu*N);
    P = SX.sym('P', 2*nx); 
    theta = SX.sym('theta', nx+nu+nx+1+1);  % theta: [Q; R; W; eta; circle_r]

    % Unpack theta
    Q_sym        = theta(1:nx);              
    R_sym        = theta(nx+1:nx+nu);        
    W_sym        = theta(nx+nu+1:2*nx+nu);   
    eta_circle_sym = theta(2*nx+nu+1);       
    circle_r_sym   = theta(2*nx+nu+2);       

    x0_sym = P(1:nx); 
    xref_sym = P(nx+1:2*nx);
    xk = x0_sym;
    J = SX(0);

    for k=1:N
        uk = U((k-1)*nu+1:k*nu);
        dx = xk - xref_sym;
        J = J + sum(Q_sym .* (dx.^2)) + sum(R_sym .* (uk.^2));

        z = xk(1:2);
        h = 1 - ((z - circle_c).' * (z - circle_c)) / circle_r_sym^2;
        psi = 0.5 * eta_circle_sym * if_else(h>0, h^2, 0);
        J = J + psi;

        xk = euler_step_symbolic(xk, uk, dt, L);
    end

    dxN = xk - xref_sym;
    J = J + sum(W_sym .* (dxN.^2));

    grad_J = gradient(J, U);

    Function('tl_f_and_grad_u_theta', ...
        {U, P, theta}, {J, grad_J}) ...
        .generate('tl_f_and_grad_u_theta.c', struct('mex', true));
    mex('-outdir','mex','tl_f_and_grad_u_theta.c');
end

%% ------------------- Initialize PANOC/PANDA -------------------
problem.dimension = N*nu;
problem.constraint_type = 'costum';
problem.constraint = @(x,gamma) indBox_manual(x,ub,lb,gamma);

solver_params.tolerance = 1e-4;
solver_params.buffer_size = 10;
solver_params.max_iterations = 800;
solver_params.max_stable_iter = 80;

panoc('init',problem,solver_params);

%% ------------------- Closed-loop computation -------------------
record = struct([]);
X_cl = zeros(nx,Tsim+1);
X_cl(:,1) = x_init;
xcur = x_init;

for t=1:Tsim
    P_current = [xcur;x_ref];
    Uguess = U_zero;

    % Oracle with theta
    teacher_oracle = @(U) tl_oracle_wrapper_theta(U, P_current, theta_val);

    tic;
    [iter,gamma] = panoc('solve', Uguess, teacher_oracle);
    runtime = toc;

    Usol = Uguess(:);
    X_pred = rollout_numeric(xcur,Usol,nx,nu,N,dt,L);
    U_pred = reshape(Usol,nu,N);

    record(t).P      = P_current;
    record(t).x0     = xcur;
    record(t).u_star = Usol + 0*Usol;
    record(t).X_star = X_pred + 0*X_pred;
    record(t).U_star = U_pred + 0*U_pred;
    record(t).iter   = iter;
    record(t).runtime= runtime;

    xcur = X_pred(:,2);
    X_cl(:,t+1) = xcur;

    fprintf('Step %02d/%02d | iter = %d | time = %.6f s\n', t,Tsim,iter,runtime);
end

panoc('cleanup');

%% ------------------- Save and plot -------------------
teacher_pars.theta = theta_val;
scene.x0 = x_init; scene.xref = x_ref; scene.circle_c = circle_c;

save('teacher_record.mat','record','teacher_pars','scene','X_cl');

plot_result(scene,circle_r_val,X_cl,X_pred);

clear mex;

%% ==================== Helper functions ====================
function [f,g] = tl_oracle_wrapper_theta(U,P,theta)
    [f,g] = tl_f_and_grad_u_theta(U,P,theta);
    f = full(f); f = double(f); f = f(1);
    g = full(g); g = double(g); g = g(:);
end

function X = rollout_numeric(x0,U,nx,nu,N,dt,L)
    U = U(:); X = zeros(nx,N+1); X(:,1)=x0(:);
    xk = x0(:);
    for k=1:N
        uk = U((k-1)*nu+1:k*nu);
        xk = euler_step_numeric(xk,uk,dt,L);
        X(:,k+1) = xk;
    end
end

function xnext = euler_step_numeric(x,u,dt,L)
    dx = dyn_numeric(x,u,L);
    xnext = x + dt*dx;
end

function dx = dyn_numeric(x,u,L)
    th = x(3); ux = u(1); uy = u(2);
    thdot = (uy*cos(th) - ux*sin(th))/L;
    pxdot = ux + L*sin(th)*thdot;
    pydot = uy - L*cos(th)*thdot;
    dx = [pxdot;pydot;thdot];
end

function xnext = euler_step_symbolic(x,u,dt,L)
    dx = dyn_symbolic(x,u,L);
    xnext = x + dt*dx;
end

function dx = dyn_symbolic(x,u,L)
    th = x(3); ux = u(1); uy = u(2);
    thdot = (uy*cos(th) - ux*sin(th))/L;
    pxdot = ux + L*sin(th)*thdot;
    pydot = uy - L*cos(th)*thdot;
    dx = [pxdot;pydot;thdot];
end

function plot_result(scene,circle_r,X_cl,X_pred)
    figure('Color','w'); hold on; grid on; box on; axis equal;
    plot(X_cl(1,:), X_cl(2,:), 'LineWidth',2);
    plot(X_pred(1,:), X_pred(2,:), '--','LineWidth',1.5);
    plot(scene.x0(1),scene.x0(2),'ko','MarkerSize',8,'LineWidth',1.5);
    plot(scene.xref(1),scene.xref(2),'kp','MarkerSize',10,'LineWidth',1.5);
    tt = linspace(0,2*pi,300);
    plot(scene.circle_c(1)+circle_r*cos(tt), scene.circle_c(2)+circle_r*sin(tt),'k--','LineWidth',1.5);
    xlabel('$p_x$','Interpreter','latex'); ylabel('$p_y$','Interpreter','latex');
    legend('closed-loop','last prediction','initial','target','obstacle','Location','best');
    title('Variable-parameter target generation','Interpreter','latex');
end