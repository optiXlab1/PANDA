# PANDA: A Matrix-Free Differentiable Optimization Solver

## 1. Project Overview
We propose a matrix-free differentiable optimization solver, **PANDA**. Currently, the forward optimization is implemented in C, while the backward process is implemented in MATLAB. PANDA can run on MATLAB, and support for other platforms will be explored in future work.

## 2. Folder Structure

### 2.1 `PANDA_SRC`
Contains the C implementation of the PANDA solver. The code structure is inspired by [PANOC reference implementation](https://github.com/kul-optec/C_FBE_algos/tree/master/PANOC).

### 2.2 `Experiments`
Includes code for evaluating computation time and memory usage for PANDA and baseline solvers. This folder contains all scripts necessary to run the experiments and reproduce the results.

### 2.3 `Results_plot`
Contains scripts for plotting the pre-recorded experimental results.

## 3. Reproduction Workflow

### 3.1 Compile PANDA
Ensure you have **CMake** installed. Run the provided script `PANDA_SRC/Matlab_install_windows.ps1` to generate `panda.mex64`.

### 3.2 Reproduce Time Experiments
1. Move `panda.mex64` to the corresponding subfolders under `Experiments/Computation_time`.
2. Run the scripts to generate runtime records.
3. To record PANDA solver residuals, run multiple times and manually rename `panda_debug.csv`.
4. Baseline solver environments:
   | Solver  | Reference                                                    |
   | ------- | ------------------------------------------------------------ |
   | CasADi  | [https://web.casadi.org/blog/nlp_sens](https://web.casadi.org/blog/nlp_sens/) |
   | SafePDP | [https://github.com/wanxinjin/Safe-PDP](https://github.com/wanxinjin/Safe-PDP) |
   | acados  | [https://docs.acados.org](https://docs.acados.org/)          |

### 3.3 Reproduce Memory Experiments

Follow the instructions and scripts under `Experiments/Memory_consumption`, similar to the runtime experiments. Observe memory usage as described in the folder documentation.

### 3.4 Result Visualization

Use the scripts in `Results_plot` to generate figures from the collected data.

## 4. Using PANDA for Your MPC Problem

The workflow is based on `train.m`:

### 4.1 Define Your Problem
- Use CasADi to define your outer loss `L`, cost `J`, control `U`, state `X`, parameters `theta`, etc.

### 4.2 Compile `panda_backward` Functions

Compile the necessary function handles to MEX files:

```matlab
Function('tl_f_and_grad_u', {U_sym, theta_sym, variable_sym}, {J, grad_J}) ...
    .generate('tl_f_and_grad_u.c', struct('mex',true));

Function('tl_L_and_grad_u', {U_sym, variable_sym}, {Lval, grad_L}) ...
    .generate('tl_L_and_grad_u.c', struct('mex',true));

Function('tl_vjp_f_u_theta', {v_sym, U_sym, theta_sym, variable_sym}, {vjp_theta}) ...
    .generate('tl_vjp_f_u_theta.c', struct('mex',true));

Function('tl_hvp_f_u', {v_sym, U_sym, theta_sym, variable_sym}, {Hv}) ...
    .generate('tl_hvp_f_u.c', struct('mex',true));

mex('-outdir', mex_dir, 'tl_f_and_grad_u.c');
mex('-outdir', mex_dir, 'tl_L_and_grad_u.c');
mex('-outdir', mex_dir, 'tl_vjp_f_u_theta.c');
mex('-outdir', mex_dir, 'tl_hvp_f_u.c');
```

### 4.3 Forward and Backward Solve

```matlab
% Define solver and problem hyperparameters
problem.dimension      = N*nu;
problem.constraint_type = 'custom';
problem.constraint      = @(x,gamma) indBox_manual(x, ub, lb, gamma);

solver_params.tolerance       = 1e-4;
solver_params.buffer_size     = 10;
solver_params.max_iterations  = 1000;
solver_params.max_stable_iter = 80;

% Function interfaces
Jprox_u    = @(x,gamma) proximal_operator('box_grad', x, ub, lb, gamma);
oracle_L   = @(u,variable) tl_L_and_grad_u(u, variable);
oracle_vjp = @(v,u,theta,variable) tl_vjp_f_u_theta(v, u, theta, variable);
oracle_hvp = @(v,u,theta,variable) tl_hvp_f_u(v, u, theta, variable);

% Backward iteration hyperparameters
opts.tol   = 1e-4;
opts.maxit = 200;

% Training hyperparameters
epochs = 120;
lr     = 2e-4;
theta  = theta_val;

% Initialize
u_guess_bank = zeros(problem.dimension, batch_size);
panoc('init', problem, solver_params);

for ep = 1:epochs
    grad_acc = zeros(size(theta));
    loss_acc = 0;

    for b = 1:batch_size
        variable_b = pack_record_variable(record(batch_idx(b)), nx, nu, N, x_ref_default, circle_c_default);

        %% Forward pass
        oracle_fg = @(u) tl_f_and_grad_u(u, theta, variable_b);
        [u_star, iters, gamma] = solve_once(u_guess_bank(:,b), oracle_fg);

        %% Backward pass
        [Lval, dLdtheta, info] = panda_backward(u_star, theta, variable_b, gamma, ...
                                                Jprox_u, oracle_hvp, oracle_L, oracle_vjp, opts);

        % Accumulate gradients
        grad_acc = grad_acc + double(full(dLdtheta(:)));
        loss_acc = loss_acc + double(full(Lval));

        u_guess_bank(:,b) = zeros(problem.dimension,1);
    end

    % Parameter update
    theta = max(theta - lr * grad_acc / batch_size, 0.01);
end

panoc('cleanup');
```

### 4.4 Notes

- Adjust hyperparameters and batch size as needed for your problem.
- Ensure all generated MEX files are in MATLAB path before calling `panda.mex64` or `panda_backward.m`.
- Follow the example structure of `train.m` to integrate PANDA with your MPC problem.