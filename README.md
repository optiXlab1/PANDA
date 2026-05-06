# PANDA: A Matrix-Free Differentiable NMPC Solver

***PANDA***, short for ***P***roximal ***A***veraged quasi-***N***ewton with a***D***aptive Linesearch ***A***lgorithm, is a unified matrix-free differentiable NMPC solver. It is designed to reduce the computation time and memory overhead of forward optimization and backward sensitivity propagation.

## 1. Project Overview

### 1.1. `PANDA_SRC`

This folder contains the C implementation of the PANDA forward solver. The code structure is inspired by the PANOC implementation in [PANOC](https://github.com/kul-optec/C_FBE_algos/tree/master/PANOC).

The MATLAB installation script for compiling the MEX interface is also provided in this folder.

### 1.2. `Experiments`

This folder contains the experimental code for evaluating computation time and memory overhead. It includes the scripts for `PANDA`, baseline solvers, and the corresponding experimental procedures.

Main subfolders include:

- `Computation_time`: scripts for runtime evaluation, including stepsize validation experiment and  benchmark comparision;
- `Memory_consumption`: scripts and instructions for memory-consumption evaluation.

### 1.3. `Results_plot`

This folder contains the experimental data reported in the paper and the plotting scripts used to visualize the results.

## 2. Requirements

The current version is tested with MATLAB on Windows. Please make sure the following tools are available:

- MATLAB;
- CMake;
- a C/C++ compiler supported by MATLAB MEX;
- CasADi for MATLAB;
- baseline solver environments, if all comparison experiments are reproduced.

For the baseline solvers, please refer to:

| Solver | Reference |
| --- | --- |
| CasADi | [https://web.casadi.org/blog/nlp_sens](https://web.casadi.org/blog/nlp_sens/) |
| SafePDP | [https://github.com/wanxinjin/Safe-PDP](https://github.com/wanxinjin/Safe-PDP) |
| acados | [https://docs.acados.org](https://docs.acados.org/) |

## 3. Reproduction Workflow

### 3.1. Compile PANDA

First, compile the PANDA MEX interface.

Please make sure that CMake and a MATLAB-supported compiler are installed. Then run:

```powershell
PANDA_SRC/Matlab_install_windows.ps1
```

This script generates:

```text
panda.mex64
```

### 3.2. Reproduce Computation-Time Experiments

First, copy the generated `panda.mex64` file to the corresponding subfolders under:

```text
Experiments/Computation_time
```

#### 3.2.1. Adaptive Stepsize Validation

Copy `panda.mex64` to:

```
Experiments/Computation_time/Adaptive_stepsize_validation
```

Then run:

```
PANDA_PANOC.m
```

This script generates the summary file:

```
PANOC_PANDA_summary.csv
```

and the solver residual debug file:

```
panda_debug.csv
```

To obtain residual curves at different time instants, modify the corresponding time-instant setting in the script and run it multiple times. After each run, manually rename `panda_debug.csv`, for example:

```
PANDA_RES_0.csv
PANDA_RES_spike.csv
```

The residual files used in the paper have already been collected in `Results_plot`.

#### 3.2.2. Benchmark Comparison

Copy `panda.mex64` to:

```
Experiments/Computation_time/Benchmark_comparision
```

Then first run:

```
generate_target.m
```

to generate the demonstration data:

```
teacher_record.mat
```

After that, run:

```
train.m
```

to generate the PANDA training records, including runtime and loss information.

For the baseline methods, please configure the corresponding environments according to the official documentation listed in Section 2. Then run the baseline scripts provided in this repository to generate their training records.

### 3.3. Reproduce Memory-Consumption Experiments

Please first read the documentation under:

```text
Experiments/Memory_consumption
```

The memory experiments follow a similar procedure to the computation-time experiments. Run the corresponding scripts and observe the memory overhead according to the protocol described in the folder documentation.

In general, the reported memory focuses on the solver-side working memory required by numerical solving and backward/sensitivity computation. It does not include symbolic or modeling-level problem-construction memory.

### 3.4. Visualize Results

Use the scripts under:

```text
Results_plot
```

to generate figures from the collected experimental data.

## 4. How to Use PANDA for Your Own MPC Problem

The recommended workflow follows the structure of `train.m`.

### 4.1. Define Your MPC Problem with CasADi

Use CasADi to define your own optimal control problem, including:

- control variable `U`;
- state trajectory `X`;
- learnable parameter `theta`;
- inner objective `J`;
- outer imitation or task loss `Lval`;
- gradient of the inner objective with respect to `U`;
- gradient of the outer loss with respect to `U`;
- Hessian-vector product for the inner objective;
- vector-Jacobian product with respect to `theta`.

A typical setting is:

```matlab
grad_J = gradient(J, U_sym);
grad_L = gradient(Lval, U_sym);

vjp_theta = jtimes(grad_J, theta_sym, v_sym, true);
Hv        = jtimes(grad_J, U_sym, v_sym, false);
```

### 4.2. Generate MEX Functions for the Backward Pass

The backward pass requires several function handles. Generate them using CasADi and compile them into MEX files.

```matlab
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
```

### 4.3. Set PANDA Solver Parameters

Define the problem dimension, constraints, and solver hyperparameters.

```matlab
% Solver hyperparameters
problem.dimension       = N*nu;
problem.constraint_type = 'custom';
problem.constraint      = @(x,gamma) indBox_manual(x, ub, lb, gamma);

solver_params.tolerance       = 1e-4;
solver_params.buffer_size     = 10;
solver_params.max_iterations  = 1000;
solver_params.max_stable_iter = 80;
```

Here, `problem.constraint` defines the proximal operator or projection associated with the nonsmooth term or hard constraints. For box constraints, `indBox_manual` is used.

### 4.4. Define Function Interfaces for the Backward Pass

Define the proximal Jacobian and the function handles required by `panda_backward.m`.

```matlab
% Function interfaces for backward sensitivity propagation
Jprox_u    = @(x,gamma) proximal_operator('box_grad', x, ub, lb, gamma);
oracle_L   = @(u,variable) tl_L_and_grad_u(u, variable);
oracle_vjp = @(v,u,theta,variable) tl_vjp_f_u_theta(v, u, theta, variable);
oracle_hvp = @(v,u,theta,variable) tl_hvp_f_u(v, u, theta, variable);

% Iterative linear solver hyperparameters for the backward pass
opts.tol   = 1e-4;
opts.maxit = 200;
```

### 4.5. Run Forward Optimization and Backward Sensitivity Propagation

Initialize the forward solver by:

```matlab
panda('init', problem, solver_params);
```

Then run the training loop.

```matlab
% Training hyperparameters
epochs = 120;
lr     = 2e-4;
theta  = theta_val;

u_guess_bank = zeros(problem.dimension, batch_size);

for ep = 1:epochs

    grad_acc = zeros(size(theta));
    loss_acc = 0;

    for b = 1:batch_size

        idx = batch_idx(b);

        variable_b = pack_record_variable( ...
            record(idx), ...
            nx, nu, N, ...
            x_ref_default, ...
            circle_c_default);

        %% Forward pass
        oracle_fg = @(u) tl_f_and_grad_u(u, theta, variable_b);

        [u_star, iters, gamma] = solve_once( ...
            u_guess_bank(:,b), ...
            oracle_fg);

        %% Backward pass
        [Lval, dLdtheta, info] = panda_backward( ...
            u_star, theta, variable_b, gamma, ...
            Jprox_u, oracle_hvp, oracle_L, oracle_vjp, opts);

        Lval     = double(full(Lval));
        dLdtheta = double(full(dLdtheta(:)));

        grad_acc = grad_acc + dLdtheta;
        loss_acc = loss_acc + Lval;

        % Reset or update warm start
        u_guess_bank(:,b) = zeros(problem.dimension,1);
    end

    % Average gradient over the batch
    grad_acc = grad_acc / batch_size;
    loss_acc = loss_acc / batch_size;

    % Parameter update
    theta = theta - lr * grad_acc;

    % Optional projection for positive parameters
    theta = max(theta, 0.01);
end
```

After training or solving, release the solver memory by:

```matlab
panda('cleanup');
```

### 4.6. Notes for Custom Problems

- Make sure `panda.mex64` and all generated CasADi MEX files are in the MATLAB path.
- For a new MPC problem, users mainly need to redefine the CasADi symbolic expressions, generate the required MEX functions, and keep the forward/backward interface consistent with `train.m`.
- If box constraints are used, the provided proximal operator and proximal Jacobian can be reused.
- For other constraints or nonsmooth terms, the corresponding proximal operator and proximal Jacobian should be implemented.
- The current version supports MATLAB-based workflows. Support for other platforms will be improved in future work.
