# Memory Benchmark

This folder contains the memory benchmark used to compare the memory overhead of different optimization and differentiable optimization solvers.

Measuring the memory overhead of the solving process is not straightforward, because different solvers use different wrapping styles and allocate memory at different stages. In this benchmark, the reported memory refers to the solver-side working memory required by the numerical solving process and, when applicable, by the backward or sensitivity computation. It does not include the memory used to construct the symbolic/modeling-level problem structure.

The solver-side working memory includes buffers, work arrays, matrix/vector storage, linear-solver workspaces, factorization-related memory, and other numerical memory required during the actual solve. Some solvers allocate such memory during the numerical solve, while others allocate part of it during solver construction. Therefore, the measurement protocol is adapted to each solver.

One-time overheads that are unrelated to repeated numerical solving, such as library loading, MEX loading, generated-code loading, symbolic function initialization, and runtime-level initialization, are excluded whenever possible.

## PANDA

PANDA has a relatively explicit memory-allocation pattern in our implementation. The forward solver is initialized directly, and the backward pass is called separately.

Therefore, for PANDA, the memory is measured from the first solver initialization, the first forward solve, and the backward pass. This includes the forward solve-time working memory and the memory required by the matrix-free backward computation.

## SafePDP

For SafePDP, part of the solver-side working memory can be allocated during solver construction. However, the first construction may also contain one-time initialization overheads that are not directly related to the numerical solve.

Therefore, the first construction is used only as a warm-up. The reported memory is measured from the second construction and the first numerical solve of the measured solver instance. This keeps the solver-side working memory that is preallocated during construction, while reducing the influence of unrelated one-time overheads.

## acados

acados follows a similar pattern to SafePDP. Part of the memory required by the numerical solve, such as solver workspaces and QP-related memory, can be allocated during solver construction. The first construction may also include generated solver loading, capsule creation, external library initialization, and other one-time overheads.

Therefore, the first construction is used only as a warm-up. The reported memory is measured from the second construction, the first numerical solve, and the sensitivity evaluation when applicable.

## CasADi

For CasADi, the backward computation in this benchmark is evaluated through a `Function`-based implementation. The relevant solver-side working memory is mainly observed during the first numerical solve/evaluation, where the backend solver allocates work arrays, matrix/vector storage, and other solve-time numerical memory.

Therefore, the reported memory is measured from the memory change before and after the first solve/evaluation. The symbolic/modeling-level problem construction is not included.

## Notes

The benchmark is intended to compare the memory-scaling behavior with respect to the prediction horizon. Absolute memory values may depend on the operating system, MATLAB/Python runtime, compiler, BLAS/LAPACK backend, external solver versions, and memory profiler behavior.

The reported values should therefore be interpreted as approximate solver-side working-memory overheads under the above measurement protocol, rather than exact internal memory allocations of each solver.