#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "../globals/globals.h"

/* solver parameters */
struct solver_parameters {
    unsigned int max_iterations;   /* maximum PANOC+ iterations */
    real_t       tolerance;        /* stopping tolerance */
    unsigned int buffer_size;      /* L-BFGS memory size */
    unsigned int max_stable_iter;  /* used by gamma enlargement */
};

/* optimization problem */
struct optimizer_problem {
    unsigned int dimension;

    /*
     * proxg:
     *   input  = x - gamma * grad f(x)
     *   output = prox_g(input)
     *   return = g(prox_g(input))
     */
    real_t (*proxg)(real_t* input, real_t gamma);

    /*
     * cost_gradient_function:
     *   input  = x
     *   output = grad f(x)
     *   return = f(x)
     */
    real_t (*cost_gradient_function)(const real_t* input, real_t* output_gradient);

    struct solver_parameters solver_params;
};

/* initialize optimizer */
int optimizer_init(struct optimizer_problem* problem_);

/* initialize optimizer with user-provided proximal operator */
int optimizer_init_with_custom_constraint(
    struct optimizer_problem* problem_,
    real_t (*proxg)(real_t* x, real_t gamma)
);

/* cleanup optimizer */
int optimizer_cleanup(void);

/* solve problem, overwriting solution in-place */
int solve_problem(real_t* solution);

/* get final accepted gamma */
real_t get_gamma(void);

#endif /* OPTIMIZER_H */