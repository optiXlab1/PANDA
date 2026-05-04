#ifndef PANDA_H
#define PANDA_H

#include "../globals/globals.h"
#include "panda_types.h"

/*
 * Initialize / cleanup the PANDA solver.
 */
int panda_init(const panda_params* params);
int panda_cleanup(void);

/*
 * Set the initial point x0.
 * This evaluates and stores the initial previous state.
 */
int panda_set_initial(const real_t* x0);

/*
 * Perform one PANDA iteration.
 *
 * Returns the infinity norm of the normalized residual:
 *     ||(x-z)/gamma||_inf
 * of the accepted new state.
 */
real_t panda_step(void);

/*
 * Reset iteration-dependent memory such as L-BFGS history.
 * This should be called when solving a new problem instance.
 */
int panda_reset(void);

/*
 * Accessors to the currently accepted state.
 */
const real_t* panda_get_solution(void);   /* returns x */
const real_t* panda_get_residual(void);   /* returns x-z */
real_t panda_get_gamma(void);
real_t panda_get_tau(void);

#endif