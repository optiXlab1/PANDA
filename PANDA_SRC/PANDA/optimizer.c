#include "../globals/globals.h"
#include "function_evaluator.h"
#include "panda.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "../include/optimizer.h"

static struct optimizer_problem* problem; /* data related to the problem */
static unsigned char initialized = FALSE;
static real_t gamma;

static void save_solution(real_t* solution);
static void save_gamma(real_t t_gamma);
static void log_residual(FILE* fp, unsigned int iter, real_t residual);

int solve_problem(real_t* solution)
{
    unsigned int i_panda;
    real_t current_residual;
    int status;
    FILE* residual_fp = NULL;

    if (initialized == FALSE) {
        return FAILURE;
    }

    residual_fp = fopen("panda_residual_log.csv", "w");
    
    status = panda_set_initial(solution);
    if (status == FAILURE) {
        fclose(residual_fp);
        return FAILURE;
    }

    current_residual = problem->solver_params.tolerance * 10.0;

    for (i_panda = 0; i_panda < problem->solver_params.max_iterations; i_panda++) {
        if (current_residual < problem->solver_params.tolerance) {
            break;
        }

        current_residual = panda_step();

        log_residual(residual_fp, i_panda, current_residual);

        if (current_residual > MACHINE_ACCURACY) {
            save_solution(solution);
        }
    }

    fclose(residual_fp);

    save_gamma(panda_get_gamma());
    panda_reset();

    return i_panda;
}

static void save_gamma(real_t t_gamma)
{
    gamma = t_gamma;
}

static void save_solution(real_t* solution)
{
    unsigned int i;
    const real_t* accepted_solution = panda_get_solution();

    for (i = 0; i < problem->dimension; i++) {
        solution[i] = accepted_solution[i];
    }
}

real_t get_gamma()
{
    return gamma;
}

int optimizer_init(struct optimizer_problem* problem_)
{
    panda_params params;

    if (initialized == TRUE) {
        optimizer_cleanup();
    }

    problem = problem_;

    if (function_evaluator_init(problem) == FAILURE) {
        return FAILURE;
    }

    params.n = problem->dimension;
    params.alpha = 1.0 - PROXIMAL_GRAD_DESC_SAFETY_VALUE;
    params.beta = 0.5;
    params.minimum_gamma = 1e-6;
    params.max_backtracks = FBE_LINESEARCH_MAX_ITERATIONS;
    params.lbfgs_memory = problem->solver_params.buffer_size;
    params.enable_gamma_enlarge =
        (problem->solver_params.max_stable_iter > 0) ? TRUE : FALSE;
    params.max_stable_iter = problem->solver_params.max_stable_iter;
    params.residual_enlarge_threshold = 1e-3;
    params.gamma_enlarge_factor = 2.0;

    if (panda_init(&params) == FAILURE) {
        function_evaluator_cleanup();
        return FAILURE;
    }

    initialized = TRUE;
    return SUCCESS;
}

static void log_residual(FILE* fp, unsigned int iter, real_t residual)
{
    if (fp != NULL) {
        fprintf(fp, "%u,%.16e\n", iter, (double)residual);
        fflush(fp);
    }
}

int optimizer_init_with_costum_constraint(struct optimizer_problem* problem_,
                                          real_t (*proxg)(real_t* x, real_t gamma))
{
    problem_->proxg = proxg;
    return optimizer_init(problem_);
}

int optimizer_cleanup(void)
{
    panda_cleanup();
    function_evaluator_cleanup();
    initialized = FALSE;
    return SUCCESS;
}