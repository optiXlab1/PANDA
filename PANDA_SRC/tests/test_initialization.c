#include "panda.h"
#include "panda_eval.h"
#include "function_evaluator.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int quartic = 0;

static real_t cost_gradient(const real_t* x, real_t* gradient)
{
    if (quartic) {
        gradient[0] = x[0] + 400.0*x[0]*x[0]*x[0] - 1.0;
        return 0.5*x[0]*x[0] + 100.0*pow(x[0], 4.0) - x[0];
    }
    gradient[0] = 4.0*x[0];
    return 2.0*x[0]*x[0];
}

static real_t identity_prox(real_t* x, real_t gamma)
{
    (void)x;
    (void)gamma;
    return 0.0;
}

int main(void)
{
    struct optimizer_problem problem;
    panda_params params;
    real_t x = 0.0, gamma, fz, rhs, residual;
    unsigned int i;

    memset(&problem, 0, sizeof(problem));
    problem.dimension = 1;
    problem.proxg = identity_prox;
    problem.cost_gradient_function = cost_gradient;
    function_evaluator_init(&problem);
    memset(&params, 0, sizeof(params));
    params.n = 1;
    params.alpha = 0.95;
    params.beta = 0.5;
    params.minimum_gamma = 1e-8;
    params.max_backtracks = 80;
    params.lbfgs_memory = 10;
    params.gamma_enlarge_factor = 2.0;

    if (panda_init(&params) == FAILURE) return 1;
    gamma = panda_eval_initial_gamma(&x, params.alpha);
    if (fabs(gamma - 0.95/4.0) > 1e-8) {
        fprintf(stderr, "Incorrect initial Lipschitz estimate: gamma=%g\n", gamma);
        return 2;
    }
    if (panda_set_initial(&x) == FAILURE) return 3;
    panda_cleanup();

    quartic = 1;
    if (panda_init(&params) == FAILURE || panda_set_initial(&x) == FAILURE) return 4;
    gamma = panda_get_gamma();
    fz = 0.5*gamma*gamma + 100.0*pow(gamma,4.0) - gamma;
    rhs = -gamma + 0.5*params.alpha*gamma;
    if (fz > rhs + 1e-10 || gamma >= 0.5) {
        fprintf(stderr, "Initial upper model violated: gamma=%g\n", gamma);
        return 5;
    }
    for (i=0; i<500; ++i) {
        residual = panda_step();
        if (!isfinite(residual) || residual < 0.0) return 6;
        if (residual < 1e-7) break;
    }
    if (i == 500) return 7;
    panda_cleanup();

    params.minimum_gamma = 1.0;
    if (panda_init(&params) == FAILURE) return 8;
    if (panda_set_initial(&x) != FAILURE || panda_step() >= 0.0) return 9;
    panda_cleanup();
    function_evaluator_cleanup();
    puts("Initialization regression checks passed.");
    return 0;
}
