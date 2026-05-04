#ifndef PANDA_ARGUMENTS_CHECK_H
#define PANDA_ARGUMENTS_CHECK_H
#include "mex.h"

int check_input_arguments(int nrhs, const mxArray* prhs[]);
int check_input_arguments_init_mode(int nrhs, const mxArray *prhs[]);

int validate_mode(const mxArray *prhs[]);

#endif