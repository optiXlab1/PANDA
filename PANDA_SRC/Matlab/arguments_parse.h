#ifndef PANDA_ARGUMENTS_PARSE_H
#define PANDA_ARGUMENTS_PARSE_H

#include "mex.h"
#include"../include/optimizer.h"

int parse_problem(const mxArray *prhs[],struct optimizer_problem* problem);
int parse_solver(const mxArray *prhs[],struct optimizer_problem* problem);

char* parser_get_name_cost_function(void);

#endif
