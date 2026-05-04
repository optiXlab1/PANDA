#include "arguments_check.h"
#include "panda.h"
#include "mex.h"

int check_input_arguments(int nrhs, const mxArray* prhs[]){
    const char * error_message_invalid_mode = "The first input argument must be either 'init','solve' or 'cleanup'";
    if (!mxIsChar(prhs[0])) {
        mexErrMsgTxt(error_message_invalid_mode);
    }
    const int mode = validate_mode(prhs);
    switch (mode)
    {
        case INVALID_MODE:
            mexErrMsgTxt(error_message_invalid_mode);
            break;
        case INIT_MODE:
            check_input_arguments_init_mode(nrhs, prhs);
            break;
    }
    return mode;
}

int validate_mode(const mxArray *prhs[]){
    const char* mode_in_char = mxGetChars(prhs[0]);

    if (mode_in_char[0] == 'i') return INIT_MODE;
    if (mode_in_char[0] == 'c') return CLEANUP_MODE;
    if (mode_in_char[0] == 's') return SOLVE_MODE;
    return INVALID_MODE;
}

int check_input_arguments_init_mode(int nrhs, const mxArray *prhs[]){
    const char* error_message_invalid_init_args = "invalid options with init, use panoc(mode,problem,solver_params)";

    if(nrhs != 3) mexErrMsgTxt(error_message_invalid_init_args);
    if (!mxIsStruct(prhs[1]) || !mxIsStruct(prhs[2])) mexErrMsgTxt(error_message_invalid_init_args);

    return SUCCESS;
}