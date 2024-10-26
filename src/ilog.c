#ifndef ILOG_H

#define ILOG_H

#include "ican.h"

#define ILOG_WARN_ENABLED 1
#define ILOG_INFO_ENABLED 1
#define ILOG_DEBUG_ENABLED 1
#define ILOG_TRACE_ENABLED 1

#if ILOG_RELASE == 1
#define ILOG_DEBUG_ENABLED 0
#define ILOG_TRACE_ENABLED 0
#endif

void ILOG(ILogLevels level, const char* message, ...) {
    if(level == ILOG_LEVEL_WARN && ILOG_WARN_ENABLED != 1) return;
    if(level == ILOG_LEVEL_INFO && ILOG_INFO_ENABLED != 1) return;
    if(level == ILOG_LEVEL_DEBUG && ILOG_DEBUG_ENABLED != 1) return;
    if(level == ILOG_LEVEL_TRACE && ILOG_TRACE_ENABLED != 1) return;
    const char* level_strings[6] = {"[FATAL]: ", "[ERROR]: ", "[WARN]:  ", "[INFO]:  ", "[DEBUG]: ", "[TRACE]: "};

    char out_message[32000];
    memset(out_message, 0, sizeof(out_message));

    __builtin_va_list arg_ptr;
    va_start(arg_ptr, message);
    vsnprintf(out_message, 32000, message, arg_ptr);
    va_end(arg_ptr);

    char out_message2[32001];
    sprintf(out_message2, "%s%s\n", level_strings[level], out_message);

    printf("%s", out_message2);
}

#endif // !ILOG_H