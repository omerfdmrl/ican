#ifndef LOG_H

#define LOG_H

#include "ican.h"

#define LOG_WARN_ENABLED 1
#define LOG_INFO_ENABLED 1
#define LOG_DEBUG_ENABLED 1
#define LOG_TRACE_ENABLED 1

#if LOG_RELASE == 1
#define LOG_DEBUG_ENABLED 0
#define LOG_TRACE_ENABLED 0
#endif

void LOG(ILogLevels level, const char* message, ...) {
    if(level == LOG_LEVEL_WARN && LOG_WARN_ENABLED != 1) return;
    if(level == LOG_LEVEL_INFO && LOG_INFO_ENABLED != 1) return;
    if(level == LOG_LEVEL_DEBUG && LOG_DEBUG_ENABLED != 1) return;
    if(level == LOG_LEVEL_TRACE && LOG_TRACE_ENABLED != 1) return;
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

#endif // !LOG_H