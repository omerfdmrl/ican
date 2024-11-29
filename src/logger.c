#include "logger.h"

void LOG(LogLevels level, const char* message, ...) {
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

void report_assertion_failure(const char* expression, const char* message, const char* file, int line) {
    if(strlen(message) > 0) {
        LOG(LOG_LEVEL_FATAL, "Assertion Failure: %s, message: '%s', in file:%s:%d\n", expression, message, file, line);
    }else {
        LOG(LOG_LEVEL_FATAL, "Assertion Failure: %s, in file:%s:%d\n", expression, file, line);
    }
}