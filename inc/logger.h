#ifndef ICAN_ASSERT_H

#define ICAN_ASSERT_H

#include "ican.h"

typedef enum{
    LOG_LEVEL_FATAL = 0,
    LOG_LEVEL_ERROR = 1,
    LOG_LEVEL_WARN = 2,
    LOG_LEVEL_INFO = 3,
    LOG_LEVEL_DEBUG = 4,
    LOG_LEVEL_TRACE = 5
} LogLevels;


#define LOG_WARN_ENABLED 1
#define LOG_INFO_ENABLED 1
#define LOG_DEBUG_ENABLED 1
#define LOG_TRACE_ENABLED 1

#if LOG_RELASE == 1
#define LOG_DEBUG_ENABLED 0
#define LOG_TRACE_ENABLED 0
#endif

#define ASSERT(expr)                                         \
    {                                                                \
        if (expr) {                                                  \
        } else {                                                     \
            report_assertion_failure(#expr, "", __FILE__, __LINE__); \
            __builtin_trap();                                        \
        }                                                            \
    }

#define ASSERT_MSG(expr, message)                                 \
    {                                                                     \
        if (expr) {                                                       \
        } else {                                                          \
            report_assertion_failure(#expr, message, __FILE__, __LINE__); \
            __builtin_trap();                                             \
        }                                                                 \
    }

void report_assertion_failure(const char* expression, const char* message, const char* file, int line);
void LOG(LogLevels level, const char* message, ...);

#endif // !ICAN_ASSERT_H