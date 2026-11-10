#include "mipk_pipeline_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/*
 * Where diagnostics go.  A pipeline call that accepts a diagnostic buffer
 * arms the sink for the calling thread before doing any work and disarms it
 * on the way out, so a message written by a module deep inside open() lands
 * in the buffer of the application that asked for it.  With no buffer armed
 * the message goes to stderr, which is what a quick command-line experiment
 * wants.
 */
typedef struct {
    char *buffer;
    size_t size;
    size_t used;
} diag_sink;

static _Thread_local diag_sink g_sink;

void mipk_diag_begin(char *buffer, size_t size)
{
    g_sink.buffer = buffer;
    g_sink.size = buffer == NULL ? 0 : size;
    g_sink.used = 0;
    if (g_sink.size != 0) buffer[0] = '\0';
}

void mipk_diag_end(void)
{
    g_sink.buffer = NULL;
    g_sink.size = 0;
    g_sink.used = 0;
}

void mipk_diag(const char *format, ...)
{
    char line[512];
    va_list arguments;
    va_start(arguments, format);
    const int length = vsnprintf(line, sizeof(line), format, arguments);
    va_end(arguments);
    if (length < 0) return;
    if (g_sink.buffer == NULL || g_sink.size == 0) {
        fprintf(stderr, "mipk: %s\n", line);
        return;
    }
    const size_t text = (size_t)length < sizeof(line) ? (size_t)length :
        sizeof(line) - 1U;
    /* Messages are separated by newlines; the buffer stays NUL terminated. */
    size_t room = g_sink.size - 1U - g_sink.used;
    if (g_sink.used != 0 && room != 0) {
        g_sink.buffer[g_sink.used++] = '\n';
        --room;
    }
    const size_t copy = text < room ? text : room;
    memcpy(g_sink.buffer + g_sink.used, line, copy);
    g_sink.used += copy;
    g_sink.buffer[g_sink.used] = '\0';
}
