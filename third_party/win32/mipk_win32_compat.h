/*
 * Force-included into every C/C++ file of an MSVC build (/FI).  Maps the few
 * POSIX calls mipk makes onto their CRT spellings so the sources need no
 * #ifdefs of their own.
 */
#ifndef MIPK_WIN32_COMPAT_H
#define MIPK_WIN32_COMPAT_H

#ifdef _WIN32
#include <errno.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifndef S_ISREG
#define S_ISREG(mode) (((mode) & _S_IFMT) == _S_IFREG)
#endif
#ifndef S_ISDIR
#define S_ISDIR(mode) (((mode) & _S_IFMT) == _S_IFDIR)
#endif

#ifndef __cplusplus
#define strdup _strdup
#endif

/* Pair with mipk_aligned_free(); the CRT cannot free() these. */
static __inline int posix_memalign(void **memory, size_t alignment,
    size_t size)
{
    *memory = _aligned_malloc(size, alignment);
    return *memory == NULL ? ENOMEM : 0;
}
#define mipk_aligned_free _aligned_free

/* realpath(path, NULL): a malloc'd absolute path.  Trailing separators are
 * dropped because the CRT stat() fails on a trailing backslash. */
static __inline char *mipk_win32_realpath(const char *path, char *resolved)
{
    char *full = _fullpath(resolved, path, resolved == NULL ? 0 : _MAX_PATH);
    if (full == NULL) return NULL;
    size_t length = strlen(full);
    while (length > 3U && (full[length - 1U] == '\\' ||
            full[length - 1U] == '/')) {
        full[--length] = '\0';
    }
    return full;
}
#define realpath mipk_win32_realpath
#endif

#endif
