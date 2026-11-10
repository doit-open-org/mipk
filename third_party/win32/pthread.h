/*
 * The subset of pthreads mipk uses, on Win32 primitives.  Only built into
 * MSVC builds (third_party/CMakeLists.txt puts this directory on the include
 * path when WIN32 is set); every other target uses the system pthread.h.
 */
#ifndef MIPK_WIN32_PTHREAD_H
#define MIPK_WIN32_PTHREAD_H

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

typedef SRWLOCK pthread_mutex_t;
typedef void pthread_mutexattr_t;
typedef INIT_ONCE pthread_once_t;

#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT
#define PTHREAD_ONCE_INIT INIT_ONCE_STATIC_INIT

static __inline int pthread_mutex_init(pthread_mutex_t *mutex,
    const pthread_mutexattr_t *attributes)
{
    (void)attributes;
    InitializeSRWLock(mutex);
    return 0;
}

static __inline int pthread_mutex_destroy(pthread_mutex_t *mutex)
{
    (void)mutex;
    return 0;
}

static __inline int pthread_mutex_lock(pthread_mutex_t *mutex)
{
    AcquireSRWLockExclusive(mutex);
    return 0;
}

static __inline int pthread_mutex_trylock(pthread_mutex_t *mutex)
{
    return TryAcquireSRWLockExclusive(mutex) ? 0 : 16; /* EBUSY */
}

static __inline int pthread_mutex_unlock(pthread_mutex_t *mutex)
{
    ReleaseSRWLockExclusive(mutex);
    return 0;
}

static BOOL CALLBACK mipk_win32_once_trampoline(PINIT_ONCE once,
    PVOID routine, PVOID *context)
{
    (void)once;
    (void)context;
    ((void (*)(void))routine)();
    return TRUE;
}

static __inline int pthread_once(pthread_once_t *once, void (*routine)(void))
{
    return InitOnceExecuteOnce(once, mipk_win32_once_trampoline,
        (PVOID)routine, NULL) ? 0 : 22; /* EINVAL */
}

#endif
