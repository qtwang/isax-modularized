//
// Created by Qitong Wang on 2020/6/28.
//

#ifndef ISAX_GLOBALS_H
#define ISAX_GLOBALS_H

#include <time.h>
#include <float.h>
#include <stdlib.h>
#include <pthread.h>


#define DEBUG


#define FINE_TIMING
#define TIMING

#ifdef FINE_TIMING
#ifndef TIMING
#define TIMING
#endif
#endif

#ifdef TIMING

#define CLK_ID CLOCK_MONOTONIC
#define NSEC_INSEC 1000000000L

int clock_code;

typedef struct TimeDiff {
    long tv_nsec;
    long tv_sec;
} TimeDiff;

inline void getTimeDiff(TimeDiff * t_diff, struct timespec t_start, struct timespec t_stop) {
    t_diff->tv_nsec = t_stop.tv_nsec - t_start.tv_nsec;
    t_diff->tv_sec = t_stop.tv_sec - t_start.tv_sec;
    if (t_diff->tv_nsec < 0) {
        t_diff->tv_sec -= 1;
        t_diff->tv_nsec += NSEC_INSEC;
    }
}

#endif


#define FINE_PROFILING
#define PROFILING

#ifdef FINE_PROFILING
#ifndef PROFILING
#define PROFILING
#endif
#endif

#ifdef PROFILING
unsigned int leaf_counter_profiling;
unsigned int sum2sax_counter_profiling;
unsigned int l2square_counter_profiling;
pthread_mutex_t *log_lock_profiling;
unsigned int query_id_profiling;
#endif


#define CLOGGER_ID 0


#define SAX_SIMD_ALIGNED_LENGTH 16

#define VALUE_MAX 1e7
#define VALUE_MIN -1e7
#define VALUE_EPSILON 1e-7

typedef float Value;
// TODO only supports sax_cardinality <= 8
typedef unsigned char SAXWord;
// TODO Why SAXMask is int? Why not char?
typedef unsigned int SAXMask;
typedef size_t ID;


#define VALUE_L(left, right) ((right) - (left) > VALUE_EPSILON)
#define VALUE_G(left, right) ((left) - (right) > VALUE_EPSILON)
#define VALUE_LEQ(left, right) ((left) - (right) <= VALUE_EPSILON)
#define VALUE_GEQ(left, right) ((right) - (left) <= VALUE_EPSILON)
#define VALUE_EQ(left, right) (VALUE_LEQ(left, right) && VALUE_GEQ(left, right))
#define VALUE_NEQ(left, right) (VALUE_L(left, right) || VALUE_G(left, right))

#define SWAP(T, a, b) do { T tmp = a; a = b; b = tmp; } while (0)

static inline int VALUE_COMPARE(void const *left, void const *right) {
    if (VALUE_L(*(Value *) left, *(Value *) right)) {
        return -1;
    }

    if (VALUE_G(*(Value *) left, *(Value *) right)) {
        return 1;
    }

    return 0;
}

#endif //ISAX_GLOBALS_H
