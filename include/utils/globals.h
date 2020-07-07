//
// Created by Qitong Wang on 2020/6/28.
//

#ifndef ISAX_GLOBALS_H
#define ISAX_GLOBALS_H

#include <float.h>
#include <stdlib.h>
#include <pthread.h>


#define FINE_TIMING
#define TIMING

#ifdef FINE_TIMING
#ifndef TIMING
#define TIMING
#endif
#endif


#define PROFILING

#ifdef PROFILING
unsigned int visited_leaves_counter_profiling;
unsigned int visited_series_counter_profiling;
unsigned int calculated_series_counter_profiling;
pthread_mutex_t *log_lock_profiling;
unsigned int query_id_profiling;
#endif


#define CLOGGER_ID 0


typedef float Value;
// TODO only supports sax_cardinality <= 8
typedef unsigned char SAXWord;
typedef unsigned int SAXMask;


#define VALUE_L(left, right) (right - left > FLT_EPSILON)
#define VALUE_G(left, right) (left - right > FLT_EPSILON)
#define VALUE_LEQ(left, right) (!VALUE_G(left, right))
#define VALUE_GEQ(left, right) (!(VALUE_L(left, right)))
#define VALUE_EQ(left, right) ((left - right <= FLT_EPSILON) && ((right - left <= FLT_EPSILON)))


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
