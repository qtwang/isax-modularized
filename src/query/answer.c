//
// Created by Qitong Wang on 2020/7/10.
//

#include "answer.h"


void heapifyTopDown(Value *heap, unsigned int parent, unsigned int size) {
    unsigned int left = (parent << 1u) + 1, right = left + 1;

    if (right < size) {
        unsigned int next = left;
        if (VALUE_L(heap[left], heap[right])) {
            next = right;
        }

        if (VALUE_L(heap[parent], heap[next])) {
            Value tmp = heap[next];
            heap[next] = heap[parent];
            heap[parent] = tmp;

            heapifyTopDown(heap, next, size);
        }
    } else if (left < size && VALUE_L(heap[parent], heap[left])) {
        Value tmp = heap[left];
        heap[left] = heap[parent];
        heap[parent] = tmp;
    }
}


void heapifyBottomUp(Value *heap, unsigned int child) {
    if (child != 0) {
        unsigned int parent = (child - 1) >> 1u;

        if (VALUE_L(heap[parent], heap[child])) {
            Value tmp = heap[child];
            heap[child] = heap[parent];
            heap[parent] = tmp;
        }

        heapifyBottomUp(heap, parent);
    }
}


int checkNUpdateBSF(Answer *answer, Value distance) {
    if (answer->size < answer->k) {
        answer->distances[answer->size] = distance;
        heapifyBottomUp(answer->distances, answer->size);

        answer->size += 1;
    } else if (VALUE_L(distance, answer->distances[0])) {
        answer->distances[0] = distance;
        heapifyTopDown(answer->distances, 0, answer->size);
    } else {
        return 1;
    }

#ifdef PROFILING
    pthread_mutex_lock(log_lock_profiling);
    clog_info(CLOG(CLOGGER_ID), "query %d - updated BSF = %f at %d l2square / %d sum2sax / %d entered",
              query_id_profiling, distance, l2square_counter_profiling, sum2sax_counter_profiling,
              leaf_counter_profiling);
    pthread_mutex_unlock(log_lock_profiling);
#endif

    return 0;
}


Answer *initializeAnswer(Config const *config) {
    Answer *answer = malloc(sizeof(Answer));

    answer->size = 0;
    answer->k = config->k;
    answer->distances = malloc(sizeof(Value) * config->k);
    answer->distances[0] = VALUE_MAX;

    answer->lock = malloc(sizeof(pthread_rwlock_t));
    assert(pthread_rwlock_init(answer->lock, NULL) == 0);

    return answer;
}


void resetAnswer(Answer *answer) {
    answer->size = 0;
    answer->distances[0] = VALUE_MAX;
}


void resetAnswerBy(Answer *answer, Value initial_bsf_distance) {
    answer->size = 1;
    answer->distances[0] = initial_bsf_distance;
}


void freeAnswer(Answer *answer) {
    free(answer->distances);
    pthread_rwlock_destroy(answer->lock);
    free(answer->lock);
    free(answer);
}


void logAnswer(unsigned int query_id, Answer *answer) {
    if (answer->size == 0) {
        clog_info(CLOG(CLOGGER_ID), "query %d NO closer neighbors than %f", query_id, answer->distances[0]);
    } else {
        for (unsigned int i = 0; i < answer->size; ++i) {
            clog_info(CLOG(CLOGGER_ID), "query %d - %d / %luNN = %f", query_id, i, answer->k, answer->distances[i]);
        }
    }
}


Value getBSF(Answer *answer) {
    return answer->distances[0];
}
