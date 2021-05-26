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
            SWAP(Value, heap[parent], heap[next]);

            heapifyTopDown(heap, next, size);
        }
    } else if (left < size && VALUE_L(heap[parent], heap[left])) {
        SWAP(Value, heap[parent], heap[left]);
    }
}


void heapifyBottomUp(Value *heap, unsigned int child) {
    if (child != 0) {
        unsigned int parent = (child - 1) >> 1u;

        if (VALUE_L(heap[parent], heap[child])) {
            SWAP(Value, heap[parent], heap[child]);
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


void heapifyTopDownWithID(Value *heap, ssize_t *ids, unsigned int parent, unsigned int size) {
    unsigned int left = (parent << 1u) + 1, right = left + 1;

    if (right < size) {
        unsigned int next = left;
        if (VALUE_L(heap[left], heap[right])) {
            next = right;
        }

        if (VALUE_L(heap[parent], heap[next])) {
            SWAP(Value, heap[parent], heap[next]);
            SWAP(ssize_t, ids[parent], ids[next]);

            heapifyTopDownWithID(heap, ids, next, size);
        }
    } else if (left < size && VALUE_L(heap[parent], heap[left])) {
        SWAP(Value, heap[parent], heap[left]);
        SWAP(ssize_t, ids[parent], ids[left]);
    }
}


void heapifyBottomUpWithID(Value *heap, ssize_t *ids, unsigned int child) {
    if (child != 0) {
        unsigned int parent = (child - 1) >> 1u;

        if (VALUE_L(heap[parent], heap[child])) {
            SWAP(Value, heap[parent], heap[child]);
            SWAP(ssize_t, ids[parent], ids[child]);

            heapifyBottomUpWithID(heap, ids, parent);
        }
    }
}


int checkBSF(Answer *answer, Value distance) {
    return answer->size < answer->k || VALUE_L(distance, answer->distances[0]);
}


void updateBSFWithID(Answer *answer, Value distance, ssize_t id) {
    if (answer->size < answer->k) {
        answer->distances[answer->size] = distance;
        answer->ids[answer->size] = id;

        heapifyBottomUpWithID(answer->distances, answer->ids, answer->size);

        answer->size += 1;
    } else {
        answer->distances[0] = distance;
        answer->ids[0] = id;

        heapifyTopDownWithID(answer->distances, answer->ids, 0, answer->size);
    }

#ifdef PROFILING
    pthread_mutex_lock(log_lock_profiling);
    clog_info(CLOG(CLOGGER_ID), "query %d - updated BSF = %f by %d after %d l2square / %d sum2sax / %d entered",
              query_id_profiling, distance, id, l2square_counter_profiling, sum2sax_counter_profiling,
              leaf_counter_profiling);
    pthread_mutex_unlock(log_lock_profiling);
#endif
}


Answer *initializeAnswer(Config const *config) {
    Answer *answer = malloc(sizeof(Answer));

    answer->size = 0;
    answer->k = config->k;
    answer->distances = malloc(sizeof(Value) * config->k);
    answer->distances[0] = VALUE_MAX;

    if (config->with_id) {
        answer->ids = malloc(sizeof(ssize_t) * config->k);
    } else {
        answer->ids = NULL;
    }

    answer->lock = malloc(sizeof(pthread_rwlock_t));
    assert(pthread_rwlock_init(answer->lock, NULL) == 0);

    return answer;
}


void resetAnswer(Answer *answer) {
    answer->size = 1;

    answer->distances[0] = VALUE_MAX;
}


void resetAnswerBy(Answer *answer, Value initial_bsf_distance) {
    answer->size = 1;

    answer->distances[0] = initial_bsf_distance;
    answer->ids[0] = -1;
}


void freeAnswer(Answer *answer) {
    free(answer->distances);
    free(answer->ids);

    pthread_rwlock_destroy(answer->lock);
    free(answer->lock);

    free(answer);
}


void logAnswer(unsigned int query_id, Answer *answer) {
//    if (answer->size == 0) {
//        clog_info(CLOG(CLOGGER_ID), "query %d NO closer neighbors than initial %f", query_id, answer->distances[0]);
//    }

    if (answer->ids) {
        for (unsigned int i = 0; i < answer->size; ++i) {
            clog_info(CLOG(CLOGGER_ID), "query %d - %d / %luNN = %f by %d",
                      query_id, i, answer->k, answer->distances[i], answer->ids[i]);
        }
    } else {
        for (unsigned int i = 0; i < answer->size; ++i) {
            clog_info(CLOG(CLOGGER_ID), "query %d - %d / %luNN = %f",
                      query_id, i, answer->k, answer->distances[i]);
        }
    }
}


Value getBSF(Answer *answer) {
    return answer->distances[0];
}
