//
// Created by Qitong Wang on 2020/7/10.
//

#ifndef ISAX_ANSWER_H
#define ISAX_ANSWER_H

#include <stdlib.h>
#include <pthread.h>

#include "globals.h"
#include "config.h"
#include "clog.h"


typedef struct Answer {
    pthread_rwlock_t *lock;

    Value *distances; // max-heap
    ssize_t *ids; // auxiliary max-heap

    unsigned int size;
    unsigned int k;
} Answer;


Answer *initializeAnswer(Config const *config);

void resetAnswer(Answer *answer);
void resetAnswerBy(Answer *answer, Value initial_bsf_distance);

void freeAnswer(Answer *answer);

Value getBSF(Answer * answer);

int checkNUpdateBSF(Answer * answer, Value distance);

int checkBSF(Answer *answer, Value distance);
void updateBSFWithID(Answer *answer, Value distance, ssize_t id);

void logAnswer(unsigned int query_id, Answer *answer);


#endif //ISAX_ANSWER_H
