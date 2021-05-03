//
// Created by Qitong Wang on 2020/6/28.
//

#ifndef ISAX_INDEX_H
#define ISAX_INDEX_H

#include <time.h>

#include "globals.h"
#include "node.h"
#include "config.h"
#include "clog.h"
#include "breakpoints.h"
#include "paa.h"
#include "sax.h"


typedef struct Index {
    Node **roots;

    unsigned int roots_size;
    unsigned int num_leaves;

    Value const *values;

    ID database_size;
    unsigned int series_length;

    SAXWord const *saxs;
    Value const *breakpoints;
    Value const *summarizations;
    ssize_t *pos2id;

    unsigned int sax_length;
    unsigned int sax_cardinality;
} Index;

Node *route(Node const *parent, SAXWord const *sax, unsigned int num_segments);

unsigned int rootSAX2ID(SAXWord const *saxs, unsigned int num_segments, unsigned int cardinality);

Index *initializeIndex(Config const *config);

void freeIndex(Index *index);

void logIndex(Index *index);

#endif //ISAX_INDEX_H
