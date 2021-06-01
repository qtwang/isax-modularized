//
// Created by Qitong Wang on 2021/5/31.
//

#ifndef ISAX_MULTINDEX_H
#define ISAX_MULTINDEX_H

#include "index.h"


typedef struct MultIndex {
    unsigned int num_indices;
    Index **indices;
    Value const *centers;

    Value const *values;
    SAXWord const *saxs;
//    Value const *breakpoints;
    Value const *summarizations;

    ID database_size;
    ID *cluster_sizes;
    ID *cluster_offsets;
    ssize_t *inverse_permutation;

    unsigned int series_length;
    unsigned int sax_length;
    unsigned int sax_cardinality;
} MultIndex;

MultIndex *initializeMultIndex(Config const *config);

void freeMultIndex(MultIndex *multindex);

void logMultIndex(MultIndex *multindex);

#endif //ISAX_MULTINDEX_H
