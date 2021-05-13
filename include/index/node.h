//
// Created by Qitong Wang on 2020/6/28.
//

#ifndef ISAX_NODE_H
#define ISAX_NODE_H

#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>

#include "globals.h"
#include "clog.h"
#include "distance.h"

typedef struct Node {
    pthread_mutex_t *lock;
    SAXWord *sax;
    SAXMask *masks;
    SAXMask *squeezed_masks;

    ID *ids;
    ID start_id;
    unsigned int size;
    unsigned int capacity;

    struct Node *left;
    struct Node *right;

    Value compactness;

    Value* upper_envelops;
    Value* lower_envelops;
} Node;


Node *initializeNode(SAXWord *saxWord, SAXMask *saxMask);

void inspectNode(Node *node, unsigned int *num_series, unsigned int *num_leaves, unsigned int *num_roots);

Value getCompactness(Node *leaf_node, Value const *values, unsigned int series_length);

void insertNode(Node *leaf, ID id, unsigned int initial_leaf_size, unsigned int leaf_size);

void freeNode(Node *node, bool free_mask, bool free_sax);

#endif //ISAX_NODE_H
