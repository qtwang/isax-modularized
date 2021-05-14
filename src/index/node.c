//
// Created by Qitong Wang on 2020/6/28.
//

#include <math.h>
#include "node.h"


Node *initializeNode(SAXWord *sax, SAXMask *masks) {
    Node *node = malloc(sizeof(Node));

    node->sax = sax;
    node->masks = masks;
    node->squeezed_masks = NULL;

    node->upper_envelops = NULL;
    node->lower_envelops = NULL;

    node->ids = NULL;
    node->start_id = 0;

    node->size = 0;
    node->capacity = 0;

    node->left = NULL;
    node->right = NULL;

    node->lock = malloc(sizeof(pthread_mutex_t));
    assert(pthread_mutex_init(node->lock, NULL) == 0);

    node->compactness = -1;

    return node;
}


void insertNode(Node *leaf, ID id, unsigned int initial_leaf_size, unsigned int leaf_size) {
    if (leaf->capacity == 0) {
        leaf->ids = malloc(sizeof(ID) * (leaf->capacity = initial_leaf_size));
    } else if (leaf->size == leaf->capacity) {
        if ((leaf->capacity *= 2) > leaf_size) {
            leaf->capacity = leaf_size;
        }
        leaf->ids = realloc(leaf->ids, sizeof(ID) * leaf->capacity);
    }

    leaf->ids[leaf->size++] = id;
}


void inspectNode(Node *node, unsigned int *num_series, unsigned int *num_leaves, unsigned int *num_roots) {
    if (node != NULL) {
        if (node->size != 0) {
            if (num_roots != NULL) {
                *num_roots += 1;
            }

#ifdef PROFILING
            clog_info(CLOG(CLOGGER_ID), "index - node %lu = %lu", *num_leaves, node->size);
#endif

            *num_leaves += 1;
            *num_series += node->size;
        } else if (node->left != NULL) {
            if (num_roots != NULL) {
                *num_roots += 1;
            }

            inspectNode(node->left, num_series, num_leaves, NULL);
            inspectNode(node->right, num_series, num_leaves, NULL);
        }
    }
}


Value getCompactness(Node *leaf_node, Value const *values, unsigned int series_length) {
    if (leaf_node->size == 0) {
        return -1;
    } else if (leaf_node->size == 1) {
        return 0;
    } else if (leaf_node->compactness > 0) {
        return leaf_node->compactness;
    }

    double sum = 0;
    Value *local_m256_fetched_cache = aligned_alloc(256, sizeof(Value) * 8);

    Value const *outer_current_series = values + series_length * leaf_node->start_id;
    Value const *inner_current_series;
    Value const *stop = values + series_length * (leaf_node->start_id + leaf_node->size);

    while (outer_current_series < stop) {
        inner_current_series = outer_current_series + series_length;

        while (inner_current_series < stop) {
            sum += sqrt(l2SquareSIMD(
                    series_length, outer_current_series, inner_current_series, local_m256_fetched_cache));

            inner_current_series += series_length;
        }

        outer_current_series += series_length;
    }

    leaf_node->compactness = (Value) (sum / (double) (leaf_node->size * (leaf_node->size - 1) / 2.));

    free(local_m256_fetched_cache);
    return leaf_node->compactness;
}


void freeNode(Node *node, bool free_mask, bool free_sax) {
    if (node != NULL) {
        if (node->left != NULL) {
            freeNode(node->left, false, false);
            freeNode(node->right, true, true);
        }

        if (free_mask) {
            free(node->masks);
        }

        if (free_sax) {
            free(node->sax);
        }

        free(node->squeezed_masks);
        free(node->upper_envelops);
        free(node->lower_envelops);

        free(node->ids);

        pthread_mutex_destroy(node->lock);
        free(node->lock);

        free(node);
    }
}
