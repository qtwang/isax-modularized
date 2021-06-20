//
// Created by Qitong Wang on 2020/7/3.
//

#include "query_engine.h"


typedef struct QueryCache {
    Index const *index;

    unsigned int series_length;
    unsigned int sax_length;

    Node const *const *leaves;
    unsigned int *leaf_indices;
    Value *leaf_distances;
    unsigned int num_leaves;

    Answer *answer;
    Value const *query_values;
    Value const *query_summarization;
    Node *resident_node;

    ID *shared_leaf_id;
    unsigned int leaf_block_size;
    unsigned int query_block_size;

    Value *m256_fetched_cache;
    Value scale_factor;

    bool sort_leaves;
    bool lower_bounding;

    unsigned int series_limitations;
} QueryCache;


void queryNodeThreadCore(Answer *answer, Node const *node, Value const *values, unsigned int series_length,
                         SAXWord const *saxs, unsigned int sax_length, Value const *breakpoints, Value scale_factor,
                         Value const *query_values, Value const *query_summarization, Value *m256_fetched_cache,
                         pthread_rwlock_t *lock, ssize_t *pos2id) {
    Value const *current_series;
    SAXWord const *current_sax;
    Value local_l2SquareSAX, local_l2Square, local_bsf = getBSF(answer);
    unsigned long pos;

    for (current_series = values + series_length * node->start_id,
                 current_sax = saxs + SAX_SIMD_ALIGNED_LENGTH * node->start_id;
         current_series < values + series_length * (node->start_id + node->size);
         current_series += series_length, current_sax += SAX_SIMD_ALIGNED_LENGTH) {
#ifdef PROFILING
        __sync_fetch_and_add(&sum2sax_counter_profiling, 1);
#endif
        local_l2SquareSAX = l2SquareValue2SAX8SIMD(sax_length, query_summarization, current_sax,
                                                   breakpoints, scale_factor, m256_fetched_cache);

        if (VALUE_G(local_bsf, local_l2SquareSAX)) {
#ifdef PROFILING
            __sync_fetch_and_add(&l2square_counter_profiling, 1);
#endif
            local_l2Square = l2SquareEarlySIMD(series_length, query_values, current_series, local_bsf,
                                               m256_fetched_cache);

            if (VALUE_G(local_bsf, local_l2Square)) {
                pthread_rwlock_wrlock(lock);

                if (pos2id) {
                    if (checkBSF(answer, local_l2Square)) {
                        pos = node->start_id +
                              (current_series - values - series_length * node->start_id) / series_length;
                        updateBSFWithID(answer, local_l2Square, pos2id[pos]);
                    }
                } else {
                    checkNUpdateBSF(answer, local_l2Square);
                }

                pthread_rwlock_unlock(lock);
                local_bsf = getBSF(answer);
            }
        }
    }
}


void queryNodeNotBoundingThreadCore(Answer *answer, Node const *node, Value const *values, unsigned int series_length,
                                    Value const *query_values, Value *m256_fetched_cache, pthread_rwlock_t *lock,
                                    ssize_t *pos2id) {
    Value const *current_series;
    Value local_l2Square, local_bsf = getBSF(answer);
    unsigned long pos;

    for (current_series = values + series_length * node->start_id;
         current_series < values + series_length * (node->start_id + node->size);
         current_series += series_length) {
#ifdef PROFILING
        __sync_fetch_and_add(&l2square_counter_profiling, 1);
#endif
        local_l2Square = l2SquareEarlySIMD(series_length, query_values, current_series, local_bsf,
                                           m256_fetched_cache);

        if (VALUE_G(local_bsf, local_l2Square)) {
            pthread_rwlock_wrlock(lock);

            if (pos2id) {
                if (checkBSF(answer, local_l2Square)) {
                    pos = node->start_id + (current_series - values - series_length * node->start_id) / series_length;
                    updateBSFWithID(answer, local_l2Square, pos2id[pos]);
                }
            } else {
                checkNUpdateBSF(answer, local_l2Square);
            }

            pthread_rwlock_unlock(lock);
            local_bsf = getBSF(answer);
        }
    }
}


void *queryThread(void *cache) {
    QueryCache *queryCache = (QueryCache *) cache;

    Value const *values = NULL;
    SAXWord const *saxs = NULL;
    Value const *breakpoints = NULL;
    ssize_t *pos2id = NULL;

    if (queryCache->index != NULL) {
        breakpoints = queryCache->index->breakpoints;
        values = queryCache->index->values;
        saxs = queryCache->index->saxs;
        pos2id = queryCache->index->pos2id;
    }

    unsigned int series_length = queryCache->series_length;
    unsigned int sax_length = queryCache->sax_length;

    Node const *const *leaves = queryCache->leaves;
    Value *leaf_distances = queryCache->leaf_distances;
    unsigned int *leaf_indices = queryCache->leaf_indices;

    Value const *query_summarization = queryCache->query_summarization;
    Value const *query_values = queryCache->query_values;

    Answer *answer = queryCache->answer;
    pthread_rwlock_t *lock = answer->lock;

    Value *m256_fetched_cache = queryCache->m256_fetched_cache;
    Value scale_factor = queryCache->scale_factor;

    unsigned int block_size = queryCache->query_block_size;
    unsigned int num_leaves = queryCache->num_leaves;
    ID *shared_index_id = queryCache->shared_leaf_id;

    bool sort_leaves = queryCache->sort_leaves;
    bool lower_bounding = queryCache->lower_bounding;

    unsigned int series_limitations = queryCache->series_limitations;

    ID leaf_id;
    unsigned int index_id, stop_index_id;
    Value local_bsf;

    while ((index_id = __sync_fetch_and_add(shared_index_id, block_size)) < num_leaves) {
        stop_index_id = index_id + block_size;
        if (stop_index_id > num_leaves) {
            stop_index_id = num_leaves;
        }

        pthread_rwlock_rdlock(lock);
        local_bsf = getBSF(answer);
        pthread_rwlock_unlock(lock);

        while (index_id < stop_index_id) {
            leaf_id = leaf_indices[index_id];

            // TODO correctness-risks traded off for efficiency
            // only using local_bsf suffers from that approximate nearest neighbours < k and never updated
            // could use VALUE_G instead of VALUE_GEQ if not for efficiency
            if (VALUE_G(local_bsf, leaf_distances[leaf_id]) || (!lower_bounding && leaf_distances[leaf_id] < 1e7)) {
#ifdef PROFILING
                __sync_fetch_and_add(&leaf_counter_profiling, 1);

                if (series_limitations != 0 && (sum2sax_counter_profiling > series_limitations ||
                                                l2square_counter_profiling > series_limitations)) {
                    return NULL;
                }
#endif
                if (queryCache->index == NULL) {
                    Index *current_index = (Index *) leaves[leaf_id]->index;

                    breakpoints = current_index->breakpoints;
                    values = current_index->values;
                    saxs = current_index->saxs;
                    pos2id = current_index->pos2id;
                }

                if (lower_bounding) {
                    queryNodeThreadCore(answer, leaves[leaf_id], values, series_length, saxs, sax_length, breakpoints,
                                        scale_factor, query_values, query_summarization, m256_fetched_cache, lock,
                                        pos2id);
                } else {
                    queryNodeNotBoundingThreadCore(answer, leaves[leaf_id], values, series_length, query_values,
                                                   m256_fetched_cache, lock, pos2id);
                }
            } else if (sort_leaves && lower_bounding) {
                return NULL;
            }

            index_id += 1;
        }
    }

    return NULL;
}


void *leafThread(void *cache) {
    QueryCache *queryCache = (QueryCache *) cache;


    Value const *breakpoints = NULL;
    if (queryCache->index != NULL) {
        breakpoints = queryCache->index->breakpoints;
    }
    unsigned int sax_length = queryCache->sax_length;

    Node *resident_node = queryCache->resident_node;
    Value *leaf_distances = queryCache->leaf_distances;

    Value const *query_summarization = queryCache->query_summarization;
    Value scale_factor = queryCache->scale_factor;
    Value *m256_fetched_cache = queryCache->m256_fetched_cache;

    unsigned int block_size = queryCache->leaf_block_size;
    unsigned int num_leaves = queryCache->num_leaves;
    ID *shared_leaf_id = queryCache->shared_leaf_id;

    ID leaf_id, stop_leaf_id;
    Node const *leaf;

    while ((leaf_id = __sync_fetch_and_add(shared_leaf_id, block_size)) < num_leaves) {
        stop_leaf_id = leaf_id + block_size;
        if (stop_leaf_id > num_leaves) {
            stop_leaf_id = num_leaves;
        }

        for (unsigned int i = leaf_id; i < stop_leaf_id; ++i) {
            leaf = queryCache->leaves[i];

            if (queryCache->index == NULL) {
                breakpoints = ((Index *) leaf->index)->breakpoints;
            }

            if (resident_node != NULL && leaf == resident_node) {
                leaf_distances[i] = VALUE_MAX;
            } else {
                if (leaf->upper_envelops != NULL) {
                    leaf_distances[i] = l2SquareValue2EnvelopSIMD(sax_length, query_summarization,
                                                                  leaf->upper_envelops, leaf->lower_envelops,
                                                                  scale_factor, m256_fetched_cache);
                } else if (leaf->squeezed_masks != NULL) {
                    leaf_distances[i] = l2SquareValue2SAXByMaskSIMD(sax_length, query_summarization, leaf->sax,
                                                                    leaf->squeezed_masks, breakpoints, scale_factor,
                                                                    m256_fetched_cache);
                } else {
                    leaf_distances[i] = l2SquareValue2SAXByMaskSIMD(sax_length, query_summarization, leaf->sax,
                                                                    leaf->masks, breakpoints, scale_factor,
                                                                    m256_fetched_cache);
                }
            }
        }
    }

    return NULL;
}


void enqueueLeaf(Node *node, Node **leaves, unsigned int *num_leaves, Index *index) {
    if (node != NULL) {
        if (node->size != 0) {
            if (index != NULL) {
                node->index = index;
            }

            leaves[*num_leaves] = node;
            *num_leaves += 1;
        } else if (node->left != NULL) {
            enqueueLeaf(node->left, leaves, num_leaves, index);
            enqueueLeaf(node->right, leaves, num_leaves, index);
        }
    }
}


void queryNode(Answer *answer, Node const *node, Value const *values, unsigned int series_length,
               SAXWord const *saxs, unsigned int sax_length, Value const *breakpoints, Value scale_factor,
               Value const *query_values, Value const *query_summarization, Value *m256_fetched_cache,
               ssize_t *pos2id) {
    Value const *outer_current_series = values + series_length * node->start_id;
    SAXWord const *outer_current_sax = saxs + SAX_SIMD_ALIGNED_LENGTH * node->start_id;
    Value local_l2SquareSAX8, local_l2Square, local_bsf = getBSF(answer);
    unsigned long pos;

    while (outer_current_series < values + series_length * (node->start_id + node->size)) {
#ifdef PROFILING
        sum2sax_counter_profiling += 1;
#endif
        local_l2SquareSAX8 = l2SquareValue2SAX8SIMD(sax_length, query_summarization, outer_current_sax,
                                                    breakpoints, scale_factor, m256_fetched_cache);

        if (VALUE_G(local_bsf, local_l2SquareSAX8)) {
#ifdef PROFILING
            l2square_counter_profiling += 1;
#endif
            local_l2Square = l2SquareEarlySIMD(series_length, query_values, outer_current_series, local_bsf,
                                               m256_fetched_cache);

            if (VALUE_G(local_bsf, local_l2Square)) {
                if (pos2id) {
                    if (checkBSF(answer, local_l2Square)) {
                        pos = node->start_id +
                              (outer_current_series - values - series_length * node->start_id) / series_length;
                        updateBSFWithID(answer, local_l2Square, pos2id[pos]);
                    }
                } else {
                    checkNUpdateBSF(answer, local_l2Square);
                }

                local_bsf = getBSF(answer);
            }
        }

        outer_current_series += series_length;
        outer_current_sax += SAX_SIMD_ALIGNED_LENGTH;
    }
}


void queryNodeNotBounding(Answer *answer, Node const *node, Value const *values, unsigned int series_length,
                          Value const *query_values, Value *m256_fetched_cache, ssize_t *pos2id) {
    Value const *outer_current_series = values + series_length * node->start_id;
    Value local_l2Square, local_bsf = getBSF(answer);
    unsigned long pos;

    while (outer_current_series < values + series_length * (node->start_id + node->size)) {
#ifdef PROFILING
        l2square_counter_profiling += 1;
#endif
        local_l2Square = l2SquareEarlySIMD(series_length, query_values, outer_current_series, local_bsf,
                                           m256_fetched_cache);

        if (VALUE_G(local_bsf, local_l2Square)) {
            if (pos2id) {
                if (checkBSF(answer, local_l2Square)) {
                    pos = node->start_id +
                          (outer_current_series - values - series_length * node->start_id) / series_length;
                    updateBSFWithID(answer, local_l2Square, pos2id[pos]);
                }
            } else {
                checkNUpdateBSF(answer, local_l2Square);
            }

            local_bsf = getBSF(answer);
        }

        outer_current_series += series_length;
    }
}


void conductQueries(Config const *config, QuerySet const *querySet, Index const *index) {
    Answer *answer = initializeAnswer(config);

    Value const *values = index->values;
    SAXWord const *saxs = index->saxs;
    ssize_t *pos2id = index->pos2id;
    Value const *breakpoints = index->breakpoints;
    unsigned int series_length = config->series_length;
    unsigned int sax_length = config->sax_length;
    unsigned int sax_cardinality = config->sax_cardinality;
    Value scale_factor = config->scale_factor;

    ID shared_leaf_id;

    unsigned int max_threads = config->max_threads;
    QueryCache queryCache[max_threads];

#ifdef FINE_TIMING
    struct timespec start_timestamp, stop_timestamp;
    TimeDiff time_diff;
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif

    Node **leaves = malloc(sizeof(Node *) * index->num_leaves);
    unsigned int num_leaves = 0;
    for (unsigned int j = 0; j < index->roots_size; ++j) {
        enqueueLeaf(index->roots[j], leaves, &num_leaves, NULL);
    }
    assert(num_leaves == index->num_leaves);

#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "query - fetch leaves = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
#endif

    unsigned int *leaf_indices = malloc(sizeof(unsigned int) * num_leaves);
    for (unsigned int j = 0; j < num_leaves; ++j) {
        leaf_indices[j] = j;
    }

    Value *leaf_distances = malloc(sizeof(Value) * num_leaves);
    unsigned int leaf_block_size = 1 + num_leaves / (max_threads << 1u);
    unsigned int query_block_size = 2 + num_leaves / (max_threads << 3u);

    for (unsigned int i = 0; i < max_threads; ++i) {
        queryCache[i].answer = answer;
        queryCache[i].index = index;

        queryCache[i].num_leaves = num_leaves;
        queryCache[i].leaves = (Node const *const *) leaves;
        queryCache[i].leaf_indices = leaf_indices;
        queryCache[i].leaf_distances = leaf_distances;

        queryCache[i].scale_factor = scale_factor;
        queryCache[i].m256_fetched_cache = aligned_alloc(256, sizeof(Value) * 8);

        queryCache[i].shared_leaf_id = &shared_leaf_id;
        queryCache[i].sort_leaves = config->sort_leaves;

        queryCache[i].series_limitations = config->series_limitations;
        queryCache[i].lower_bounding = config->lower_bounding;

        queryCache[i].series_length = config->series_length;
        queryCache[i].sax_length = config->sax_length;
    }

    Value *local_m256_fetched_cache = queryCache[0].m256_fetched_cache;

    Value const *query_values, *query_summarization;
    SAXWord const *query_sax;
    Value local_bsf;
    Node *node;

    for (unsigned int i = 0; i < querySet->query_size; ++i) {
        if (querySet->initial_bsf_distances == NULL) {
            resetAnswer(answer);
        } else {
            resetAnswerBy(answer, querySet->initial_bsf_distances[i]);
            clog_info(CLOG(CLOGGER_ID), "query %d - initial 1bsf = %f", i, querySet->initial_bsf_distances[i]);
        }

        query_values = querySet->values + series_length * i;
        query_summarization = querySet->summarizations + sax_length * i;
        query_sax = querySet->saxs + SAX_SIMD_ALIGNED_LENGTH * i;

#ifdef PROFILING
        query_id_profiling = i;
        leaf_counter_profiling = 0;
        sum2sax_counter_profiling = 0;
        l2square_counter_profiling = 0;
#endif
#ifdef FINE_TIMING
        clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
        // TODO to support cardinality > 8
        node = index->roots[rootSAX2ID(query_sax, sax_length, 8)];
//        node = index->roots[rootSAX2ID(query_sax, sax_length, sax_cardinality)];
        local_bsf = getBSF(answer);

        if (node != NULL) {
            while (node->left != NULL) {
                node = route(node, query_sax, sax_length);
            }
#ifdef PROFILING
            leaf_counter_profiling += 1;

#ifdef FINE_PROFILING
            for (unsigned int j = 0; j < index->sax_length; ++j) {
                clog_info(CLOG(CLOGGER_ID), "query %d - resident leaf segment %d = %s - %d", i, j,
                          char2bin(node->sax[j]), BITS_BY_MASK[node->masks[j]]);
            }
#endif

            if (config->leaf_compactness) {
                clog_info(CLOG(CLOGGER_ID), "query %d - resident leaf size %d compactness %f",
                          i, node->size, getCompactness(node, values, series_length));
            }

            if (config->log_leaf_only) {
                continue;
            }
#endif
            if (config->lower_bounding) {
                queryNode(answer, node, values, series_length, saxs, sax_length, breakpoints, scale_factor,
                          query_values, query_summarization, local_m256_fetched_cache, pos2id);
            } else {
                queryNodeNotBounding(answer, node, values, series_length, query_values, local_m256_fetched_cache,
                                     pos2id);
            }
        } else {
            clog_info(CLOG(CLOGGER_ID), "query %d - no resident node", i);
        }
#ifdef FINE_TIMING
        clock_code = clock_gettime(CLK_ID, &stop_timestamp);
        getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
        //TODO why this takes longer than nearest-leaf approximate search and exact search?
        clog_info(CLOG(CLOGGER_ID), "query %d - resident-leaf approximate search = %ld.%lds", i, time_diff.tv_sec,
                  time_diff.tv_nsec);
#endif
        if ((config->exact_search && !(VALUE_EQ(local_bsf, 0) && answer->size == answer->k)) || node == NULL) {
#ifdef FINE_TIMING
            clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
            pthread_t leaves_threads[max_threads];
            shared_leaf_id = 0;

            for (unsigned int j = 0; j < max_threads; ++j) {
                queryCache[j].query_values = query_values;
                queryCache[j].query_summarization = query_summarization;
                queryCache[j].resident_node = node;
                queryCache[j].leaf_block_size = leaf_block_size;

                pthread_create(&leaves_threads[j], NULL, leafThread, (void *) &queryCache[j]);
            }

            for (unsigned int j = 0; j < max_threads; ++j) {
                pthread_join(leaves_threads[j], NULL);
            }

            if (config->sort_leaves) {
                qSortFirstHalfIndicesBy(leaf_indices, leaf_distances, 0, (int) (num_leaves - 1), local_bsf);
            }
#ifdef FINE_TIMING
            clock_code = clock_gettime(CLK_ID, &stop_timestamp);
            getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
            clog_info(CLOG(CLOGGER_ID), "query %d - cal&sort leaf distances = %ld.%lds", i,
                      time_diff.tv_sec, time_diff.tv_nsec);
#endif

            if (node == NULL) {
                node = leaves[leaf_indices[0]];
                leaf_distances[leaf_indices[0]] = VALUE_MAX;
#ifdef PROFILING
                leaf_counter_profiling += 1;

                if (config->leaf_compactness) {
                    for (unsigned int j = 0; j < index->sax_length; ++j) {
                        clog_info(CLOG(CLOGGER_ID), "query %d - nearest leaf segment %d = %d - %d", i, j, node->sax[j],
                                  node->masks[j]);
                    }

                    clog_info(CLOG(CLOGGER_ID), "query %d - nearest leaf size %d compactness %f", i, node->size,
                              getCompactness(node, values, series_length));
                }

                if (config->log_leaf_only) {
                    continue;
                }
#endif
#ifdef FINE_TIMING
                clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
                if (config->lower_bounding) {
                    queryNode(answer, node, values, series_length, saxs, sax_length, breakpoints, scale_factor,
                              query_values, query_summarization, local_m256_fetched_cache, pos2id);
                } else {
                    queryNodeNotBounding(answer, node, values, series_length, query_values, local_m256_fetched_cache,
                                         pos2id);
                }

                if (config->sort_leaves) {
                    unsigned int tmp_index = leaf_indices[0], bsf_position = num_leaves - 1;

                    if (VALUE_L(local_bsf, leaf_distances[leaf_indices[num_leaves - 1]])) {
                        bsf_position = bSearchByIndicesFloor(local_bsf, leaf_indices, leaf_distances,
                                                             0, num_leaves - 1);
                    }

                    memmove(leaf_indices, leaf_indices + 1, sizeof(unsigned int) * bsf_position);
                    leaf_indices[bsf_position] = tmp_index;
                }
#ifdef FINE_TIMING
                clock_code = clock_gettime(CLK_ID, &stop_timestamp);
                getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
                clog_info(CLOG(CLOGGER_ID), "query %d - nearest-leaf approximate search = %ld.%lds", i,
                          time_diff.tv_sec, time_diff.tv_nsec);
#endif
            }

            if (config->exact_search && !(VALUE_EQ(local_bsf, 0) && answer->size == answer->k)) {
#ifdef PROFILING
                clog_info(CLOG(CLOGGER_ID), "query %d - %d l2square / %d sum2sax in 1st leaf",
                          i + querySet->query_size, l2square_counter_profiling, sum2sax_counter_profiling);
#endif
#ifdef FINE_TIMING
                clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
                pthread_t query_threads[max_threads];
                shared_leaf_id = 0;

                for (unsigned int j = 0; j < max_threads; ++j) {
                    queryCache[j].query_block_size = query_block_size;

                    pthread_create(&query_threads[j], NULL, queryThread, (void *) &queryCache[j]);
                }

                for (unsigned int j = 0; j < max_threads; ++j) {
                    pthread_join(query_threads[j], NULL);
                }
#ifdef FINE_TIMING
                clock_code = clock_gettime(CLK_ID, &stop_timestamp);
                getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
                clog_info(CLOG(CLOGGER_ID), "query %d - exact search = %ld.%lds", i, time_diff.tv_sec,
                          time_diff.tv_nsec);
#endif
            }
        }
#ifdef PROFILING
        clog_info(CLOG(CLOGGER_ID), "query %d - %d l2square / %d sum2sax / %d entered", i,
                  l2square_counter_profiling, sum2sax_counter_profiling, leaf_counter_profiling);
#endif
        logAnswer(i, answer);
    }

    for (unsigned int i = 0; i < max_threads; ++i) {
        free(queryCache[i].m256_fetched_cache);
    }

    freeAnswer(answer);

    free(leaves);
    free(leaf_distances);
    free(leaf_indices);
}


void conductQueriesMI(Config const *config, QuerySet const *querySet, MultIndex const *multindex) {
#ifdef FINE_TIMING
    struct timespec start_timestamp, stop_timestamp;
    TimeDiff time_diff;
#endif
    Answer *answer = initializeAnswer(config);

    unsigned int series_length = config->series_length;
    unsigned int sax_length = config->sax_length;
    Value scale_factor = config->scale_factor;

    ID shared_leaf_id = 0;
    unsigned int max_threads = config->max_threads;
    QueryCache queryCache[max_threads * config->num_indices];

    for (unsigned int i = 0; i < config->num_indices; ++i) {
        Node **leaves = malloc(sizeof(Node *) * multindex->indices[i]->num_leaves);
        unsigned int num_leaves = 0;
        for (unsigned int j = 0; j < multindex->indices[i]->roots_size; ++j) {
            enqueueLeaf(multindex->indices[i]->roots[j], leaves, &num_leaves, NULL);
        }
        assert(num_leaves == multindex->indices[i]->num_leaves);

        Value *leaf_distances = malloc(sizeof(Value) * num_leaves);
        unsigned int *leaf_indices = malloc(sizeof(unsigned int) * num_leaves);
        for (unsigned int j = 0; j < num_leaves; ++j) {
            leaf_indices[j] = j;
        }

        unsigned int leaf_block_size = 1 + num_leaves / (max_threads << 1u);
        unsigned int query_block_size = 2 + num_leaves / (max_threads << 3u);

        for (unsigned int j = 0; j < max_threads; ++j) {
            unsigned int k = i * max_threads + j;

            queryCache[k].answer = answer;
            queryCache[k].index = multindex->indices[i];

            queryCache[k].series_length = config->series_length;
            queryCache[k].sax_length = config->sax_length;

            queryCache[k].num_leaves = num_leaves;
            queryCache[k].leaves = (Node const *const *) leaves;
            queryCache[k].leaf_indices = leaf_indices;
            queryCache[k].leaf_distances = leaf_distances;

            queryCache[k].scale_factor = scale_factor;
            queryCache[k].m256_fetched_cache = aligned_alloc(sizeof(__m256i), sizeof(Value) * 8);

            queryCache[k].shared_leaf_id = &shared_leaf_id;
            queryCache[k].sort_leaves = config->sort_leaves;

            queryCache[k].series_limitations = config->series_limitations;
            queryCache[k].lower_bounding = config->lower_bounding;

            queryCache[k].leaf_block_size = leaf_block_size;
            queryCache[k].query_block_size = query_block_size;

            queryCache[k].resident_node = NULL;
        }
    }

    Value *distance2clusters = malloc(sizeof(Value) * config->num_indices);
    unsigned int *order_clusters = malloc(sizeof(unsigned int) * config->num_indices);
    for (unsigned int j = 0; j < config->num_indices; ++j) {
        order_clusters[j] = j;
    }

    for (unsigned int i = 0; i < querySet->query_size; ++i) {
#ifdef PROFILING
        query_id_profiling = i;
        leaf_counter_profiling = 0;
        sum2sax_counter_profiling = 0;
        l2square_counter_profiling = 0;
#endif
#ifdef FINE_TIMING
        clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
        if (querySet->initial_bsf_distances == NULL) {
            resetAnswer(answer);
        } else {
            resetAnswerBy(answer, querySet->initial_bsf_distances[i]);
            clog_info(CLOG(CLOGGER_ID), "query %d - initial 1bsf = %f", i, querySet->initial_bsf_distances[i]);
        }

        Value const *query_values = querySet->values + series_length * i;
        Value const *query_summarization = querySet->summarizations + sax_length * i;

        for (unsigned int j = 0; j < config->num_indices; ++j) {
            distance2clusters[j] = l2Square(series_length, query_values, multindex->centers + series_length * j);
        }
        qSortIndicesBy(order_clusters, distance2clusters, 0, (int) config->num_indices - 1);
//#ifdef DEBUG
//        for (unsigned int j = 0; j < config->num_indices; ++j) {
//            clog_debug(CLOG(CLOGGER_ID), "query %d cluster %d = %f",
//                       i, j, distance2clusters[j]);
//        }
//        for (unsigned int j = 0; j < config->num_indices; ++j) {
//            clog_debug(CLOG(CLOGGER_ID), "query %d cluster %d = %f",
//                       i, order_clusters[j], distance2clusters[order_clusters[j]]);
//        }
//#endif
        for (unsigned int j = 0; j < config->num_indices * max_threads; ++j) {
            queryCache[j].query_values = query_values;
            queryCache[j].query_summarization = query_summarization;
        }

        for (unsigned int k = 0; k < config->num_indices; ++k) {
            unsigned int index_id = order_clusters[k];
            unsigned int cache_offset = index_id * max_threads;
            pthread_t leaves_threads[max_threads];
            shared_leaf_id = 0;

            for (unsigned int j = 0; j < max_threads; ++j) {
                pthread_create(&leaves_threads[j], NULL, leafThread, (void *) &queryCache[cache_offset + j]);
            }
            for (unsigned int j = 0; j < max_threads; ++j) {
                pthread_join(leaves_threads[j], NULL);
            }

            if (config->sort_leaves) {
                qSortIndicesBy(queryCache[cache_offset].leaf_indices, queryCache[cache_offset].leaf_distances,
                               0, (int) (queryCache[cache_offset].num_leaves - 1));
            }

            pthread_t query_threads[max_threads];
            shared_leaf_id = 0;

            for (unsigned int j = 0; j < max_threads; ++j) {
                pthread_create(&query_threads[j], NULL, queryThread, (void *) &queryCache[cache_offset + j]);
            }
            for (unsigned int j = 0; j < max_threads; ++j) {
                pthread_join(query_threads[j], NULL);
            }
#ifdef PROFILING
            clog_info(CLOG(CLOGGER_ID), "query %d - %d l2square / %d sum2sax / %d entered / %d examined", i,
                      l2square_counter_profiling, sum2sax_counter_profiling, leaf_counter_profiling, k + 1);
#endif
        }
#ifdef FINE_TIMING
        clock_code = clock_gettime(CLK_ID, &stop_timestamp);
        getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
        clog_info(CLOG(CLOGGER_ID), "query %d - total search = %ld.%lds", i, time_diff.tv_sec, time_diff.tv_nsec);
#endif
#ifdef PROFILING
        clog_info(CLOG(CLOGGER_ID), "query %d - %d l2square / %d sum2sax / %d entered", i,
                  l2square_counter_profiling, sum2sax_counter_profiling, leaf_counter_profiling);
#endif
        logAnswer(i, answer);
    }

    for (unsigned int i = 0; i < config->num_indices; ++i) {
        unsigned int cache_offset = i * max_threads;

        free((Node **) queryCache[cache_offset].leaves);
        free(queryCache[cache_offset].leaf_distances);
        free(queryCache[cache_offset].leaf_indices);

        for (unsigned int j = 0; j < max_threads; ++j) {
            free(queryCache[cache_offset + j].m256_fetched_cache);
        }
    }

    freeAnswer(answer);
}


void conductQueriesLeavesMI(Config const *config, QuerySet const *querySet, MultIndex const *multindex) {
#ifdef FINE_TIMING
    struct timespec start_timestamp, stop_timestamp;
    TimeDiff time_diff;
#endif
    Answer *answer = initializeAnswer(config);

    unsigned int series_length = config->series_length;
    unsigned int sax_length = config->sax_length;
    Value scale_factor = config->scale_factor;

    unsigned int num_leaves = 0, leaf_counter = 0;
    for (unsigned int i = 0; i < config->num_indices; ++i) {
        num_leaves += multindex->indices[i]->num_leaves;
    }
    Node **leaves = malloc(sizeof(Node *) * num_leaves);
    for (unsigned int i = 0; i < config->num_indices; ++i) {
        for (unsigned int j = 0; j < multindex->indices[i]->roots_size; ++j) {
            enqueueLeaf(multindex->indices[i]->roots[j], leaves, &leaf_counter, multindex->indices[i]);
        }
    }
    assert(num_leaves == leaf_counter);

    Value *leaf_distances = malloc(sizeof(Value) * num_leaves);
    unsigned int *leaf_indices = malloc(sizeof(unsigned int) * num_leaves);
    for (unsigned int j = 0; j < num_leaves; ++j) {
        leaf_indices[j] = j;
    }

    ID shared_leaf_id = 0;
    unsigned int max_threads = config->max_threads;
    QueryCache queryCache[max_threads];

    unsigned int leaf_block_size = 1 + num_leaves / (max_threads << 1u);
    unsigned int query_block_size = 2 + num_leaves / (max_threads << 3u);

    for (unsigned int j = 0; j < max_threads; ++j) {
        queryCache[j].answer = answer;
        queryCache[j].index = NULL;

        queryCache[j].series_length = config->series_length;
        queryCache[j].sax_length = config->sax_length;

        queryCache[j].num_leaves = num_leaves;
        queryCache[j].leaves = (Node const *const *) leaves;
        queryCache[j].leaf_indices = leaf_indices;
        queryCache[j].leaf_distances = leaf_distances;

        queryCache[j].scale_factor = scale_factor;
        queryCache[j].m256_fetched_cache = aligned_alloc(sizeof(__m256i), sizeof(Value) * 8);

        queryCache[j].shared_leaf_id = &shared_leaf_id;
        queryCache[j].sort_leaves = config->sort_leaves;

        queryCache[j].series_limitations = config->series_limitations;
        queryCache[j].lower_bounding = config->lower_bounding;

        queryCache[j].leaf_block_size = leaf_block_size;
        queryCache[j].query_block_size = query_block_size;

        queryCache[j].resident_node = NULL;
    }

    for (unsigned int i = 0; i < querySet->query_size; ++i) {
#ifdef PROFILING
        query_id_profiling = i;
        leaf_counter_profiling = 0;
        sum2sax_counter_profiling = 0;
        l2square_counter_profiling = 0;
#endif
#ifdef FINE_TIMING
        clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
        if (querySet->initial_bsf_distances == NULL) {
            resetAnswer(answer);
        } else {
            resetAnswerBy(answer, querySet->initial_bsf_distances[i]);
            clog_info(CLOG(CLOGGER_ID), "query %d - initial 1bsf = %f", i, querySet->initial_bsf_distances[i]);
        }

        Value const *query_values = querySet->values + series_length * i;
        Value const *query_summarization = querySet->summarizations + sax_length * i;

        for (unsigned int j = 0; j < max_threads; ++j) {
            queryCache[j].query_values = query_values;
            queryCache[j].query_summarization = query_summarization;
        }

        pthread_t leaves_threads[max_threads];
        shared_leaf_id = 0;

        for (unsigned int j = 0; j < max_threads; ++j) {
            pthread_create(&leaves_threads[j], NULL, leafThread, (void *) &queryCache[j]);
        }
        for (unsigned int j = 0; j < max_threads; ++j) {
            pthread_join(leaves_threads[j], NULL);
        }

        if (config->sort_leaves) {
            qSortIndicesBy(leaf_indices, leaf_distances, 0, (int) (num_leaves - 1));
        }

        for (unsigned j = 0; getBSF(answer) < VALUE_MAX && j < num_leaves; ++j) {
#ifdef PROFILING
            leaf_counter_profiling += 1;
#endif
            for (unsigned int k = 0; k < max_threads; ++k) {
                leaf_distances[leaf_indices[j]] = VALUE_MAX;
            }

            Index *resident_index = (Index *) leaves[leaf_indices[j]]->index;

            queryNode(answer, leaves[leaf_indices[j]], resident_index->values, series_length, resident_index->saxs,
                      sax_length, resident_index->breakpoints, scale_factor, query_values, query_summarization,
                      queryCache[0].m256_fetched_cache, resident_index->pos2id);
        }

        pthread_t query_threads[max_threads];
        shared_leaf_id = 0;

        for (unsigned int j = 0; j < max_threads; ++j) {
            pthread_create(&query_threads[j], NULL, queryThread, (void *) &queryCache[j]);
        }
        for (unsigned int j = 0; j < max_threads; ++j) {
            pthread_join(query_threads[j], NULL);
        }
#ifdef FINE_TIMING
        clock_code = clock_gettime(CLK_ID, &stop_timestamp);
        getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
        clog_info(CLOG(CLOGGER_ID), "query %d - total search = %ld.%lds", i, time_diff.tv_sec, time_diff.tv_nsec);
#endif
#ifdef PROFILING
        clog_info(CLOG(CLOGGER_ID), "query %d - %d l2square / %d sum2sax / %d entered", i,
                  l2square_counter_profiling, sum2sax_counter_profiling, leaf_counter_profiling);
#endif
        logAnswer(i, answer);
    }

    for (unsigned int i = 0; i < max_threads; ++i) {
        free(queryCache[i].m256_fetched_cache);
    }

    free(leaves);
    free(leaf_distances);
    free(leaf_indices);
    freeAnswer(answer);
}