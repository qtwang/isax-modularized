//
// Created by Qitong Wang on 2021/5/31.
//

#include "multindex.h"


void permute(void *values, ID *permutation, ID num_segments, unsigned int bytesize_value, unsigned int length_segment) {
    unsigned int bytesize_segment = bytesize_value * length_segment;
    Value *values_cache = aligned_alloc(256, bytesize_segment);

    for (ID i = 0, next, tmp; i < num_segments; ++i) {
        next = i;

        while (permutation[next] >= 0) {
            memcpy(values_cache, values + bytesize_segment * i, bytesize_segment);
            memcpy(values + bytesize_segment * i, values + bytesize_segment * permutation[next], bytesize_segment);
            memcpy(values + bytesize_segment * permutation[next], values_cache, bytesize_segment);

            tmp = permutation[next];
            permutation[next] -= num_segments;
            next = tmp;
        }
    }

    free(values_cache);
}


MultIndex *initializeMultIndex(Config const *config) {
    initializeM256IConstants();

    MultIndex *multindex = malloc(sizeof(MultIndex));
    if (multindex == NULL) {
        clog_error(CLOG(CLOGGER_ID), "could not allocate memory to initialize a multindex");
        exit(EXIT_FAILURE);
    }

    multindex->num_indices = config->num_indices;
    multindex->database_size = config->database_size;
    multindex->series_length = config->series_length;
    multindex->sax_length = config->sax_length;
    multindex->sax_cardinality = config->sax_cardinality;

#ifdef FINE_TIMING
    struct timespec start_timestamp, stop_timestamp;
    TimeDiff time_diff;
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
    Value *values = aligned_alloc(256, sizeof(Value) * config->series_length * config->database_size);

    FILE *file_values = fopen(config->database_filepath, "rb");
    size_t read_values = fread(values, sizeof(Value), config->series_length * config->database_size, file_values);
    fclose(file_values);
    assert(read_values == config->series_length * config->database_size);
#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "index - load series = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
    ID *permutation = NULL;

    if (config->cluster_indicators_filepath != NULL && config->cluster_centers_filepath != NULL) {
        Value *centers = aligned_alloc(256, sizeof(Value) * config->series_length * config->num_indices);

        file_values = fopen(config->cluster_centers_filepath, "rb");
        read_values = fread(values, sizeof(Value), config->series_length * config->num_indices, file_values);
        fclose(file_values);
        assert(read_values == config->series_length * config->num_indices);

        multindex->centers = (Value const *) centers;

        int32_t *indicators = aligned_alloc(256, sizeof(int32_t) * config->database_size);

        file_values = fopen(config->cluster_indicators_filepath, "rb");
        read_values = fread(values, sizeof(int32_t), config->database_size, file_values);
        fclose(file_values);
        assert(read_values == config->database_size);

        multindex->cluster_sizes = malloc(sizeof(ID) * config->num_indices);
        for (unsigned int i = 0; i < config->num_indices; ++i) {
            multindex->cluster_sizes[i] = 0;
        }
        for (unsigned int i = 0; i < config->database_size; ++i) {
            multindex->cluster_sizes[indicators[i]] += 1;
        }

        multindex->cluster_offsets = malloc(sizeof(ID) * config->num_indices);
        multindex->cluster_offsets[0] = 0;
        for (unsigned int i = 1; i < config->num_indices; ++i) {
            multindex->cluster_offsets[i] = multindex->cluster_sizes[i - 1];
        }

        ID *offset_iterators = malloc(sizeof(ID) * config->num_indices);
        memcpy(offset_iterators, multindex->cluster_offsets, sizeof(ID) * config->num_indices);
        permutation = aligned_alloc(sizeof(ID), sizeof(ID) * config->database_size);
        for (unsigned int i = 0; i < config->database_size; ++i) {
            permutation[i] = offset_iterators[indicators[i]];
            offset_iterators[indicators[i]] += 1;
        }
        multindex->inverse_permutation = aligned_alloc(sizeof(ID), sizeof(ID) * config->database_size);
        for (unsigned int i = 0; i < config->database_size; ++i) {
            multindex->inverse_permutation[permutation[i]] = i;
        }

        permute(values, permutation, config->database_size, sizeof(Value), config->series_length);
        multindex->values = (Value const *) values;

        free(offset_iterators);
    } else {
        clog_error(CLOG(CLOGGER_ID), "not yet support internal clustering");
        exit(EXIT_FAILURE);
    }
#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "index - load clusters = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
    Value *summarizations = aligned_alloc(256, sizeof(Value) * config->sax_length * config->database_size);

    if (config->database_summarization_filepath != NULL) {
        FILE *file_summarizations = fopen(config->database_summarization_filepath, "rb");
        read_values = fread(summarizations, sizeof(Value), config->sax_length * config->database_size,
                            file_summarizations);
        fclose(file_summarizations);
        assert(read_values == config->sax_length * config->database_size);

        assert(permutation != NULL);
        for (ID i = 0; i < config->database_size; ++i) {
            permutation[i] += config->database_size;
        }
        permute(summarizations, permutation, config->database_size, sizeof(Value), config->sax_length);
    } else {
        summarizations = piecewiseAggregate(multindex->values, config->database_size, config->series_length,
                                            config->sax_length, config->max_threads);
    }

    if (permutation != NULL) {
        free(permutation);
        permutation = NULL;
    }
#ifdef FINE_TIMING
    char *method4summarizations = "load";
    if (config->database_summarization_filepath == NULL) {
        method4summarizations = "calculate";
    }
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "index - %s summarizations = %ld.%lds", method4summarizations, time_diff.tv_sec,
              time_diff.tv_nsec);
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
    multindex->indices = malloc(sizeof(Index) * config->num_indices);
    if (multindex->indices == NULL) {
        clog_error(CLOG(CLOGGER_ID), "could not allocate memory to initialize the internal indices");
        exit(EXIT_FAILURE);
    }

    SAXWord *saxs = aligned_alloc(128, sizeof(SAXWord) * SAX_SIMD_ALIGNED_LENGTH * config->database_size);

    for (unsigned int i = 0; i < config->num_indices; ++i) {
        multindex->indices[i]->series_length = config->series_length;
        multindex->indices[i]->sax_length = config->sax_length;
        multindex->indices[i]->sax_cardinality = config->sax_cardinality;

        multindex->indices[i]->database_size = multindex->cluster_sizes[i];
        multindex->indices[i]->values = multindex->values + config->series_length * multindex->cluster_offsets[i];
        multindex->indices[i]->summarizations = summarizations + config->sax_length * multindex->cluster_offsets[i];

        multindex->indices[i]->num_leaves = 0;

        multindex->indices[i]->roots_size = 1u << config->sax_length;
        multindex->indices[i]->roots = malloc(sizeof(Node *) * multindex->indices[i]->roots_size);
        SAXMask *root_masks = aligned_alloc(128, sizeof(SAXMask) * config->sax_length);
        for (unsigned int j = 0; j < config->sax_length; ++j) {
            root_masks[j] = (SAXMask) (1u << (config->sax_cardinality - 1));
        }
        for (unsigned int j = 0; j < multindex->indices[j]->roots_size; ++j) {
            multindex->indices[i]->roots[j] = initializeNode(rootID2SAX(j, config->sax_length, config->sax_cardinality),
                                                             root_masks);
        }

        if (config->use_adhoc_breakpoints) {
            if (config->share_breakpoints) {
                multindex->indices[i]->breakpoints = getSharedAdhocBreakpoints8(multindex->indices[i]->summarizations,
                                                                                multindex->cluster_sizes[i],
                                                                                config->sax_length);
            } else {
                multindex->indices[i]->breakpoints = getAdhocBreakpoints8(multindex->indices[i]->summarizations,
                                                                          multindex->cluster_sizes[i],
                                                                          config->sax_length, config->max_threads);
            }
        } else {
            multindex->indices[i]->breakpoints = getNormalBreakpoints8(config->sax_length);
        }

        summarizations2SAX16(saxs + config->sax_length * multindex->cluster_offsets[i],
                             summarizations, multindex->indices[i]->breakpoints, config->database_size,
                             config->sax_length, config->sax_cardinality, config->max_threads);
        multindex->indices[i]->saxs = saxs + config->sax_length * multindex->cluster_offsets[i];
    }
#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "index - initialize %d internal indices = %ld.%lds",
              config->num_indices, time_diff.tv_sec, time_diff.tv_nsec);
#endif
    if (!config->split_by_summarizations && !config->peel_leaves) {
        free(summarizations);
        multindex->summarizations = NULL;
        for (unsigned int i = 0; i < config->num_indices; ++i) {
            multindex->indices[i]->summarizations = NULL;
        }
    }

    return multindex;
}


void freeMultIndex(MultIndex *multindex) {
    free((Value *) multindex->centers);
    free((Value *) multindex->values);
    free((SAXWord *) multindex->saxs);
    free((Value *) multindex->summarizations);

    free(multindex->cluster_sizes);
    free(multindex->cluster_offsets);
    free(multindex->inverse_permutation);

    for (unsigned int i = 0; i < multindex->num_indices; ++i) {
        free((Value *) multindex->indices[i]->breakpoints);
        free(multindex->indices[i]->pos2id);

        bool first_root = true;
        for (unsigned int j = 0; j < multindex->indices[i]->roots_size; ++j) {
            if (multindex->indices[i]->roots[j] != NULL) {
                if (first_root) {
                    freeNode(multindex->indices[i]->roots[j], true, true);
                    first_root = false;
                } else {
                    freeNode(multindex->indices[i]->roots[j], false, true);
                }
            }
        }

        free(multindex->indices[i]->roots);
    }
}


void logMultIndex(MultIndex *multindex) {
    for (unsigned int i = 0; i < multindex->num_indices; ++i) {
        clog_info(CLOG(CLOGGER_ID), "multindex - %d series in %d internal index", multindex->cluster_sizes[i], i);
        logIndex(multindex->indices[i]);
    }
}