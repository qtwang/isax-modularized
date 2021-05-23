//
// Created by Qitong Wang on 2020/6/28.
//

#include "index.h"


SAXWord *rootID2SAX(unsigned int id, unsigned int num_segments, unsigned int cardinality) {
    SAXWord *sax = aligned_alloc(128, sizeof(SAXWord) * SAX_SIMD_ALIGNED_LENGTH);
    unsigned int offset = cardinality - 1;

    for (unsigned int i = 1; i <= num_segments; ++i) {
        sax[num_segments - i] = (SAXWord) ((id & 1u) << offset);
        id >>= 1u;
    }

    return sax;
}


unsigned int rootSAX2ID(SAXWord const *saxs, unsigned int num_segments, unsigned int cardinality) {
    unsigned int id = 0, offset = cardinality - 1;

    for (unsigned int i = 0; i < num_segments; ++i) {
        id <<= 1u;
        id += (unsigned int) (saxs[i] >> offset);
    }

    return id;
}


Index *initializeIndex(Config const *config) {
    initializeM256IConstants();

    Index *index = malloc(sizeof(Index));
    if (index == NULL) {
        clog_error(CLOG(CLOGGER_ID), "could not allocate memory to initialize index");
        exit(EXIT_FAILURE);
    }

    index->series_length = config->series_length;
    index->sax_length = config->sax_length;
    index->sax_cardinality = config->sax_cardinality;
    index->database_size = config->database_size;
    index->num_leaves = 0;

#ifdef FINE_TIMING
    struct timespec start_timestamp, stop_timestamp;
    TimeDiff time_diff;
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif

    index->roots_size = 1u << config->sax_length;
    index->roots = malloc(sizeof(Node *) * index->roots_size);
    SAXMask *root_masks = aligned_alloc(128, sizeof(SAXMask) * config->sax_length);
    for (unsigned int i = 0; i < config->sax_length; ++i) {
        root_masks[i] = (SAXMask) (1u << (config->sax_cardinality - 1));
    }
    for (unsigned int i = 0; i < index->roots_size; ++i) {
        index->roots[i] = initializeNode(rootID2SAX(i, config->sax_length, config->sax_cardinality),
                                         root_masks);
    }

#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "index - initialize roots = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif

    Value *values = aligned_alloc(256, sizeof(Value) * config->series_length * config->database_size);

    FILE *file_values = fopen(config->database_filepath, "rb");
    size_t read_values = fread(values, sizeof(Value), config->series_length * config->database_size, file_values);
    fclose(file_values);
    assert(read_values == config->series_length * config->database_size);

    index->values = (Value const *) values;

#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "index - load series = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif

    Value *summarizations = aligned_alloc(256, sizeof(Value) * config->sax_length * config->database_size);

    if (config->database_summarization_filepath != NULL) {
        FILE *file_summarizations = fopen(config->database_summarization_filepath, "rb");
        read_values = fread(summarizations, sizeof(Value), config->sax_length * config->database_size,
                            file_summarizations);
        fclose(file_summarizations);
        assert(read_values == config->sax_length * config->database_size);
    } else {
        summarizations = piecewiseAggregate(index->values, config->database_size, config->series_length,
                                            config->sax_length, config->max_threads);
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

    if (config->use_adhoc_breakpoints) {
        if (config->share_breakpoints) {
            index->breakpoints = getSharedAdhocBreakpoints8(summarizations, config->database_size, config->sax_length);
        } else {
            index->breakpoints = getAdhocBreakpoints8(summarizations, config->database_size, config->sax_length,
                                                      config->max_threads);
        }
    } else {
        index->breakpoints = getNormalBreakpoints8(config->sax_length);
    }

#ifdef FINE_TIMING
    char *method4breakpoints = "normal";
    if (config->use_adhoc_breakpoints) {
        method4breakpoints = "adhoc";
    }
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "index - load %s breakpoints = %ld.%lds", method4breakpoints, time_diff.tv_sec,
              time_diff.tv_nsec);
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif

    index->saxs = (SAXWord const *) summarizations2SAX16(summarizations, index->breakpoints, index->database_size,
                                                         index->sax_length, index->sax_cardinality, config->max_threads);

#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "index - calculate SAXs = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
#endif

    if (config->split_by_summarizations || config->peel_leaves) {
        index->summarizations = (Value const *) summarizations;
    } else {
        free(summarizations);
        index->summarizations = NULL;
    }

    return index;
}


void freeIndex(Index *index) {
    free((Value *) index->values);
    free((SAXWord *) index->saxs);
    free((Value *) index->breakpoints);
    free(index->pos2id); // which might be NULL

    bool first_root = true;
    for (unsigned int i = 0; i < index->roots_size; ++i) {
        if (index->roots[i] != NULL) {
            if (first_root) {
                freeNode(index->roots[i], true, true);
                first_root = false;
            } else {
                freeNode(index->roots[i], false, true);
            }
        }
    }

    free(index->roots);
}


void logIndex(Index *index) {
    unsigned int num_series = 0, num_roots = 0;
    for (unsigned int i = 0; i < index->roots_size; ++i) {
        inspectNode(index->roots[i], &num_series, &index->num_leaves, &num_roots);
    }

    clog_info(CLOG(CLOGGER_ID), "index - %d series in %d leaves from %d / %d roots", num_series, index->num_leaves,
              num_roots, index->roots_size);
}
