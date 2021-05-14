//
// Created by Qitong Wang on 2020/7/3.
//

#include "query.h"


QuerySet *initializeQuery(Config const *config, Index const *index) {
    QuerySet *queries = malloc(sizeof(QuerySet));

    queries->query_size = config->query_size;

#ifdef FINE_TIMING
    struct timespec start_timestamp, stop_timestamp;
    TimeDiff time_diff;
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif

    Value *values = aligned_alloc(256, sizeof(Value) * config->series_length * config->query_size);
    FILE *file_values = fopen(config->query_filepath, "rb");
    size_t read_values = fread(values, sizeof(Value), config->series_length * config->query_size, file_values);
    fclose(file_values);
    assert(read_values == config->series_length * config->query_size);

    queries->values = (Value const *) values;

    if (config->query_bsf_distance_filepath != NULL) {
        Value *initial_bsf_distances = aligned_alloc(64, sizeof(Value) * config->query_size);
        FILE *file_bsf_distances = fopen(config->query_bsf_distance_filepath, "rb");
        read_values = fread(initial_bsf_distances, sizeof(Value), config->query_size, file_bsf_distances);
        fclose(file_bsf_distances);
        assert(read_values == config->query_size);

        queries->initial_bsf_distances = (Value const *) initial_bsf_distances;
    } else {
        queries->initial_bsf_distances = NULL;
    }

#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "query - load series = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif

    if (config->query_summarization_filepath != NULL) {
        Value *summarizations = aligned_alloc(256, sizeof(Value) * config->sax_length * config->query_size);
        FILE *file_summarizations = fopen(config->query_summarization_filepath, "rb");
        read_values = fread(summarizations, sizeof(Value), config->sax_length * config->query_size,
                            file_summarizations);
        fclose(file_summarizations);
        assert(read_values == config->sax_length * config->query_size);

        queries->summarizations = (Value const *) summarizations;
    } else {
        queries->summarizations = (Value const *) piecewiseAggregate(queries->values, config->query_size,
                                                                     config->series_length, config->sax_length,
                                                                     config->max_threads);
    }

#ifdef FINE_TIMING
    char *method4summarizations = "load";
    if (config->database_summarization_filepath == NULL) {
        method4summarizations = "calculate";
    }
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "query - %s summarizations = %ld.%lds", method4summarizations, time_diff.tv_sec,
              time_diff.tv_nsec);
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif

    queries->saxs = (SAXWord const *) summarizations2SAX16(queries->summarizations, index->breakpoints,
                                                           queries->query_size, config->sax_length,
                                                           config->sax_cardinality, config->max_threads);

#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "query - calculate SAXs = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
#endif

    return queries;
}


void freeQuery(QuerySet *queries) {
    free((Value *) queries->values);
    free((Value *) queries->summarizations);
    free((Value *) queries->initial_bsf_distances);
    free((Value *) queries->saxs);
    free(queries);
}
