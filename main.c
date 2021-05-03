//
// Created by Qitong Wang on 2020/6/28.
//

#include <stdio.h>
#include <time.h>
#include <pthread.h>

#ifndef CLOG_MAIN
#define CLOG_MAIN
#endif

#include "clog.h"
#include "globals.h"
#include "config.h"
#include "index.h"
#include "index_engine.h"
#include "query.h"
#include "query_engine.h"


int main(int argc, char **argv) {
    Config const *config = initializeConfig(argc, argv);

    int clog_init_code = clog_init_path(CLOGGER_ID, config->log_filepath);
    if (clog_init_code != 0) {
        fprintf(stderr, "Logger initialization failed, log_filepath = %s\n", config->log_filepath);
        exit(EXIT_FAILURE);
    }

    logConfig(config);

#ifdef TIMING
    struct timespec start_timestamp, stop_timestamp;
    TimeDiff time_diff;

    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif

    Index *index = initializeIndex(config);
    buildIndex(config, index);
    finalizeIndex(config, index);

#ifdef TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);

    clog_info(CLOG(CLOGGER_ID), "index - overall = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
#endif

    logIndex(index);

#ifdef PROFILING
    log_lock_profiling = malloc(sizeof(pthread_mutex_t));
    assert(pthread_mutex_init(log_lock_profiling, NULL) == 0);
#endif

#ifdef TIMING
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif

    QuerySet *queries = initializeQuery(config, index);
    conductQueries(queries, index, config);

#ifdef TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);

    clog_info(CLOG(CLOGGER_ID), "query - overall = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
#endif

#ifdef PROFILING
    pthread_mutex_destroy(log_lock_profiling);
    free(log_lock_profiling);
#endif

    freeIndex(index);
    freeQuery(queries);
    free((Config *) config);

    clog_free(CLOGGER_ID);
    return 0;
}
