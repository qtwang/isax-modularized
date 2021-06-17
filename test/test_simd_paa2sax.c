//
// Created by Qitong Wang on 2020/7/10.
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
#include "sax.h"

int main(int argc, char **argv) {
    Config const *config = initializeConfig(argc, argv);

    int clog_init_code = clog_init_path(CLOGGER_ID, config->log_filepath);
    if (clog_init_code != 0) {
        fprintf(stderr, "Logger initialization failed, log_filepath = %s\n", config->log_filepath);
        exit(EXIT_FAILURE);
    }

    logConfig(config);

    clock_t start_clock = clock();

    Index *index = initializeIndex(config);

    clog_info(CLOG(CLOGGER_ID), "index - load only = %lums", (clock() - start_clock) * 1000 / CLOCKS_PER_SEC);

    logIndex(config, index);

    Value scale_factor = (Value) config->series_length / (Value) config->sax_length;
    Value min_paa2sax = VALUE_MAX, local_paa2sax;
    Value *m256_fetched_cache = malloc(sizeof(Value) * 8);
    start_clock = clock();

    for (unsigned int i = 1; i < index->database_size; ++i) {
        local_paa2sax = l2SquareValue2SAX8(index->sax_length, index->summarizations,
                                           index->saxs + index->sax_length * i, index->breakpoints, scale_factor);

        if (VALUE_L(local_paa2sax, min_paa2sax)) {
            min_paa2sax = local_paa2sax;
        }
    }

    clog_info(CLOG(CLOGGER_ID), "test - l2SquareValue2SAX8 = %f by %lums", min_paa2sax,
              (clock() - start_clock) * 1000 / CLOCKS_PER_SEC);

    min_paa2sax = VALUE_MAX;

    start_clock = clock();

    for (unsigned int i = 1; i < index->database_size; ++i) {
        local_paa2sax = l2SquareValue2SAX8SIMD(index->sax_length, index->summarizations,
                                               index->saxs + index->sax_length * i, index->breakpoints, scale_factor,
                                               m256_fetched_cache);

        if (VALUE_L(local_paa2sax, min_paa2sax)) {
            min_paa2sax = local_paa2sax;
        }
    }

    clog_info(CLOG(CLOGGER_ID), "test - l2SquareValue2SAX8SIMD = %f by %lums", min_paa2sax,
              (clock() - start_clock) * 1000 / CLOCKS_PER_SEC);

    SAXMask masks8[16] = {1u, 1u << 6u, 1u, 1u, 1u, 1u << 3u, 1u, 1u,
                          1u << 2u, 1u, 1u, 1u << 4u, 1u, 1u, 1u, 1u};
    min_paa2sax = VALUE_MAX;

    start_clock = clock();

    for (unsigned int i = 1; i < index->database_size; ++i) {
        local_paa2sax = l2SquareValue2SAXByMask(index->sax_length, index->summarizations,
                                                index->saxs + index->sax_length * i, masks8, index->breakpoints,
                                                scale_factor);

        if (VALUE_L(local_paa2sax, min_paa2sax)) {
            min_paa2sax = local_paa2sax;
        }
    }

    clog_info(CLOG(CLOGGER_ID), "test - l2SquareValue2SAXByMask = %f by %lums", min_paa2sax,
              (clock() - start_clock) * 1000 / CLOCKS_PER_SEC);

    min_paa2sax = VALUE_MAX;

    start_clock = clock();

    for (unsigned int i = 1; i < index->database_size; ++i) {
        local_paa2sax = l2SquareValue2SAXByMaskSIMD(index->sax_length, index->summarizations,
                                                    index->saxs + index->sax_length * i, masks8, index->breakpoints,
                                                    scale_factor, m256_fetched_cache);

        if (VALUE_L(local_paa2sax, min_paa2sax)) {
            min_paa2sax = local_paa2sax;
        }
    }

    clog_info(CLOG(CLOGGER_ID), "test - l2SquareValue2SAXByMaskSIMD = %f by %lums", min_paa2sax,
              (clock() - start_clock) * 1000 / CLOCKS_PER_SEC);

    free(m256_fetched_cache);

    freeIndex(index);
    free((Config *) config);

    clog_free(CLOGGER_ID);
    return 0;
}
