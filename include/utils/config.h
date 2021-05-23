//
// Created by Qitong Wang on 2020/6/28.
//

#ifndef ISAX_CONFIG_H
#define ISAX_CONFIG_H

// TODO CPU_ZERO and others are specified for Linux, except Mac
#define _GNU_SOURCE

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <pthread.h>
#include <stdbool.h>
#include <assert.h>

#include "globals.h"
#include "clog.h"


typedef struct Config {
    char *database_filepath;
    char *database_summarization_filepath;
    char *query_filepath;
    char *query_summarization_filepath;
    char *query_bsf_distance_filepath;
    char *log_filepath;

    ID database_size;
    unsigned int query_size;

    unsigned int series_length;
    unsigned int sax_length;
    unsigned int sax_cardinality;

    unsigned int initial_leaf_size;
    unsigned int leaf_size;

    bool use_adhoc_breakpoints;
    bool share_breakpoints;
    bool exact_search;
    bool sort_leaves;
    bool split_by_summarizations;

    unsigned int k; // kNN

    unsigned int cpu_cores;
    unsigned int numa_cores;
    unsigned int max_threads;
    unsigned int skipped_cores;
    unsigned int numa_id;

    unsigned int index_block_size;

    unsigned int series_limitations;

    Value scale_factor;

    bool leaf_compactness;
    bool lower_bounding;
    bool log_leaf_only;
    bool with_id;

    bool squeeze_leaves;
    bool peel_leaves;
} Config;


Config *initializeConfig(int argc, char **argv);

void logConfig(Config const *config);

#endif //ISAX_CONFIG_H
