//
// Created by Qitong Wang on 2020/6/28.
//

#include "config.h"


const struct option longopts[] = {
        {"database_filepath",               required_argument, NULL, 1},
        {"database_summarization_filepath", required_argument, NULL, 2},
        {"query_filepath",                  required_argument, NULL, 3},
        {"query_summarization_filepath",    required_argument, NULL, 4},
        {"database_size",                   required_argument, NULL, 5},
        {"query_size",                      required_argument, NULL, 6},
        {"sax_length",                      required_argument, NULL, 7},
        {"sax_cardinality",                 required_argument, NULL, 8},
        {"cpu_cores",                       required_argument, NULL, 9},
        {"log_filepath",                    required_argument, NULL, 10},
        {"series_length",                   required_argument, NULL, 11},
        {"adhoc_breakpoints",               no_argument,       NULL, 12},
        {"numa_cores",                      required_argument, NULL, 13},
        {"index_block_size",                required_argument, NULL, 14},
        {"leaf_size",                       required_argument, NULL, 15},
        {"initial_leaf_size",               required_argument, NULL, 16},
        {"exact_search",                    no_argument,       NULL, 17},
        {"k",                               required_argument, NULL, 18},
        {"sort_leaves",                     no_argument,       NULL, 19},
        {"split_by_summarizations",         no_argument,       NULL, 20},
        {"scale_factor",                    required_argument, NULL, 21},
        {"skipped_cores",                   required_argument, NULL, 22},
        {"numa_id",                         required_argument, NULL, 23},
        {"series_limitations",              required_argument, NULL, 24},
        {"leaf_compactness",                no_argument,       NULL, 25},
        {"not_lower_bounding",              no_argument,       NULL, 26},
        {"log_leaf_only",                   no_argument,       NULL, 27},
        {"share_breakpoints",               no_argument,       NULL, 28},
        {"query_bsf_filepath",              required_argument, NULL, 29},
        {"squeeze_leaves",                  no_argument,       NULL, 30},
        {"peel_leaves",                     no_argument,       NULL, 31},
        {"with_id",                         no_argument,       NULL,    32},
        {NULL,                              no_argument,       NULL, 0}
};


int initializeThreads(Config *config, unsigned int cpu_cores, unsigned int numa_cores, unsigned int skipped_cores,
                      unsigned int numa_id) {
    config->max_threads = cpu_cores;

    cpu_set_t mask, get;

    CPU_ZERO(&mask);
    CPU_ZERO(&get);

    // for andromache(Intel(R) Xeon(R) Gold 6134 CPU @ 3.20GHz), system(cpu)-dependent, check by lscpu
    unsigned int step = 3 - numa_cores;
    for (unsigned int i = 0; i < cpu_cores; ++i) {
        CPU_SET(numa_id + (skipped_cores + i) * step, &mask);
    }

    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask) != 0) {
        fprintf(stderr, "set thread affinity failed\n");
    }

    if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &get) != 0) {
        fprintf(stderr, "get thread affinity failed\n");
    }

    return 0;
}


Config *initializeConfig(int argc, char **argv) {
    Config *config = malloc(sizeof(Config));

    config->database_filepath = NULL;
    config->database_summarization_filepath = NULL;
    config->query_filepath = NULL;
    config->query_summarization_filepath = NULL;
    config->query_bsf_distance_filepath = NULL;
    config->log_filepath = "./isax.log";

    config->series_length = 256;

    config->database_size = 0;
    config->query_size = 0;

    config->initial_leaf_size = 1024;
    config->leaf_size = 8000;

    config->index_block_size = 20000;

    config->sax_cardinality = 8;
    config->sax_length = 16;
    config->scale_factor = -1;

    config->use_adhoc_breakpoints = false;
    config->share_breakpoints = false;
    config->exact_search = false;
    config->sort_leaves = false;
    config->split_by_summarizations = false;

    config->k = 1;

    config->cpu_cores = 1;
    config->numa_cores = 1;
    config->numa_id = 0;
    config->skipped_cores = 0;

    config->series_limitations = 0;

    config->leaf_compactness = false;
    config->lower_bounding = true;
    config->with_id = false;

    config->squeeze_leaves = false;
    config->peel_leaves = false;

    char *string_parts;
    int opt, longindex = 0;
    while ((opt = getopt_long(argc, argv, "", longopts, &longindex)) != -1) {
        switch (opt) {
            case 1:
                config->database_filepath = optarg;
                break;
            case 2:
                config->database_summarization_filepath = optarg;
                break;
            case 3:
                config->query_filepath = optarg;
                break;
            case 4:
                config->query_summarization_filepath = optarg;
                break;
            case 5:
                config->database_size = (ID) strtoull(optarg, &string_parts, 10);
                break;
            case 6:
                config->query_size = (unsigned int) strtoul(optarg, &string_parts, 10);
                break;
            case 7:
                config->sax_length = (unsigned int) strtoul(optarg, &string_parts, 10);
                break;
            case 8:
                config->sax_cardinality = (unsigned int) strtol(optarg, &string_parts, 10);
                break;
            case 9:
                config->cpu_cores = (int) strtol(optarg, &string_parts, 10);
                break;
            case 10:
                config->log_filepath = optarg;
                break;
            case 11:
                config->series_length = (unsigned int) strtol(optarg, &string_parts, 10);
                break;
            case 12:
                config->use_adhoc_breakpoints = true;
                break;
            case 13:
                config->numa_cores = (unsigned int) strtol(optarg, &string_parts, 10);
                break;
            case 14:
                config->index_block_size = (unsigned int) strtoul(optarg, &string_parts, 10);
                break;
            case 15:
                config->leaf_size = (unsigned int) strtoul(optarg, &string_parts, 10);
                break;
            case 16:
                config->initial_leaf_size = (unsigned int) strtoul(optarg, &string_parts, 10);
                break;
            case 17:
                config->exact_search = true;
                break;
            case 18:
                config->k = (unsigned int) strtol(optarg, &string_parts, 10);
                break;
            case 19:
                config->sort_leaves = true;
                break;
            case 20:
                config->split_by_summarizations = true;
                break;
            case 21:
                config->scale_factor = strtof(optarg, &string_parts);
                break;
            case 22:
                config->skipped_cores = (unsigned int) strtol(optarg, &string_parts, 10);
                break;
            case 23:
                config->numa_id = (unsigned int) strtol(optarg, &string_parts, 10);
                break;
            case 24:
                config->series_limitations = (unsigned int) strtol(optarg, &string_parts, 10);
                break;
            case 25:
                config->leaf_compactness = true;
                break;
            case 26:
                config->lower_bounding = false;
                break;
            case 27:
                config->log_leaf_only = true;
                break;
            case 28:
                config->share_breakpoints = true;
                break;
            case 29:
                config->query_bsf_distance_filepath = optarg;
                break;
            case 30:
                config->squeeze_leaves = true;
                break;
            case 31:
                config->peel_leaves = true;
                break;
            case 32:
                config->with_id = true;
                break;
            default:
                exit(EXIT_FAILURE);
        }
    }

//    assert(config->series_length % config->sax_length == 0 && config->series_length % 8 == 0);
    assert(config->series_length % 8 == 0);
//    assert(config->sax_length == 8 || config->sax_length == 16);
    assert(config->sax_length >= 8 || config->sax_length <= 16);
    assert(config->sax_cardinality == 8);
    assert(config->database_size > 0);
    assert(config->query_size > 0);
    assert(config->index_block_size > 0);
    assert(config->series_length > 0);
    assert(config->leaf_size > 0 && config->initial_leaf_size > 0 && config->initial_leaf_size <= config->leaf_size);
    assert(config->k >= 0 && config->k <= 1024);

    if (VALUE_EQ(config->scale_factor, -1)) {
        config->scale_factor = (Value) config->series_length / (Value) config->sax_length;
    } else {
        assert(config->scale_factor > 0);
    }

    assert(config->cpu_cores > 0 && config->numa_cores > 0 &&
           (config->numa_id == 0 || config->numa_id == 1) &&
           ((config->numa_cores == 2 && config->skipped_cores + config->cpu_cores <= 32) ||
            (config->numa_cores == 1 && config->skipped_cores + config->cpu_cores <= 16)));

    initializeThreads(config, config->cpu_cores, config->numa_cores, config->skipped_cores, config->numa_id);
    return config;
}


void logConfig(Config const *config) {
    clog_info(CLOG(CLOGGER_ID), "config - database_filepath = %s", config->database_filepath);
    clog_info(CLOG(CLOGGER_ID), "config - database_summarization_filepath = %s",
              config->database_summarization_filepath);
    clog_info(CLOG(CLOGGER_ID), "config - query_filepath = %s", config->query_filepath);
    clog_info(CLOG(CLOGGER_ID), "config - query_summarization_filepath = %s", config->query_summarization_filepath);
    clog_info(CLOG(CLOGGER_ID), "config - query_bsf_distance_filepath = %s", config->query_bsf_distance_filepath);
    clog_info(CLOG(CLOGGER_ID), "config - log_filepath = %s", config->log_filepath);

    clog_info(CLOG(CLOGGER_ID), "config - series_length = %u", config->series_length);
    clog_info(CLOG(CLOGGER_ID), "config - database_size = %lu", config->database_size);
    clog_info(CLOG(CLOGGER_ID), "config - query_size = %u", config->query_size);
    clog_info(CLOG(CLOGGER_ID), "config - sax_length = %u", config->sax_length);
    clog_info(CLOG(CLOGGER_ID), "config - sax_cardinality = %d", config->sax_cardinality);
    clog_info(CLOG(CLOGGER_ID), "config - adhoc_breakpoints = %d", config->use_adhoc_breakpoints);
    clog_info(CLOG(CLOGGER_ID), "config - share_breakpoints = %d", config->share_breakpoints);

    clog_info(CLOG(CLOGGER_ID), "config - exact_search = %d", config->exact_search);
    clog_info(CLOG(CLOGGER_ID), "config - k = %d", config->k);

    clog_info(CLOG(CLOGGER_ID), "config - leaf_size = %u", config->leaf_size);
    clog_info(CLOG(CLOGGER_ID), "config - initial_leaf_size = %u", config->initial_leaf_size);
    clog_info(CLOG(CLOGGER_ID), "config - sort_leaves = %d", config->sort_leaves);
    clog_info(CLOG(CLOGGER_ID), "config - split_by_summarizations = %d", config->split_by_summarizations);
    clog_info(CLOG(CLOGGER_ID), "config - squeeze_leaves = %d", config->squeeze_leaves);
    clog_info(CLOG(CLOGGER_ID), "config - peel_leaves = %d", config->peel_leaves);

    clog_info(CLOG(CLOGGER_ID), "config - cpu_cores = %d", config->cpu_cores);
    clog_info(CLOG(CLOGGER_ID), "config - numa_cores = %d", config->numa_cores);
    clog_info(CLOG(CLOGGER_ID), "config - skipped_cores = %d", config->skipped_cores);
    clog_info(CLOG(CLOGGER_ID), "config - numa_id = %d", config->numa_id);
    clog_info(CLOG(CLOGGER_ID), "config - index_block_size = %u", config->index_block_size);
}
