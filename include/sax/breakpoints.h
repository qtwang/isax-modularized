//
// Created by Qitong Wang on 2020/6/28.
//

#ifndef ISAX_BREAKPOINTS_H
#define ISAX_BREAKPOINTS_H

#include <float.h>
#include <stdlib.h>
#include <pthread.h>
#include <immintrin.h>

#include "globals.h"
#include "clog.h"
#include "config.h"


__m256i M256I_1;
__m256i M256I_BREAKPOINTS_OFFSETS_0_7, M256I_BREAKPOINTS_OFFSETS_8_15;
__m256i M256I_BREAKPOINTS8_OFFSETS_0_7, M256I_BREAKPOINTS8_OFFSETS_8_15;
__m256i *M256I_OFFSETS_BY_SEGMENTS;


static unsigned int const OFFSETS_BY_CARDINALITY[9] = {
        0, 3, 8, 17,
        34, 67, 132, 261,
        518
};


static unsigned int const OFFSETS_BY_SEGMENTS[17] = {
        0, 518, 518 * 2, 518 * 3, 518 * 4,
        518 * 5, 518 * 6, 518 * 7, 518 * 8, 518 * 9,
        518 * 10, 518 * 11, 518 * 12, 518 * 13, 518 * 14,
        518 * 15, 518 * 16
};


static unsigned int const OFFSETS_BY_MASK[129] = {
        0, 261, 132, 0, 67, 0, 0, 0, 34, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0
};


static unsigned int const LENGTHS_BY_MASK[129] = {
        0, 257, 129, 0, 65, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 3
};


void initializeM256IConstants();

Value const *getNormalBreakpoints8(unsigned int num_segments);

Value const *getAdhocBreakpoints8(Value const *summarizations, size_t size, unsigned int num_segments,
                                  unsigned int num_threads);

Value const *getSharedAdhocBreakpoints8(Value const *summarizations, size_t size, unsigned int num_segments);

#endif //ISAX_BREAKPOINTS_H
