//
// Created by Qitong Wang on 2020/6/28.
//

#include "breakpoints.h"


static Value const NORMAL_BREAKPOINTS_8[255] = {
        -2.10003938216135, -2.06623181534436, -2.03283385699339, -1.99984225624759, -1.96725376224613,
        -1.93506512412816, -1.90327309103284, -1.87187441209932, -1.84086583646678, -1.81024411327436,
        -1.78000599166123, -1.75014822076655, -1.72066754972946, -1.69156072768914, -1.66282450378474,
        -1.63445562715541, -1.60645084694032, -1.57880691227863, -1.55152057230949, -1.52458857617206,
        -1.49800767300551, -1.47177461194898, -1.44588614214165, -1.42033901272266, -1.39512997283118,
        -1.37025577160636, -1.34571315818736, -1.32149888171335, -1.29760969132347, -1.27404233615690,
        -1.25079356535278, -1.22786012805028, -1.20523877338855, -1.18292625050675, -1.16091930854405,
        -1.13921469663960, -1.11780916393255, -1.09669945956208, -1.07588233266733, -1.05535453238746,
        -1.03511280786164, -1.01515390822902, -0.995474582628758, -0.976071580200017, -0.956941650081954,
        -0.938081541413728, -0.919488003334496, -0.901157784983417, -0.883087635499651, -0.865274304022356,
        -0.847714539690689, -0.830405091643811, -0.813342709020879, -0.796524140961053, -0.779946136603490,
        -0.763605445087349, -0.747498815551789, -0.731622997135969, -0.715974738979046, -0.700550790220181,
        -0.685347899998530, -0.670362817453254, -0.655592291723510, -0.641033071948457, -0.626681907267254,
        -0.612535546819059, -0.598590739743031, -0.584844235178328, -0.571292782264110, -0.557933130139534,
        -0.544762027943759, -0.531776224815945, -0.518972469895249, -0.506347512320829, -0.493898101231846,
        -0.481620985767457, -0.469512915066820, -0.457570638269096, -0.445790904513441, -0.434170462939015,
        -0.422706062684976, -0.411394452890483, -0.400232382694694, -0.389216601236769, -0.378343857655865,
        -0.367610901091141, -0.357014480681756, -0.346551345566869, -0.336218244885638, -0.326011927777221,
        -0.315929143380778, -0.305966640835466, -0.296121169280445, -0.286389477854873, -0.276768315697909,
        -0.267254431948711, -0.257844575746437, -0.248535496230247, -0.239323942539300, -0.230206663812752,
        -0.221180409189764, -0.212241927809494, -0.203387968811101, -0.194615281333742, -0.185920614516577,
        -0.177300717498764, -0.168752339419462, -0.160272229417829, -0.151857136633025, -0.143503810204207,
        -0.135208999270534, -0.126969452971165, -0.118781920445259, -0.110643150831974, -0.102549893270468,
        -0.0944988968999002, -0.0864869108594294, -0.0785106842882140, -0.0705669663254127, -0.0626525061101840,
        -0.0547640527816864, -0.0468983554790787, -0.0390521633415194, -0.0312222255081671, -0.0234052911181804,
        -0.0155981093107179, -0.00779742922493818, 0.00, 0.00779742922493841, 0.0155981093107181, 0.0234052911181806,
        0.0312222255081673, 0.0390521633415196, 0.0468983554790789, 0.0547640527816867, 0.0626525061101842,
        0.0705669663254129, 0.0785106842882143, 0.0864869108594296, 0.0944988968999004, 0.102549893270468,
        0.110643150831974, 0.118781920445259, 0.126969452971166, 0.135208999270535, 0.143503810204207,
        0.151857136633025, 0.160272229417830, 0.168752339419462, 0.177300717498764, 0.185920614516577,
        0.194615281333742, 0.203387968811101, 0.212241927809494, 0.221180409189765, 0.230206663812752,
        0.239323942539300, 0.248535496230247, 0.257844575746437, 0.267254431948711, 0.276768315697909,
        0.286389477854873, 0.296121169280445, 0.305966640835466, 0.315929143380778, 0.326011927777221,
        0.336218244885638, 0.346551345566869, 0.357014480681756, 0.367610901091141, 0.378343857655865,
        0.389216601236769, 0.400232382694694, 0.411394452890483, 0.422706062684976, 0.434170462939015,
        0.445790904513441, 0.457570638269096, 0.469512915066821, 0.481620985767457, 0.493898101231846,
        0.506347512320829, 0.518972469895249, 0.531776224815945, 0.544762027943760, 0.557933130139534,
        0.571292782264110, 0.584844235178328, 0.598590739743031, 0.612535546819059, 0.626681907267254,
        0.641033071948457, 0.655592291723510, 0.670362817453254, 0.685347899998530, 0.700550790220181,
        0.715974738979046, 0.731622997135969, 0.747498815551789, 0.763605445087349, 0.779946136603489,
        0.796524140961053, 0.813342709020879, 0.830405091643811, 0.847714539690689, 0.865274304022356,
        0.883087635499651, 0.901157784983417, 0.919488003334496, 0.938081541413728, 0.956941650081954,
        0.976071580200017, 0.995474582628758, 1.01515390822902, 1.03511280786164, 1.05535453238746, 1.07588233266733,
        1.09669945956208, 1.11780916393255, 1.13921469663960, 1.16091930854405, 1.18292625050675, 1.20523877338855,
        1.22786012805028, 1.25079356535278, 1.27404233615690, 1.29760969132347, 1.32149888171335, 1.34571315818736,
        1.37025577160636, 1.39512997283118, 1.42033901272266, 1.44588614214165, 1.47177461194898, 1.49800767300551,
        1.52458857617206, 1.55152057230949, 1.57880691227863, 1.60645084694032, 1.63445562715541, 1.66282450378474,
        1.69156072768914, 1.72066754972946, 1.75014822076655, 1.78000599166123, 1.81024411327436, 1.84086583646678,
        1.87187441209932, 1.90327309103284, 1.93506512412816, 1.96725376224613, 1.99984225624759, 2.03283385699339,
        2.06623181534436, 2.10003938216135
};


void initializeM256IConstants() {
    M256I_1 = _mm256_set1_epi32(1);

    M256I_OFFSETS_BY_SEGMENTS = (__m256i *) OFFSETS_BY_SEGMENTS;

    __m256i m256i_cardinality8_offsets = _mm256_i32gather_epi32(OFFSETS_BY_MASK, M256I_1, 4);

    M256I_BREAKPOINTS8_OFFSETS_0_7 = _mm256_add_epi32(_mm256_load_si256(M256I_OFFSETS_BY_SEGMENTS),
                                                      m256i_cardinality8_offsets);
    M256I_BREAKPOINTS8_OFFSETS_8_15 = _mm256_add_epi32(_mm256_load_si256(M256I_OFFSETS_BY_SEGMENTS + 1),
                                                       m256i_cardinality8_offsets);

    M256I_BREAKPOINTS_OFFSETS_0_7 = _mm256_load_si256(M256I_OFFSETS_BY_SEGMENTS);
    M256I_BREAKPOINTS_OFFSETS_8_15 = _mm256_load_si256(M256I_OFFSETS_BY_SEGMENTS + 1);
}


typedef struct BreakpointsCache {
    Value const *summarizations;
    size_t size;

    Value *breakpoints;

    unsigned int *shared_processed_counter;
    unsigned int num_segments;
} BreakpointsCache;


void extractBreakpoints8(Value *breakpoints, Value const *values, unsigned int length) {
    Value *current_breakpoint;
    Value const *current_value;

    for (unsigned int i = 0; i < 8; ++i) {
        breakpoints[OFFSETS_BY_CARDINALITY[i]] = VALUE_MIN;
        breakpoints[OFFSETS_BY_CARDINALITY[i + 1] - 1] = VALUE_MAX;

        unsigned int range_length = 1 + (length >> (i + 1));
        for (current_breakpoint = breakpoints + OFFSETS_BY_CARDINALITY[i] + 1, current_value = values + range_length;
             current_breakpoint < breakpoints + OFFSETS_BY_CARDINALITY[i + 1] - 1;
             current_breakpoint += 1, current_value += range_length) {
            *current_breakpoint = *current_value;
        }
    }
}


void *getAdhocBreakpoints8Thread(void *cache) {
    BreakpointsCache *breakpointsCache = (BreakpointsCache *) cache;
    unsigned int num_segments = breakpointsCache->num_segments;
    size_t size = breakpointsCache->size;

    Value *values_per_segment = malloc(sizeof(Value) * size), *current_value;
    Value const *current_summarization;

    unsigned int current_segment;
    while ((current_segment = __sync_fetch_and_add(breakpointsCache->shared_processed_counter, 1)) < num_segments) {
        for (current_summarization = breakpointsCache->summarizations + current_segment,
                     current_value = values_per_segment;
             current_summarization < breakpointsCache->summarizations + num_segments * size;
             current_summarization += num_segments, current_value += 1) {
            *current_value = *current_summarization;
        }

        qsort(values_per_segment, size, sizeof(Value), VALUE_COMPARE);

        extractBreakpoints8(breakpointsCache->breakpoints + OFFSETS_BY_SEGMENTS[current_segment],
                            values_per_segment, size);
    }

    free(values_per_segment);
    return NULL;
}


void profileBreakpoints(Value const *breakpoints, unsigned int num_segments) {
    for (unsigned int i = 0; i < num_segments; ++i) {
        Value sum = 0, sum_square = 0;
        for (unsigned int j = OFFSETS_BY_SEGMENTS[i] + OFFSETS_BY_CARDINALITY[7] + 1;
             j < OFFSETS_BY_SEGMENTS[i] + OFFSETS_BY_CARDINALITY[8] - 1;
             ++j) {
            sum += breakpoints[j];
            sum_square += breakpoints[j] * breakpoints[j];
        }

        sum /= 255;
        clog_info(CLOG(CLOGGER_ID), "index - mean / variance of %d = %f / %f", i, sum,
                  sum_square / 255 - sum * sum);
    }
}


Value const *getAdhocBreakpoints8(Value const *summarizations, size_t size, unsigned int num_segments,
                                  unsigned int num_threads) {
    Value *breakpoints = aligned_alloc(64, sizeof(Value) * OFFSETS_BY_SEGMENTS[num_segments]);

    if (num_threads > num_segments) {
        num_threads = num_segments;
    }

    pthread_t threads[num_threads];
    BreakpointsCache breakpointsCache[num_threads];

    for (unsigned int shared_processed_counter = 0, i = 0; i < num_threads; ++i) {
        breakpointsCache[i].breakpoints = breakpoints;
        breakpointsCache[i].summarizations = summarizations;
        breakpointsCache[i].size = size;
        breakpointsCache[i].shared_processed_counter = &shared_processed_counter;
        breakpointsCache[i].num_segments = num_segments;

        pthread_create(&threads[i], NULL, getAdhocBreakpoints8Thread, (void *) &breakpointsCache[i]);
    }

    for (unsigned int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

#ifdef FINE_PROFILING
    profileBreakpoints(breakpoints, num_segments);
#endif

    return breakpoints;
}


Value const *getSharedAdhocBreakpoints8(Value const *summarizations, size_t size, unsigned int num_segments) {
    Value *breakpoints = aligned_alloc(64, sizeof(Value) * OFFSETS_BY_SEGMENTS[num_segments]);

    Value *values = malloc(sizeof(Value) * size * num_segments);
    memcpy(values, summarizations, sizeof(Value) * size * num_segments);
    qsort(values, size, sizeof(Value), VALUE_COMPARE);

    for (unsigned int i = 0; i < num_segments; ++i) {
        extractBreakpoints8(breakpoints + OFFSETS_BY_SEGMENTS[i], values, size * num_segments);
    }

    free(values);

#ifdef PROFILING
    profileBreakpoints(breakpoints, num_segments);
#endif

    return breakpoints;
}


Value const *getNormalBreakpoints8(unsigned int num_segments) {
    Value *breakpoints = aligned_alloc(256, sizeof(Value) * OFFSETS_BY_SEGMENTS[num_segments]);

    extractBreakpoints8(breakpoints, NORMAL_BREAKPOINTS_8, 255);
    for (unsigned int i = 1; i < num_segments; ++i) {
        memcpy(breakpoints + OFFSETS_BY_SEGMENTS[i], breakpoints, sizeof(Value) * OFFSETS_BY_CARDINALITY[8]);
    }

#ifdef PROFILING
    profileBreakpoints(breakpoints, num_segments);
#endif

    return breakpoints;
}
