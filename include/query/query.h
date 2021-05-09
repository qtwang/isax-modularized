//
// Created by Qitong Wang on 2020/7/3.
//

#ifndef ISAX_QUERY_H
#define ISAX_QUERY_H

#include <time.h>
#include <stdlib.h>

#include "globals.h"
#include "config.h"
#include "paa.h"
#include "breakpoints.h"
#include "sax.h"
#include "index.h"
#include "clog.h"


typedef struct QuerySet {
    Value const *values;
    Value const *summarizations;
    Value const *initial_bsf_distances;
    SAXWord const *saxs;

    unsigned int query_size;
} QuerySet;


QuerySet *initializeQuery(Config const *config, Index const *index);

void freeQuery(QuerySet *queries);


#endif //ISAX_QUERY_H
