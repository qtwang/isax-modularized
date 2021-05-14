//
// Created by Qitong Wang on 2020/7/3.
//

#ifndef ISAX_QUERY_ENGINE_H
#define ISAX_QUERY_ENGINE_H

#include <time.h>
#include <math.h>

#include "globals.h"
#include "config.h"
#include "index.h"
#include "distance.h"
#include "query.h"
#include "clog.h"
#include "answer.h"
#include "sort.h"
#include "str.h"


void conductQueries(QuerySet const *querySet, Index const *index, Config const *config);

#endif //ISAX_QUERY_ENGINE_H
