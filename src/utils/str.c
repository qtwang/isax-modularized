//
// Created by Qitong Wang on 2021/5/14.
//

#include "str.h"


char *char2bin(SAXWord symbol) {
    size_t bits = sizeof(SAXWord) * CHAR_BIT;

    char *str = malloc(bits + 1);
    if (!str) {
        return NULL;
    }
    str[bits] = 0;

    // type punning because signed shift is implementation-defined
    for (unsigned u = *(unsigned *) &symbol; bits--; u >>= 1) {
        str[bits] = u & 1 ? '1' : '0';
    }

    return str;
}
