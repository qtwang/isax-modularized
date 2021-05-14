cmake_minimum_required(VERSION 3.16)

add_library(utils distance.c sort.c config.c str.c ../../include/utils/str.h)

# -mavx -mavx2 -mavx512 -msse4.2 are supported by Intel(R) Xeon(R) Gold 6134 CPU @ 3.20GHz in andromache
target_compile_options(utils PUBLIC -march=native -msse4.2 -mavx)