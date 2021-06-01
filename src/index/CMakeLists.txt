cmake_minimum_required(VERSION 3.16)

add_library(index index_engine.c index.c node.c ../../include/index/multindex.h multindex.c)
