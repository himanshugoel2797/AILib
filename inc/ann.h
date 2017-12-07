// Copyright (c) 2017 Himanshu Goel
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AILIB_ANN_H
#define AILIB_ANN_H

#include "mat.h"

typedef struct ann ann_t;
struct ann {
    int layers;
    int input_count;
    int output_count;
    int max_h;
    int *layer_sizes;
    mat_t *weights;
};

ann_t ann_create(int, int*, int, int);
int ann_activate(ann_t, float*, float*);
void ann_setseed(unsigned int);

#endif