// Copyright (c) 2017 Himanshu Goel
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AILIB_MAT_H
#define AILIB_MAT_H

typedef struct mat mat_t;
struct mat{
    int width;
    int height;
    int stride;
    float *data;
};

mat_t mat_create(int, int);
void mat_set(mat_t, int, int, float);
float mat_get(mat_t, int, int);
int mat_mult(mat_t, mat_t, mat_t*);

#endif