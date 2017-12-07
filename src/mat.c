/**
 * Copyright (c) 2017 Himanshu Goel
 * 
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */

#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
#include "mat.h"

mat_t mat_create(int width, int height) {
    mat_t nmat;

    nmat.width = width;
    nmat.height = height;
    nmat.stride = height;
    if(nmat.stride % 8 != 0)
        nmat.stride += (8 - nmat.stride % 8);

    size_t alloc_sz = width * nmat.stride * sizeof(float);
    
    if(alloc_sz % 32 != 0)
        alloc_sz += (32 - alloc_sz % 32);

    nmat.alloc_sz = alloc_sz;
    nmat.data = aligned_alloc(32, alloc_sz);
    memset(nmat.data, 0, alloc_sz);

    return nmat;
}

void mat_delete(mat_t mat) {
    free(mat.data);
}

void mat_set(mat_t mat, int x, int y, float val) {
    mat.data[mat.stride * x + y] = val;
}

float mat_get(mat_t mat, int x, int y) {
    return mat.data[mat.stride * x + y];
}

void mat_clear(mat_t mat) {
    memset(mat.data, 0, mat.alloc_sz);
}

int mat_mult(mat_t a, mat_t b, mat_t *c) {
    if(a.width != b.height)
        return -1;

    if(a.width == 1 && a.height == 1){
        mat_set(*c, 0, 0, mat_get(a, 0, 0) * mat_get(b, 0, 0));
        return 0;
    }

    if(b.width == 1) {  //Vector and matrix multiplication

        for(int j = 0; j < a.stride; j+= 8){

            float *src = &a.data[j];
            float *src_b = b.data;
            const int stride = a.stride;

            __m256 mat_prev = _mm256_setzero_ps();
            __m256 mat_prev1 = _mm256_setzero_ps();
            __m256 mat_prev2 = _mm256_setzero_ps();
            __m256 mat_prev3 = _mm256_setzero_ps();

            int i = 0;

            for(; i + 4 < a.width; i+=4){
                mat_prev = _mm256_fmadd_ps(_mm256_load_ps(src), _mm256_set1_ps(*src_b), mat_prev);
                mat_prev1 = _mm256_fmadd_ps(_mm256_load_ps(src + stride), _mm256_set1_ps(*(src_b + 1)), mat_prev1);
                mat_prev2 = _mm256_fmadd_ps(_mm256_load_ps(src + 2 * stride), _mm256_set1_ps(*(src_b + 2)), mat_prev2);
                mat_prev3 = _mm256_fmadd_ps(_mm256_load_ps(src + 3 * stride), _mm256_set1_ps(*(src_b + 3)), mat_prev3);

                src += 4 * stride;
                src_b += 4;
            }

            mat_prev = _mm256_add_ps(mat_prev, mat_prev1);
            mat_prev2 = _mm256_add_ps(mat_prev2, mat_prev3);

            switch(a.width - (i + 1)) {
                case 3: mat_prev = _mm256_fmadd_ps(_mm256_load_ps(src), _mm256_set1_ps(*src_b), mat_prev);
                case 2: mat_prev2 = _mm256_fmadd_ps(_mm256_load_ps(src + stride), _mm256_set1_ps(*(src_b + 1)), mat_prev2);
                case 1: mat_prev = _mm256_fmadd_ps(_mm256_load_ps(src + 2 * stride), _mm256_set1_ps(*(src_b + 2)), mat_prev);
                case 0: ;
            }

            _mm256_store_ps(&c->data[j], _mm256_add_ps(mat_prev, mat_prev2));
        }
        return 0;
    }

    return -1;
}