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

    //matrix multiplication
    for(int q = 0; q < b.width; q++){
        for(int j = 0; j < a.stride; j+= 8){

            float *src = &a.data[j];
            float *src_b = &b.data[b.stride * q];
            const int stride = a.stride;

            __m256 mat_prev = _mm256_setzero_ps();
            __m256 mat_prev1 = _mm256_setzero_ps();
            __m256 mat_prev2 = _mm256_setzero_ps();
            __m256 mat_prev3 = _mm256_setzero_ps();

            int repeat = a.width / 4;
            int left = a.width % 4;

            while(repeat--){
                mat_prev = _mm256_fmadd_ps(_mm256_load_ps(src), _mm256_set1_ps(*src_b), mat_prev);
                mat_prev1 = _mm256_fmadd_ps(_mm256_load_ps(src + stride), _mm256_set1_ps(*(src_b + 1)), mat_prev1);
                mat_prev2 = _mm256_fmadd_ps(_mm256_load_ps(src + 2 * stride), _mm256_set1_ps(*(src_b + 2)), mat_prev2);
                mat_prev3 = _mm256_fmadd_ps(_mm256_load_ps(src + 3 * stride), _mm256_set1_ps(*(src_b + 3)), mat_prev3);

                src += 4 * stride;
                src_b += 4;
            }

            mat_prev = _mm256_add_ps(mat_prev, mat_prev1);
            mat_prev2 = _mm256_add_ps(mat_prev2, mat_prev3);

            switch(left) {
                case 3: mat_prev = _mm256_fmadd_ps(_mm256_load_ps(src + 2 * stride), _mm256_set1_ps(*(src_b + 2)), mat_prev);
                case 2: mat_prev2 = _mm256_fmadd_ps(_mm256_load_ps(src + stride), _mm256_set1_ps(*(src_b + 1)), mat_prev2);
                case 1: mat_prev = _mm256_fmadd_ps(_mm256_load_ps(src), _mm256_set1_ps(*src_b), mat_prev);
                case 0: ;
            }

            _mm256_store_ps(&c->data[j], _mm256_add_ps(mat_prev, mat_prev2));
        }
        return 0;
    }

    return -1;
}

int mat_multadd(mat_t a, mat_t b, mat_t d, mat_t *c) {
    if(a.width != b.height)
        return -1;

    if(a.width == 1 && a.height == 1){
        mat_set(*c, 0, 0, mat_get(a, 0, 0) * mat_get(b, 0, 0));
        return 0;
    }

    if(b.width == 1) {  //Vector and matrix multiplication

        for(int j = 0; j < a.stride; j+= 8){

            float *src = &a.data[j];
            float *src_d = &d.data[j];
            float *src_b = b.data;
            const int stride = a.stride;

            __m256 mat_prev = _mm256_load_ps(src_d);
            __m256 mat_prev1 = _mm256_setzero_ps();
            __m256 mat_prev2 = _mm256_setzero_ps();
            __m256 mat_prev3 = _mm256_setzero_ps();

            int repeat = a.width / 4;
            int left = a.width % 4;

            while(repeat--){
                mat_prev = _mm256_fmadd_ps(_mm256_load_ps(src), _mm256_set1_ps(*src_b), mat_prev);
                mat_prev1 = _mm256_fmadd_ps(_mm256_load_ps(src + stride), _mm256_set1_ps(*(src_b + 1)), mat_prev1);
                mat_prev2 = _mm256_fmadd_ps(_mm256_load_ps(src + 2 * stride), _mm256_set1_ps(*(src_b + 2)), mat_prev2);
                mat_prev3 = _mm256_fmadd_ps(_mm256_load_ps(src + 3 * stride), _mm256_set1_ps(*(src_b + 3)), mat_prev3);

                src += 4 * stride;
                src_b += 4;
            }

            mat_prev = _mm256_add_ps(mat_prev, mat_prev1);
            mat_prev2 = _mm256_add_ps(mat_prev2, mat_prev3);

            switch(left) {
                case 3: mat_prev = _mm256_fmadd_ps(_mm256_load_ps(src + 2 * stride), _mm256_set1_ps(*(src_b + 2)), mat_prev);
                case 2: mat_prev2 = _mm256_fmadd_ps(_mm256_load_ps(src + stride), _mm256_set1_ps(*(src_b + 1)), mat_prev2);
                case 1: mat_prev = _mm256_fmadd_ps(_mm256_load_ps(src), _mm256_set1_ps(*src_b), mat_prev);
                case 0: ;
            }

            _mm256_store_ps(&c->data[j], _mm256_add_ps(mat_prev, mat_prev2));
        }
        return 0;
    }

    return -1;
}


int mat_transpose(mat_t a, mat_t *c) {
    if(a.width != c->height)
        return -1;

    if(a.height != c->width)
        return -1;

    for(int x = 0; x < a.width; x++)
        for(int y = 0; y < a.height; y++)
            mat_set(*c, y, x, mat_get(a, x, y));

    return 0;
}

int mat_subscalar(mat_t a, float v, mat_t *c) {
    __m256 sub = _mm256_set1_ps(v);
    for(int i = 0; i < a.stride; i+=8) {

        float *src_a = &a.data[i];
        float *dst_c = &c->data[i];

        int repeat = a.width / 4;
        int left = a.width % 4;
        int stride = a.stride;

        while(repeat--) {
            _mm256_store_ps(dst_c, _mm256_sub_ps(_mm256_load_ps(src_a), sub));
            _mm256_store_ps(dst_c + stride, _mm256_sub_ps(_mm256_load_ps(src_a + stride), sub));
            _mm256_store_ps(dst_c + 2 * stride, _mm256_sub_ps(_mm256_load_ps(src_a + 2 * stride), sub));
            _mm256_store_ps(dst_c + 3 * stride, _mm256_sub_ps(_mm256_load_ps(src_a + 3 * stride), sub));

            dst_c += 4 * stride;
            src_a += 4 * stride;
        }

        switch(left) {
            case 3:_mm256_store_ps(dst_c + 2 * stride, _mm256_sub_ps(_mm256_load_ps(src_a + 2 * stride), sub));
            case 2:_mm256_store_ps(dst_c + stride, _mm256_sub_ps(_mm256_load_ps(src_a + stride), sub));
            case 1:_mm256_store_ps(dst_c, _mm256_sub_ps(_mm256_load_ps(src_a), sub));
            case 0: ;
        }

        return 0;
    }
    return -1;
}

int mat_hadamard(mat_t a, mat_t b, mat_t *c) {
    for(int i = 0; i < a.stride; i+=8) {
        __m256 a_v = _mm256_load_ps(&a.data[i]);
        __m256 b_v = _mm256_load_ps(&b.data[i]);

        //multiply the two
        __m256 res = _mm256_mul_ps(a_v, b_v);
        _mm256_store_ps(&c->data[i], res);
    }
    return 0;
}