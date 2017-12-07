/**
 * Copyright (c) 2017 Himanshu Goel
 * 
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */

#include "mat.h"
#include <x86intrin.h>

ann_t ann_create(int layers, int *layer_sizes, int input_cnt, int output_cnt) {
    ann_t ann;
    ann.layers = layers;
    ann.layer_sizes = malloc(layers * sizeof(int));
    ann.weights = malloc(layers * sizeof(mat_t));
    memcpy(ann.layer_sizes, layer_sizes, layers * sizeof(int));
    ann.input_count = input_cnt;
    ann.output_count = output_cnt;

    int max_h = 0;
    for(int i = 0; i < layers; i++) {
        if(layer_sizes[i] > max_h)
            layer_sizes[i] = max_h;
    }
    ann.max_h = max_h;

    for(int i = 0; i < layers; i++) {
        ann.weights[i] = mat_create(max_h, max_h);
    }
}

int mat_mult_softsign(mat_t a, mat_t b, mat_t *c) {
    if(a.width != b.height)
        return -1;

    if(a.width == 1 && a.height == 1){
        mat_set(*c, 0, 0, mat_get(a, 0, 0) * mat_get(b, 0, 0));
        return 0;
    }

    if(b.width == 1) {  //Vector and matrix multiplication

        __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        __m256 one = _mm256_set1_ps(1);

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

            __m256 mat_net = _mm256_add_ps(mat_prev, mat_prev2);
            mat_net = _mm256_mul_ps(_mm256_rcp_ps(_mm256_add_ps(_mm256_and_ps(mat_net, mask), one)), mat_net);
            _mm256_store_ps(&c->data[j], mat_net);
        }
        return 0;
    }

    return -1;
}

int ann_activate(ann_t ann, float* inputs, float* outputs){
    mat_t res = mat_create(1, max_h);
    for(int i = 0; i < ann.input_count; i++) {
        mat_set(res, 0, i, inputs[i]);
    }

    mat_t res2 = mat_create(1, max_h);
    for(int i = 0; i < ann.layers; i++) {
        mat_clear(res2);
        if(mat_mult_softsign(ann.weights[i], res, &res2) != 0)
            return -1;

        mat_t tmp = res;
        res = res2;
        res2 = tmp;
    }

    for(int i = 0; i < ann.output_count; i++) {
        outputs[i] = mat_get(res, 0, i);
    }

    return 0;
}