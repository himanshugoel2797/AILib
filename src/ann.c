/**
 * Copyright (c) 2017 Himanshu Goel
 * 
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */

#include "mat.h"
#include <x86intrin.h>


int mat_mult_softsign(mat_t a, mat_t b, mat_t *c) {
    if(a.width != b.height)
        return -1;

    if(b.width == 1) {  //Vector and matrix multiplication
        mat_t d = mat_create(1, a.height);

        __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        __m256 one = _mm256_set1_ps(1);

        for(int j = 0; j < a.height; j+= 8){

            __m256 mat_prev = _mm256_mul_ps(_mm256_load_ps(&a.data[j]), _mm256_set1_ps(b.data[0]));
            for(int i = 1; i < a.width; i++)
                mat_prev = _mm256_fmadd_ps(_mm256_load_ps(&a.data[a.stride * i + j]), _mm256_set1_ps(b.data[i]), mat_prev);

            //mask away the sign bit
            //add one
            //take the reciprocal
            //multiply by the original

            mat_prev = _mm256_mul_ps(_mm256_rcp_ps(_mm256_add_ps(_mm256_and_ps(mat_prev, mask), one)), mat_prev);
            _mm256_store_ps(&d.data[j], mat_prev);
        }
        *c = d;
        return 0;
    }

    return -1;
}