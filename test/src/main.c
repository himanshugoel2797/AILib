/**
 * Copyright (c) 2017 Himanshu Goel
 * 
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */

#include "mat.h"

#include <stdio.h>
#include <time.h>

int main(){

    mat_t a = mat_create(20, 20);
    mat_set(a, 0, 0, 1);
    mat_set(a, 1, 0, 2);
    mat_set(a, 0, 1, 3);
    mat_set(a, 1, 1, 4);

    mat_t b = mat_create(1, 20);
    mat_set(b, 0, 0, 1);
    mat_set(b, 0, 1, 2);

    mat_t res = mat_create(1, 20);

    struct timespec start_time;
    clock_gettime(CLOCK_REALTIME, &start_time);

    for(long i = 0; i < 500000000; i++){
        mat_mult(a, b, &res);
    }

    struct timespec end_time;
    clock_gettime(CLOCK_REALTIME, &end_time);

    printf("Seconds: %ld\r\n", end_time.tv_sec - start_time.tv_sec);

    printf("[0][0] = %f\r\n", mat_get(res, 0, 0));
    printf("[1][0] = %f\r\n", mat_get(res, 1, 0));
    printf("[0][1] = %f\r\n", mat_get(res, 0, 1));
    printf("[1][1] = %f\r\n", mat_get(res, 1, 1));

    return 0;
}