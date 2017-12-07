/**
 * Copyright (c) 2017 Himanshu Goel
 * 
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */

#include "mat.h"
#include "ann.h"

#include <stdio.h>
#include <time.h>

int mat_mult_softsign(mat_t a, mat_t b, mat_t *c);

int main(){

    int layers[] = {1};
    ann_t net = ann_create(1, layers, 1, 1);

    float inputs[] = {5};
    float outputs[1];
    if(ann_activate(net, inputs, outputs) != 0)
        printf("ERROR\r\n");

    printf("RESULT:%f\r\n", outputs[0] );

    return 0;
}