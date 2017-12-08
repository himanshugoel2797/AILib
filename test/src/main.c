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

int main(){
    
    ann_setseed(0);


    float inputs[4][2] = {{0, 0}, {0.0, 0.25}, {0.5, 0}, {0.5, 0.5}};
    float outputs[4][1] = {{0}, {0.25}, {0.5}, {1}};

    int layers[] = {2, 1};
    ann_t net = ann_create(2, layers, 0.05);

    for(int k = 0; k < 5; k++){
        for(int i = 0; i < 4; i++){
            float in[2] = {0.5, 0.25};
            float res[1];
            if(ann_activate(net, in, res) != 0)
                printf("ERROR\r\n");
            
            printf("RESULT0:%f\r\n", res[0] );
        }

        for(int i = 0; i < 1000000; i++){
            if(ann_train(net, inputs[i % 4], outputs[i % 4]) != 0)
                printf("ERROR\r\n");
        }

        printf("\r\n\r\n");

    }
        ann_delete(net);

/*
    mat_t a = mat_create(1, 2);
    mat_t b = mat_create(2, 2);
    mat_t c = mat_create(1, 2);

    mat_set(a, 0, 0, 1);
    mat_set(a, 0, 1, 2);

    mat_set(b, 0, 0, 1);
    mat_set(b, 1, 0, 2);
    mat_set(b, 0, 1, 3);
    mat_set(b, 1, 1, 4);

    mat_mult(b, a, &c);
    printf("RESULT: %f\r\n", mat_get(c, 0, 0));
    printf("RESULT: %f\r\n", mat_get(c, 0, 1));*/

    return 0;
}