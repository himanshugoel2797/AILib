/**
 * Copyright (c) 2017 Himanshu Goel
 * 
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */

#include "ann.h"
#include "mat.h"
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <x86intrin.h>

static unsigned int seed = 0;
#define CORNER_CNT 8
static float corners[CORNER_CNT];
static int prev_idx = 0;

static unsigned int get_rand(){
    seed = (seed * 1664525 + 1013904223);
    return seed;
}

static float ann_rand(){
    prev_idx = (prev_idx + 1) % CORNER_CNT;
    corners[prev_idx] = fmod(get_rand(), 1024) / 1024;

    float avg = 0;
    for(int i = 0; i < CORNER_CNT; i++)
        avg += corners[i];

    return avg / CORNER_CNT;
}

void ann_setseed(unsigned int s) {
    seed = s;
    for(int i = 0; i < CORNER_CNT; i++)
        ann_rand();
}

ann_t ann_create(int layers, int *layer_sizes, int input_cnt) {
    ann_t ann;
    ann.layers = layers;
    ann.layer_sizes = malloc(layers * sizeof(int));
    ann.weights = malloc(layers * sizeof(mat_t));
    memcpy(ann.layer_sizes, layer_sizes, layers * sizeof(int));
    ann.input_count = input_cnt;
    ann.output_count = layer_sizes[layers - 1];

    int w = input_cnt;

    for(int i = 0; i < layers; i++) {
        int h = layer_sizes[i];

        ann.weights[i] = mat_create(w, h);
        for(int x = 0; x < w; x++)
            for(int y = 0; y < h; y++){
                float val = ann_rand();
                mat_set(ann.weights[i], x, y, val);
            }

        w = h;
    }

    return ann;
}

void ann_randomizelayer(ann_t ann, int layer) {
    int w = ann.input_count;

    for(int i = 0; i < ann.layers; i++) {
        int h = ann.layer_sizes[i];

        if(i == layer){
            for(int x = 0; x < w; x++)
                for(int y = 0; y < h; y++){
                    float val = ann_rand();
                    mat_set(ann.weights[i], x, y, val);
                }
        }

        w = h;
    }
}

mat_t ann_getlayer(ann_t ann, int layer) {
    return ann.weights[layer];
}

void ann_setlayer(ann_t ann, int layer, mat_t val) {
    for(int x = 0; x < ann.max_h; x++)
        for(int y = 0; y < ann.max_h; y++)
            mat_set(ann.weights[layer], x, y, mat_get(val, x, y));
}

void ann_delete(ann_t ann) {
    free(ann.layer_sizes);

    for(int i = 0; i < ann.layers; i++)
        mat_delete(ann.weights[i]);

    free(ann.weights);
}

static void ann_softsign(mat_t a, mat_t *c) {

    __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 one = _mm256_set1_ps(1);

    for(int i = 0; i < a.stride; i+=8) {
        __m256 mat_net = _mm256_load_ps(&a.data[i]);
        mat_net = _mm256_mul_ps(_mm256_rcp_ps(_mm256_add_ps(_mm256_and_ps(mat_net, mask), one)), mat_net);
        _mm256_store_ps(&c->data[i], mat_net);
    }
}

static void ann_output_error(mat_t expected, mat_t output, mat_t *c) {

    __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 one = _mm256_set1_ps(1);

    for(int i = 0; i < expected.stride; i+=8) {
        __m256 expected_v = _mm256_load_ps(&expected.data[i]);
        __m256 output_v = _mm256_load_ps(&output.data[i]);

        //take the difference and put it in one register
        __m256 diff = _mm256_sub_ps(output_v, expected_v);

        //compute the softsign_deriv for the output
        __m256 net = _mm256_add_ps(_mm256_and_ps(output_v, mask), one);
        __m256 softsign_deriv = _mm256_rcp_ps(_mm256_mul_ps(net, net));

        //multiply the two
        __m256 res = _mm256_mul_ps(diff, softsign_deriv);
        _mm256_store_ps(&c->data[i], res);
    }
} 

static void ann_hadamard(mat_t a, mat_t z, mat_t *c) {

    __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 one = _mm256_set1_ps(1);

    for(int i = 0; i < a.stride; i+=8) {
        __m256 a_v = _mm256_load_ps(&a.data[i]);
        __m256 z_v = _mm256_load_ps(&z.data[i]);

        //compute the softsign_deriv for the output
        __m256 net = _mm256_add_ps(_mm256_and_ps(z_v, mask), one);
        __m256 softsign_deriv = _mm256_rcp_ps(_mm256_mul_ps(net, net));

        //multiply the two
        __m256 res = _mm256_mul_ps(a_v, softsign_deriv);
        _mm256_store_ps(&c->data[i], res);
    }
}

int ann_activate(ann_t ann, float* inputs, float* outputs){
    mat_t res = mat_create(1, ann.input_count);
    for(int i = 0; i < ann.input_count; i++) {
        mat_set(res, 0, i, inputs[i]);
    }

    for(int i = 0; i < ann.layers; i++) {
        mat_t res2 = mat_create(1, ann.layer_sizes[i]);
        if(mat_mult(ann.weights[i], res, &res2) != 0)
            return -1;

        ann_softsign(res2, &res2);

        mat_delete(res);
        res = res2;
    }

    for(int i = 0; i < ann.output_count; i++) {
        outputs[i] = mat_get(res, 0, i);
    }

    return 0;
}

float trans_deriv(float o) {
    return 1 / powf(1 + fabs(o), 2); 
}

int ann_train(ann_t ann, float* input, float *expected_outputs) {

    mat_t *errors = malloc((ann.layers + 1) * sizeof(mat_t));
    mat_t *z = malloc((ann.layers + 1) * sizeof(mat_t));
    mat_t *a = malloc((ann.layers + 1) * sizeof(mat_t));

    mat_t expect_output_vec = mat_create(1, ann.output_count);
    for(int i = 0; i < ann.output_count; i++)
        mat_set(expect_output_vec, 0, i, expected_outputs[i]);


    //Setup output vector
    z[ann.layers] = mat_create(1, ann.layer_sizes[ann.layers - 1]);

    //Setup input vector
    a[0] = mat_create(1, ann.input_count);
    for(int i = 0; i < ann.input_count; i++)
        mat_set(a[0], 0, i, input[i]);

    for(int i = 0; i < ann.layers; i++) {
        z[i] = mat_create(1, ann.layer_sizes[i]);
        a[i] = mat_create(1, ann.layer_sizes[i]);

        if(i + 1 < ann.layers)
            z[i + 1] = mat_create(1, ann.layer_sizes[i + 1]);

        errors[i] = mat_create(1, ann.layer_sizes[i]);

        if(mat_mult(ann.weights[i], a[i], &z[i + 1]) != 0)
            return -1;
        ann_softsign(z[i + 1], &a[i + 1]);
    }

    //compute the output error
    //(output - expected_output) hadamard trans_deriv(output)
    ann_output_error(expect_output_vec, a[ann.layers], &errors[ann.layers]);

    //backpropagate the error
    for(int i = ann.layers - 1; i > 0; i--){
        mat_t w_trans = mat_create(ann.weights[i + 1].height, ann.weights[i + 1].width);
        mat_transpose(ann.weights[i + 1], &w_trans);

        if(mat_mult(w_trans, errors[i + 1], &errors[i]) != 0){
            printf("ERROR");
            return -1;
        }

        ann_hadamard(errors[i], z[i], &errors[i]);
    }
    
    for(int i = ann.layers - 1; i > 0; i--) {
        mat_t a_trans = mat_create(a[i - 1].height, a[i - 1].width);
        mat_t nabla = mat_create(1, 1);

        mat_transpose(a[i], &a_trans);
        mat_mult(errors[i], a_trans, &nabla);
        mat_subscalar(ann.weights[i], 0.05 * mat_get(nabla, 0, 0), &ann.weights[i]);
    }

    return 0;

}