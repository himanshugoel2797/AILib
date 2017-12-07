/**
 * Copyright (c) 2017 Himanshu Goel
 * 
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */

#include "ga.h"
#include <stdlib.h>
#include <math.h>

static unsigned int seed = 0;
#define CORNER_CNT 8
static float corners[CORNER_CNT];
static int prev_idx = 0;

static uint8_t get_rand(){
    seed = (seed + 1013904223) * 1664525;
    return seed;
}

static float ga_rand(){
    prev_idx = (prev_idx + 1) % CORNER_CNT;
    corners[prev_idx] = fmod(get_rand(), 1024) / 1024.0f;

    float avg = 0;
    for(int i = 0; i < CORNER_CNT; i++)
        avg += corners[i];

    return avg / CORNER_CNT;
}

void ga_setseed(unsigned int s) {
    seed = s;
    for(int i = 0; i < CORNER_CNT; i++)
        ga_rand();
}

ga_t ga_create(int pop_sz, float mutation_rate, MemberIniter init, FitnessFunction fitness, MemberMutate mutator, MemberMerge merger, MemberKill murderer) {
    ga_t ga;
    ga.pop_sz = pop_sz;
    ga.mutation_rate = mutation_rate;
    ga.init = init;
    ga.fitness = fitness;
    ga.mutator = mutator;
    ga.merger = merger;
    ga.murderer = murderer;
    ga.population = malloc(pop_sz * sizeof(void*));
    ga.fitness_vals = malloc(pop_sz * sizeof(float));
    ga.current_pop_sz = 0;
    ga.generation = 0;

    while(ga.current_pop_sz != ga.pop_sz) {
        ga.population[ga.current_pop_sz] = init( ga.generation << 16 | ga.current_pop_sz);
        ga.fitness_vals[ga.current_pop_sz] = fitness(ga.population[ga.current_pop_sz]);
        ga.current_pop_sz++;
    }
}

int ga_iteration(ga_t ga, void** fittest) {
    int cur_idx = 0;
    while(ga.current_pop_sz != ga.pop_sz) {
        //Produce children based on closest fitness
        float target_fitness = ga.fitness_vals[cur_idx];

        //Find member with closest fitness
        float fitness_diff_threshold = 0.2;
        int closest_fitness_idx = cur_idx;
        for(int i = 0; i < ga.current_pop_sz; i++){
            if(i == cur_idx)
                continue;

            if(fabs(ga.fitness_vals[i] - target_fitness) < fitness_diff_threshold){
                closest_fitness_idx = i;
                break;
            }
        }

        //produce a child from these two and add it to the population
        ga.population[ga.current_pop_sz] = ga.merger(ga.population[cur_idx], ga.population[closest_fitness_idx]);

        //mutate randomly
        if(ga_rand() <= ga.mutation_rate)
            ga.population[ga.current_pop_sz] = ga.mutator(ga.population[ga.current_pop_sz]);

        //update fitness values
        ga.fitness_vals[ga.current_pop_sz] = ga.fitness(ga.population[ga.current_pop_sz]);

        ga.current_pop_sz++;
        cur_idx++;

        if(cur_idx >= ga.current_pop_sz)
            break;
    }

    //randomly kill and replace some members
    float max_fitness = 0;
    int max_fitness_idx = -1;

    for(int i = 0; i < ga.current_pop_sz; i++) {
        if(ga_rand() <= 0.5f * ga.fitness_vals[i]){
            ga.murderer(ga.population[i]);
            
            //replace killed with new individuals
            ga.population[i] = ga.init(ga.generation << 16 | i);
            ga.fitness_vals[i] = ga.fitness(ga.population[i]);
        }

        if(ga.fitness_vals[i] > max_fitness) {
            max_fitness = ga.fitness_vals[i];
            max_fitness_idx = i;
        }
    }

    //find highest fitness and return it
    *fittest = ga.population[max_fitness_idx];
}