// Copyright (c) 2017 Himanshu Goel
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AILIB_GA_H
#define AILIB_GA_H


typedef void* (*MemberIniter)(int);
typedef float (*FitnessFunction)(void *);
typedef void* (*MemberMutate)(void *);
typedef void* (*MemberMerge)(void *, void *);
typedef void (*MemberKill)(void*);

typedef struct ga ga_t;
struct ga {
    MemberIniter init;
    FitnessFunction fitness;
    MemberMutate mutator;
    MemberMerge merger;
    MemberKill murderer;
    void** population;
    float* fitness_vals;
    float mutation_rate;
    int pop_sz;
    int generation;
    int current_pop_sz;
};

ga_t ga_create(int pop_sz, float mutation_rate, MemberIniter init, FitnessFunction fitness, MemberMutate mutator, MemberMerge merger, MemberKill murderer);
int ga_iteration(ga_t ga, void** fittest);
void ga_setseed(unsigned int);

#endif