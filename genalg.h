#pragma once

#include <vector>

/**
 * \brief chromosome definition
 */
class chromosome{
public:
    std::vector<double> genes;
    
    chromosome(int size, double *initialization_data = nullptr);
    ~chromosome();
    
    void mutate(double mutation_rate);
    void crossover_with(chromosome *ch, double prob);
};

/**
 * \brief definition of a genetic algorithm
 */
class genetic_algorithm{
public:
    double (*computeError)(chromosome*, double *params, int parameter_size);
    std::vector<chromosome*> chromosomes;
    chromosome *best;
    double error;
    
    genetic_algorithm(int chromosomes_count, int chromosome_size,
                      double (*compute_error)(chromosome*, double* params, int parameter_size), double* initialization_data = nullptr);
    ~genetic_algorithm();
    
    void work_iterations(int iterations, double prob, double mut_rate, double *params, int parameter_size);
    void work_error(double error, double prob, double mut_rate, double *params, int parameter_size, int max_iterations = 100000);
};

