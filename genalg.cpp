#include "genalg.h"

chromosome::chromosome(int size, double *initialization_data){
    for (auto i = 0;i<size;i++){
        if (initialization_data != nullptr){
            this->genes.push_back(initialization_data[i]);
        }else{
        	this->genes.push_back(0.0);
        }
    }
}

chromosome::~chromosome(){
    this->genes.clear();
}

void chromosome::mutate(const double mutation_rate){
    for (auto& gene : this->genes)
    {
        gene += mutation_rate*(static_cast<double>(rand() % 100000)/100000.0f)-static_cast<double>(rand() % 100000)/100000.0f*mutation_rate;
    }
}

void chromosome::crossover_with(chromosome *ch, const double prob){
    for (unsigned i = 0; i < this->genes.size(); i++){
        if (static_cast<double>(rand() % 1000)/1000.0f<=prob)
            this->genes[i] = ch->genes[i];
    }
}
    
genetic_algorithm::genetic_algorithm(const int chromosomes_count, const int chromosome_size, double (*compute_error)(chromosome*, double *params, int parameter_size), double *initialization_data):
    error(0)
{
    this->computeError = compute_error;
    for (auto i = 0; i < chromosomes_count; i++)
    {
        this->chromosomes.push_back(new chromosome(chromosome_size, initialization_data));
    }
    this->best = this->chromosomes[0];
}

genetic_algorithm::~genetic_algorithm(){
    for (auto& chromosome : this->chromosomes)
    {
        delete chromosome;
    }
    this->chromosomes.clear();
}
    
void genetic_algorithm::work_iterations(const int iterations, const double prob, const double mut_rate, double *params, const int parameter_size){
    auto min_err = this->computeError(this->best, params, parameter_size);
    auto mutation_rate = mut_rate;
    auto new_best = this->best;
    for (auto i = 0; i < iterations; i++){
        for (auto& chromosome : this->chromosomes)
        {
            if (chromosome == this->best) continue;
            chromosome->crossover_with(this->best, prob);
            chromosome->mutate(mutation_rate);
            const auto ack_err = this->computeError(chromosome, params, parameter_size);
            if (ack_err < min_err && ack_err >= 0.0) {
                min_err = ack_err;
                new_best = chromosome;
                printf("min error: %f, mutation rate: %f\n", min_err, mutation_rate);
                fflush(stdout);
                mutation_rate *= 2.0;
            }
        }
        this->best = new_best;
        if (i % 100 == 0) mutation_rate /= 10.0;
        if (mutation_rate < 0.001) mutation_rate = 10.0;
    }
}

void genetic_algorithm::work_error(const double error, double prob, const double mut_rate, double* params,
                                   const int parameter_size, const int max_iterations)
{
    auto min_err = this->computeError(this->best, params, parameter_size);
    auto mutation_rate = mut_rate;
    auto new_best = this->best;
    auto iteration = 0;
    while(min_err > error){
        for (auto& chromosome : this->chromosomes)
        {
            if (chromosome == this->best) continue;
            chromosome->crossover_with(this->best, prob);
            chromosome->mutate(mutation_rate);
            const double ack_err = this->computeError(chromosome, params, parameter_size);
            if (ack_err < min_err && ack_err >= 0.0) {
                min_err = ack_err;
                new_best = chromosome;
                printf("min error: %f, mutation rate: %f, iteration: %d\n", min_err, mutation_rate, iteration);
                fflush(stdout);
            }
        }
        if (this->best == new_best){
            mutation_rate *= 0.1;
        }else{
            mutation_rate *= 1000.0;
        }
        this->best = new_best;
        iteration++;
        if (iteration > max_iterations) break;
        if (mutation_rate < 0.0001) mutation_rate = mut_rate;
    }
    this->error = min_err;
}

