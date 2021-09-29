
#ifndef DF_EXPLORER_MIMC_H
#define DF_EXPLORER_MIMC_H


#include <stdint.h>
#include <gmpxx.h>
#include <vector>

typedef struct 
{
    std::string hash;
    int64_t x;
    int64_t y;
} location_hash_t;


void init();

int32_t get_block_size();

void gpu_explore_chunk(int64_t bottom_left_x,
                       int64_t bottom_left_y,
                       uint32_t side_length,
                       uint32_t key,
                       uint32_t rarity,
                       std::vector<location_hash_t> &hashes);

void cpu_mimc_sponge(const std::vector<int64_t> &inputs,
                     size_t n_outputs,
                     uint32_t key,
                     std::vector<mpz_class> &outputs);

void cpu_bulk_mimc_sponge(const std::vector<std::vector<int64_t>> &inputs,
                          size_t n_outputs,
                          uint32_t key,
                          std::vector<std::vector<mpz_class>> &outputs);

#endif //DF_EXPLORER_MIMC_H
