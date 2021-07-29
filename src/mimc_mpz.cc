
#include <stdint.h>
#include <gmpxx.h>
#include <vector>

#include "mimc_constants.h"

typedef struct
{
    mpz_class k;
    mpz_class l;
    mpz_class r;
} feistel_state_t;


void inject(feistel_state_t &state, int64_t elt)
{
    state.l = (state.l + elt);
}

void mix(feistel_state_t &state)
{
    mpz_class t;
    for (size_t i = 0; i < MimcConstants::rounds - 1; ++i) {
        t = (state.l + state.k + MimcConstants::c_at(i));
        mpz_powm_ui(t.get_mpz_t(), t.get_mpz_t(), 5, MimcConstants::get_p().get_mpz_t());
        t = (t + state.r) %  MimcConstants::get_p();
        state.r = state.l;
        state.l = t;
    }
    t = (state.l + state.k);
    mpz_powm_ui(t.get_mpz_t(), t.get_mpz_t(), 5, MimcConstants::get_p().get_mpz_t());
    state.r = (t + state.r) %  MimcConstants::get_p();
}

void cpu_mimc_sponge(const std::vector<int64_t> &inputs,
                     size_t n_outputs,
                     uint32_t key, std::vector<mpz_class> &outputs)
{
    feistel_state_t state{key, 0, 0};

    for (int64_t input : inputs) {
        inject(state, input);
        mix(state);
    }

    outputs.push_back(state.l);
    for (size_t i = 0; i < n_outputs - 1; i++) {
        mix(state);
        outputs.emplace_back(state.l);
    }
}

void cpu_bulk_mimc_sponge(const std::vector<std::vector<int64_t>> &inputs,
                          size_t n_outputs,
                          uint32_t key,
                          std::vector<std::vector<mpz_class>> &outputs)
{
    for (const auto & input : inputs) {
        std::vector<mpz_class> out_item;
        cpu_mimc_sponge(input, n_outputs, key, out_item);
        outputs.push_back(out_item);
    }
}
