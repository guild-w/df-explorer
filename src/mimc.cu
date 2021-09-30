#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <gmpxx.h>
#include "cgbn/cgbn.h"
#include "utility/cpu_support.h"
#include "utility/gpu_support.h"

#include "mimc_constants.h"
#include "mimc.h"


// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI 8
#define BITS 256
#define DEFAULT_TPB 128
#define MAX_CUDA_OUT 4096
#define MAX_IPB 256 // MAX_TPB / MIN_tpi 1024/4

typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;
typedef cgbn_mem_t<BITS> bn_mem_t;
typedef typename env_t::cgbn_t bn_t;

template<typename T>
struct feistel_state_t{
    T k;
    T l;
    T r;
} ;

typedef struct {
    int64_t x;
    int64_t y;
    uint32_t side_length;
    uint32_t key;
    uint32_t rarity;
} explore_in_t;

typedef struct {
    bn_mem_t hash;
    int64_t x;
    int64_t y;
} explore_out_item_t;

typedef struct {
    explore_out_item_t planets[MAX_CUDA_OUT];
    uint32_t count;
} explore_out_t;


namespace {

    int32_t g_tpb = DEFAULT_TPB;
    __constant__ bn_mem_t g_device_p;
    __constant__ bn_mem_t g_device_c[MimcConstants::rounds];
}

__device__ void add_mod(env_t &bn_env, bn_t &r, const bn_t &a, const bn_t &b, const bn_t &modulus)
{
    cgbn_add(bn_env, r, a, b);
    if (cgbn_compare(bn_env, r, modulus) == 1) {
        cgbn_sub(bn_env, r, r, modulus);
    }
}

__device__ void mix(env_t &bn_env, feistel_state_t<bn_t> &state)
{
    bn_t bn_t_5, p, t, ci;
    cgbn_load(bn_env, p, &g_device_p);
    cgbn_set_ui32(bn_env, bn_t_5, 5);

    for (int32_t i = 0; i < MimcConstants::rounds - 1; ++i) {
        cgbn_load(bn_env, ci, g_device_c + i);
        add_mod(bn_env, t, state.l, state.k, p);
        add_mod(bn_env, t, t, ci, p);
        cgbn_modular_power(bn_env, t, t, bn_t_5, p);

        add_mod(bn_env, t, t, state.r, p);
        cgbn_set(bn_env, state.r, state.l);
        cgbn_set(bn_env, state.l, t);
    }
    add_mod(bn_env, t, state.l, state.k, p);
    cgbn_modular_power(bn_env, t, t, bn_t_5, p);
    add_mod(bn_env, state.r, t, state.r, p);
}

__device__  void inject(env_t &bn_env, feistel_state_t<bn_t> &state, bn_t elt)
{
    bn_t p;
    cgbn_load(bn_env, p, &g_device_p);
    add_mod(bn_env, state.l, state.l, elt, p);
}

__device__ void coords_to_bn(env_t &bn_env, bn_t &r, int64_t num)
{
#ifdef DF_INT32_COORDS
    if (num > 0) {
        cgbn_set_ui32(bn_env, r, num);
        return;
    }
    bn_t p;
    cgbn_load(bn_env, p, &g_device_p);
    cgbn_sub_ui32(bn_env, r, p, abs(num));
#else
    uint32_t low = llabs(num) & 0xffffffff;
    uint32_t high = llabs(num) >> 32;
    cgbn_set_ui32(bn_env, r, 0);
    cgbn_insert_bits_ui32(bn_env, r, r, 0, 32, low);
    cgbn_insert_bits_ui32(bn_env, r, r, 32, 32, high);
    if (num < 0) {
        bn_t p;
        cgbn_load(bn_env, p, &g_device_p);
        cgbn_sub(bn_env, r, p, r);
    }
#endif //DF_INT32_COORDS
}

__device__  bool is_planet(env_t &bn_env, const bn_t &hash, uint32_t rarity)
{
    bn_t threshold, p;
    cgbn_load(bn_env, p, &g_device_p);
    cgbn_div_ui32(bn_env, threshold, p, rarity);
    if (cgbn_compare(bn_env, hash, threshold) == -1) {
        return true;
    }
    return false;
}

__global__ void kernel_explore(const explore_in_t * __restrict__ explore_params,
                               feistel_state_t<bn_mem_t> *states,
                               explore_out_t * __restrict__ explore_out,
                               uint32_t count)
{
    uint32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    if (instance >= count) {
        return;
    }

    context_t bn_context(cgbn_no_checks);
    env_t bn_env(bn_context.env<env_t>());

    bn_t hash, bn_y;
    int64_t index_x = (instance / explore_params->side_length);
    int64_t x = explore_params->x + index_x;
    int64_t y = explore_params->y + (instance % explore_params->side_length);
    coords_to_bn(bn_env, bn_y, y);

    feistel_state_t<bn_t> state{};
    cgbn_load(bn_env, state.l, &(states[index_x].l));
    cgbn_load(bn_env, state.r,  &(states[index_x].r));
    cgbn_set_ui32(bn_env, state.k, explore_params->key);

    inject(bn_env, state, bn_y);
    mix(bn_env, state);
    cgbn_set(bn_env, hash, state.l);

    if (!is_planet(bn_env, hash, explore_params->rarity)) {
        return;
    }

    __shared__ uint32_t result_index[MAX_IPB];
    uint32_t ii = threadIdx.x / TPI;
    uint32_t group_thread = threadIdx.x & TPI-1;
    if (0 == group_thread) {
        result_index[ii] = atomicInc((uint32_t*)&(explore_out->count), 0xffffffff);
    }
    __syncthreads();

    uint32_t i = result_index[ii];
    if (i >= MAX_CUDA_OUT) {
        return;
    }
    explore_out->planets[i].x = x;
    explore_out->planets[i].y = y;
    cgbn_store(bn_env, &(explore_out->planets[i].hash), hash);
}

__global__ void kernel_absorb_x(const explore_in_t * explore_params, feistel_state_t<bn_mem_t> *result, uint32_t count)
{
    uint32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    if (instance >= count) {
        return;
    }

    context_t bn_context(cgbn_no_checks);
    env_t bn_env(bn_context.env<env_t>());

    bn_t bn_x;
    int64_t x = explore_params->x + instance;
    coords_to_bn(bn_env, bn_x, x);

    feistel_state_t<bn_t> state{};
    cgbn_set_ui32(bn_env, state.l, 0);
    cgbn_set_ui32(bn_env, state.r, 0);
    cgbn_set_ui32(bn_env, state.k, explore_params->key);
    inject(bn_env, state, bn_x);
    mix(bn_env, state);

    cgbn_store(bn_env, &(result[instance].l), state.l);
    cgbn_store(bn_env, &(result[instance].r), state.r);
}

void init_device_constants(int32_t device_id)
{
    bn_mem_t p;
    bn_mem_t c[MimcConstants::rounds];
    from_mpz(MimcConstants::get_p().get_mpz_t(), p._limbs, BITS / 32);
    for (int32_t i = 0; i < MimcConstants::rounds; ++i) {
        from_mpz(MimcConstants::c_at(i).get_mpz_t(), c[i]._limbs, BITS / 32);
    }
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMemcpyToSymbol(g_device_p, &p, sizeof(bn_mem_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_device_c, c, sizeof(bn_mem_t) * MimcConstants::rounds));
}

int32_t get_env_i32(const char *name, int32_t default_value=INT32_MAX)
{
    char *value_str;
    value_str = getenv(name);
    if (value_str == nullptr) {
        return default_value;
    }
    return atoi(value_str);
}

void init() {
    g_tpb = get_env_i32("MIMC_CUDA_BLOCK_SIZE", DEFAULT_TPB);
    if (g_tpb <= 0 || g_tpb > 1024) {
        g_tpb = DEFAULT_TPB;
    }
    int32_t device_id = get_env_i32("MIMC_CUDA_DEVICE", 0);

    printf("cuda block size: %d\n", g_tpb);
    printf("cuda device id: %d\n", device_id);

    init_device_constants(device_id);
}

void get_result(explore_out_t * cuda_result, std::vector<location_hash_t> &hashes)
{
    mpz_class h;
    for (int32_t i = 0; i < cuda_result->count; ++i) {
        to_mpz(h.get_mpz_t(), cuda_result->planets[i].hash._limbs, BITS / 32);
        hashes.push_back({h.get_str(), cuda_result->planets[i].x, cuda_result->planets[i].y});
    }
}

void gpu_explore_chunk(int64_t bottom_left_x,
                       int64_t bottom_left_y,
                       uint32_t side_length,
                       uint32_t key,
                       uint32_t rarity,
                       std::vector<location_hash_t> &hashes)
{
    uint32_t TPB = g_tpb;
    uint32_t IPB = TPB / TPI; // IPB is instances per block

    explore_in_t in_params {
        .x = bottom_left_x,
        .y = bottom_left_y,
        .side_length = side_length,
        .key = key,
        .rarity = rarity
    };

    explore_in_t * gpu_in_params;
    CUDA_CHECK(cudaMalloc((void **)&gpu_in_params, sizeof(explore_in_t)));
    CUDA_CHECK(cudaMemcpy(gpu_in_params, &in_params, sizeof(explore_in_t), cudaMemcpyHostToDevice));

    //count == side_length
    feistel_state_t<bn_mem_t> *states;
    CUDA_CHECK(cudaMalloc((void **)&states, sizeof(feistel_state_t<bn_mem_t>) * side_length));
    kernel_absorb_x<<<(side_length + IPB - 1) / IPB, TPB>>>(gpu_in_params, states, side_length);
    CUDA_CHECK(cudaDeviceSynchronize());

    explore_out_t *out;
    CUDA_CHECK(cudaHostAlloc((void **)&out, sizeof(explore_out_t), cudaHostAllocDefault));
    out->count = 0;
    explore_out_t *gpu_out;
    CUDA_CHECK(cudaMalloc((void **)&gpu_out, sizeof(explore_out_t)));
    CUDA_CHECK(cudaMemcpy(gpu_out, out, sizeof(explore_out_t), cudaMemcpyHostToDevice));

    uint32_t count = side_length * side_length;
    kernel_explore<<<(count + IPB - 1) / IPB, TPB>>>(gpu_in_params, states, gpu_out, count);
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy the result back from gpuMemory
    CUDA_CHECK(cudaMemcpy(out, gpu_out, sizeof(explore_out_t), cudaMemcpyDeviceToHost));
    get_result(out, hashes);

    CUDA_CHECK(cudaFreeHost(out));
    CUDA_CHECK(cudaFree(gpu_in_params));
    CUDA_CHECK(cudaFree(states));
    CUDA_CHECK(cudaFree(gpu_out));
}
