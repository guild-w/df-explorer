
#include <stdint.h>
#include "mimc.h"
#include "mimc_constants.h"
#include "darkforest.h"

using json = nlohmann::json;

void check_task_params(const darkforest::ExploreTask &task)
{
    if (task.chunkFootprint.sideLength < 256 || task.chunkFootprint.sideLength > 2048) {
        throw std::exception();
    }
}

void darkforest::explore_chunk(const darkforest::ExploreTask &task, darkforest::ExploreResult &result)
{
    check_task_params(task);

    std::vector<location_hash_t> hashes;
    gpu_explore_chunk(task.chunkFootprint.bottomLeft.x,
                      task.chunkFootprint.bottomLeft.y,
                      task.chunkFootprint.sideLength,
                      task.planetHashKey,
                      task.planetRarity,
                      hashes);

     std::vector<darkforest::Planet> planets;

     planets.reserve(hashes.size());
     for (auto & hash : hashes) {
         planets.push_back({hash.hash, hash.x, hash.y});
     }

    result.chunkFootprint = task.chunkFootprint;
    result.planetLocations = planets;
}

void init_device()
{
    init();
}

void clear_device()
{
}