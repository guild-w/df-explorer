
#ifndef DF_EXPLORER_DARKFOREST_H
#define DF_EXPLORER_DARKFOREST_H

#include <stdint.h>
#include <vector>

#include "nlohmann/json.hpp"

namespace darkforest {

struct Coords {
    int64_t x;
    int64_t y;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Coords, x, y)

struct Planet {
    std::string hash;
    Coords coords;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Planet, coords, hash)

struct ChunkFootprint {
    Coords bottomLeft;
    uint32_t sideLength;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ChunkFootprint, bottomLeft, sideLength)

struct ExploreTask {
    ChunkFootprint chunkFootprint;
    uint32_t  planetRarity;
    uint32_t planetHashKey;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ExploreTask, chunkFootprint, planetRarity, planetHashKey)

struct ExploreResult {
    ChunkFootprint chunkFootprint;
    std::vector<Planet> planetLocations;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ExploreResult, chunkFootprint, planetLocations)


void explore_chunk(const darkforest::ExploreTask &task, darkforest::ExploreResult &result);

}

void init_device();
void clear_device();

#endif //DF_EXPLORER_DARKFOREST_H
