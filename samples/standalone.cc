#include <iostream>
#include "darkforest.h"
#include "mimc.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

double time_it(const std::function<void()> &func)
{
    clock_t start = clock();
    func();
    clock_t end = clock();
    double time_use = (double)(end - start) / CLOCKS_PER_SEC;
    printf("explore time : %lf sec\n",time_use);
    return time_use;
}

int main ()
{
    init_device();
    darkforest::ExploreResult result;
    darkforest::ExploreTask task = json::parse(R"({"chunkFootprint":{"bottomLeft":{"x":-256,"y":0},"sideLength":256},"planetRarity":16384,"planetHashKey":420})");
    json req = task;
    printf("explore task: %s\n", req.dump().c_str());

    double time_use = time_it(std::bind(darkforest::explore_chunk, task, std::ref(result)));

    json r = result;
    printf("explore result: %s\n", r.dump().c_str());

    int64_t count = task.chunkFootprint.sideLength * task.chunkFootprint.sideLength;
    printf("explore rate: %.0f H/s\n", count/time_use);
    clear_device();
}