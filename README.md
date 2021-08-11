# df-explorer
a dark forest cuda explorer

## Build
df-explorer CGBN requires CUDA toolkit https://developer.nvidia.com/cuda-toolkit

+ clone with submodules
```bash
git clone --recurse-submodules https://github.com/guild-w/df-explorer.git
```  

+ install GMP
```bash
sudo apt install libgmp-dev
```
+ build the server component `workflow`
```bash
cd df-explorer/thirdparty/workflow/ && make
```

+ and back to project root build the df-explorer
```bash
cd ../../
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```
+ start server and use `http://localhost:8880/explore` to connect to your miner with the plugin "Remote Explore"
```bash
./df-explorer
```

**It is highly recommended to increase the chunk size in "Remote Explore"** 
