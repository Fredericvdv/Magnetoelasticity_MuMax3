#include <stdint.h>
#include "float3.h"

// set dst to zero in cells where mask != 0
extern "C" __global__ void
copymask(float* __restrict__  dst, float* maskLUT, float* valuesLUT, uint8_t* regions, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        if (maskLUT[regions[i]] != 0) {
            dst[i] = valuesLUT[regions[i]];
            //printf("sf %f \n", valuesLUT[regions[i]]);
            //printf(valuesLUT[regions[i]]);
        }
    }
}
