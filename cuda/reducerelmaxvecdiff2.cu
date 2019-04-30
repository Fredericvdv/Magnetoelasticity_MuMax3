#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "stdio.h"



extern "C" __global__ void
RelMaxVecDiff(float* out, float* __restrict__ x1, float* __restrict__ y1, float* __restrict__ z1,
                  float* __restrict__ x2, float* __restrict__ y2, float* __restrict__ z2,
                  int Nx, int Ny, int Nz) {
    //positie van cell waar we in aan het kijkken zijn is: ix,iy,iz
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    //if cell position is out of mesh --> do nothing
    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return ;
    }

    int I = idx(ix, iy, iz);

    if (x1[I] == 0) {
        out[I] = 0.0;
    } else {
        out[I] = pow2((x1[I] - x2[I])/x1[I]);
    }

    if (y1[I] == 0) {
        out[I] += 0.0;
    } else {
        out[I] += pow2((y1[I] - y2[I])/y1[I]);
    }

    if (z1[I] == 0) {
        out[I] += 0.0;
    } else {
        out[I] += pow2((z1[I] - z2[I])/z1[I]);
    }

    
    out[I] = 5.0;
}

