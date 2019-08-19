#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "stdio.h"

extern "C" __global__ void
Normstress(float* __restrict__ sx, float* __restrict__ sy, float* __restrict__ sz, 
                 float* __restrict__ ex, float* __restrict__ ey, float* __restrict__ ez,
                 int Nx, int Ny, int Nz, float* __restrict__  C1_, float  C1_mul, float* __restrict__  C2_, float  C2_mul) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    //Do nothing if cell position is not in mesh
    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I = idx(ix, iy, iz);
    float c1 = amul(C1_, C1_mul, I);
    float c2 = amul(C2_, C2_mul, I);

    sx[I] = ex[I]*c1 + c2*(ey[I]+ez[I]);
    sy[I] = ey[I]*c1 + c2*(ex[I]+ez[I]);
    sz[I] = ez[I]*c1 + c2*(ex[I]+ey[I]);
}