#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "stdio.h"

extern "C" __global__ void
Shearstress(float* __restrict__ sxy, float* __restrict__ syz, float* __restrict__ szx, 
                 float* __restrict__ exy, float* __restrict__ eyz, float* __restrict__ ezx,
                 int Nx, int Ny, int Nz, float* __restrict__  C3_, float  C3_mul) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    //Do nothing if cell position is not in mesh
    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I = idx(ix, iy, iz);
    float C3 = amul(C3_, C3_mul, I);
    
    sxy[I] = 2*exy[I]*C3;
    syz[I] = 2*eyz[I]*C3;
    szx[I] = 2*ezx[I]*C3;
}