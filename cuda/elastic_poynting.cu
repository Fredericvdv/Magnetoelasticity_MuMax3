#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "stdio.h"

extern "C" __global__ void
poynting(float* __restrict__ px, float* __restrict__ py, float* __restrict__ pz, 
                 float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
                 float* __restrict__ sxx, float* __restrict__ syy, float* __restrict__ szz,
                 float* __restrict__ sxy, float* __restrict__ syz, float* __restrict__ szx,
                 int Nx, int Ny, int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    //Do nothing if cell position is not in mesh
    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I = idx(ix, iy, iz);

    px[I] = -1*(vx[I]*sxx[I] + vy[I]*sxy[I] + vz[I]*szx[I]) ;
    py[I] = -1*(vx[I]*sxy[I] + vy[I]*syy[I] + vz[I]*syz[I]) ;
    pz[I] = -1*(vx[I]*szx[I] + vy[I]*syz[I] + vz[I]*szz[I]) ;
}