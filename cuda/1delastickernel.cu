#include "float3.h"
#include "stencil.h"

// ~ Bibliothee can C inladen of zichtbaar maken
// Nx aantal cellen in x-richting
extern "C" __global__ void
SecondDerivative(float* __restrict__ dmx, float* __restrict__ dmy, float* __restrict__ dmz, 
float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
int Nx, int Ny, int Nz) {
    //positie van cell waar we in aan het kijkken zijn is: ix,iy,iz
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }
    // central cell
    int I = idx(ix, iy, iz);
    float3 m0 = make_float3(mx[I], my[I], mz[I]);

    //output gelijkstellen aan de magnetizatie
    dmx[I] = m0.x;
    dmy[I] = m0.y;
    dmz[I] = m0.z;



}


