#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "stdio.h"

extern "C" __global__ void
ElsticEnergy(float* __restrict__ energy, 
                 float* __restrict__ exx, float* __restrict__ eyy, float* __restrict__ ezz,
                 float* __restrict__ exy, float* __restrict__ eyz, float* __restrict__ exz,
                 int Nx, int Ny, int Nz,  
                 float* __restrict__  C1_, float  C1_mul, float* __restrict__  C2_, float  C2_mul, 
                 float* __restrict__  C3_, float  C3_mul) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    //Do nothing if cell position is not in mesh
    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    // Central cell
    int I = idx(ix, iy, iz);
    float c1 = amul(C1_, C1_mul, I);
    float c2 = amul(C2_, C2_mul, I);
    float c3 = amul(C3_, C3_mul, I);

    energy[I] = c1*0.5*(exx[I]*exx[I]+eyy[I]*eyy[I]+ezz[I]*ezz[I]);
    energy[I] += c2*(exx[I]*eyy[I]+eyy[I]*ezz[I]+exx[I]*ezz[I]);
    energy[I] += c3*2*(exy[I]*exy[I]+eyz[I]*eyz[I]+exz[I]*exz[I]);
}

