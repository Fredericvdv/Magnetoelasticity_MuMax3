#include <stdint.h>
#include "float3.h"
#include "stencil.h"

// ~ Bibliothee can C inladen of zichtbaar maken
// Nx aantal cellen in x-richting
extern "C" __global__ void
SecondDerivative(float* __restrict__ dmx, float* __restrict__ dmy, float* __restrict__ dmz, 
float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
int Nx, int Ny, int Nz, float c, uint8_t PBC) {
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

    if (is0(m0)) {
        return;
    }

    //Index left
    int i_l; //left neighbor index
    int i_r; //right neighbor index
    float3 m_l; //neighbor mag
    float3 m_r; //neighbor mag
    float3 out;

    //Neighbours
    i_l = idx(lclampx(ix-1),iy,iz);
    i_r = idx(lclampx(ix+1),iy,iz);
    m_l = make_float3(mx[i_l],my[i_l],mz[i_l]);
    m_l = ( is0(m_l)? m0: m_l);
    m_r = make_float3(mx[i_r],my[i_r],mz[i_r]);
    m_r = ( is0(m_r)? m0: m_r);

    out = c*(m_l - 2*m0 + m_r);     //Second derivative
    //out = c*(m_l - m_r);      //First derivative





    //output gelijkstellen aan de magnetizatie
    dmx[I] = out.x;
    dmy[I] = out.y;
    dmz[I] = out.z;



}


