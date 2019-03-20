#include <stdint.h>
#include "float3.h"
#include "stencil.h"

// ~ Bibliothee can C inladen of zichtbaar maken
// Nx aantal cellen in x-richting
extern "C" __global__ void
SecondDerivative(float* __restrict__ dux, float* __restrict__ duy, float* __restrict__ duz, 
float* __restrict__ ux, float* __restrict__ uy, float* __restrict__ uz,
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
    float3 u0 = make_float3(ux[I], uy[I], uz[I]);

    if (is0(u0)) {
        return;
    }

    //Index left
    int i_l; //left neighbor index
    int i_r; //right neighbor index
    float3 u_l; //neighbor mag
    float3 u_r; //neighbor mag
    float3 out;

    //Neighbours
    i_l = idx(lclampx(ix-1),iy,iz);
    i_r = idx(lclampx(ix+1),iy,iz);
    u_l = make_float3(ux[i_l],uy[i_l],uz[i_l]);
    u_l = ( is0(u_l)? u0: u_l);
    u_r = make_float3(ux[i_r],uy[i_r],uz[i_r]);
    u_r = ( is0(u_r)? u0: u_r);

    out = c*(u_l - 2*u0 + u_r);     //Second derivative
    //out = c*(m_l - m_r);      //First derivative
    //out = c* m_r;
    //out = c* m_l;           //OK

    //out = c*m0;           // OK


    //output gelijkstellen aan de magnetizatie
    dux[I] = out.x;
    duy[I] = out.y;
    duz[I] = out.z;



}


