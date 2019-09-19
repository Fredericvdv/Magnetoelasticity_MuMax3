#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "stdio.h"

extern "C" __global__ void
Bndryy(float* __restrict__ ux, float* __restrict__ uy, float* __restrict__ uz,
                 int Nx, int Ny, int Nz, float wx, float wy, float c1, float c2) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    //Do nothing if cell position is not in mesh
    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    // Central cell
    int I = idx(ix, iy, iz);



    //Neighbor cell
    int I_ = idx(1, 1, iz);

    float d_ = 0;

    //x-interface
    if (ix==0) {
        if (iy==0) {
            //Left-down corner
            I_ = idx(ix+1, iy, iz);
            ux[I] = ux[I_];
            I_ = idx(ix, iy+1, iz);
            uy[I] = uy[I_];
        } else if (iy==Ny-1) {
            //Left-upper corner
            I_ = idx(ix+1, iy, iz);
            ux[I] = ux[I_];
            I_ = idx(ix, iy-1, iz);
            uy[I] = uy[I_];
        } else {
            //Left interface
            d_=0;
            I_ = idx(ix+1, iy, iz);
            d_ = ux[I_];
            I_ = idx(ix+1, iy+1, iz);
            d_ += c2*0.5*wx/(wy*c1)* uy[I_];
            I_ = idx(ix+1, iy-1, iz);
            d_ -= c2*0.5*wx/(wy*c1)* uy[I_];
            ux[I] = d_;
        }
    } else if (ix==Nx-1) {
        if (iy==0) {
            //Right-down corner
            I_ = idx(ix-1, iy, iz);
            ux[I] = ux[I_];
            I_ = idx(ix, iy+1, iz);
            uy[I] = uy[I_];
        } else if (iy==Ny-1) {
            //Right-upper corner
            I_ = idx(ix-1, iy, iz);
            ux[I] = ux[I_];
            I_ = idx(ix, iy-1, iz);
            uy[I] = uy[I_];
        } else {
            //Right interface
            d_=0;
            I_ = idx(ix-1, iy, iz);
            d_ = ux[I_];
            I_ = idx(ix-1, iy+1, iz);
            d_ -= c2*0.5*wx/(wy*c1)* uy[I_];
            I_ = idx(ix-1, iy-1, iz);
            d_ += c2*0.5*wx/(wy*c1)* uy[I_];
            ux[I] = d_;
        }
    } else if (iy==0) {
        //down interface
        d_=0;
        I_ = idx(ix, iy+1, iz);
        d_ = uy[I_];
        I_ = idx(ix+1, iy+1, iz);
        d_ += c2*0.5*wy/(wx*c1)* ux[I_];
        I_ = idx(ix-1, iy+1, iz);
        d_ -= c2*0.5*wy/(wx*c1)* ux[I_];
        uy[I] = d_;
    } else if (iy==Ny-1) {
        //Upper-inteface
        d_=0;
        I_ = idx(ix, iy-1, iz);
        d_ = uy[I_];
        I_ = idx(ix+1, iy-1, iz);
        d_ -= c2*0.5*wy/(wx*c1)* ux[I_];
        I_ = idx(ix-1, iy-1, iz);
        d_ += c2*0.5*wy/(wx*c1)* ux[I_];
        uy[I] = d_;
        }
    }