#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "stdio.h"

extern "C" __global__ void
ShearStrain(float* __restrict__ exy, float* __restrict__ eyz, float* __restrict__ ezx, 
                 float* __restrict__ ux, float* __restrict__ uy, float* __restrict__ uz,
                 int Nx, int Ny, int Nz, float wx, float wy, float wz,float* __restrict__  C1_, float  C1_mul) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    //Do nothing if cell position is not in mesh
    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    // Central cell
    int I = idx(ix, iy, iz);
    float C = amul(C1_, C1_mul, I);
    
    //Neighbor cell
    int I_ = idx(ix, iy, iz);
    float C_ = amul(C1_, C1_mul, I_);

    exy[I] = 0.0;
    eyz[I] = 0.0;
    ezx[I] = 0.0;
 
    //X-direction
    if (ix < Nx-1) {
        I_ = idx(ix+1, iy, iz);
        C_ = amul(C1_, C1_mul, I_);
        if (!(C_ == 0 || C ==0)) {
            exy[I] += 0.5*0.5*wx*(uy[I_]-uy[I]);
            ezx[I] += 0.5*0.5*wx*(uz[I_]-uz[I]);
        }
    } 
    //If there is left neighbour
    if (ix > 0) {
        I_ = idx(ix-1, iy, iz);
        C_ = amul(C1_, C1_mul, I_);
        if (!(C_ == 0 || C ==0)) {
            exy[I] += 0.5*0.5*wx*(uy[I]-uy[I_]);
            ezx[I] += 0.5*0.5*wx*(uz[I]-uz[I_]);
        }
    }


    //y-direction
    if (iy < Ny-1) {
        I_ = idx(ix, iy+1, iz);
        C_ = amul(C1_, C1_mul, I_);
        if (!(C_ == 0 || C ==0)) {
            exy[I] += 0.5*0.5*wy*(ux[I_]-ux[I]);
            eyz[I] += 0.5*0.5*wy*(uz[I_]-uz[I]);
        }
    } 
    //If there is left neighbour
    if (iy > 0) {
        I_ = idx(ix, iy-1, iz);
        C_ = amul(C1_, C1_mul, I_);
        if (!(C_ == 0 || C ==0)) {
            exy[I] += 0.5*0.5*wy*(ux[I]-ux[I_]);
            eyz[I] += 0.5*0.5*wy*(uz[I]-uz[I_]);
        }
    }

    
    //z-direction
    if (iz < Nz-1) {
        I_ = idx(ix, iy, iz+1);
        C_ = amul(C1_, C1_mul, I_);
        if (!(C_ == 0 || C ==0)) {
            ezx[I] += 0.5*wz*(ux[I_]-ux[I]);
            eyz[I] += 0.5*wz*(uy[I_]-uy[I]);
        }
    } 
    //If there is left neighbour
    if (iz > 0) {
        I_ = idx(ix, iy, iz-1);
        C_ = amul(C1_, C1_mul, I_);
        if (!(C_ == 0 || C ==0)) {
            ezx[I] += 0.5*wz*(ux[I]-ux[I_]);
            eyz[I] += 0.5*wz*(uy[I]-uy[I_]);
        }
    }
}