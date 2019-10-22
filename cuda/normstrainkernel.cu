#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "stdio.h"

extern "C" __global__ void
NormStrain(float* __restrict__ ex, float* __restrict__ ey, float* __restrict__ ez, 
                 float* __restrict__ ux, float* __restrict__ uy, float* __restrict__ uz,
                 int Nx, int Ny, int Nz, float wx, float wy, float wz,float* __restrict__  C1_, float  C1_mul, uint8_t PBC) {

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

    ex[I] = 0.0;
    ey[I] = 0.0;
    ez[I] = 0.0;
 
    //X-direction
    if (ix==0) {
        I_ = idx(hclampx(ix+1), iy, iz);
        ex[I] += wx*(ux[I_]-ux[I]);
    } else if (ix==Nx-1) {
        I_ = idx(lclampx(ix-1), iy, iz);
        ex[I] += wx*(ux[I]-ux[I_]);
    } else {
        I_ = idx(hclampx(ix+1), iy, iz);
        ex[I] += 0.5*wx*ux[I_];
        I_ = idx(lclampx(ix-1), iy, iz);
        ex[I] -= 0.5*wx*ux[I_];
    }

    //Y-direction
    if (iy==0) {
        I_ = idx(ix, hclampy(iy+1), iz);
        ey[I] += wy*(uy[I_]-uy[I]);
    } else if (iy==Ny-1) {
        I_ = idx(ix, lclampy(iy-1), iz);
        ey[I] += wy*(uy[I]-uy[I_]);
    } else {
        I_ = idx(ix, hclampy(iy+1), iz);
        ey[I] += 0.5*wy*uy[I_];
        I_ = idx(ix, lclampy(iy-1), iz);
        ey[I] -= 0.5*wy*uy[I_];
    }


    // //X-direction
    // //Right neighbor
    // I_ = idx(hclampx(ix+1), iy, iz);
    // C_ = amul(C1_, C1_mul, I_);
    // if (!(C_ == 0 || C ==0)) {
    //     ex[I] += 0.5*wx*(ux[I_]-ux[I]);
    // }
    // //If there is left neighbour
    // I_ = idx(lclampx(ix-1), iy, iz);
    // C_ = amul(C1_, C1_mul, I_);
    // if (!(C_ == 0 || C ==0)) {
    //     ex[I] += 0.5*wx*(ux[I]-ux[I_]);
    // }

    // //y-direction
    // I_ = idx(ix, hclampy(iy+1), iz);
    // C_ = amul(C1_, C1_mul, I_);
    // if (!(C_ == 0 || C ==0)) {
    //     ey[I] += 0.5*wy*(uy[I_]-uy[I]);
    // }
    // //If there is left neighbour
    // I_ = idx(ix, lclampy(iy-1), iz);
    // C_ = amul(C1_, C1_mul, I_);
    // if (!(C_ == 0 || C ==0)) {
    //     ey[I] += 0.5*wy*(uy[I]-uy[I_]);
    // }

    
    // //z-direction
    // I_ = idx(ix, iy, hclampz(iz+1));
    // C_ = amul(C1_, C1_mul, I_);
    // if (!(C_ == 0 || C ==0)) {
    //     ez[I] += 0.5*wz*(uz[I_]-uz[I]);
    // }
    // //If there is left neighbour
    // I_ = idx(ix, iy, lclampz(iz-1));
    // C_ = amul(C1_, C1_mul, I_);
    // if (!(C_ == 0 || C ==0)) {
    //     ez[I] += 0.5*wz*(uz[I]-uz[I_]);
    // }
}