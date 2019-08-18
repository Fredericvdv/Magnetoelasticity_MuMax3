#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "stdio.h"

extern "C" __global__ void
Elastodynamic1(float* __restrict__ dux, float* __restrict__ duy, float* __restrict__ duz, 
                 float* __restrict__ ux, float* __restrict__ uy, float* __restrict__ uz,
                 int Nx, int Ny, int Nz, float wx, float wy, float wz, 
                 float* __restrict__  C1_, float  C1_mul, float* __restrict__  C2_, float  C2_mul, 
                 float* __restrict__  C3_, float  C3_mul, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    //Do nothing if cell position is not in mesh
    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    // Central cell
    int I = idx(ix, iy, iz);
    float3 u0 = make_float3(ux[I], uy[I], uz[I]);
    float3 cc = make_float3(0.0,0.0,0.0);
    
    //Neighbor cell
    int I_ = idx(ix, iy, iz);
    float3 u_ = make_float3(0.0,0.0,0.0);
    float3 cc_ =make_float3(0.0,0.0,0.0);

    float3 d_ = make_float3(0.0,0.0,0.0);

    //Set output to zero at start
    dux[I] = 0.0 ;
    duy[I] = 0.0 ;
    duz[I] = 0.0 ;

    //Check if you are in vacuum region
    if (amul(C1_, C1_mul, I)==0) {
        return;
    }

    //dxx
    d_ = make_float3(0.0,0.0,0.0);
    cc = make_float3(amul(C1_, C1_mul, I),amul(C3_, C3_mul, I),amul(C3_, C3_mul, I));
    //Right neighbor
    I_ = idx(hclampx(ix+1), iy, iz);
    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    cc_ = make_float3(amul(C1_, C1_mul, I_),amul(C3_, C3_mul, I_), amul(C3_, C3_mul, I_));
    //Harmonic mean, takes also vacuum regions into account because product will be zero
    cc_ = 2*haddiv(had(cc,cc_),(cc+cc_));
    d_ = wx*wx*had(cc_,(u_-u0));
    //Left neighbour
    I_ = idx(lclampx(ix-1), iy, iz);
    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    cc_ = make_float3(amul(C1_, C1_mul, I_),amul(C3_, C3_mul, I_), amul(C3_, C3_mul, I_));
    cc_ = 2*haddiv(had(cc,cc_),(cc+cc_));
    d_ += wx*wx*had(cc_,(u_-u0));
    
    dux[I] += d_.x ;
    duy[I] += d_.y ;
    duz[I] += d_.z ;

    //dyy
    d_ = make_float3(0.0,0.0,0.0);
    cc = make_float3(amul(C3_, C3_mul, I),amul(C1_, C1_mul, I),amul(C3_, C3_mul, I));
    //Right neighbor
    I_ = idx(ix, hclampy(iy+1), iz);
    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    cc_ = make_float3(amul(C3_, C3_mul, I),amul(C1_, C1_mul, I),amul(C3_, C3_mul, I));
    //Harmonic mean, takes also vacuum regions into account because product will be zero
    cc_ = 2*haddiv(had(cc,cc_),(cc+cc_));
    d_ = wy*wy*had(cc_,(u_-u0));
    //Left neighbour
    I_ = idx(ix, lclampy(iy-1), iz);
    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    cc_ = make_float3(amul(C3_, C3_mul, I),amul(C1_, C1_mul, I),amul(C3_, C3_mul, I));
    cc_ = 2*haddiv(had(cc,cc_),(cc+cc_));
    d_ += wy*wy*had(cc_,(u_-u0));
    
    dux[I] += d_.x ;
    duy[I] += d_.y ;
    duz[I] += d_.z ;


    // //dzz
    // d_ = make_float3(0.0,0.0,0.0);
    // cc = make_float3(amul(C3_, C3_mul, I),amul(C3_, C3_mul, I),amul(C1_, C1_mul, I));
    // //If there is a right neighbor
    // I_ = idx(ix, iy, hclampz(iz+1));
    // u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    // cc_ = make_float3(amul(C3_, C3_mul, I_),amul(C3_, C3_mul, I_), amul(C1_, C1_mul, I_));
    // cc_ = 2*haddiv(had(cc,cc_),(cc+cc_));
    // d_ = wz*wz*had(cc_,(u_-u0));
    // //If there is left neighbour
    // I_ = idx(ix, iy, lclampz(iz-1));
    // u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    // cc_ = make_float3(amul(C3_, C3_mul, I_),amul(C3_, C3_mul, I_), amul(C1_, C1_mul, I_));
    // cc_ = 2*haddiv(had(cc,cc_),(cc+cc_));
    // d_ = wz*wz*had(cc_,(u_-u0));
    
    // dux[I] += d_.x ;
    // duy[I] += d_.y ;
    // duz[I] += d_.z ;


    // Output should be equal to:
    // dux[I] = dxx.x + dxy.y + dxz.z + dyy.x + dyx.y + dzz.x + dzx.z;
    // duy[I] = dyy.y + dyx.x + dyz.z + dxx.y + dxy.x + dzz.y + dzy.z;
    // duz[I] = dzz.z + dzx.x + dzy.y + dxx.z + dxz.x + dyy.z + dyz.y; 
}
