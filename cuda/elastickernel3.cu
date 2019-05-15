#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "stdio.h"

extern "C" __global__ void
Elastodynamic3(float* __restrict__ dux, float* __restrict__ duy, float* __restrict__ duz, 
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
    float3 d2 =make_float3(0.0,0.0,0.0);


    //Shear components: part II

    //dzx
    d_ = make_float3(0.0,0.0,0.0);
    cc = make_float3(amul(C2_, C2_mul, I),0.0,amul(C3_, C3_mul, I));    
    if (iz < Nz-1) {
        I_ = idx(ix, iy, iz+1);
        cc_ = make_float3(amul(C2_, C2_mul, I_), 0.0,amul(C3_, C3_mul, I_));
        if (ix < Nx-1) {
            I_ = idx(ix+1, iy, iz+1);
            d2 = make_float3(ux[I_], uy[I_], uz[I_]);
            I_ = idx(ix, iy, iz+1);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ += 0.5*wx*0.5*wz*had(cc_,d2-u_); 
            
            I_ = idx(ix+1, iy, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ -= 0.5*wx*0.5*wz*had(cc,u_-u0);
        }
        if (ix > 0) {
            I_ = idx(ix-1, iy, iz+1);
            d2 = make_float3(ux[I_], uy[I_], uz[I_]);
            I_ = idx(ix, iy, iz+1);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ += 0.5*wx*0.5*wz*had(cc_,u_-d2); 
            
            I_ = idx(ix-1, iy, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ -= 0.5*wx*0.5*wz*had(cc,u0-u_);
        }
    }
    if (iz > 0) {
        I_ = idx(ix, iy, iz-1);
        cc_ = make_float3(amul(C2_, C2_mul, I_), 0.0,amul(C3_, C3_mul, I_));
        if (ix < Nx-1) {
            I_ = idx(ix+1, iy, iz-1);
            d2 = make_float3(ux[I_], uy[I_], uz[I_]);
            I_ = idx(ix, iy, iz-1);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ -= 0.5*wx*0.5*wz*had(cc_,d2-u_); 
            
            I_ = idx(ix+1, iy, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ += 0.5*wx*0.5*wz*had(cc,u_-u0);
        }
        //If there is neighbour below
        if (ix > 0) {
            I_ = idx(ix-1, iy, iz-1);
            d2 = make_float3(ux[I_], uy[I_], uz[I_]);
            I_ = idx(ix, iy, iz-1);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ -= 0.5*wx*0.5*wz*had(cc_,u_-d2); 
            
            I_ = idx(ix-1, iy, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ += 0.5*wx*0.5*wz*had(cc,u0-u_);
        }
    }
    
    dux[I] += d_.z ;
    duy[I] += 0.0 ;
    duz[I] += d_.x ;


        //dyz
        d_ = make_float3(0.0,0.0,0.0);
        cc = make_float3(0.0, amul(C3_, C3_mul, I), amul(C2_, C2_mul, I));    
        //If there is a neighbor to the right
        if (iy < Ny-1) {
            I_ = idx(ix, iy+1, iz);
            cc_ = make_float3( 0.0,amul(C3_, C3_mul, I_),amul(C2_, C2_mul, I_));
            if (iz < Nz-1) {
                I_ = idx(ix, iy+1, iz+1);
                d2 = make_float3(ux[I_], uy[I_], uz[I_]);
                I_ = idx(ix, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d_ += 0.5*wy*0.5*wz*had(cc_,d2-u_); 
                
                I_ = idx(ix, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d_ -= 0.5*wy*0.5*wz*had(cc,u_-u0);
            }
            if (iz > 0) {
                I_ = idx(ix, iy+1, iz-1);
                d2 = make_float3(ux[I_], uy[I_], uz[I_]);
                I_ = idx(ix, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d_ += 0.5*wy*0.5*wz*had(cc_,u_-d2); 
                
                I_ = idx(ix, iy, iz-1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d_ -= 0.5*wy*0.5*wz*had(cc,u0-u_);
            }
        }
        if (iy > 0) {
            I_ = idx(ix, iy-1, iz);
            cc_ = make_float3(0.0,amul(C3_, C3_mul, I_),amul(C2_, C2_mul, I_));
            if (iz < Nz-1) {
                I_ = idx(ix, iy-1, iz+1);
                d2 = make_float3(ux[I_], uy[I_], uz[I_]);
                I_ = idx(ix, iy-1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d_ -= 0.5*wy*0.5*wz*had(cc_,d2-u_); 
                
                I_ = idx(ix, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d_ += 0.5*wy*0.5*wz*had(cc,u_-u0);
            }
            if (iz > 0) {
                I_ = idx(ix, iy-1, iz-1);
                d2 = make_float3(ux[I_], uy[I_], uz[I_]);
                I_ = idx(ix, iy-1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d_ -= 0.5*wy*0.5*wz*had(cc_,u_-d2); 
                
                I_ = idx(ix, iy, iz-1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d_ += 0.5*wy*0.5*wz*had(cc,u0-u_);
            }
        }

    dux[I] += 0.0 ;
    duy[I] += d_.z ;
    duz[I] += d_.y ;


    //dzy
    d_ = make_float3(0.0,0.0,0.0);
    cc = make_float3(0.0, amul(C2_, C2_mul, I),amul(C3_, C3_mul, I));    
    if (iz < Nz-1) {
        I_ = idx(ix, iy, iz+1);
        cc_ = make_float3( 0.0, amul(C2_, C2_mul, I_),amul(C3_, C3_mul, I_));
        if (iy < Ny-1) {
            I_ = idx(ix, iy+1, iz+1);
            d2 = make_float3(ux[I_], uy[I_], uz[I_]);
            I_ = idx(ix, iy, iz+1);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ += 0.5*wy*0.5*wz*had(cc_,d2-u_); 
            
            I_ = idx(ix, iy+1, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ -= 0.5*wy*0.5*wz*had(cc,u_-u0);
        }
        if (iy > 0) {
            I_ = idx(ix, iy-1, iz+1);
            d2 = make_float3(ux[I_], uy[I_], uz[I_]);
            I_ = idx(ix, iy, iz+1);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ += 0.5*wy*0.5*wz*had(cc_,u_-d2); 
            
            I_ = idx(ix, iy-1, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ -= 0.5*wy*0.5*wz*had(cc,u0-u_);
        }
    }
    if (iz > 0) {
        I_ = idx(ix, iy, iz-1);
        cc_ = make_float3( 0.0, amul(C2_, C2_mul, I_),amul(C3_, C3_mul, I_));
        if (iy < Ny-1) {
            I_ = idx(ix, iy+1, iz-1);
            d2 = make_float3(ux[I_], uy[I_], uz[I_]);
            I_ = idx(ix, iy, iz-1);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ -= 0.5*wy*0.5*wz*had(cc_,d2-u_); 
            
            I_ = idx(ix, iy+1, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ += 0.5*wy*0.5*wz*had(cc,u_-u0);
        }
        //If there is neighbour below
        if (iy > 0) {
            I_ = idx(ix, iy-1, iz-1);
            d2 = make_float3(ux[I_], uy[I_], uz[I_]);
            I_ = idx(ix, iy, iz-1);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ -= 0.5*wy*0.5*wz*had(cc_,u_-d2); 
            
            I_ = idx(ix, iy-1, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ += 0.5*wy*0.5*wz*had(cc,u0-u_);
        }
    }

    dux[I] += 0.0 ;
    duy[I] += d_.z ;
    duz[I] += d_.y ;


    //Output should be equal to:
    // dux[I] = dxx.x + dxy.y + dxz.z + dyy.x + dyx.y + dzz.x + dzx.z;
    // duy[I] = dyy.y + dyx.x + dyz.z + dxx.y + dxy.x + dzz.y + dzy.z;
    // duz[I] = dzz.z + dzx.x + dzy.y + dxx.z + dxz.x + dyy.z + dyz.y; 
}
