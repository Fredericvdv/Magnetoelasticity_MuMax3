#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "stdio.h"

extern "C" __global__ void
Elastodynamic2(float* __restrict__ dux, float* __restrict__ duy, float* __restrict__ duz, 
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

    //Check if you are in a vacuum
    if (amul(C1_, C1_mul, I)==0) {
        return;
    }
    
    //Neighbor cell
    int I_ = idx(ix, iy, iz);
    float3 u_ = make_float3(0.0,0.0,0.0);
    float3 cc_ =make_float3(0.0,0.0,0.0);
    float3 cc__ =make_float3(0.0,0.0,0.0);

    float3 d_ = make_float3(0.0,0.0,0.0);
    float3 u__ =make_float3(0.0,0.0,0.0);

    dux[I] = 0.0 ;
    duy[I] = 0.0 ;
    duz[I] = 0.0 ;


    //Normal components

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


    // //Shear components: part I   
    // //dxy without boundaries
    // d_ = make_float3(0.0,0.0,0.0);
    // cc = make_float3(amul(C2_, C2_mul, I)+amul(C3_, C3_mul, I),amul(C2_, C2_mul, I)+amul(C3_, C3_mul, I),0);  
    // //(i+1,j+1)
    // I_ = idx(hclampx(ix+1),hclampy(iy+1), iz);
    // d_ += make_float3(ux[I_], uy[I_], uz[I_]);
    // //(i-1,j-1)
    // I_ = idx(lclampx(ix-1),lclampy(iy-1), iz);
    // d_ += make_float3(ux[I_], uy[I_], uz[I_]);
    // //(i+1,j-1)
    // I_ = idx(hclampx(ix+1),lclampy(iy-1), iz);
    // d_ -= make_float3(ux[I_], uy[I_], uz[I_]);
    // //(i-1,j+1)
    // I_ = idx(lclampx(ix-1),hclampy(iy+1), iz);
    // d_ -= make_float3(ux[I_], uy[I_], uz[I_]);

    // d_ = had(cc,d_)*0.25*wx*wy;

    /////////////////////////
    // //dxy without boundaries in 6th order
    // d_ = make_float3(0.0,0.0,0.0);
    // cc = make_float3(amul(C2_, C2_mul, I)+amul(C3_, C3_mul, I),amul(C2_, C2_mul, I)+amul(C3_, C3_mul, I),0);  
    // //(i+1,j+1)
    // I_ = idx(hclampx(ix+1),hclampy(iy+1), iz);
    // d_ += 270*make_float3(ux[I_], uy[I_], uz[I_]);
    // //(i-1,j-1)
    // I_ = idx(lclampx(ix-1),lclampy(iy-1), iz);
    // d_ += 270*make_float3(ux[I_], uy[I_], uz[I_]);
    // //(i+1,j-1)
    // I_ = idx(hclampx(ix+1),lclampy(iy-1), iz);
    // d_ -= 270*make_float3(ux[I_], uy[I_], uz[I_]);
    // //(i-1,j+1)
    // I_ = idx(lclampx(ix-1),hclampy(iy+1), iz);
    // d_ -= 270*make_float3(ux[I_], uy[I_], uz[I_]);

    // //(i+2,j+2)
    // I_ = idx(hclampx(ix+2),hclampy(iy+2), iz);
    // d_ -= 27*make_float3(ux[I_], uy[I_], uz[I_]);
    // //(i-2,j-2)
    // I_ = idx(lclampx(ix-2),lclampy(iy-2), iz);
    // d_ -= 27*make_float3(ux[I_], uy[I_], uz[I_]);
    // //(i+2,j-2)
    // I_ = idx(hclampx(ix+2),lclampy(iy-2), iz);
    // d_ += 27*make_float3(ux[I_], uy[I_], uz[I_]);
    // //(i-1,j+2)
    // I_ = idx(lclampx(ix-2),hclampy(iy+2), iz);
    // d_ += 27*make_float3(ux[I_], uy[I_], uz[I_]);

    // //(i+3,j+3)
    // I_ = idx(hclampx(ix+3),hclampy(iy+3), iz);
    // d_ += 2*make_float3(ux[I_], uy[I_], uz[I_]);
    // //(i-3,j-3)
    // I_ = idx(lclampx(ix-3),lclampy(iy-3), iz);
    // d_ += 2*make_float3(ux[I_], uy[I_], uz[I_]);
    // //(i+3,j-3)
    // I_ = idx(hclampx(ix+3),lclampy(iy-3), iz);
    // d_ -= 2*make_float3(ux[I_], uy[I_], uz[I_]);
    // //(i-3,j+3)
    // I_ = idx(lclampx(ix-3),hclampy(iy+3), iz);
    // d_ -= 2*make_float3(ux[I_], uy[I_], uz[I_]);

    // d_ = had(cc,d_)*0.0013888*wx*wy;

    // dux[I] += d_.x ;
    // duy[I] += d_.y ;
    // duz[I] += 0.0 ;

    //////////////

    //Shear components: part I   

    //dxy
    d_ = make_float3(0.0,0.0,0.0);
    cc = make_float3(amul(C3_, C3_mul, I),amul(C2_, C2_mul, I),0);  
    //Check if it is necessary to do computation
    if ((ix < Nx-1 || PBCx==1) && amul(C1_, C1_mul, idx(hclampx(ix+1), iy, iz))!=0) {
        if ((iy < Ny-1 || PBCy==1) && amul(C1_, C1_mul, idx(ix, hclampy(iy+1), iz))!=0) {
            if (amul(C1_, C1_mul, idx(hclampx(ix+1), hclampy(iy+1), iz))!=0) {
                //(ix+1, iy+1) - (ix+1, iy) 
                I_ = idx(hclampx(ix+1), iy, iz);
                cc_ = make_float3(amul(C3_, C3_mul, I_),amul(C2_, C2_mul, I_), 0.0);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I_ = idx(hclampx(ix+1), hclampy(iy+1), iz);
                cc__ = make_float3(amul(C3_, C3_mul, I_),amul(C2_, C2_mul, I_), 0.0);
                u__ = make_float3(ux[I_], uy[I_], uz[I_]);
                cc_ = 2*haddiv(had(cc_,cc__),(cc_+cc__));
                d_ += 0.5*wx*0.5*wy*had(cc_,u__-u_);
                //(ix, iy+1) - (ix, iy) 
                I_ = idx(ix, hclampy(iy+1), iz);
                cc_ = make_float3(amul(C3_, C3_mul, I_),amul(C2_, C2_mul, I_), 0.0);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                cc_ = 2*haddiv(had(cc,cc_),(cc+cc_));
                d_ -= 0.5*wx*0.5*wy*had(cc_,u_-u0);
            }
        }
        //Check if there is vacuum cell in the lower-right corner
        if ((iy > 0 || PBCy==1) && amul(C1_, C1_mul, idx(ix, lclampy(iy-1), iz))!=0) {
            if (amul(C1_, C1_mul, idx(hclampx(ix+1), lclampy(iy-1), iz))!=0) {
                //(ix+1, iy) - (ix+1, iy-1) 
                I_ = idx(hclampx(ix+1), iy, iz);
                cc_ = make_float3(amul(C3_, C3_mul, I_),amul(C2_, C2_mul, I_), 0.0);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I_ = idx(hclampx(ix+1), lclampy(iy-1), iz);
                cc__ = make_float3(amul(C3_, C3_mul, I_),amul(C2_, C2_mul, I_), 0.0);
                u__ = make_float3(ux[I_], uy[I_], uz[I_]);
                cc_ = 2*haddiv(had(cc_,cc__),(cc_+cc__));
                d_ += 0.5*wx*0.5*wy*had(cc_,u_-u__);
                //(ix, iy) - (ix, iy-1)
                I_ = idx(ix, lclampy(iy-1), iz);
                cc_ = make_float3(amul(C3_, C3_mul, I_),amul(C2_, C2_mul, I_), 0.0);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                cc_ = 2*haddiv(had(cc,cc_),(cc+cc_));
                d_ -= 0.5*wx*0.5*wy*had(cc_,u0-u_);
            }
        }
    }
    if ((ix > 0 || PBCx==1) && amul(C1_, C1_mul, idx(lclampx(ix-1), iy, iz))!=0) {
        //Check if there is vacuum cell in the higher-left corner
        if ((iy < Ny-1 || PBCy==1) && amul(C1_, C1_mul, idx(ix, hclampy(iy+1), iz))!=0) {
            if (amul(C1_, C1_mul, idx(lclampx(ix-1), hclampy(iy+1), iz))!=0) {
                //(ix-1, iy+1) - (ix-1, iy)
                I_ = idx(lclampx(ix-1), iy, iz);
                cc_ = make_float3(amul(C3_, C3_mul, I_),amul(C2_, C2_mul, I_), 0.0);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I_ = idx(lclampx(ix-1), hclampy(iy+1), iz);
                cc__ = make_float3(amul(C3_, C3_mul, I_),amul(C2_, C2_mul, I_), 0.0);
                u__ = make_float3(ux[I_], uy[I_], uz[I_]);
                cc_ = 2*haddiv(had(cc_,cc__),(cc_+cc__));
                d_ -= 0.5*wx*0.5*wy*had(cc_,u__-u_);
                //(ix, iy+1) - (ix, iy)
                I_ = idx(ix, hclampy(iy+1), iz);
                cc_ = make_float3(amul(C3_, C3_mul, I_),amul(C2_, C2_mul, I_), 0.0);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                cc_ = 2*haddiv(had(cc,cc_),(cc+cc_));
                d_ += 0.5*wx*0.5*wy*had(cc_,u_-u0);
            }
        }
        //Check if there is vacuum cell in the lower-left corner
        if ((iy > 0 || PBCy==1) && amul(C1_, C1_mul, idx(ix, lclampy(iy-1), iz))!=0) {
            if (amul(C1_, C1_mul, idx(lclampx(ix-1), lclampy(iy-1), iz))!=0) {
                //(ix-1, iy) - (ix-1, iy-1)
                I_ = idx(lclampx(ix-1), iy, iz);
                cc_ = make_float3(amul(C3_, C3_mul, I_),amul(C2_, C2_mul, I_), 0.0);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I_ = idx(lclampx(ix-1), lclampy(iy-1), iz);
                cc__ = make_float3(amul(C3_, C3_mul, I_),amul(C2_, C2_mul, I_), 0.0);
                u__ = make_float3(ux[I_], uy[I_], uz[I_]);
                cc_ = 2*haddiv(had(cc_,cc__),(cc_+cc__));
                d_ -= 0.5*wx*0.5*wy*had(cc_,u_-u__);
                //(ix, iy) - (ix, iy-1)
                I_ = idx(ix, lclampy(iy-1), iz);
                cc_ = make_float3(amul(C3_, C3_mul, I_),amul(C2_, C2_mul, I_), 0.0);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                cc_ = 2*haddiv(had(cc,cc_),(cc+cc_));
                d_ += 0.5*wx*0.5*wy*had(cc_,u0-u_);
            }
        }
    }

    dux[I] += d_.y ;
    duy[I] += d_.x ;
    duz[I] += 0.0 ;



    //dyx
    d_ = make_float3(0.0,0.0,0.0);
    cc = make_float3(amul(C2_, C2_mul, I),amul(C3_, C3_mul, I),0);  
    if ((iy < Ny-1 || PBCy==1) && amul(C1_, C1_mul, idx(ix, hclampy(iy+1), iz))!=0) {
        if ((ix < Nx-1 || PBCx==1) && amul(C1_, C1_mul, idx(hclampx(ix+1), iy, iz))!=0) {
            if (amul(C1_, C1_mul, idx(hclampx(ix+1), hclampy(iy+1), iz))!=0) {
                //(ix+1, iy+1) - (ix, iy+1)
                I_ = idx(ix, hclampy(iy+1), iz);
                cc_ = make_float3(amul(C2_, C2_mul, I_),amul(C3_, C3_mul, I_), 0.0);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I_ = idx(hclampx(ix+1), hclampy(iy+1), iz);
                cc__ = make_float3(amul(C2_, C2_mul, I_),amul(C3_, C3_mul, I_), 0.0);
                u__ = make_float3(ux[I_], uy[I_], uz[I_]);
                cc_ = 2*haddiv(had(cc_,cc__),(cc_+cc__));
                d_ += 0.5*wx*0.5*wy*had(cc_,u__-u_);
                //(ix+1, iy) - (ix, iy)
                I_ = idx(hclampx(ix+1), iy, iz);
                cc_ = make_float3(amul(C2_, C2_mul, I_),amul(C3_, C3_mul, I_), 0.0);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                cc_ = 2*haddiv(had(cc,cc_),(cc+cc_));
                d_ -= 0.5*wx*0.5*wy*had(cc_,u_-u0);
            }
        }
        if ((ix > 0 || PBCx==1) && amul(C1_, C1_mul, idx(lclampx(ix-1), iy, iz))!=0) {
            if (amul(C1_, C1_mul, idx(hclampx(ix+1), lclampy(iy-1), iz))!=0) {
                //(ix-1, iy+1) - (ix, iy+1)
                I_ = idx(ix, hclampx(iy+1), iz);
                cc_ = make_float3(amul(C2_, C2_mul, I_),amul(C3_, C3_mul, I_), 0.0);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I_ = idx(lclampx(ix-1), hclampy(iy+1), iz);
                cc_ = make_float3(amul(C2_, C2_mul, I_),amul(C3_, C3_mul, I_), 0.0);
                u__ = make_float3(ux[I_], uy[I_], uz[I_]);
                cc_ = 2*haddiv(had(cc_,cc__),(cc_+cc__));
                d_ += 0.5*wx*0.5*wy*had(cc_,u_-u__);
                //(ix-1, iy) - (ix, iy)
                I_ = idx(lclampx(ix-1), iy, iz);
                cc_ = make_float3(amul(C2_, C2_mul, I_),amul(C3_, C3_mul, I_), 0.0);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                cc_ = 2*haddiv(had(cc,cc_),(cc+cc_));
                d_ -= 0.5*wx*0.5*wy*had(cc_,u0-u_);
            }
        }
    }
    if ((iy > 0 || PBCy==1) && amul(C1_, C1_mul, idx(ix,lclampy(iy-1), iz))!=0) {
        if ((ix < Nx-1 || PBCx==1) && amul(C1_, C1_mul, idx(hclampx(ix+1), iy, iz))!=0) {
            if (amul(C1_, C1_mul, idx(hclampx(ix+1), lclampy(iy-1), iz))!=0) {
                //(ix+1, iy-1) - (ix, iy-1)
                I_ = idx(ix, lclampy(iy-1), iz);
                cc_ = make_float3(amul(C2_, C2_mul, I_),amul(C3_, C3_mul, I_), 0.0);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I_ = idx(hclampx(ix+1), lclampy(iy-1), iz);
                cc_ = make_float3(amul(C2_, C2_mul, I_),amul(C3_, C3_mul, I_), 0.0);
                u__ = make_float3(ux[I_], uy[I_], uz[I_]);
                cc_ = 2*haddiv(had(cc_,cc__),(cc_+cc__));
                d_ -= 0.5*wx*0.5*wy*had(cc_,u__-u_);
                //(ix+1, iy) - (ix, iy)
                I_ = idx(hclampx(ix+1), iy, iz);
                cc_ = make_float3(amul(C2_, C2_mul, I_),amul(C3_, C3_mul, I_), 0.0);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                cc_ = 2*haddiv(had(cc,cc_),(cc+cc_));
                d_ += 0.5*wx*0.5*wy*had(cc_,u_-u0);
            }
        }
        if ((ix > 0 || PBCx==1) && amul(C1_, C1_mul, idx(lclampx(ix-1), iy, iz))!=0) {
            if (amul(C1_, C1_mul, idx(lclampx(ix-1), lclampy(iy-1), iz))!=0) {
                //(ix, iy-1) - (ix-1, iy-1)
                I_ = idx(ix, lclampy(iy-1), iz);
                cc_ = make_float3(amul(C2_, C2_mul, I_),amul(C3_, C3_mul, I_), 0.0);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I_ = idx(lclampx(ix-1), lclampy(iy-1), iz);
                cc_ = make_float3(amul(C2_, C2_mul, I_),amul(C3_, C3_mul, I_), 0.0);
                u__ = make_float3(ux[I_], uy[I_], uz[I_]);
                cc_ = 2*haddiv(had(cc_,cc__),(cc_+cc__));
                d_ -= 0.5*wx*0.5*wy*had(cc_,u_-u__);
                //(ix, iy) - (ix-1, iy)
                I_ = idx(lclampx(ix-1), iy, iz);
                cc_ = make_float3(amul(C2_, C2_mul, I_),amul(C3_, C3_mul, I_), 0.0);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                cc_ = 2*haddiv(had(cc,cc_),(cc+cc_));
                d_ += 0.5*wx*0.5*wy*had(cc_,u0-u_);
            }
        }
    }

    dux[I] += d_.y ;
    duy[I] += d_.x ;
    duz[I] += 0.0 ;




    // //dxz
    // d_ = make_float3(0.0,0.0,0.0);
    // cc = make_float3(amul(C3_, C3_mul, I),0.0, amul(C2_, C2_mul, I));    
    // if (ix < Nx-1) {
    //     I_ = idx(ix+1, iy, iz);
    //     if (amul(C1_, C1_mul, I_)!=0) {
    //         cc_ = make_float3(amul(C3_, C3_mul, I_), 0.0,amul(C2_, C2_mul, I_));
    //         if (iz < Nz-1) {
    //             I_ = idx(ix+1, iy, iz+1);
    //             if (amul(C1_, C1_mul, I_)!=0) {
    //                 u__ = make_float3(ux[I_], uy[I_], uz[I_]);
    //                 I_ = idx(ix+1, iy, iz);
    //                 u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //                 d_ += 0.5*wx*0.5*wz*had(cc_,u__-u_); 
    //             }
                
    //             I_ = idx(ix, iy, iz+1);
    //             if (amul(C1_, C1_mul, I_)!=0) {
    //                 u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //                 d_ -= 0.5*wx*0.5*wz*had(cc,u_-u0);
    //             }
    //         }
    //         if (iz > 0) {
    //             I_ = idx(ix+1, iy, iz-1);
    //             if (amul(C1_, C1_mul, I_)!=0) {
    //                 u__ = make_float3(ux[I_], uy[I_], uz[I_]);
    //                 I_ = idx(ix+1, iy, iz);
    //                 u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //                 d_ += 0.5*wx*0.5*wz*had(cc_,u_-u__); 
    //             }
                
    //             I_ = idx(ix, iy, iz-1);
    //             if (amul(C1_, C1_mul, I_)!=0) {
    //                 u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //                 d_ -= 0.5*wx*0.5*wz*had(cc,u0-u_);
    //             }
    //         }
    //     }
    // }
    // if (ix > 0) {
    //     I_ = idx(ix-1, iy, iz);
    //     if (amul(C1_, C1_mul, I_)!=0) {
    //         cc_ = make_float3(amul(C3_, C3_mul, I_), 0.0,amul(C2_, C2_mul, I_));
    //         if (iz < Nz-1) {
    //             I_ = idx(ix-1, iy, iz+1);
    //             if (amul(C1_, C1_mul, I_)!=0) {
    //                 u__ = make_float3(ux[I_], uy[I_], uz[I_]);
    //                 I_ = idx(ix-1, iy, iz);
    //                 u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //                 d_ -= 0.5*wx*0.5*wz*had(cc_,u__-u_); 
    //             }
                
    //             I_ = idx(ix, iy, iz+1);
    //             if (amul(C1_, C1_mul, I_)!=0) {
    //                 u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //                 d_ += 0.5*wx*0.5*wz*had(cc,u_-u0);
    //             }
    //         }
    //         if (iz > 0) {
    //             I_ = idx(ix-1, iy, iz-1);
    //             if (amul(C1_, C1_mul, I_)!=0) {
    //                 u__ = make_float3(ux[I_], uy[I_], uz[I_]);
    //                 I_ = idx(ix-1, iy, iz);
    //                 u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //                 d_ -= 0.5*wx*0.5*wz*had(cc_,u_-u__); 
    //             }
                
    //             I_ = idx(ix, iy, iz-1);
    //             if (amul(C1_, C1_mul, I_)!=0) {
    //                 u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //                 d_ += 0.5*wx*0.5*wz*had(cc,u0-u_);
    //             }
    //         }
    //     }
    // }

    // dux[I] += d_.z ;
    // duy[I] += 0.0 ;
    // duz[I] += d_.x ;


    //Output should be equal to:
    // dux[I] = dxx.x + dxy.y + dxz.z + dyy.x + dyx.y + dzz.x + dzx.z;
    // duy[I] = dyy.y + dyx.x + dyz.z + dxx.y + dxy.x + dzz.y + dzy.z;
    // duz[I] = dzz.z + dzx.x + dzy.y + dxx.z + dxz.x + dyy.z + dyz.y; 
}
