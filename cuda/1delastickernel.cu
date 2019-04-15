#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "stdio.h"

// ~ Bibliothee can C inladen of zichtbaar maken
// Nx aantal cellen in x-richting
extern "C" __global__ void
SecondDerivative(float* __restrict__ dux, float* __restrict__ duy, float* __restrict__ duz, 
                 float* __restrict__ ux, float* __restrict__ uy, float* __restrict__ uz,
                 int Nx, int Ny, int Nz, float wx, float wy, float wz, 
                 float* __restrict__  C1_, float  C1_mul, float* __restrict__  C2_, float  C2_mul, 
                 float* __restrict__  C3_, float  C3_mul, uint8_t PBC) {
                    
    printf("Hello world from CUDA\n");


    //positie van cell waar we in aan het kijkken zijn is: ix,iy,iz
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    //if cell position is out of mesh --> do nothing
    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    // Central cell
    int I = idx(ix, iy, iz);
    float3 u0 = make_float3(ux[I], uy[I], uz[I]);
    float  c1 = amul(C1_, C1_mul, I);
    float  c2 = amul(C2_, C2_mul, I);
    float  c3 = amul(C3_, C3_mul, I);
    //uint8_t r0 = regions[I];


    //TODO: implement boundary conditions
    //Neighbours
    //i_l = idx(lclampx(ix-1),iy,iz);
    //i_r = idx(hclampx(ix+1),iy,iz);
    //u_l = make_float3(ux[i_l],uy[i_l],uz[i_l]);
    //u_l = ( is0(u_l)? u0: u_l);
    //u_r = make_float3(ux[i_r],uy[i_r],uz[i_r]);
    //u_r = ( is0(u_r)? u0: u_r);
    
    //initialize derivatives
    //Higher neighbor
    int I_ = idx(ix, iy, iz);
    float3 u_ = make_float3(0.0,0.0,0.0);
    float  c1_ = 0.0;
    float  c2_ = 0.0;
    float  c3_ = 0.0;

    //Lower neighbor
    int I__ = idx(ix, iy, iz);
    float3 u__ = make_float3(0.0,0.0,0.0);
    float  c1__ = 0.0;
    float  c2__ = 0.0;
    float  c3__ = 0.0;

    //make vector to use the right constant for the right component of u 
    float3 cc = make_float3(0.0,0.0,0.0);
    float3 cc_ =make_float3(0.0,0.0,0.0);
    float3 cc__ =make_float3(0.0,0.0,0.0);

    float3 d_ = make_float3(0.0,0.0,0.0);

    float3 d1 =make_float3(0.0,0.0,0.0);
    float3 d2 =make_float3(0.0,0.0,0.0);
    float3 d3 =make_float3(0.0,0.0,0.0);

    //////////////////////////////////////////
    //Calculate derivatives
    //dxx
    //if there are no neighbours: keep the second deriative to zero = do nothing
    if (ix-1<0 && ix+1>=Nx) {
        d_ = make_float3(0.0,0.0,0.0);
    }
    else {
        //Only neighbour to the right
        if (ix-1<0) {
            I_ = idx(ix+1, iy, iz);
            I__ = idx(ix, iy, iz);
        //Only neighbour to the left
        } else if (ix+1>=Nx) {
            I_ = idx(ix, iy, iz);
            I__ = idx(ix-1, iy, iz);
        //Neighbours on both sides
        } else {
            I_ = idx(ix+1, iy, iz);
            I__ = idx(ix-1, iy, iz);          
        }
    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    c1_ = amul(C1_, C1_mul, I_);
    c2_ = amul(C2_, C2_mul, I_);
    c3_ = amul(C3_, C3_mul, I_);
    cc_ = make_float3(c1_,c3_,c3_);
    u__ = make_float3(ux[I__], uy[I__], uz[I__]);
    c1__ = amul(C1_, C1_mul, I__);
    c2__ = amul(C2_, C2_mul, I__);
    c3__ = amul(C3_, C3_mul, I__);
    cc__ = make_float3(c1__,c3__,c3__);
    d_ = 0.5*wx*wx*(had((cc+cc_),(u_-u0))+had((cc+cc__),(u__-u0)));
    }

    dux[I] += d_.x ;
    duy[I] += d_.y ;
    duz[I] += d_.z ;

    //dyy
    cc = make_float3(c3,c1,c3);
    if (iy-1<0 && iy+1>=Ny) {
        d_ = make_float3(0.0,0.0,0.0);
    }
    else {
        if (iy-1<0) {
            I_ = idx(ix, iy+1, iz);
            I__ = idx(ix, iy, iz);
        } else if (iy+1>=Ny) {
            I_ = idx(ix, iy, iz);
            I__ = idx(ix, iy-1, iz);
        } else {
            I_ = idx(ix, iy+1, iz);
            I__ = idx(ix, iy-1, iz);
        }
    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    c1_ = amul(C1_, C1_mul, I_);
    c2_ = amul(C2_, C2_mul, I_);
    c3_ = amul(C3_, C3_mul, I_);
    cc_ = make_float3(c3_,c1_,c3_);
    u__ = make_float3(ux[I__], uy[I__], uz[I__]);
    c1__ = amul(C1_, C1_mul, I__);
    c2__ = amul(C2_, C2_mul, I__);
    c3__ = amul(C3_, C3_mul, I__);
    cc__ = make_float3(c3__,c1__,c3__);
    d_ = 0.5*wy*wy*(had((cc+cc_),(u_-u0))+had((cc+cc__),(u__-u0)));
    }

    dux[I] += d_.x ;
    duy[I] += d_.y ;
    duz[I] += d_.z ;

    //dzz
    cc = make_float3(c3,c3,c1);
    if (iz-1<0 && iz+1>=Nz) {
        d_ = make_float3(0.0,0.0,0.0);
    }
    else {
        if (iz-1<0) {
            I_ = idx(ix, iy, iz+1);
            I__ = idx(ix, iy, iz);
        } else if (iz+1>=Nz) {
            I_ = idx(ix, iy, iz);
            I__ = idx(ix, iy, iz-1);
        } else {
            I_ = idx(ix, iy, iz+1);
            I__ = idx(ix, iy, iz-1);
        }
    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    c1_ = amul(C1_, C1_mul, I_);
    c2_ = amul(C2_, C2_mul, I_);
    c3_ = amul(C3_, C3_mul, I_);
    cc_ = make_float3(c3_,c3_,c1_);
    u__ = make_float3(ux[I__], uy[I__], uz[I__]);
    c1__ = amul(C1_, C1_mul, I__);
    c2__ = amul(C2_, C2_mul, I__);
    c3__ = amul(C3_, C3_mul, I__);
    cc__ = make_float3(c3__,c3__,c1__);
    d_ = 0.5*wz*wz*(had((cc+cc_),(u_-u0))+had((cc+cc__),(u__-u0)));
    }

    dux[I] += d_.x ;
    duy[I] += d_.y ;
    duz[I] += d_.z ;

    
    //dxy
    cc = make_float3(c3,c2,0);    
    //if there are no neighbours in x/y-direction: keep the second deriative to zero
    if ((ix-1<0 && ix+1>=Nx) || (iy-1<0 && iy+1>=Ny)) {
        d_ = make_float3(0.0,0.0,0.0);
    }
    else {
        //No neighbor in the right direction
        if (ix-1<0) {
            if (iy-1<0) {
                //Calculate change in y-direction at postion ix = d2
                //0.5*wy*(FWD + BWD) with BWD=0
                I_ = idx(ix, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wy*(u_-u0);
                //Calculate change in y-direction at postion ix+1 = d1
                //rectangular mesh: if (ix+1,iy) and (ix,iy+1) are present, then (ix+1,iy+1) is also present
                I_ = idx(ix+1, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix+1, iy, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wy*(u_-u__);
            } else if (iy+1>=Ny) {
                I_ = idx(ix, iy-1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wy*(u0-u_);

                I_ = idx(ix+1, iy, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix+1, iy-1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wy*(u_-u__);
            } else {
                I_ = idx(ix, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy-1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d2 = 0.5*wy*(u_-u__);

                I_ = idx(ix+1, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix+1, iy-1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wy*(u_-u__);
            }
            I_ = idx(ix+1, iy, iz);
            c2_ = amul(C2_, C2_mul, I_);
            c3_ = amul(C3_, C3_mul, I_);
            cc_ = make_float3(c3_ ,c2_ ,0.0);
            // dxy = 0.5*wx*(FWD - BWD) with BWD = 0 beause ix-1<0
            d_ = 0.5*wx*(had(cc_,d1)-had(cc,d2));

        } else if (ix+1>=Nx) {
            if (iy-1<0) {
                //Calculate change in y-direction at postion ix = d2
                I_ = idx(ix, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wy*(u_-u0);
                //Calculate change in y-direction at postion ix-1 = d1
                I_ = idx(ix-1, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wy*(u_-u__);
            } else if (iy+1>=Ny) {
                I_ = idx(ix, iy-1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wy*(u0-u_);

                I_ = idx(ix-1, iy, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy-1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wy*(u_-u__);
            } else {
                I_ = idx(ix, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy-1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d2 = 0.5*wy*(u_-u__);

                I_ = idx(ix-1, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy-1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wy*(u_-u__);
            }
            I_ = idx(ix-1, iy, iz);
            c2_ = amul(C2_, C2_mul, I_);
            c3_ = amul(C3_, C3_mul, I_);
            cc_ = make_float3(c3_ ,c2_ ,0.0);
            d_ = 0.5*wx*(had(cc,d2)-had(cc_,d3));
        } else {
            if (iy-1<0) {
                //Calculate change in y-direction at postion ix+1 = d1
                I_ = idx(ix+1, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix+1, iy, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wy*(u_-u__);
                //Calculate change in y-direction at postion ix-1 = d3
                I_ = idx(ix-1, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wy*(u_-u__);
            } else if (iy+1>=Ny) {
                I_ = idx(ix+1, iy, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix+1, iy-1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wy*(u_-u__);

                I_ = idx(ix-1, iy, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy-1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wy*(u_-u__);
            } else {
                I_ = idx(ix+1, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix+1, iy-1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wy*(u_-u__);
                I_ = idx(ix-1, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy-1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wy*(u_-u__);
            }
            I_ = idx(ix+1, iy, iz);
            c2_ = amul(C2_, C2_mul, I_);
            c3_ = amul(C3_, C3_mul, I_);
            cc_ = make_float3(c3_ ,c2_ ,0.0);
            I__ = idx(ix-1, iy, iz);
            c2__ = amul(C2_, C2_mul, I__);
            c3__ = amul(C3_, C3_mul, I__);
            cc__ = make_float3(c3__ ,c2__ ,0.0);
            d_ = 0.5*wx*(had(cc_,d1)-had(cc__,d3));
        }
    }

    dux[I] += d_.y ;
    duy[I] += d_.x ;
    duz[I] += 0.0 ;

    //dyx
    cc = make_float3(c2,c3,0);
    if ((ix-1<0 && ix+1>=Nx) || (iy-1<0 && iy+1>=Ny)) {
        d_ = make_float3(0.0,0.0,0.0);
    }
    else {
        if (iy-1<0) {
            if (ix-1<0) {
                I_ = idx(ix+1, iy, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wx*(u_-u0);

                I_ = idx(ix+1, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy+1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wx*(u_-u__);
            } else if (ix+1>=Nx) {
                I_ = idx(ix-1, iy, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wx*(u0-u_);

                I_ = idx(ix, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy+1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wx*(u_-u__);
            } else {
                I_ = idx(ix+1, iy, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d2 = 0.5*wx*(u_-u__);

                I_ = idx(ix+1, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy+1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wx*(u_-u__);
            }
            I_ = idx(ix, iy+1, iz);
            c2_ = amul(C2_, C2_mul, I_);
            c3_ = amul(C3_, C3_mul, I_);
            cc_ = make_float3(c2_ ,c3_ ,0.0);
            d_ = 0.5*wy*(had(cc_,d1)-had(cc,d2));

        } else if (iy+1>=Ny) {
            if (ix-1<0) {
                I_ = idx(ix+1, iy, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wx*(u_-u0);

                I_ = idx(ix+1, iy-1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy-1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wx*(u_-u__);
            } else if (ix+1>=Nx) {
                I_ = idx(ix-1, iy, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wx*(u0-u_);

                I_ = idx(ix, iy-1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy-1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wx*(u_-u__);
            } else {
                I_ = idx(ix+1, iy, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d2 = 0.5*wx*(u_-u__);

                I_ = idx(ix+1, iy-1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy-1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wx*(u_-u__);
            }
            I_ = idx(ix, iy-1, iz);
            c2_ = amul(C2_, C2_mul, I_);
            c3_ = amul(C3_, C3_mul, I_);
            cc_ = make_float3(c2_ ,c3_ ,0.0);
            d_ = 0.5*wy*(had(cc,d2)-had(cc_,d3));

        } else {
            if (ix-1<0) {
                //Calculate change in y-direction at postion ix+1 = d1
                I_ = idx(ix+1, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy+1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wx*(u_-u__);
                //Calculate change in y-direction at postion ix-1 = d3
                I_ = idx(ix+1, iy-1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy-1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wx*(u_-u__);
                
            } else if (ix+1>=Nx) {
                I_ = idx(ix, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy+1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wx*(u_-u__);

                I_ = idx(ix, iy-1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy-1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wx*(u_-u__);

            } else {
                I_ = idx(ix+1, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy+1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wx*(u_-u__);

                I_ = idx(ix+1, iy-1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy-1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wx*(u_-u__);

            }
            I_ = idx(ix, iy+1, iz);
            c2_ = amul(C2_, C2_mul, I_);
            c3_ = amul(C3_, C3_mul, I_);
            cc_ = make_float3(c2_ ,c3_ ,0.0);
            I__ = idx(ix, iy-1, iz);
            c2__ = amul(C2_, C2_mul, I__);
            c3__ = amul(C3_, C3_mul, I__);
            cc__ = make_float3(c2__ ,c3__ ,0.0);
            d_ = 0.5*wy*(had(cc_,d1)-had(cc__,d3));
        }
    }
    
    dux[I] += d_.y ;
    duy[I] += d_.x ;
    duz[I] += 0.0 ;


    //dxz
    cc = make_float3(c3,0,c2);
    if ((ix-1<0 && ix+1>=Nx) || (iz-1<0 && iz+1>=Nz)) {
        d_ = make_float3(0.0,0.0,0.0);
    }
    else {
        if (ix-1<0) {
            if (iz-1<0) {
                I_ = idx(ix, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wz*(u_-u0);

                I_ = idx(ix+1, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix+1, iy, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wz*(u_-u__);
            } else if (iz+1>=Nz) {
                I_ = idx(ix, iy, iz-1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wz*(u0-u_);

                I_ = idx(ix+1, iy, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix+1, iy, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wz*(u_-u__);
            } else {
                I_ = idx(ix, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d2 = 0.5*wz*(u_-u__);

                I_ = idx(ix+1, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix+1, iy, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wz*(u_-u__);
            }
            I_ = idx(ix+1, iy, iz);
            c2_ = amul(C2_, C2_mul, I_);
            c3_ = amul(C3_, C3_mul, I_);
            cc_ = make_float3(c3_ , 0.0, c2_);
            d_ = 0.5*wx*(had(cc_,d1)-had(cc,d2));

        } else if (ix+1>=Nx) {
            if (iz-1<0) {
                I_ = idx(ix, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wz*(u_-u0);

                I_ = idx(ix-1, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wz*(u_-u__);
            } else if (iz+1>=Nz) {
                I_ = idx(ix, iy, iz-1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wz*(u0-u_);

                I_ = idx(ix-1, iy, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wz*(u_-u__);
            } else {
                I_ = idx(ix, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d2 = 0.5*wz*(u_-u__);

                I_ = idx(ix-1, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wz*(u_-u__);
            }
            I_ = idx(ix-1, iy, iz);
            c2_ = amul(C2_, C2_mul, I_);
            c3_ = amul(C3_, C3_mul, I_);
            cc_ = make_float3(c3_ ,0.0,c2_ );
            d_ = 0.5*wx*(had(cc,d2)-had(cc_,d3));

        } else {
            if (iz-1<0) {
                I_ = idx(ix+1, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix+1, iy, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wz*(u_-u__);

                I_ = idx(ix-1, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wz*(u_-u__);
            } else if (iz+1>=Nz) {
                I_ = idx(ix+1, iy, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix+1, iy, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wz*(u_-u__);

                I_ = idx(ix-1, iy, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wz*(u_-u__);
            } else {
                I_ = idx(ix+1, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix+1, iy, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wz*(u_-u__);

                I_ = idx(ix-1, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wz*(u_-u__);
            }
            I_ = idx(ix+1, iy, iz);
            c2_ = amul(C2_, C2_mul, I_);
            c3_ = amul(C3_, C3_mul, I_);
            cc_ = make_float3(c3_ ,0.0,c2_ );
            I__ = idx(ix-1, iy, iz);
            c2__ = amul(C2_, C2_mul, I__);
            c3__ = amul(C3_, C3_mul, I__);
            cc__ = make_float3(c3__ ,0.0,c2__ );
            d_ = 0.5*wx*(had(cc_,d1)-had(cc__,d3));
        }
    }

    dux[I] += d_.z ;
    duy[I] += 0.0 ;
    duz[I] += d_.x ;

    //dzx
    cc = make_float3(c2,0,c3);
    if ((ix-1<0 && ix+1>=Nx) || (iz-1<0 && iz+1>=Nz)) {
        d_ = make_float3(0.0,0.0,0.0);
    }
    else {
        if (iz-1<0) {
            if (ix-1<0) {
                I_ = idx(ix+1, iy, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wx*(u_-u0);

                I_ = idx(ix+1, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy, iz+1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wx*(u_-u__);
            } else if (ix+1>=Nx) {
                I_ = idx(ix-1, iy, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wx*(u0-u_);

                I_ = idx(ix, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy, iz+1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wx*(u_-u__);
            } else {
                I_ = idx(ix+1, iy, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d2 = 0.5*wx*(u_-u__);

                I_ = idx(ix+1, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy, iz+1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wx*(u_-u__);
            }
            I_ = idx(ix, iy, iz+1);
            c2_ = amul(C2_, C2_mul, I_);
            c3_ = amul(C3_, C3_mul, I_);
            cc_ = make_float3(c2_ ,0.0, c3_);
            d_ = 0.5*wz*(had(cc_,d1)-had(cc,d2));
        } else if (iz+1>=Nz) {
            if (ix-1<0) {
                I_ = idx(ix+1, iy, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wx*(u_-u0);

                I_ = idx(ix+1, iy, iz-1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wx*(u_-u__);
            } else if (ix+1>=Nx) {
                I_ = idx(ix-1, iy, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wx*(u0-u_);

                I_ = idx(ix, iy, iz-1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wx*(u_-u__);
            } else {
                I_ = idx(ix+1, iy, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d2 = 0.5*wx*(u_-u__);

                I_ = idx(ix+1, iy, iz-1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wx*(u_-u__);
            }
            I_ = idx(ix, iy, iz-1);
            c2_ = amul(C2_, C2_mul, I_);
            c3_ = amul(C3_, C3_mul, I_);
            cc_ = make_float3(c2_ ,0.0,c3_ );
            d_ = 0.5*wz*(had(cc,d2)-had(cc_,d3));
        } else {
            if (ix-1<0) {
                I_ = idx(ix+1, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy, iz+1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wx*(u_-u__);

                I_ = idx(ix+1, iy, iz-1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wx*(u_-u__);
            } else if (ix+1>=Nx) {
                I_ = idx(ix, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy, iz+1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wx*(u_-u__);

                I_ = idx(ix, iy, iz-1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wx*(u_-u__);
            } else {
                I_ = idx(ix+1, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy, iz+1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wx*(u_-u__);

                I_ = idx(ix+1, iy, iz-1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix-1, iy, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wx*(u_-u__);
            }
            I_ = idx(ix, iy, iz+1);
            c2_ = amul(C2_, C2_mul, I_);
            c3_ = amul(C3_, C3_mul, I_);
            cc_ = make_float3(c2_ ,0.0,c3_ );
            I__ = idx(ix, iy, iz-1);
            c2__ = amul(C2_, C2_mul, I__);
            c3__ = amul(C3_, C3_mul, I__);
            cc__ = make_float3(c2__ ,0.0,c3__ );
            d_ = 0.5*wz*(had(cc_,d1)-had(cc__,d3));
        }
    }

    dux[I] += d_.z ;
    duy[I] += 0.0 ;
    duz[I] += d_.x ;

    //dyz
    cc = make_float3(0,c3,c2);
    if ((iy-1<0 && iy+1>=Ny) || (iz-1<0 && iz+1>=Nz)) {
        d_ = make_float3(0.0,0.0,0.0);
    }
    else {
        if (iy-1<0) {
            if (iz-1<0) {
                I_ = idx(ix, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wz*(u_-u0);

                I_ = idx(ix, iy+1, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy+1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wz*(u_-u__);
            } else if (iz+1>=Nz) {
                I_ = idx(ix, iy, iz-1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wz*(u0-u_);

                I_ = idx(ix, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy+1, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wz*(u_-u__);
            } else {
                I_ = idx(ix, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d2 = 0.5*wz*(u_-u__);

                I_ = idx(ix, iy+1, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy+1, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wz*(u_-u__);
            }
            I_ = idx(ix, iy+1, iz);
            c2_ = amul(C2_, C2_mul, I_);
            c3_ = amul(C3_, C3_mul, I_);
            cc_ = make_float3( 0.0,c3_ , c2_);
            d_ = 0.5*wy*(had(cc_,d1)-had(cc,d2));
        } else if (iy+1>=Ny) {
            if (iz-1<0) {
                I_ = idx(ix, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wz*(u_-u0);

                I_ = idx(ix, iy-1, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy-1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wz*(u_-u__);
            } else if (iz+1>=Nz) {
                I_ = idx(ix, iy, iz-1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wz*(u0-u_);

                I_ = idx(ix, iy-1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy-1, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wz*(u_-u__);

            } else {
                I_ = idx(ix, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d2 = 0.5*wz*(u_-u__);

                I_ = idx(ix, iy-1, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy-1, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wz*(u_-u__);
            }
            I_ = idx(ix, iy-1, iz);
            c2_ = amul(C2_, C2_mul, I_);
            c3_ = amul(C3_, C3_mul, I_);
            cc_ = make_float3(0.0,c3_, c2_ );
            d_ = 0.5*wy*(had(cc,d2)-had(cc_,d3));
        } else {
            if (iz-1<0) {
                I_ = idx(ix, iy+1, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy+1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wz*(u_-u__);

                I_ = idx(ix, iy-1, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy-1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wz*(u_-u__);
            } else if (iz+1>=Nz) {
                I_ = idx(ix, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy+1, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wz*(u_-u__);

                I_ = idx(ix, iy-1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy-1, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wz*(u_-u__);
            } else {
                I_ = idx(ix, iy+1, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy+1, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wz*(u_-u__);

                I_ = idx(ix, iy-1, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy-1, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wz*(u_-u__);
            }
            I_ = idx(ix, iy+1, iz);
            c2_ = amul(C2_, C2_mul, I_);
            c3_ = amul(C3_, C3_mul, I_);
            cc_ = make_float3(0.0,c3_,c2_ );
            I__ = idx(ix, iy-1, iz);
            c2__ = amul(C2_, C2_mul, I__);
            c3__ = amul(C3_, C3_mul, I__);
            cc__ = make_float3(0.0,c3__,c2__ );
            d_ = 0.5*wy*(had(cc_,d1)-had(cc__,d3));
        }
    }

    dux[I] += 0.0 ;
    duy[I] += d_.z ;
    duz[I] += d_.y ;

    //dzy
    cc = make_float3(0,c2,c3);
    if ((iy-1<0 && iy+1>=Ny) || (iz-1<0 && iz+1>=Nz)) {
        d_ = make_float3(0.0,0.0,0.0);
    }
    else {
        if (iz-1<0) {
            if (iy-1<0) {
                I_ = idx(ix, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wy*(u_-u0);

                I_ = idx(ix, iy+1, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy, iz+1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wy*(u_-u__);

            } else if (iy+1>=Ny) {
                I_ = idx(ix, iy-1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wy*(u0-u_);

                I_ = idx(ix, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy-1, iz+1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wy*(u_-u__);
            } else {
                I_ = idx(ix, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy-1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d2 = 0.5*wy*(u_-u__);

                I_ = idx(ix, iy+1, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy-1, iz+1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wy*(u_-u__);
            }
            I_ = idx(ix, iy, iz+1);
            c2_ = amul(C2_, C2_mul, I_);
            c3_ = amul(C3_, C3_mul, I_);
            cc_ = make_float3( 0.0,c2_ ,c3_);
            d_ = 0.5*wz*(had(cc_,d1)-had(cc,d2));
        } else if (iz+1>=Nz) {
            if (iy-1<0) {
                I_ = idx(ix, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wy*(u_-u0);

                I_ = idx(ix, iy+1, iz-1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wy*(u_-u__);
            } else if (iy+1>=Ny) {
                I_ = idx(ix, iy-1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                d2 = 0.5*wy*(u0-u_);

                I_ = idx(ix, iy, iz-1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy-1, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wy*(u_-u__);
            } else {
                I_ = idx(ix, iy+1, iz);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy-1, iz);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d2 = 0.5*wy*(u_-u__);

                I_ = idx(ix, iy+1, iz-1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy-1, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wy*(u_-u__);
            }
            I_ = idx(ix, iy, iz-1);
            c2_ = amul(C2_, C2_mul, I_);
            c3_ = amul(C3_, C3_mul, I_);
            cc_ = make_float3(0.0,c2_ ,c3_);
            d_ = 0.5*wz*(had(cc,d2)-had(cc_,d3));
        } else {
            if (iy-1<0) {
                I_ = idx(ix, iy+1, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy, iz+1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wy*(u_-u__);

                I_ = idx(ix, iy+1, iz-1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wy*(u_-u__);
            } else if (iy+1>=Ny) {
                I_ = idx(ix, iy, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy-1, iz+1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wy*(u_-u__);

                I_ = idx(ix, iy, iz-1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy-1, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wy*(u_-u__);
            } else {
                I_ = idx(ix, iy+1, iz+1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy-1, iz+1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d1 = 0.5*wy*(u_-u__);

                I_ = idx(ix, iy+1, iz-1);
                u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                I__ = idx(ix, iy-1, iz-1);
                u__ = make_float3(ux[I__], uy[I__], uz[I__]);
                d3 = 0.5*wy*(u_-u__);
            }
            I_ = idx(ix, iy, iz+1);
            c2_ = amul(C2_, C2_mul, I_);
            c3_ = amul(C3_, C3_mul, I_);
            cc_ = make_float3( 0.0 ,c2_ ,c3_);
            I__ = idx(ix, iy, iz-1);
            c2__ = amul(C2_, C2_mul, I__);
            c3__ = amul(C3_, C3_mul, I__);
            cc__ = make_float3(0.0 ,c2__ ,c3__);
            d_ = 0.5*wz*(had(cc_,d1)-had(cc__,d3));
        }
    }

    dux[I] += 0.0 ;
    duy[I] += d_.z ;
    duz[I] += d_.y ;

    //output gelijkstellen aan de magnetizatie
    // dux[I] = dxx.x + dxy.y + dxz.z + dyy.x + dyx.y + dzz.x + dzx.z;
    // duy[I] = dyy.y + dyx.x + dyz.z + dxx.y + dxy.x + dzz.y + dzy.z;
    // duz[I] = dzz.z + dzx.x + dzy.y + dxx.z + dxz.x + dyy.z + dyz.y; 
}
