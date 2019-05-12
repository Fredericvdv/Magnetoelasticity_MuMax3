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
                    


    //positie van cell waar we in aan het kijkken zijn is: ix,iy,iz
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    //if cell position is out of mesh --> do nothing
    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

        //TODO: implement boundary conditions
    //Neighbours
    //i_l = idx(lclampx(ix-1),iy,iz);
    //i_r = idx(hclampx(ix+1),iy,iz);
    //u_l = make_float3(ux[i_l],uy[i_l],uz[i_l]);
    //u_l = ( is0(u_l)? u0: u_l);
    //u_r = make_float3(ux[i_r],uy[i_r],uz[i_r]);
    //u_r = ( is0(u_r)? u0: u_r);

    // Central cell
    int I = idx(ix, iy, iz);
    float3 u0 = make_float3(ux[I], uy[I], uz[I]);
    //float  c1 = amul(C1_, C1_mul, I);
    //float  c2 = amul(C2_, C2_mul, I);
    //float  c3 = amul(C3_, C3_mul, I);
    //uint8_t r0 = regions[I];
    
    //initialize derivatives
    //Higher neighbor
    int I_ = idx(ix, iy, iz);
    float3 u_ = make_float3(0.0,0.0,0.0);
    //float  c1_ = 0.0;
    //float  c2_ = 0.0;
    //float  c3_ = 0.0;

    //float veri = 1.0e20;

    //make vector to use the right constant for the right component of u 
    float3 cc = make_float3(0.0,0.0,0.0);
    float3 cc_ =make_float3(0.0,0.0,0.0);

    float3 d_ = make_float3(0.0,0.0,0.0);

    float3 d2 =make_float3(0.0,0.0,0.0);

    dux[I] = 0.0 ;
    duy[I] = 0.0 ;
    duz[I] = 0.0 ;

    //###############################################
    //###############################################
    //###############################################
    //dxx
    d_ = make_float3(0.0,0.0,0.0);
    cc = make_float3(amul(C1_, C1_mul, I),amul(C3_, C3_mul, I),amul(C3_, C3_mul, I));
    //If there is a right neighbor
    if (ix < Nx-1) {
        I_ = idx(ix+1, iy, iz);
        u_ = make_float3(ux[I_], uy[I_], uz[I_]);
        cc_ = make_float3(amul(C1_, C1_mul, I_),amul(C3_, C3_mul, I_), amul(C3_, C3_mul, I_));
        d_ = 0.5*wx*wx*had((cc+cc_),(u_-u0));
    } 
    //If there is left neighbour
    if (ix > 0) {
        I_ = idx(ix-1, iy, iz);
        u_ = make_float3(ux[I_], uy[I_], uz[I_]);
        cc_ = make_float3(amul(C1_, C1_mul, I_),amul(C3_, C3_mul, I_), amul(C3_, C3_mul, I_));
        d_ += 0.5*wx*wx*had((cc+cc_),(u_-u0));
    }
    
    dux[I] += d_.x ;
    duy[I] += d_.y ;
    //duz[I] += d_.z ;


    //dyy
    d_ = make_float3(0.0,0.0,0.0);
    cc = make_float3(amul(C3_, C3_mul, I),amul(C1_, C1_mul, I),amul(C3_, C3_mul, I));
    //If there is a right neighbor
    if (iy < Ny-1) {
        I_ = idx(ix, iy+1, iz);
        u_ = make_float3(ux[I_], uy[I_], uz[I_]);
        cc_ = make_float3(amul(C3_, C3_mul, I_),amul(C1_, C1_mul, I_),amul(C3_, C3_mul, I_));
        d_ = 0.5*wy*wy*had((cc+cc_),(u_-u0));
    } 
    //If there is left neighbour
    if (iy > 0) {
        I_ = idx(ix, iy-1, iz);
        u_ = make_float3(ux[I_], uy[I_], uz[I_]);
        cc_ = make_float3(amul(C3_, C3_mul, I_),amul(C1_, C1_mul, I_),amul(C3_, C3_mul, I_));
        d_ += 0.5*wy*wy*had((cc+cc_),(u_-u0));
    }
    
    dux[I] += d_.x ;
    duy[I] += d_.y ;
    //duz[I] += d_.z ;


    //dzz
    d_ = make_float3(0.0,0.0,0.0);
    cc = make_float3(amul(C3_, C3_mul, I),amul(C3_, C3_mul, I),amul(C1_, C1_mul, I));
    //If there is a right neighbor
    if (iz < Nz-1) {
        I_ = idx(ix, iy, iz+1);
        u_ = make_float3(ux[I_], uy[I_], uz[I_]);
        cc_ = make_float3(amul(C3_, C3_mul, I_),amul(C3_, C3_mul, I_),amul(C1_, C1_mul, I_));
        d_ = 0.5*wz*wz*had((cc+cc_),(u_-u0));
    } 
    //If there is left neighbour
    if (iz > 0) {
        I_ = idx(ix, iy, iz-1);
        u_ = make_float3(ux[I_], uy[I_], uz[I_]);
        cc_ = make_float3(amul(C3_, C3_mul, I_),amul(C3_, C3_mul, I_),amul(C1_, C1_mul, I_));
        d_ += 0.5*wz*wz*had((cc+cc_),(u_-u0));
    }
    
    dux[I] += d_.x ;
    duy[I] += d_.y ;
    duz[I] += d_.z ;
    
    
    
    //dxy
    d_ = make_float3(0.0,0.0,0.0);
    cc = make_float3(amul(C3_, C3_mul, I),amul(C2_, C2_mul, I),0);    
    //If there is a neighbor to the right
    if (ix < Nx-1) {
        I_ = idx(ix+1, iy, iz);
        cc_ = make_float3(amul(C3_, C3_mul, I_),amul(C2_, C2_mul, I_), 0.0);
        //If there is neighbour above
        if (iy < Ny-1) {
            //Calculate change in y-direction at postion ix+1 = d1
            //0.5*wy*(FWD + BWD) with BWD=0
            I_ = idx(ix+1, iy+1, iz);
            d2 = make_float3(ux[I_], uy[I_], uz[I_]);
            I_ = idx(ix+1, iy, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ += 0.5*wx*0.5*wy*had(cc_,d2-u_); 
            
            //Calculate change in y-direction at postion ix = d2 
            //rectangular mesh: if (ix+1,iy) and (ix,iy+1) are present, then (ix+1,iy+1) is also present
            I_ = idx(ix, iy+1, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ -= 0.5*wx*0.5*wy*had(cc,u_-u0);
        }
        //If there is neighbour below
        if (iy > 0) {
            //Calculate change in y-direction at postion ix+1 = d1
            //0.5*wy*(FWD + BWD) with BWD=0
            I_ = idx(ix+1, iy-1, iz);
            d2 = make_float3(ux[I_], uy[I_], uz[I_]);
            I_ = idx(ix+1, iy, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ += 0.5*wx*0.5*wy*had(cc_,u_-d2); 
            
            //Calculate change in y-direction at postion ix = d2 
            I_ = idx(ix, iy-1, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ -= 0.5*wx*0.5*wy*had(cc,u0-u_);
        }
    }
    //If there is left neighbour
    if (ix > 0) {
        I_ = idx(ix-1, iy, iz);
        cc_ = make_float3(amul(C3_, C3_mul, I_),amul(C2_, C2_mul, I_), 0.0);
        //If there is neighbour above
        if (iy < Ny-1) {
            //Calculate change in y-direction at postion ix-1 = d1
            //0.5*wy*(FWD + BWD) with BWD=0
            I_ = idx(ix-1, iy+1, iz);
            d2 = make_float3(ux[I_], uy[I_], uz[I_]);
            I_ = idx(ix-1, iy, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ -= 0.5*wx*0.5*wy*had(cc_,d2-u_); 
            
            //Calculate change in y-direction at postion ix = d2 
            //rectangular mesh: if (ix+1,iy) and (ix,iy+1) are present, then (ix+1,iy+1) is also present
            I_ = idx(ix, iy+1, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ += 0.5*wx*0.5*wy*had(cc,u_-u0);
        }
        //If there is neighbour below
        if (iy > 0) {
            //Calculate change in y-direction at postion ix+1 = d1
            //0.5*wy*(FWD + BWD) with BWD=0
            I_ = idx(ix-1, iy-1, iz);
            d2 = make_float3(ux[I_], uy[I_], uz[I_]);
            I_ = idx(ix-1, iy, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ -= 0.5*wx*0.5*wy*had(cc_,u_-d2); 
            
            //Calculate change in y-direction at postion ix = d2 
            I_ = idx(ix, iy-1, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ += 0.5*wx*0.5*wy*had(cc,u0-u_);
        }
    }

    //printf("dxy.y = %g", d_.y);
    //printf("dxy.x = %g", d_.x);
    dux[I] += d_.y ;
    duy[I] += d_.x ;
    duz[I] += 0.0 ;


    //dyx
    d_ = make_float3(0.0,0.0,0.0);
    cc = make_float3(amul(C2_, C2_mul, I),amul(C3_, C3_mul, I),0);    
    //If there is a neighbor to the right
    if (iy < Ny-1) {
        I_ = idx(ix, iy+1, iz);
        cc_ = make_float3(amul(C2_, C2_mul, I_),amul(C3_, C3_mul, I_), 0.0);
        //If there is neighbour above
        if (ix < Nx-1) {
            I_ = idx(ix+1, iy+1, iz);
            d2 = make_float3(ux[I_], uy[I_], uz[I_]);
            I_ = idx(ix, iy+1, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ += 0.5*wx*0.5*wy*had(cc_,d2-u_); 
            
            I_ = idx(ix+1, iy, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ -= 0.5*wx*0.5*wy*had(cc,u_-u0);
        }
        //If there is neighbour below
        if (ix > 0) {
            I_ = idx(ix-1, iy+1, iz);
            d2 = make_float3(ux[I_], uy[I_], uz[I_]);
            I_ = idx(ix, iy+1, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ += 0.5*wx*0.5*wy*had(cc_,u_-d2); 
            
            I_ = idx(ix-1, iy, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ -= 0.5*wx*0.5*wy*had(cc,u0-u_);
        }
    }
    //If there is left neighbour
    if (iy > 0) {
        I_ = idx(ix, iy-1, iz);
        cc_ = make_float3(amul(C2_, C2_mul, I_),amul(C3_, C3_mul, I_), 0.0);
        //If there is neighbour above
        if (ix < Nx-1) {
            I_ = idx(ix+1, iy-1, iz);
            d2 = make_float3(ux[I_], uy[I_], uz[I_]);
            I_ = idx(ix, iy-1, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ -= 0.5*wx*0.5*wy*had(cc_,d2-u_); 
            
            I_ = idx(ix+1, iy, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ += 0.5*wx*0.5*wy*had(cc,u_-u0);
        }
        //If there is neighbour below
        if (ix > 0) {
            I_ = idx(ix-1, iy-1, iz);
            d2 = make_float3(ux[I_], uy[I_], uz[I_]);
            I_ = idx(ix, iy-1, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ -= 0.5*wx*0.5*wy*had(cc_,u_-d2); 
            
            I_ = idx(ix-1, iy, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ += 0.5*wx*0.5*wy*had(cc,u0-u_);
        }
    }

    dux[I] += d_.y ;
    duy[I] += d_.x ;
    duz[I] += 0.0 ;




    // //dxz
    // d_ = make_float3(0.0,0.0,0.0);
    // cc = make_float3(amul(C3_, C3_mul, I),0.0, amul(C2_, C2_mul, I));    
    // //If there is a neighbor to the right
    // if (ix < Nx-1) {
    //     I_ = idx(ix+1, iy, iz);
    //     cc_ = make_float3(amul(C3_, C3_mul, I_), 0.0,amul(C2_, C2_mul, I_));
    //     if (iz < Nz-1) {
    //         I_ = idx(ix+1, iy, iz+1);
    //         d2 = make_float3(ux[I_], uy[I_], uz[I_]);
    //         I_ = idx(ix+1, iy, iz);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ += 0.5*wx*0.5*wz*had(cc_,d2-u_); 
            
    //         I_ = idx(ix, iy, iz+1);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ -= 0.5*wx*0.5*wz*had(cc,u_-u0);
    //     }
    //     if (iz > 0) {
    //         I_ = idx(ix+1, iy, iz-1);
    //         d2 = make_float3(ux[I_], uy[I_], uz[I_]);
    //         I_ = idx(ix+1, iy, iz);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ += 0.5*wx*0.5*wz*had(cc_,u_-d2); 
            
    //         I_ = idx(ix, iy, iz-1);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ -= 0.5*wx*0.5*wz*had(cc,u0-u_);
    //     }
    // }
    // if (ix > 0) {
    //     I_ = idx(ix-1, iy, iz);
    //     cc_ = make_float3(amul(C3_, C3_mul, I_), 0.0,amul(C2_, C2_mul, I_));
    //     if (iz < Nz-1) {
    //         I_ = idx(ix-1, iy, iz+1);
    //         d2 = make_float3(ux[I_], uy[I_], uz[I_]);
    //         I_ = idx(ix-1, iy, iz);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ -= 0.5*wx*0.5*wz*had(cc_,d2-u_); 
            
    //         I_ = idx(ix, iy, iz+1);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ += 0.5*wx*0.5*wz*had(cc,u_-u0);
    //     }
    //     if (iz > 0) {
    //         I_ = idx(ix-1, iy, iz-1);
    //         d2 = make_float3(ux[I_], uy[I_], uz[I_]);
    //         I_ = idx(ix-1, iy, iz);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ -= 0.5*wx*0.5*wz*had(cc_,u_-d2); 
            
    //         I_ = idx(ix, iy, iz-1);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ += 0.5*wx*0.5*wz*had(cc,u0-u_);
    //     }
    // }

    // dux[I] += d_.z ;
    // duy[I] += 0.0 ;
    // duz[I] += d_.x ;


    // //dzx
    // d_ = make_float3(0.0,0.0,0.0);
    // cc = make_float3(amul(C2_, C2_mul, I),0.0,amul(C3_, C3_mul, I));    
    // if (iz < Nz-1) {
    //     I_ = idx(ix, iy, iz+1);
    //     cc_ = make_float3(amul(C2_, C2_mul, I_), 0.0,amul(C3_, C3_mul, I_));
    //     if (ix < Nx-1) {
    //         I_ = idx(ix+1, iy, iz+1);
    //         d2 = make_float3(ux[I_], uy[I_], uz[I_]);
    //         I_ = idx(ix, iy, iz+1);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ += 0.5*wx*0.5*wz*had(cc_,d2-u_); 
            
    //         I_ = idx(ix+1, iy, iz);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ -= 0.5*wx*0.5*wz*had(cc,u_-u0);
    //     }
    //     if (ix > 0) {
    //         I_ = idx(ix-1, iy, iz+1);
    //         d2 = make_float3(ux[I_], uy[I_], uz[I_]);
    //         I_ = idx(ix, iy, iz+1);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ += 0.5*wx*0.5*wz*had(cc_,u_-d2); 
            
    //         I_ = idx(ix-1, iy, iz);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ -= 0.5*wx*0.5*wz*had(cc,u0-u_);
    //     }
    // }
    // if (iz > 0) {
    //     I_ = idx(ix, iy, iz-1);
    //     cc_ = make_float3(amul(C2_, C2_mul, I_), 0.0,amul(C3_, C3_mul, I_));
    //     if (ix < Nx-1) {
    //         I_ = idx(ix+1, iy, iz-1);
    //         d2 = make_float3(ux[I_], uy[I_], uz[I_]);
    //         I_ = idx(ix, iy, iz-1);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ -= 0.5*wx*0.5*wz*had(cc_,d2-u_); 
            
    //         I_ = idx(ix+1, iy, iz);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ += 0.5*wx*0.5*wz*had(cc,u_-u0);
    //     }
    //     //If there is neighbour below
    //     if (ix > 0) {
    //         I_ = idx(ix-1, iy, iz-1);
    //         d2 = make_float3(ux[I_], uy[I_], uz[I_]);
    //         I_ = idx(ix, iy, iz-1);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ -= 0.5*wx*0.5*wz*had(cc_,u_-d2); 
            
    //         I_ = idx(ix-1, iy, iz);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ += 0.5*wx*0.5*wz*had(cc,u0-u_);
    //     }
    // }
    
    // dux[I] += d_.z ;
    // duy[I] += 0.0 ;
    // duz[I] += d_.x ;


    //     //dyz
    //     d_ = make_float3(0.0,0.0,0.0);
    //     cc = make_float3(0.0, amul(C3_, C3_mul, I), amul(C2_, C2_mul, I));    
    //     //If there is a neighbor to the right
    //     if (iy < Ny-1) {
    //         I_ = idx(ix, iy+1, iz);
    //         cc_ = make_float3( 0.0,amul(C3_, C3_mul, I_),amul(C2_, C2_mul, I_));
    //         if (iz < Nz-1) {
    //             I_ = idx(ix, iy+1, iz+1);
    //             d2 = make_float3(ux[I_], uy[I_], uz[I_]);
    //             I_ = idx(ix, iy+1, iz);
    //             u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //             d_ += 0.5*wy*0.5*wz*had(cc_,d2-u_); 
                
    //             I_ = idx(ix, iy, iz+1);
    //             u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //             d_ -= 0.5*wy*0.5*wz*had(cc,u_-u0);
    //         }
    //         if (iz > 0) {
    //             I_ = idx(ix, iy+1, iz-1);
    //             d2 = make_float3(ux[I_], uy[I_], uz[I_]);
    //             I_ = idx(ix, iy+1, iz);
    //             u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //             d_ += 0.5*wy*0.5*wz*had(cc_,u_-d2); 
                
    //             I_ = idx(ix, iy, iz-1);
    //             u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //             d_ -= 0.5*wy*0.5*wz*had(cc,u0-u_);
    //         }
    //     }
    //     if (iy > 0) {
    //         I_ = idx(ix, iy-1, iz);
    //         cc_ = make_float3(0.0,amul(C3_, C3_mul, I_),amul(C2_, C2_mul, I_));
    //         if (iz < Nz-1) {
    //             I_ = idx(ix, iy-1, iz+1);
    //             d2 = make_float3(ux[I_], uy[I_], uz[I_]);
    //             I_ = idx(ix, iy-1, iz);
    //             u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //             d_ -= 0.5*wy*0.5*wz*had(cc_,d2-u_); 
                
    //             I_ = idx(ix, iy, iz+1);
    //             u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //             d_ += 0.5*wy*0.5*wz*had(cc,u_-u0);
    //         }
    //         if (iz > 0) {
    //             I_ = idx(ix, iy-1, iz-1);
    //             d2 = make_float3(ux[I_], uy[I_], uz[I_]);
    //             I_ = idx(ix, iy-1, iz);
    //             u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //             d_ -= 0.5*wy*0.5*wz*had(cc_,u_-d2); 
                
    //             I_ = idx(ix, iy, iz-1);
    //             u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //             d_ += 0.5*wy*0.5*wz*had(cc,u0-u_);
    //         }
    //     }

    // dux[I] += 0.0 ;
    // duy[I] += d_.z ;
    // duz[I] += d_.y ;


    // //dzy
    // d_ = make_float3(0.0,0.0,0.0);
    // cc = make_float3(0.0, amul(C2_, C2_mul, I),amul(C3_, C3_mul, I));    
    // if (iz < Nz-1) {
    //     I_ = idx(ix, iy, iz+1);
    //     cc_ = make_float3( 0.0, amul(C2_, C2_mul, I_),amul(C3_, C3_mul, I_));
    //     if (iy < Ny-1) {
    //         I_ = idx(ix, iy+1, iz+1);
    //         d2 = make_float3(ux[I_], uy[I_], uz[I_]);
    //         I_ = idx(ix, iy, iz+1);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ += 0.5*wy*0.5*wz*had(cc_,d2-u_); 
            
    //         I_ = idx(ix, iy+1, iz);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ -= 0.5*wy*0.5*wz*had(cc,u_-u0);
    //     }
    //     if (iy > 0) {
    //         I_ = idx(ix, iy-1, iz+1);
    //         d2 = make_float3(ux[I_], uy[I_], uz[I_]);
    //         I_ = idx(ix, iy, iz+1);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ += 0.5*wy*0.5*wz*had(cc_,u_-d2); 
            
    //         I_ = idx(ix, iy-1, iz);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ -= 0.5*wy*0.5*wz*had(cc,u0-u_);
    //     }
    // }
    // if (iz > 0) {
    //     I_ = idx(ix, iy, iz-1);
    //     cc_ = make_float3( 0.0, amul(C2_, C2_mul, I_),amul(C3_, C3_mul, I_));
    //     if (iy < Ny-1) {
    //         I_ = idx(ix, iy+1, iz-1);
    //         d2 = make_float3(ux[I_], uy[I_], uz[I_]);
    //         I_ = idx(ix, iy, iz-1);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ -= 0.5*wy*0.5*wz*had(cc_,d2-u_); 
            
    //         I_ = idx(ix, iy+1, iz);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ += 0.5*wy*0.5*wz*had(cc,u_-u0);
    //     }
    //     //If there is neighbour below
    //     if (iy > 0) {
    //         I_ = idx(ix, iy-1, iz-1);
    //         d2 = make_float3(ux[I_], uy[I_], uz[I_]);
    //         I_ = idx(ix, iy, iz-1);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ -= 0.5*wy*0.5*wz*had(cc_,u_-d2); 
            
    //         I_ = idx(ix, iy-1, iz);
    //         u_ = make_float3(ux[I_], uy[I_], uz[I_]);
    //         d_ += 0.5*wy*0.5*wz*had(cc,u0-u_);
    //     }
    // }

    // dux[I] += 0.0 ;
    // duy[I] += d_.z ;
    // duz[I] += d_.y ;


    //output gelijkstellen aan de magnetizatie
    // dux[I] = dxx.x + dxy.y + dxz.z + dyy.x + dyx.y + dzz.x + dzx.z;
    // duy[I] = dyy.y + dyx.x + dyz.z + dxx.y + dxy.x + dzz.y + dzy.z;
    // duz[I] = dzz.z + dzx.x + dzy.y + dxx.z + dxz.x + dyy.z + dyz.y; 
}
