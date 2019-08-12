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

    //Check if you are in a free disp region
    if (amul(C1_, C1_mul, I)==0) {
        return;
    }
    
    //Neighbor cell
    int I_ = idx(ix, iy, iz);
    float3 u_ = make_float3(0.0,0.0,0.0);
    float3 cc_ =make_float3(0.0,0.0,0.0);

    float3 d_ = make_float3(0.0,0.0,0.0);
    float3 d2 =make_float3(0.0,0.0,0.0);


    //Shear components: part I   

    
    //dxy
    d_ = make_float3(0.0,0.0,0.0);
    cc = make_float3(amul(C3_, C3_mul, I),amul(C2_, C2_mul, I),0);    
    //Check if there is a neighbor to the right
    if (ix < Nx-1) {
        I_ = idx(ix+1, iy, iz);
        //Check if this cell corresponds to a "free" region
        if (amul(C1_, C1_mul, I_)!=0) {
            cc_ = make_float3(amul(C3_, C3_mul, I_),amul(C2_, C2_mul, I_), 0.0);
            //Check if there is neighbour above
            if (iy < Ny-1) {
                //rectangular mesh: if (ix+1,iy) and (ix,iy+1) are present, then (ix+1,iy+1) is also present
                I_ = idx(ix+1, iy+1, iz);
                //Check if this cell corresponds to a "free" region
                if (amul(C1_, C1_mul, I_)!=0) {
                    //Calculate change in y-direction at postion ix+1 = d1
                    //0.5*wy*(FWD + BWD) with BWD=0
                    d2 = make_float3(ux[I_], uy[I_], uz[I_]);
                    I_ = idx(ix+1, iy, iz);
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ += 0.5*wx*0.5*wy*had(cc_,d2-u_);
                } 
                
                I_ = idx(ix, iy+1, iz);
                if (amul(C1_, C1_mul, I_)!=0) {
                    //Calculate change in y-direction at postion ix
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ -= 0.5*wx*0.5*wy*had(cc,u_-u0);
                }
            }
            //Check if there is neighbour below
            if (iy > 0) {
                //Calculate change in y-direction at postion ix+1 = d2 
                I_ = idx(ix+1, iy-1, iz);
                if (amul(C1_, C1_mul, I_)!=0) {
                    d2 = make_float3(ux[I_], uy[I_], uz[I_]);
                    I_ = idx(ix+1, iy, iz);
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ += 0.5*wx*0.5*wy*had(cc_,u_-d2); 
                }
                
                //Calculate change in y-direction at postion ix 
                I_ = idx(ix, iy-1, iz);
                if (amul(C1_, C1_mul, I_)!=0) {
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ -= 0.5*wx*0.5*wy*had(cc,u0-u_);  
                }
            }
        }
    }
    //Check if there is left neighbour
    if (ix > 0) {
        I_ = idx(ix-1, iy, iz);
        //Check if this cell corresponds to a "free" region
        if (amul(C1_, C1_mul, I_)!=0) {
            cc_ = make_float3(amul(C3_, C3_mul, I_),amul(C2_, C2_mul, I_), 0.0);    
            //Check if there is neighbour above
            if (iy < Ny-1) {
                //rectangular mesh: if (ix-1,iy) and (ix,iy+1) are present, then (ix-1,iy+1) is also present                
                I_ = idx(ix-1, iy+1, iz);
                //Check if this cell corresponds to a "free" region
                if (amul(C1_, C1_mul, I_)!=0) {
                    //Calculate change in y-direction at postion ix-1 = d1
                    //0.5*wy*(FWD + BWD) with BWD=0
                    d2 = make_float3(ux[I_], uy[I_], uz[I_]);
                    I_ = idx(ix-1, iy, iz);
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ -= 0.5*wx*0.5*wy*had(cc_,d2-u_); 
                }
                
                I_ = idx(ix, iy+1, iz);
                if (amul(C1_, C1_mul, I_)!=0) {
                    //Calculate change in y-direction at postion ix
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);   
                    d_ += 0.5*wx*0.5*wy*had(cc,u_-u0);
                }
            }
            //Check if there is neighbour below
            if (iy > 0) {
                //Calculate change in y-direction at postion ix+1 = d2 
                I_ = idx(ix-1, iy-1, iz);
                if (amul(C1_, C1_mul, I_)!=0) {
                    d2 = make_float3(ux[I_], uy[I_], uz[I_]);
                    I_ = idx(ix-1, iy, iz);
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ -= 0.5*wx*0.5*wy*had(cc_,u_-d2); 
                }
                
                //Calculate change in y-direction at postion ix = d2 
                I_ = idx(ix, iy-1, iz);
                if (amul(C1_, C1_mul, I_)!=0) {
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ += 0.5*wx*0.5*wy*had(cc,u0-u_);
                }
            }
        }
    }

    dux[I] += d_.y ;
    duy[I] += d_.x ;
    duz[I] += 0.0 ;


    //dyx
    d_ = make_float3(0.0,0.0,0.0);
    cc = make_float3(amul(C2_, C2_mul, I),amul(C3_, C3_mul, I),0);    
    if (iy < Ny-1) {
        I_ = idx(ix, iy+1, iz);
        if (amul(C1_, C1_mul, I_)!=0) {
            cc_ = make_float3(amul(C2_, C2_mul, I_),amul(C3_, C3_mul, I_), 0.0);
            if (ix < Nx-1) {
                I_ = idx(ix+1, iy+1, iz);
                if (amul(C1_, C1_mul, I_)!=0) {
                    d2 = make_float3(ux[I_], uy[I_], uz[I_]);
                    I_ = idx(ix, iy+1, iz);
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ += 0.5*wx*0.5*wy*had(cc_,d2-u_); 
                }
                
                I_ = idx(ix+1, iy, iz);
                if (amul(C1_, C1_mul, I_)!=0) {
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ -= 0.5*wx*0.5*wy*had(cc,u_-u0);
                }
            }
            if (ix > 0) {
                I_ = idx(ix-1, iy+1, iz);
                if (amul(C1_, C1_mul, I_)!=0) {
                    d2 = make_float3(ux[I_], uy[I_], uz[I_]);
                    I_ = idx(ix, iy+1, iz);
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ += 0.5*wx*0.5*wy*had(cc_,u_-d2); 
                }
                
                I_ = idx(ix-1, iy, iz);
                if (amul(C1_, C1_mul, I_)!=0) {
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ -= 0.5*wx*0.5*wy*had(cc,u0-u_);
                }
            }
        }
    }
    if (iy > 0) {
        I_ = idx(ix, iy-1, iz);
        if (amul(C1_, C1_mul, I_)!=0) {
            cc_ = make_float3(amul(C2_, C2_mul, I_),amul(C3_, C3_mul, I_), 0.0);
            if (ix < Nx-1) {
                I_ = idx(ix+1, iy-1, iz);
                if (amul(C1_, C1_mul, I_)!=0) {
                    d2 = make_float3(ux[I_], uy[I_], uz[I_]);
                    I_ = idx(ix, iy-1, iz);
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ -= 0.5*wx*0.5*wy*had(cc_,d2-u_); 
                }
                
                I_ = idx(ix+1, iy, iz);
                if (amul(C1_, C1_mul, I_)!=0) {
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ += 0.5*wx*0.5*wy*had(cc,u_-u0);
                }
            }
            if (ix > 0) {
                I_ = idx(ix-1, iy-1, iz);
                if (amul(C1_, C1_mul, I_)!=0) {
                    d2 = make_float3(ux[I_], uy[I_], uz[I_]);
                    I_ = idx(ix, iy-1, iz);
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ -= 0.5*wx*0.5*wy*had(cc_,u_-d2); 
                }
                
                I_ = idx(ix-1, iy, iz);
                if (amul(C1_, C1_mul, I_)!=0) {
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ += 0.5*wx*0.5*wy*had(cc,u0-u_);
                }
            }
        }
    }

    dux[I] += d_.y ;
    duy[I] += d_.x ;
    duz[I] += 0.0 ;




    //dxz
    d_ = make_float3(0.0,0.0,0.0);
    cc = make_float3(amul(C3_, C3_mul, I),0.0, amul(C2_, C2_mul, I));    
    if (ix < Nx-1) {
        I_ = idx(ix+1, iy, iz);
        if (amul(C1_, C1_mul, I_)!=0) {
            cc_ = make_float3(amul(C3_, C3_mul, I_), 0.0,amul(C2_, C2_mul, I_));
            if (iz < Nz-1) {
                I_ = idx(ix+1, iy, iz+1);
                if (amul(C1_, C1_mul, I_)!=0) {
                    d2 = make_float3(ux[I_], uy[I_], uz[I_]);
                    I_ = idx(ix+1, iy, iz);
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ += 0.5*wx*0.5*wz*had(cc_,d2-u_); 
                }
                
                I_ = idx(ix, iy, iz+1);
                if (amul(C1_, C1_mul, I_)!=0) {
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ -= 0.5*wx*0.5*wz*had(cc,u_-u0);
                }
            }
            if (iz > 0) {
                I_ = idx(ix+1, iy, iz-1);
                if (amul(C1_, C1_mul, I_)!=0) {
                    d2 = make_float3(ux[I_], uy[I_], uz[I_]);
                    I_ = idx(ix+1, iy, iz);
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ += 0.5*wx*0.5*wz*had(cc_,u_-d2); 
                }
                
                I_ = idx(ix, iy, iz-1);
                if (amul(C1_, C1_mul, I_)!=0) {
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ -= 0.5*wx*0.5*wz*had(cc,u0-u_);
                }
            }
        }
    }
    if (ix > 0) {
        I_ = idx(ix-1, iy, iz);
        if (amul(C1_, C1_mul, I_)!=0) {
            cc_ = make_float3(amul(C3_, C3_mul, I_), 0.0,amul(C2_, C2_mul, I_));
            if (iz < Nz-1) {
                I_ = idx(ix-1, iy, iz+1);
                if (amul(C1_, C1_mul, I_)!=0) {
                    d2 = make_float3(ux[I_], uy[I_], uz[I_]);
                    I_ = idx(ix-1, iy, iz);
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ -= 0.5*wx*0.5*wz*had(cc_,d2-u_); 
                }
                
                I_ = idx(ix, iy, iz+1);
                if (amul(C1_, C1_mul, I_)!=0) {
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ += 0.5*wx*0.5*wz*had(cc,u_-u0);
                }
            }
            if (iz > 0) {
                I_ = idx(ix-1, iy, iz-1);
                if (amul(C1_, C1_mul, I_)!=0) {
                    d2 = make_float3(ux[I_], uy[I_], uz[I_]);
                    I_ = idx(ix-1, iy, iz);
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ -= 0.5*wx*0.5*wz*had(cc_,u_-d2); 
                }
                
                I_ = idx(ix, iy, iz-1);
                if (amul(C1_, C1_mul, I_)!=0) {
                    u_ = make_float3(ux[I_], uy[I_], uz[I_]);
                    d_ += 0.5*wx*0.5*wz*had(cc,u0-u_);
                }
            }
        }
    }

    dux[I] += d_.z ;
    duy[I] += 0.0 ;
    duz[I] += d_.x ;


    //Output should be equal to:
    // dux[I] = dxx.x + dxy.y + dxz.z + dyy.x + dyx.y + dzz.x + dzx.z;
    // duy[I] = dyy.y + dyx.x + dyz.z + dxx.y + dxy.x + dzz.y + dzy.z;
    // duz[I] = dzz.z + dzx.x + dzy.y + dxx.z + dxz.x + dyy.z + dyz.y; 
}
