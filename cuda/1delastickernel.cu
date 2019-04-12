#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// ~ Bibliothee can C inladen of zichtbaar maken
// Nx aantal cellen in x-richting
extern "C" __global__ void
SecondDerivative(float* __restrict__ dux, float* __restrict__ duy, float* __restrict__ duz, 
                 float* __restrict__ ux, float* __restrict__ uy, float* __restrict__ uz,
                 int Nx, int Ny, int Nz, float wx, float wy, float wz, 
                 float* __restrict__  c1_, float  c1_mul, float* __restrict__  c2_, float  c2_mul, 
                 float* __restrict__  c3_, float  c3_mul, uint8_t PBC) {
    
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
    float  c1 = amul(c1_, c1_mul, I);
    float  c2 = amul(c2_, c2_mul, I);
    float  c3 = amul(c3_, c3_mul, I);
    //uint8_t r0 = regions[I];


    //TODO: implement boundary conditions
    //Neighbours
    //i_l = idx(lclampx(ix-1),iy,iz);
    //i_r = idx(hclampx(ix+1),iy,iz);
    //u_l = make_float3(ux[i_l],uy[i_l],uz[i_l]);
    //u_l = ( is0(u_l)? u0: u_l);
    //u_r = make_float3(ux[i_r],uy[i_r],uz[i_r]);
    //u_r = ( is0(u_r)? u0: u_r);


    //Declaration
    int Ixh = idx(ix+1, iy, iz);
    float3 uxh = make_float3(0,0,0);
    float  c1xh = 0;
    float  c2xh = 0;
    float  c3xh = 0;

    int Ixhyh = idx(ix+1, iy+1, iz);
    float3 uxhyh = make_float3(0,0,0);
    //float  c1xhyh = 0;
    //float  c2xhyh = 0;
    //float  c3xhyh = 0;

    int Ixhyl = idx(ix+1, iy-1, iz);
    float3 uxhyl = make_float3(0,0,0);
    //float  c1xhyl = 0;
    //float  c2xhyl= 0;
    //float  c3xhyl = 0;

    int Ixhzh = idx(ix+1, iy, iz+1);
    float3 uxhzh = make_float3(0,0,0);
    //float  c1xhzh = 0;
    //float  c2xhzh= 0;
    //float  c3xhzh = 0;

    int Ixhzl = idx(ix+1, iy, iz-1);
    float3 uxhzl = make_float3(0,0,0);
    //float  c1xhzl = 0;
    //float  c2xhzl= 0;
    //float  c3xhzl = 0;

    int Ixl = idx(ix-1, iy, iz);
    float3 uxl = make_float3(0,0,0);
    float  c1xl = 0;
    float  c2xl= 0;
    float  c3xl = 0;

    int Ixlyh = idx(ix-1, iy+1, iz);
    float3 uxlyh = make_float3(0,0,0);
    //float  c1xlyh = 0;
    //float  c2xlyh = 0;
    //float  c3xlyh = 0;

    int Ixlyl = idx(ix-1, iy-1, iz);
    float3 uxlyl = make_float3(0,0,0);
    //float  c1xlyl = 0;
    //float  c2xlyl = 0;
    //float  c3xlyl = 0;

    int Ixlzh = idx(ix-1, iy, iz+1);
    float3 uxlzh = make_float3(0,0,0);
    //float  c1xlzh = 0;
    //float  c2xlzh = 0;
    //float  c3xlzh = 0;

    int Ixlzl = idx(ix-1, iy, iz-1);
    float3 uxlzl = make_float3(0,0,0);
    //float  c1xlzl = 0;
    //float  c2xlzl = 0;
    //float  c3xlzl = 0;

    int Iyh = idx(ix, iy+1, iz);
    float3 uyh = make_float3(0,0,0);
    float  c1yh = 0;
    float  c2yh = 0;
    float  c3yh = 0;

    int Iyhzh = idx(ix, iy+1, iz+1);
    float3 uyhzh = make_float3(0,0,0);
    //float  c1yhzh = 0;
    //float  c2yhzh = 0;
    //float  c3yhzh = 0;

    int Iyhzl = idx(ix, iy+1, iz-1);
    float3 uyhzl = make_float3(0,0,0);
    //float  c1yhzl = 0;
    //float  c2yhzl = 0;
    //float  c3yhzl = 0;

    int Iyl = idx(ix, iy-1, iz);
    float3 uyl = make_float3(0,0,0);
    float  c1yl = 0;
    float  c2yl = 0;
    float  c3yl = 0;

    int Iylzl = idx(ix, iy-1, iz-1);
    float3 uylzl = make_float3(0,0,0);
    //float  c1ylzl = 0;
    //float  c2ylzl = 0;
    //float  c3ylzl = 0;

    int Iylzh = idx(ix, iy-1, iz+1);
    float3 uylzh = make_float3(0,0,0);
    //float  c1ylzh = 0;
    //float  c2ylzh = 0;
    //float  c3ylzh = 0;

    int Izh = idx(ix, iy, iz+1);
    float3 uzh = make_float3(0,0,0);
    float  c1zh = 0;
    float  c2zh = 0;
    float  c3zh = 0;

    int Izl = idx(ix, iy, iz-1);
    float3 uzl = make_float3(0,0,0);
    float  c1zl = 0;
    float  c2zl = 0;
    float  c3zl = 0;


    // 18 surrounding cells    
    if (ix+1<Nx) {
         uxh = make_float3(ux[Ixh], uy[Ixh], uz[Ixh]);
          c1xh = amul(c1_, c1_mul, Ixh);
          c2xh = amul(c2_, c2_mul, Ixh);
          c3xh = amul(c3_, c3_mul, Ixh);

        if (iy+1<Nx) {
             uxhyh = make_float3(ux[Ixhyh], uy[Ixhyh], uz[Ixhyh]);
              //c1xhyh = amul(c1_, c1_mul, Ixhyh);
              //c2xhyh = amul(c2_, c2_mul, Ixhyh);
              //c3xhyh = amul(c3_, c3_mul, Ixhyh);
        }
        if (iy-1>=0) {
             uxhyl = make_float3(ux[Ixhyl], uy[Ixhyl], uz[Ixhyl]);
              //c1xhyl = amul(c1_, c1_mul, Ixhyl);
              //c2xhyl = amul(c2_, c2_mul, Ixhyl);
              //c3xhyl = amul(c3_, c3_mul, Ixhyl);
        }
        if (iz+1<Nz) {
             uxhzh = make_float3(ux[Ixhzh], uy[Ixhzh], uz[Ixhzh]);
              //c1xhzh = amul(c1_, c1_mul, Ixhzh);
              //c2xhzh = amul(c2_, c2_mul, Ixhzh);
              //c3xhzh = amul(c3_, c3_mul, Ixhzh);
        }
        if (iz-1>=0) {
             uxhzl = make_float3(ux[Ixhzl], uy[Ixhzl], uz[Ixhzl]);
              //c1xhzl = amul(c1_, c1_mul, Ixhzl);
              //c2xhzl = amul(c2_, c2_mul, Ixhzl);
              //c3xhzl = amul(c3_, c3_mul, Ixhzl);
        }   
    }
    if (ix-1>=0) {
         uxl = make_float3(ux[Ixl], uy[Ixl], uz[Ixl]);
          c1xl = amul(c1_, c1_mul, Ixl);
          c2xl = amul(c2_, c2_mul, Ixl);
          c3xl = amul(c3_, c3_mul, Ixl);

        if (iy+1<Ny) {
             uxlyh = make_float3(ux[Ixlyh], uy[Ixlyh], uz[Ixlyh]);
              //c1xlyh = amul(c1_, c1_mul, Ixlyh);
              //c2xlyh = amul(c2_, c2_mul, Ixlyh);
              //c3xlyh = amul(c3_, c3_mul, Ixlyh);
        }
        if (iy-1>=0) {
             uxlyl = make_float3(ux[Ixlyl], uy[Ixlyl], uz[Ixlyl]);
              //c1xlyl = amul(c1_, c1_mul, Ixlyl);
              //c2xlyl = amul(c2_, c2_mul, Ixlyl);
              //c3xlyl = amul(c3_, c3_mul, Ixlyl);
        }
        if (iz+1<Nz) {
             uxlzh = make_float3(ux[Ixlzh], uy[Ixlzh], uz[Ixlzh]);
              //c1xlzh = amul(c1_, c1_mul, Ixlzh);
              //c2xlzh = amul(c2_, c2_mul, Ixlzh);
              //c3xlzh = amul(c3_, c3_mul, Ixlzh);
        }
        if (iz-1>=0) {
             uxlzl = make_float3(ux[Ixlzl], uy[Ixlzl], uz[Ixlzl]);
             // c1xlzl = amul(c1_, c1_mul, Ixlzl);
              //c2xlzl = amul(c2_, c2_mul, Ixlzl);
              //c3xlzl = amul(c3_, c3_mul, Ixlzl);
        }
    }
    if (iy+1<Ny) {
         uyh = make_float3(ux[Iyh], uy[Iyh], uz[Iyh]);
          c1yh = amul(c1_, c1_mul, Iyh);
          c2yh = amul(c2_, c2_mul, Iyh);
          c3yh = amul(c3_, c3_mul, Iyh);

        if (iz+1<Nz) {
             uyhzh = make_float3(ux[Iyhzh], uy[Iyhzh], uz[Iyhzh]);
              //c1yhzh = amul(c1_, c1_mul, Iyhzh);
              //c2yhzh = amul(c2_, c2_mul, Iyhzh);
              //c3yhzh = amul(c3_, c3_mul, Iyhzh);
        }
        if (iz-1>=0) {
             uyhzl = make_float3(ux[Iyhzl], uy[Iyhzl], uz[Iyhzl]);
              //c1yhzl = amul(c1_, c1_mul, Iyhzl);
              //c2yhzl = amul(c2_, c2_mul, Iyhzl);
              //c3yhzl = amul(c3_, c3_mul, Iyhzl);
        }
    }
    if (iy-1>=0) {
         uyl = make_float3(ux[Iyl], uy[Iyl], uz[Iyl]);
          c1yl = amul(c1_, c1_mul, Iyl);
          c2yl = amul(c2_, c2_mul, Iyl);
          c3yl = amul(c3_, c3_mul, Iyl);

        if (iz-1>=0) {
             uylzl = make_float3(ux[Iylzl], uy[Iylzl], uz[Iylzl]);
              //c1ylzl = amul(c1_, c1_mul, Iylzl);
              //c2ylzl = amul(c2_, c2_mul, Iylzl);
              //c3ylzl = amul(c3_, c3_mul, Iylzl);
        }
        if (iz+1<Nz) {
             uylzh = make_float3(ux[Iylzh], uy[Iylzh], uz[Iylzh]);
              //c1ylzh = amul(c1_, c1_mul, Iylzh);
              //c2ylzh = amul(c2_, c2_mul, Iylzh);
              //c3ylzh = amul(c3_, c3_mul, Iylzh);
        }
    }
    if (iz+1<Nz) {
         uzh = make_float3(ux[Izh], uy[Izh], uz[Izh]);
          c1zh = amul(c1_, c1_mul, Izh);
          c2zh = amul(c2_, c2_mul, Izh);
          c3zh = amul(c3_, c3_mul, Izh);
    }
    if (iz-1>=0) {
         uzl = make_float3(ux[Izl], uy[Izl], uz[Izl]);
          c1zl = amul(c1_, c1_mul, Izl);
          c2zl = amul(c2_, c2_mul, Izl);
          c3zl = amul(c3_, c3_mul, Izl);
    }
    
    //initialize derivatives
    float3 dxx  = make_float3(0.0,0.0,0.0);
    float3 dyy  = make_float3(0.0,0.0,0.0);
    float3 dzz  = make_float3(0.0,0.0,0.0);
    float3 dxy  = make_float3(0.0,0.0,0.0);
    float3 dyx  = make_float3(0.0,0.0,0.0);
    float3 dyz  = make_float3(0.0,0.0,0.0);
    float3 dzy  = make_float3(0.0,0.0,0.0);
    float3 dxz  = make_float3(0.0,0.0,0.0);
    float3 dzx  = make_float3(0.0,0.0,0.0);

    //////////////////////////////////////////
    //Calculate derivatives
    //dxx
    //make vector to use the right constant for the right component of u 
    float3 cc = make_float3(c1,c3,c3);
    float3 d1 =make_float3(0.0,0.0,0.0);
    float3 d2 =make_float3(0.0,0.0,0.0);
    float3 d3 = make_float3(0.0,0.0,0.0);
    float3 cch =make_float3(0.0,0.0,0.0);
    float3 ccl = make_float3(0.0,0.0,0.0);
    //if there are no neighbours: keep the second deriative to zero = do nothing
    if (ix-1<0 && ix+1>=Nx) {}
    else {
        //Only neighbour to the right
        if (ix-1<0) {
            cch = make_float3(c1xh,c3xh,c3xh);
            dxx = 0.5*wx*wx*had((cc+cch),(uxh-u0));
        //Only neighbour to the left
        } else if (ix+1>=Nx) {
            ccl = make_float3(c1xl,c3xl,c3xl);
            dxx = 0.5*wx*wx*had((cc+ccl),(uxl-u0));
        //Neighbours on both sides
        } else {
            cch = make_float3(c1xh,c3xh,c3xh);
            ccl = make_float3(c1xl,c3xl,c3xl);
            dxx = 0.5*wx*wx*(had((cc+cch),(uxh-u0))+had((cc+ccl),(uxl-u0)));
        }
    }

    //dyy
    cc = make_float3(c3,c1,c3);
    if (iy-1<0 && iy+1>=Ny) {}
    else {
        if (iy-1<0) {
            cch = make_float3(c3yh,c1yh,c3yh);
            dyy = 0.5*wy*wy*had((cc+cch),(uyh-u0));
        } else if (iy+1>=Ny) {
            ccl = make_float3(c3yl,c1yl,c3yl);
            dyy = 0.5*wy*wy*had((cc+ccl),(uyl-u0));
        } else {
            cch = make_float3(c3yh,c1yh,c3yh);
            ccl = make_float3(c3yl,c1yl,c3yl);
            dyy = 0.5*wy*wy*(had((cc+cch),(uyh-u0))+had((cc+ccl),(uyl-u0)));
        }
    }

    //dzz
    cc = make_float3(c3,c3,c1);
    if (iz-1<0 && iz+1>=Nz) {}
    else {
        if (iz-1<0) {
            cch = make_float3(c3zh,c3zh,c1zh);
            dzz = 0.5*wz*wz*had((cc+cch),(uzh-u0));
        } else if (iz+1>=Nz) {
            ccl = make_float3(c3zl,c3zl,c1zl);
            dzz = 0.5*wz*wz*had((cc+ccl),(uzh-u0));
        } else {
            cch = make_float3(c3zh,c3zh,c1zh);
            ccl = make_float3(c3zl,c3zl,c1zl);
            dzz = 0.5*wz*wz*(had((cc+cch),(uzh-u0))+had((cc+ccl),(uzl-u0)));
        }
    }
    
    //dxy
    cc = make_float3(c3,c2,0);    
     d1 =make_float3(0.0,0.0,0.0);
     d2 =make_float3(0.0,0.0,0.0);
     d3 =make_float3(0.0,0.0,0.0);
    //if there are no neighbours in x/y-direction: keep the second deriative to zero
    if ((ix-1<0 && ix+1>=Nx) || (iy-1<0 && iy+1>=Ny)) {}
    else {
        //No neighbor in the right direction
        if (ix-1<0) {
            if (iy-1<0) {
                //Calculate change in y-direction at postion ix = d2
                //0.5*wy*(FWD + BWD) with BWD=0
                d2 = 0.5*wy*(uyh-u0);
                //Calculate change in y-direction at postion ix+1 = d1
                //rectangular mesh: if (ix+1,iy) and (ix,iy+1) are present, then (ix+1,iy+1) is also present
                d1 = 0.5*wy*(uxhyh-uxh);
            } else if (iy+1>=Ny) {
                d2 = 0.5*wy*(u0-uyl);
                d1 = 0.5*wy*(uxh-uxhyl);
            } else {
                d2 = 0.5*wy*(uyh-uyl);
                d1 = 0.5*wy*(uxhyh-uxhyl);
            }
            cch = make_float3(c3xh,c2xh,0);
            // dxy = 0.5*wx*(FWD - BWD) with BWD = 0 beause ix-1<0
            dxy = 0.5*wx*(had(cch,d1)-had(cc,d2));

        } else if (ix+1>=Nx) {
            if (iy-1<0) {
                //Calculate change in y-direction at postion ix = d2
                d2 = 0.5*wy*(uyh-u0);
                //Calculate change in y-direction at postion ix-1 = d1
                d3 = 0.5*wy*(uxlyh-uxl);
            } else if (iy+1>=Ny) {
                d2 = 0.5*wy*(u0-uyl);
                d3 = 0.5*wy*(uxl-uxlyl);
            } else {
                d2 = 0.5*wy*(uyh-uyl);
                d3 = 0.5*wy*(uxlyh-uxlyl);
            }
            ccl = make_float3(c3xl,c2xl,0);
            dxy = 0.5*wx*(had(cc,d2)-had(ccl,d3));
        } else {
            if (iy-1<0) {
                //Calculate change in y-direction at postion ix+1 = d1
                d1 = 0.5*wy*(uxhyh-uxh);
                //Calculate change in y-direction at postion ix-1 = d3
                d3 = 0.5*wy*(uxlyh-uxl);
            } else if (iy+1>=Ny) {
                d1 = 0.5*wy*(uxh-uxhyl);
                d3 = 0.5*wy*(uxl-uxlyl);
            } else {
                d1 = 0.5*wy*(uxhyh-uxhyl);
                d3 = 0.5*wy*(uxlyh-uxlyl);
            }
            ccl = make_float3(c3xl,c2xl,0);
            cch = make_float3(c3xh,c2xh,0);
            dxy = 0.5*wx*(had(cch,d1)-had(ccl,d3));
        }
    }

    //dyx
    cc = make_float3(c2,c3,0);
    d1= make_float3(0.0,0.0,0.0);
    d2= make_float3(0.0,0.0,0.0);
    d3= make_float3(0.0,0.0,0.0);
    if ((ix-1<0 && ix+1>=Nx) || (iy-1<0 && iy+1>=Ny)) {}
    else {
        if (iy-1<0) {
            if (ix-1<0) {
                d2 = 0.5*wx*(uxh-u0);
                d1 = 0.5*wx*(uxhyh-uyh);
            } else if (ix+1>=Nx) {
                d2 = 0.5*wx*(u0-uxl);
                d1 = 0.5*wx*(uyh-uxlyh);
            } else {
                d2 = 0.5*wx*(uxh-uxl);
                d1 = 0.5*wx*(uxhyh-uxlyh);
            }
            cch = make_float3(c2yh,c3yh,0);
            dyx = 0.5*wy*(had(cch,d1)-had(cc,d2));
        } else if (iy+1>=Ny) {
            if (ix-1<0) {
                d2 = 0.5*wx*(uxh-u0);
                d3 = 0.5*wx*(uxhyl-uyl);
            } else if (ix+1>=Nx) {
                d2 = 0.5*wx*(u0-uxl);
                d3 = 0.5*wx*(uyl-uxlyl);
            } else {
                d2 = 0.5*wx*(uxh-uxl);
                d3 = 0.5*wx*((uxhyl-u0)+(u0-uxlyl));
            }
            ccl = make_float3(c2yl,c3yl,0);
            dyx = 0.5*wy*(had(cc,d2)-had(ccl,d3));
        } else {
            if (ix-1<0) {
                d1 = 0.5*wx*(uxhyh-uyh);
                d3 = 0.5*wx*(uxhyl-uyl);
            } else if (ix+1>=Nx) {
                d1 = 0.5*wx*(uyh-uxlyh);
                d3 = 0.5*wx*(uyl-uxlyl);
            } else {
                d1 = 0.5*wx*(uxhyh-uxlyh);
                d3 = 0.5*wx*(uxhyl-uxlyl);
            }
            ccl = make_float3(c2yl,c3yl,0);
            cch = make_float3(c2yh,c3yh,0);
            dyx = 0.5*wy*(had(cch,d1)-had(ccl,d3));
        }
    }
    
    //dxz
    cc = make_float3(c3,0,c2);
    d1= make_float3(0.0,0.0,0.0);
    d2= make_float3(0.0,0.0,0.0);
    d3 =make_float3(0.0,0.0,0.0);
    if ((ix-1<0 && ix+1>=Nx) || (iz-1<0 && iz+1>=Nz)) {}
    else {
        if (ix-1<0) {
            if (iz-1<0) {
                d2 = 0.5*wz*(uzh-u0);
                d1 = 0.5*wz*(uxhzh-uxh);
            } else if (iz+1>=Nz) {
                d2 = 0.5*wz*(u0-uzl);
                d1 = 0.5*wz*(uxh-uxhzl);
            } else {
                d2 = 0.5*wz*(uzh-uzl);
                d1 = 0.5*wz*(uxhzh-uxhzl);
            }
            cch = make_float3(c3xh,0,c2xh);
            dxz = 0.5*wx*(had(cch,d1)-had(cc,d2));

        } else if (ix+1>=Nx) {
            if (iz-1<0) {
                d2 = 0.5*wz*(uzh-u0);
                d3 = 0.5*wz*(uxlzh-uxl);
            } else if (iz+1>=Nz) {
                d2 = 0.5*wz*(u0-uzl);
                d3 = 0.5*wz*(uxl-uxlzl);
            } else {
                d2 = 0.5*wz*(uzh-uzl);
                d3 = 0.5*wz*(uxlzh-uxlzl);
            }
            ccl = make_float3(c3xl,0,c2xh);
            dxz = 0.5*wx*(had(cc,d2)-had(ccl,d3));
        } else {
            if (iz-1<0) {
                d1 = 0.5*wz*(uxhzh-uxh);
                d3 = 0.5*wz*(uxlzh-uxl);
            } else if (iz+1>=Nz) {
                d1 = 0.5*wz*(uxh-uxhzl);
                d3 = 0.5*wz*(uxl-uxlzl);
            } else {
                d1 = 0.5*wz*(uxhzh-uxhzl);
                d3 = 0.5*wz*(uxlzh-uxlzl);
            }
            ccl = make_float3(c3xl,0,c2xh);
            cch = make_float3(c3xh,0,c2xh);
            dxz = 0.5*wx*(had(cch,d1)-had(ccl,d3));
        }
    }

    //dzx
    cc = make_float3(c2,0,c3);
    d1= make_float3(0.0,0.0,0.0);
    d2= make_float3(0.0,0.0,0.0);
    d3= make_float3(0.0,0.0,0.0);
    if ((ix-1<0 && ix+1>=Nx) || (iz-1<0 && iz+1>=Nz)) {}
    else {
        if (iz-1<0) {
            if (ix-1<0) {
                d2 = 0.5*wx*(uxh-u0);
                d1 = 0.5*wx*(uxhzh-uzh);
            } else if (ix+1>=Nx) {
                d2 = 0.5*wx*(u0-uxl);
                d1 = 0.5*wx*(uzh-uxlzh);
            } else {
                d2 = 0.5*wx*(uxh-uxl);
                d1 = 0.5*wx*(uxhzh-uxlzh);
            }
            cch = make_float3(c2zh,0,c3zh);
            dzx = 0.5*wz*(had(cch,d1)-had(cc,d2));
        } else if (iz+1>=Nz) {
            if (ix-1<0) {
                d2 = 0.5*wx*(uxh-u0);
                d3 = 0.5*wx*(uxhzl-uzl);
            } else if (ix+1>=Nx) {
                d2 = 0.5*wx*(u0-uxl);
                d3 = 0.5*wx*(uzl-uxlzl);
            } else {
                d2 = 0.5*wx*(uxh-uxl);
                d3 = 0.5*wx*(uxhzl-uxlzl);
            }
            ccl = make_float3(c2zl,0,c3zh);
            dzx = 0.5*wz*(had(cc,d2)-had(ccl,d3));
        } else {
            if (ix-1<0) {
                d1 = 0.5*wx*(uxhzh-uzh);
                d3 = 0.5*wx*(uxhzl-uzl);
            } else if (ix+1>=Nx) {
                d1 = 0.5*wx*(uzh-uxlzh);
                d3 = 0.5*wx*(uzl-uxlzl);
            } else {
                d1 = 0.5*wx*(uxhzh-uxlzh);
                d3 = 0.5*wx*(uxhzl-uxlzl);
            }
            ccl = make_float3(c2zl,0,c3zh);
            cch = make_float3(c2zh,0,c3zh);
            dzx = 0.5*wz*(had(cch,d1)-had(ccl,d3));
        }
    }

    //dyz
    cc = make_float3(0,c3,c2);
    d1= make_float3(0.0,0.0,0.0);
    d2= make_float3(0.0,0.0,0.0);
    d3= make_float3(0.0,0.0,0.0);
    if ((iy-1<0 && iy+1>=Ny) || (iz-1<0 && iz+1>=Nz)) {}
    else {
        if (iy-1<0) {
            if (iz-1<0) {
                d2 = 0.5*wz*(uzh-u0);
                d1 = 0.5*wz*(uyhzh-uyh);
            } else if (iz+1>=Nz) {
                d2 = 0.5*wz*(u0-uzl);
                d1 = 0.5*wz*(uyh-uyhzl);
            } else {
                d2 = 0.5*wz*(uzh-uzl);
                d1 = 0.5*wz*(uyhzh-uyhzl);
            }
            cch = make_float3(0,c3yh,c2yh);
            dyz = 0.5*wy*(had(cch,d1)-had(cc,d2));

        } else if (iy+1>=Ny) {
            if (iz-1<0) {
                d2 = 0.5*wz*(uzh-u0);
                d3 = 0.5*wz*(uylzh-uyl);
            } else if (iz+1>=Nz) {
                d2 = 0.5*wz*(u0-uzl);
                d3 = 0.5*wz*(uyl-uylzl);
            } else {
                d2 = 0.5*wz*(uzh-uzl);
                d3 = 0.5*wz*(uylzh-uylzl);
            }
            ccl = make_float3(0,c3yh,c2yh);
            dyz = 0.5*wy*(had(cc,d2)-had(ccl,d3));
        } else {
            if (iz-1<0) {
                d1 = 0.5*wz*(uyhzh-uyh);
                d3 = 0.5*wz*(uylzh-uyl);
            } else if (iz+1>=Nz) {
                d1 = 0.5*wz*(uyh-uyhzl);
                d3 = 0.5*wz*(uyl-uylzl);
            } else {
                d1 = 0.5*wz*(uyhzh-uyhzl);
                d3 = 0.5*wz*(uylzh-uylzl);
            }
            ccl = make_float3(0,c3yh,c2yh);
            cch = make_float3(0,c3yh,c2yh);
            dyz = 0.5*wy*(had(cch,d1)-had(ccl,d3));
        }
    }

    //dzy
    cc = make_float3(0,c2,c3);
    d1= make_float3(0.0,0.0,0.0);
    d2 = make_float3(0.0,0.0,0.0);
    d3= make_float3(0.0,0.0,0.0);
    if ((iy-1<0 && iy+1>=Ny) || (iz-1<0 && iz+1>=Nz)) {}
    else {
        if (iz-1<0) {
            if (iy-1<0) {
                d2 = 0.5*wy*(uyh-u0);
                d1 = 0.5*wy*(uyhzh-uzh);
            } else if (iy+1>=Ny) {
                d2 = 0.5*wy*(u0-uyl);
                d1 = 0.5*wy*(uzh-uylzh);
            } else {
                d2 = 0.5*wy*(uyh-uyl);
                d1 = 0.5*wy*(uyhzh-uylzh);
            }
            cch = make_float3(0,c2zh,c3zh);
            dzy = 0.5*wz*(had(cch,d1)-had(cc,d2));
        } else if (iz+1>=Nz) {
            if (iy-1<0) {
                d2 = 0.5*wy*(uyh-u0);
                d3 = 0.5*wy*(uyhzl-uzl);
            } else if (iy+1>=Ny) {
                d2 = 0.5*wy*(u0-uyl);
                d3 = 0.5*wy*(uzl-uylzl);
            } else {
                d2 = 0.5*wy*(uyh-uyl);
                d3 = 0.5*wy*(uyhzl-uylzl);
            }
            ccl = make_float3(0,c2zh,c3zh);
            dzy = 0.5*wz*(had(cc,d2)-had(ccl,d3));
        } else {
            if (iy-1<0) {
                d1 = 0.5*wy*(uyhzh-uzh);
                d3 = 0.5*wy*(uyhzl-uzl);
            } else if (iy+1>=Ny) {
                d1 = 0.5*wy*(uzh-uylzh);
                d3 = 0.5*wy*(uzl-uylzl);
            } else {
                d1 = 0.5*wy*(uyhzh-uylzh);
                d3 = 0.5*wy*(uyhzl-uylzl);
            }
            ccl = make_float3(0,c2zh,c3zh);
            cch = make_float3(0,c2zh,c3zh);
            dzy = 0.5*wz*(had(cch,d1)-had(ccl,d3));
        }
    }
    //output gelijkstellen aan de magnetizatie
    dux[I] = dxx.x + dxy.y + dxz.z + dyy.x + dyx.y + dzz.x + dzx.z;
    duy[I] = dyy.y + dyx.x + dyz.z + dxx.y + dxy.x + dzz.y + dzy.z;
    duz[I] = dzz.z + dzx.x + dzy.y + dxx.z + dxz.x + dyy.z + dyz.y; 
}


