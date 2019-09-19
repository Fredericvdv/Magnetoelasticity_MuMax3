#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "stdio.h"

extern "C" __global__ void
Elastodynamic_freebndry(float* __restrict__ dux, float* __restrict__ duy, float* __restrict__ duz, 
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
    int I__ = idx(ix, iy, iz);
    float3 u__ = make_float3(0.0,0.0,0.0);

    float3 d_ = make_float3(0.0,0.0,0.0);
    float f_ = 0;

    dux[I] = 0.0 ;
    duy[I] = 0.0 ;
    duz[I] = 0.0 ;



    if (ix==0 ) {
        if (iy==0) {
            //Left-down corner
            //Double derivative in x-direction
            d_ = make_float3(0.0,0.0,0.0);
            cc = make_float3(amul(C1_, C1_mul, I),amul(C3_, C3_mul, I),amul(C3_, C3_mul, I));
            I_ = idx(hclampx(ix+1), iy, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ = 2*wx*wx*had(cc,(u_-u0));
            
            dux[I] += d_.x ;
            duy[I] += d_.y ;
            duz[I] += 0 ;

            //Double derivative in y-direction
            d_ = make_float3(0.0,0.0,0.0);
            cc = make_float3(amul(C3_, C3_mul, I),amul(C1_, C1_mul, I),amul(C3_, C3_mul, I));
            //Right neighbor
            I_ = idx(ix, hclampy(iy+1), iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ = 2*wy*wy*had(cc,(u_-u0));
            
            dux[I] += d_.x ;
            duy[I] += d_.y ;
            duz[I] += 0 ;

        } else if (iy==Ny-1) {
            //Left-up corner
            //Double derivative in x-direction
             d_ = make_float3(0.0,0.0,0.0);
             cc = make_float3(amul(C1_, C1_mul, I),amul(C3_, C3_mul, I),amul(C3_, C3_mul, I));
             //Right neighbor
             I_ = idx(hclampx(ix+1), iy, iz);
             u_ = make_float3(ux[I_], uy[I_], uz[I_]);
             d_ = 2*wx*wx*had(cc,(u_-u0));
             
             dux[I] += d_.x ;
             duy[I] += d_.y ;
             duz[I] += 0 ;
 
             //Double derivative in y-direction
             d_ = make_float3(0.0,0.0,0.0);
             cc = make_float3(amul(C3_, C3_mul, I),amul(C1_, C1_mul, I),amul(C3_, C3_mul, I));
             //Right neighbor
             I_ = idx(ix, lclampy(iy-1), iz);
             u_ = make_float3(ux[I_], uy[I_], uz[I_]);
             d_ = 2*wy*wy*had(cc,(u_-u0));
             
             dux[I] += d_.x ;
             duy[I] += d_.y ;
             duz[I] += 0 ;

        } else {
            //Left interface
            //Double derivative in x-direction
            f_=0;
            I_ = idx(hclampx(ix+1), iy, iz);
            f_ = 2*wx*wx*(ux[I_]-ux[I]);
            I_ = idx(ix, hclampy(iy+1), iz);
            I__ = idx(ix, lclampy(iy-1), iz);
            f_ += wx*wy*amul(C2_, C2_mul, I)*(uy[I_]-uy[I__])/amul(C1_, C1_mul, I);
            dux[I] += amul(C1_, C1_mul, I)*f_ ;

            f_=0;
            I_ = idx(hclampx(ix+1), iy, iz);
            f_ = 2*wx*wx*(uy[I_]-uy[I]);
            I_ = idx(ix, hclampy(iy+1), iz);
            I__ = idx(ix, lclampy(iy-1), iz);
            f_ += wx*wy*(uy[I_]-uy[I__]);
            duy[I] += amul(C3_, C3_mul, I)*f_ ;


            //Double derivative in y-direction
            d_ = make_float3(0.0,0.0,0.0);
            cc = make_float3(amul(C3_, C3_mul, I),amul(C1_, C1_mul, I),amul(C3_, C3_mul, I));
            //Right neighbor
            I_ = idx(ix, hclampy(iy+1), iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ = wy*wy*had(cc,(u_-u0));
            //Left neighbour
            I_ = idx(ix, lclampy(iy-1), iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ += wy*wy*had(cc,(u_-u0));
            
            dux[I] += d_.x ;
            duy[I] += d_.y ;
            duz[I] += 0 ;
            
            //dxy without boundaries
            d_ = make_float3(0.0,0.0,0.0);
            I_ = idx(ix,hclampy(iy+1), iz);
            d_ += make_float3(ux[I_], uy[I_], uz[I_]);
            I_ = idx(ix, lclampy(iy-1), iz);
            d_ += make_float3(ux[I_], uy[I_], uz[I_]);
            d_ -= 2*u0;

            d_ = -(amul(C2_, C2_mul, I)+amul(C3_, C3_mul, I))*d_*wy*wy;

            dux[I] += d_.y ;
            duy[I] += d_.x ;
            duz[I] += 0.0 ;
        }

    } else if (ix==Nx-1) {
        if (iy==0) {
            //right-down corner
            //Double derivative in x-direction
            d_ = make_float3(0.0,0.0,0.0);
            cc = make_float3(amul(C1_, C1_mul, I),amul(C3_, C3_mul, I),amul(C3_, C3_mul, I));
            //Right neighbor
            I_ = idx(lclampx(ix-1), iy, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ = 2*wx*wx*had(cc,(u_-u0));
            
            dux[I] += d_.x ;
            duy[I] += d_.y ;
            duz[I] += 0 ;

            //Double derivative in y-direction
            d_ = make_float3(0.0,0.0,0.0);
            cc = make_float3(amul(C3_, C3_mul, I),amul(C1_, C1_mul, I),amul(C3_, C3_mul, I));
            //Right neighbor
            I_ = idx(ix, hclampy(iy+1), iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ = 2*wy*wy*had(cc,(u_-u0));
            
            dux[I] += d_.x ;
            duy[I] += d_.y ;
            duz[I] += 0 ;

        } else if (iy==Ny-1) {
            //right-up corner
            //Double derivative in x-direction
            d_ = make_float3(0.0,0.0,0.0);
            cc = make_float3(amul(C1_, C1_mul, I),amul(C3_, C3_mul, I),amul(C3_, C3_mul, I));
            //Right neighbor
            I_ = idx(lclampx(ix-1), iy, iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ = 2*wx*wx*had(cc,(u_-u0));
            
            dux[I] += d_.x ;
            duy[I] += d_.y ;
            duz[I] += 0 ;

            //Double derivative in y-direction
            d_ = make_float3(0.0,0.0,0.0);
            cc = make_float3(amul(C3_, C3_mul, I),amul(C1_, C1_mul, I),amul(C3_, C3_mul, I));
            //Right neighbor
            I_ = idx(ix, lclampy(iy-1), iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ = 2*wy*wy*had(cc,(u_-u0));
            
            dux[I] += d_.x ;
            duy[I] += d_.y ;
            duz[I] += 0 ;
        } else {
            //right interface
            //Double derivative in x-direction
            f_=0;
            I_ = idx(hclampx(ix-1), iy, iz);
            f_ = 2*wx*wx*(ux[I_]-ux[I]);
            I_ = idx(ix, hclampy(iy+1), iz);
            I__ = idx(ix, lclampy(iy-1), iz);
            f_ += wx*wy*amul(C2_, C2_mul, I)*(uy[I_]-uy[I__])/amul(C1_, C1_mul, I);
            dux[I] += amul(C1_, C1_mul, I)*f_ ;

            f_=0;
            I_ = idx(hclampx(ix-1), iy, iz);
            f_ = 2*wx*wx*(uy[I_]-uy[I]);
            I_ = idx(ix, hclampy(iy+1), iz);
            I__ = idx(ix, lclampy(iy-1), iz);
            f_ += wx*wy*(uy[I_]-uy[I__]);
            duy[I] += amul(C3_, C3_mul, I)*f_ ;

            //Double derivative in y-direction
            d_ = make_float3(0.0,0.0,0.0);
            cc = make_float3(amul(C3_, C3_mul, I),amul(C1_, C1_mul, I),amul(C3_, C3_mul, I));
            //Right neighbor
            I_ = idx(ix, hclampy(iy+1), iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ = wy*wy*had(cc,(u_-u0));
            //Left neighbour
            I_ = idx(ix, lclampy(iy-1), iz);
            u_ = make_float3(ux[I_], uy[I_], uz[I_]);
            d_ += wy*wy*had(cc,(u_-u0));

            dux[I] += d_.x ;
            duy[I] += d_.y ;
            duz[I] += 0 ;

            //dxy without boundaries
            d_ = make_float3(0.0,0.0,0.0);
            I_ = idx(ix,hclampy(iy+1), iz);
            d_ += make_float3(ux[I_], uy[I_], uz[I_]);
            I_ = idx(ix, lclampy(iy-1), iz);
            d_ += make_float3(ux[I_], uy[I_], uz[I_]);
            d_ -= 2*u0;

            d_ = -(amul(C2_, C2_mul, I)+amul(C3_, C3_mul, I))*d_*wy*wy;

            dux[I] += d_.y ;
            duy[I] += d_.x ;
            duz[I] += 0.0 ;
        }
    } else if (iy==0) {
        //Bottom interface
        //Double derivative in y-direction
        d_ = make_float3(0.0,0.0,0.0);
        cc = make_float3(amul(C1_, C1_mul, I),amul(C3_, C3_mul, I),amul(C3_, C3_mul, I));
        //Right neighbor
        I_ = idx(hclampx(ix+1), iy, iz);
        u_ = make_float3(ux[I_], uy[I_], uz[I_]);
        d_ = wx*wx*had(cc,(u_-u0));
        //Left neighbour
        I_ = idx(lclampx(ix-1), iy, iz);
        u_ = make_float3(ux[I_], uy[I_], uz[I_]);
        d_ += wx*wx*had(cc,(u_-u0));
        
        dux[I] += d_.x ;
        duy[I] += d_.y ;
        duz[I] += 0 ;

        //Double derivative in y-direction
        f_=0;
        I_ = idx(ix, hclampx(iy+1), iz);
        f_ = 2*wy*wy*(uy[I_]-uy[I]);
        I_ = idx(hclampx(ix+1), iy, iz);
        I__ = idx(lclampx(ix-1), iy, iz);
        f_ += wx*wy*amul(C2_, C2_mul, I)*(ux[I_]-ux[I__])/amul(C1_, C1_mul, I);
        duy[I] += amul(C1_, C1_mul, I)*f_ ;

        f_=0;
        I_ = idx(ix, hclampy(iy+1), iz);
        f_ = 2*wy*wy*(ux[I_]-ux[I]);
        I_ = idx(hclampx(ix+1), iy, iz);
        I__ = idx(lclampx(ix-1), iy, iz);
        f_ += wx*wy*(uy[I_]-uy[I__]);
        dux[I] += amul(C3_, C3_mul, I)*f_ ;

        //dxy without boundaries
        d_ = make_float3(0.0,0.0,0.0);
        I_ = idx(hclampx(ix+1), iy, iz);
        d_ += make_float3(ux[I_], uy[I_], uz[I_]);
        I_ = idx(lclampx(ix-1), iy, iz);
        d_ += make_float3(ux[I_], uy[I_], uz[I_]);
        d_ -= 2*u0;
        d_ = -(amul(C2_, C2_mul, I)+amul(C3_, C3_mul, I))*d_*wx*wx;

        dux[I] += d_.y ;
        duy[I] += d_.x ;
        duz[I] += 0.0 ;

    } else if (iy==Ny-1) {
        //Top interface
        //Double derivative in x-direction
        d_ = make_float3(0.0,0.0,0.0);
        cc = make_float3(amul(C1_, C1_mul, I),amul(C3_, C3_mul, I),amul(C3_, C3_mul, I));
        //Right neighbor
        I_ = idx(hclampx(ix+1), iy, iz);
        u_ = make_float3(ux[I_], uy[I_], uz[I_]);
        d_ = wx*wx*had(cc,(u_-u0));
        //Left neighbour
        I_ = idx(lclampx(ix-1), iy, iz);
        u_ = make_float3(ux[I_], uy[I_], uz[I_]);
        d_ += wx*wx*had(cc,(u_-u0));
        
        dux[I] += d_.x ;
        duy[I] += d_.y ;
        duz[I] += 0 ;

        //Double derivative in y-direction
        f_=0;
        I_ = idx(ix, lclampx(iy-1), iz);
        f_ = 2*wy*wy*(uy[I_]-uy[I]);
        I_ = idx(hclampx(ix+1), iy, iz);
        I__ = idx(lclampx(ix-1), iy, iz);
        f_ += wx*wy*amul(C2_, C2_mul, I)*(ux[I_]-ux[I__])/amul(C1_, C1_mul, I);
        duy[I] += amul(C1_, C1_mul, I)*f_ ;

        f_=0;
        I_ = idx(ix, lclampy(iy-1), iz);
        f_ = 2*wy*wy*(ux[I_]-ux[I]);
        I_ = idx(hclampx(ix+1), iy, iz);
        I__ = idx(lclampx(ix-1), iy, iz);
        f_ += wx*wy*(uy[I_]-uy[I__]);
        dux[I] += amul(C3_, C3_mul, I)*f_ ;

        //dxy without boundaries
        d_ = make_float3(0.0,0.0,0.0);
        I_ = idx(hclampx(ix+1), iy, iz);
        d_ += make_float3(ux[I_], uy[I_], uz[I_]);
        I_ = idx(lclampx(ix-1), iy, iz);
        d_ += make_float3(ux[I_], uy[I_], uz[I_]);
        d_ -= 2*u0;
        d_ = -(amul(C2_, C2_mul, I)+amul(C3_, C3_mul, I))*d_*wx*wx;

        dux[I] += d_.y ;
        duy[I] += d_.x ;
        duz[I] += 0.0 ;

    } else {
        //Bulk
        //Double derivative in x-direction
        d_ = make_float3(0.0,0.0,0.0);
        cc = make_float3(amul(C1_, C1_mul, I),amul(C3_, C3_mul, I),amul(C3_, C3_mul, I));
        //Right neighbor
        I_ = idx(hclampx(ix+1), iy, iz);
        u_ = make_float3(ux[I_], uy[I_], uz[I_]);
        d_ = wx*wx*had(cc,(u_-u0));
        //Left neighbour
        I_ = idx(lclampx(ix-1), iy, iz);
        u_ = make_float3(ux[I_], uy[I_], uz[I_]);
        d_ += wx*wx*had(cc,(u_-u0));
        
        dux[I] += d_.x ;
        duy[I] += d_.y ;
        duz[I] += 0 ;

        //Double derivative in y-direction
        d_ = make_float3(0.0,0.0,0.0);
        cc = make_float3(amul(C3_, C3_mul, I),amul(C1_, C1_mul, I),amul(C3_, C3_mul, I));
        //Right neighbor
        I_ = idx(ix, hclampy(iy+1), iz);
        u_ = make_float3(ux[I_], uy[I_], uz[I_]);
        d_ = wy*wy*had(cc,(u_-u0));
        //Left neighbour
        I_ = idx(ix, lclampy(iy-1), iz);
        u_ = make_float3(ux[I_], uy[I_], uz[I_]);
        d_ += wy*wy*had(cc,(u_-u0));
        
        dux[I] += d_.x ;
        duy[I] += d_.y ;
        duz[I] += 0 ;

        //dxy without boundaries
        d_ = make_float3(0.0,0.0,0.0);
        //(i+1,j+1)
        I_ = idx(hclampx(ix+1),hclampy(iy+1), iz);
        d_ += make_float3(ux[I_], uy[I_], uz[I_]);
        //(i-1,j-1)
        I_ = idx(lclampx(ix-1),lclampy(iy-1), iz);
        d_ += make_float3(ux[I_], uy[I_], uz[I_]);
        //(i+1,j-1)
        I_ = idx(hclampx(ix+1),lclampy(iy-1), iz);
        d_ -= make_float3(ux[I_], uy[I_], uz[I_]);
        //(i-1,j+1)
        I_ = idx(lclampx(ix-1),hclampy(iy+1), iz);
        d_ -= make_float3(ux[I_], uy[I_], uz[I_]);

        d_ = (amul(C2_, C2_mul, I)+amul(C3_, C3_mul, I))*d_*0.25*wx*wy;

        dux[I] += d_.y ;
        duy[I] += d_.x ;
        duz[I] += 0.0 ;
    }
}
