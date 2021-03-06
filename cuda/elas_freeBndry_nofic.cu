#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "stdio.h"

extern "C" __global__ void
Elastos(float* __restrict__ dux, float* __restrict__ duy, float* __restrict__ duz, 
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
    float3 cc_ = make_float3(0.0,0.0,0.0);

    //Neighbor cell
    int I_ = idx(ix, iy, iz);
    float3 u_ = make_float3(0.0,0.0,0.0);
    float3 d_ = make_float3(0.0,0.0,0.0);
    float d__ = 0;
    float cc__ = 0 ;

    dux[I] = 0.0 ;
    duy[I] = 0.0 ;
    duz[I] = 0.0 ;

    if (amul(C1_, C1_mul, I)==0) {
        return ;
    }


    //################################
    //x-interface
    if (ix==0 && PBCx==0) {
        if (iy==0 && PBCy==0) {
            //Left-down corner
            I_ = idx(ix+1, iy, iz);
            cc__ = 0.5*(amul(C1_, C1_mul, I)+amul(C1_, C1_mul, I_));
            dux[I] += cc__*wx*wx*(ux[I_]-ux[I]);

            I_ = idx(ix, iy+1, iz);
            cc__ = amul(C2_, C2_mul, I_);
            d__ = uy[I_];
            I_ = idx(ix+1, iy+1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ += uy[I_];
            I_ = idx(ix, iy, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= uy[I_];
            I_ = idx(ix+1, iy, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= uy[I_];
            cc__ = cc__/4;
            dux[I] += cc__*wx*wy*0.5*d__;

            I_ = idx(ix, iy+1, iz);
            cc__ = 0.5*(amul(C1_, C1_mul, I)+amul(C1_, C1_mul, I_));
            duy[I] += cc__*wy*wy*(uy[I_]-uy[I]);

            I_ = idx(ix+1, iy, iz);
            cc__ = amul(C2_, C2_mul, I_);
            d__ = ux[I_];
            I_ = idx(ix+1, iy+1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ += ux[I_];
            I_ = idx(ix, iy, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= ux[I_];
            I_ = idx(ix, iy+1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= ux[I_];
            cc__ = cc__/4;
            dux[I] += cc__*wx*wy*0.5*d__;
            
        } else if (iy==Ny-1 && PBCy==0) {
            //Left-upper corner
            I_ = idx(ix+1, iy, iz);
            cc__ = 0.5*(amul(C1_, C1_mul, I)+amul(C1_, C1_mul, I_));
            dux[I] += cc__*wx*wx*(ux[I_]-ux[I]);

            I_ = idx(ix, iy, iz);
            cc__ = amul(C2_, C2_mul, I_);
            d__ = uy[I_];
            I_ = idx(ix+1, iy, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ += uy[I_];
            I_ = idx(ix, iy-1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= uy[I_];
            I_ = idx(ix+1, iy-1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= uy[I_];
            cc__ = cc__/4;
            dux[I] += cc__*wx*wy*0.5*d__;

            I_ = idx(ix, iy-1, iz);
            cc__ = 0.5*(amul(C1_, C1_mul, I)+amul(C1_, C1_mul, I_));
            duy[I] += cc__*wy*wy*(uy[I_]-uy[I]);

            I_ = idx(ix+1, iy, iz);
            cc__ = amul(C2_, C2_mul, I_);
            d__ = ux[I_];
            I_ = idx(ix+1, iy-1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ += ux[I_];
            I_ = idx(ix, iy, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= ux[I_];
            I_ = idx(ix, iy-1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= ux[I_];
            cc__ = cc__/4;
            dux[I] += cc__*wx*wy*0.5*d__;
            
        } else {
            //Left interface
            I_ = idx(ix+1, iy, iz);
            cc__ = 0.5*(amul(C1_, C1_mul, I)+amul(C1_, C1_mul, I_));
            dux[I] += cc__*wx*wx*(ux[I_]-ux[I]);

            I_ = idx(ix, iy+1, iz);
            cc__ = amul(C2_, C2_mul, I_);
            d__ = uy[I_];
            I_ = idx(ix+1, iy+1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ += uy[I_];
            I_ = idx(ix, iy-1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= uy[I_];
            I_ = idx(ix+1, iy-1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= uy[I_];
            cc__ = cc__/4;
            dux[I] += cc__*wx*wy*0.25*d__;

            I_ = idx(ix, iy+1, iz);
            cc__ = amul(C3_, C3_mul, I_);
            d__ = ux[I_];
            I_ = idx(ix+1, iy+1, iz);
            cc__ += amul(C3_, C3_mul, I_);
            d__ += ux[I_];
            I_ = idx(ix, iy-1, iz);
            cc__ += amul(C3_, C3_mul, I_);
            d__ -= ux[I_];
            I_ = idx(ix+1, iy-1, iz);
            cc__ += amul(C3_, C3_mul, I_);
            d__ -= ux[I_];
            cc__ = cc__/4;
            duy[I] += cc__*wx*wy*0.25*d__;

            I_ = idx(ix+1, iy, iz);
            cc__ = 0.5*(amul(C3_, C3_mul, I)+amul(C3_, C3_mul, I_));
            duy[I] += cc__*wx*wx*(uy[I_]-uy[I]);

            I_ = idx(ix, iy+1, iz);
            cc__ = amul(C1_, C1_mul, I_);
            d__ = uy[I_];
            I_ = idx(ix, iy-1, iz);
            cc__ += amul(C1_, C1_mul, I_);
            d__ += uy[I_];
            cc__ += 2*amul(C1_, C1_mul, I);
            d__ -= 2*uy[I];
            cc__ = cc__/4;
            duy[I] += cc__*wy*wy*d__;

            I_ = idx(ix+1, iy+1, iz);
            cc__ = amul(C2_, C2_mul, I_);
            d__ = ux[I_];
            I_ = idx(ix, iy+1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= ux[I_];
            I_ = idx(ix+1, iy-1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= ux[I_];
            I_ = idx(ix, iy-1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ += ux[I_];
            cc__ = cc__/4;
            duy[I] += cc__*wx*wy*0.5*d__;            
        }
    } else if (ix==Nx-1 && PBCx==0) {
        if (iy==0 && PBCy==0) {
            //Right-down corner
            I_ = idx(ix-1, iy, iz);
            cc__ = 0.5*(amul(C1_, C1_mul, I)+amul(C1_, C1_mul, I_));
            dux[I] += cc__*wx*wx*(ux[I_]-ux[I]);

            I_ = idx(ix, iy+1, iz);
            cc__ = amul(C2_, C2_mul, I_);
            d__ = uy[I_];
            I_ = idx(ix-1, iy+1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ += uy[I_];
            I_ = idx(ix, iy, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= uy[I_];
            I_ = idx(ix-1, iy, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= uy[I_];
            cc__ = cc__/4;
            dux[I] += -cc__*wx*wy*0.5*d__;

            I_ = idx(ix, iy+1, iz);
            cc__ = 0.5*(amul(C1_, C1_mul, I)+amul(C1_, C1_mul, I_));
            duy[I] += cc__*wy*wy*(uy[I_]-uy[I]);

            I_ = idx(ix, iy, iz);
            cc__ = amul(C2_, C2_mul, I_);
            d__ = ux[I_];
            I_ = idx(ix, iy+1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ += ux[I_];
            I_ = idx(ix-1, iy, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= ux[I_];
            I_ = idx(ix-1, iy+1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= ux[I_];
            cc__ = cc__/4;
            dux[I] += -cc__*wx*wy*0.5*d__;

            
            
        } else if (iy==Ny-1 && PBCy==0) {
            //Right-upper corner
            I_ = idx(ix-1, iy, iz);
            cc__ = 0.5*(amul(C1_, C1_mul, I)+amul(C1_, C1_mul, I_));
            dux[I] += -cc__*wx*wx*(ux[I]-ux[I_]);

            I_ = idx(ix, iy, iz);
            cc__ = amul(C2_, C2_mul, I_);
            d__ = uy[I_];
            I_ = idx(ix-1, iy, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ += uy[I_];
            I_ = idx(ix, iy-1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= uy[I_];
            I_ = idx(ix-1, iy-1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= uy[I_];
            cc__ = cc__/4;
            dux[I] += -cc__*wx*wy*0.5*d__;

            I_ = idx(ix, iy-1, iz);
            cc__ = 0.5*(amul(C1_, C1_mul, I)+amul(C1_, C1_mul, I_));
            duy[I] += -cc__*wy*wy*(uy[I]-uy[I_]);

            I_ = idx(ix, iy, iz);
            cc__ = amul(C2_, C2_mul, I_);
            d__ = ux[I_];
            I_ = idx(ix, iy-1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ += ux[I_];
            I_ = idx(ix-1, iy, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= ux[I_];
            I_ = idx(ix-1, iy-1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= ux[I_];
            cc__ = cc__/4;
            dux[I] += -cc__*wx*wy*0.5*d__;
            
        } else {
            //Right interface
            I_ = idx(ix-1, iy, iz);
            cc__ = 0.5*(amul(C1_, C1_mul, I)+amul(C1_, C1_mul, I_));
            dux[I] += cc__*wx*wx*(ux[I_]-ux[I]);

            I_ = idx(ix, iy+1, iz);
            cc__ = amul(C2_, C2_mul, I_);
            d__ = uy[I_];
            I_ = idx(ix-1, iy+1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ += uy[I_];
            I_ = idx(ix, iy-1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= uy[I_];
            I_ = idx(ix-1, iy-1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= uy[I_];
            cc__ = cc__/4;
            dux[I] += -cc__*wx*wy*0.25*d__;

            I_ = idx(ix, iy+1, iz);
            cc__ = amul(C3_, C3_mul, I_);
            d__ = ux[I_];
            I_ = idx(ix-1, iy+1, iz);
            cc__ += amul(C3_, C3_mul, I_);
            d__ += ux[I_];
            I_ = idx(ix, iy-1, iz);
            cc__ += amul(C3_, C3_mul, I_);
            d__ -= ux[I_];
            I_ = idx(ix-1, iy-1, iz);
            cc__ += amul(C3_, C3_mul, I_);
            d__ -= ux[I_];
            cc__ = cc__/4;
            duy[I] += -cc__*wx*wy*0.25*d__;

            I_ = idx(ix-1, iy, iz);
            cc__ = 0.5*(amul(C3_, C3_mul, I)+amul(C3_, C3_mul, I_));
            duy[I] += cc__*wx*wx*(uy[I_]-uy[I]);

            I_ = idx(ix, iy+1, iz);
            cc__ = amul(C1_, C1_mul, I_);
            d__ = uy[I_];
            I_ = idx(ix, iy-1, iz);
            cc__ += amul(C1_, C1_mul, I_);
            d__ += uy[I_];
            cc__ += 2*amul(C1_, C1_mul, I);
            d__ -= 2*uy[I];
            cc__ = cc__/4;
            duy[I] += cc__*wy*wy*d__;

            I_ = idx(ix, iy+1, iz);
            cc__ = amul(C2_, C2_mul, I_);
            d__ = ux[I_];
            I_ = idx(ix-1, iy+1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= ux[I_];
            I_ = idx(ix, iy-1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ -= ux[I_];
            I_ = idx(ix-1, iy-1, iz);
            cc__ += amul(C2_, C2_mul, I_);
            d__ += ux[I_];
            cc__ = cc__/4;
            duy[I] += cc__*wx*wy*0.5*d__;
            
        }
    } else if (iy==0 && PBCy==0) {
        //down interface
        I_ = idx(hclampx(ix+1), iy, iz);
        cc__ = amul(C1_, C1_mul, I_);
        d__ = ux[I_];
        I_ = idx(lclampx(ix-1), iy, iz);
        cc__ += amul(C1_, C1_mul, I_);
        d__ += ux[I_];
        cc__ += 2*amul(C1_, C1_mul, I);
        d__ -= 2*ux[I];
        cc__ = cc__/4;
        dux[I] += cc__*wx*wx*d__;

        I_ = idx(hclampx(ix+1), iy+1, iz);
        cc__ = amul(C2_, C2_mul, I_);
        d__ = uy[I_];
        I_ = idx(hclampx(ix+1), iy, iz);
        cc__ += amul(C2_, C2_mul, I_);
        d__ -= uy[I_];
        I_ = idx(lclampx(ix-1), iy+1, iz);
        cc__ += amul(C2_, C2_mul, I_);
        d__ -= uy[I_];
        I_ = idx(lclampx(ix-1), iy, iz);
        cc__ += amul(C2_, C2_mul, I_);
        d__ += uy[I_];
        cc__ = cc__/4;
        dux[I] += cc__*wx*wy*0.5*d__;

        I_ = idx(hclampx(ix+1), iy, iz);
        cc__ = amul(C3_, C3_mul, I_);
        d__ = uy[I_];
        I_ = idx(hclampx(ix+1), iy+1, iz);
        cc__ += amul(C3_, C3_mul, I_);
        d__ += uy[I_];
        I_ = idx(lclampx(ix-1), iy, iz);
        cc__ += amul(C3_, C3_mul, I_);
        d__ -= uy[I_];
        I_ = idx(lclampx(ix-1), iy+1, iz);
        cc__ += amul(C3_, C3_mul, I_);
        d__ -= uy[I_];
        cc__ = cc__/4;
        dux[I] += cc__*wx*wy*0.25*d__;

        I_ = idx(ix, iy+1, iz);
        cc__ = 0.5*(amul(C3_, C3_mul, I)+amul(C3_, C3_mul, I_));
        dux[I] += cc__*wy*wy*(ux[I_]-ux[I]);

        I_ = idx(ix, iy+1, iz);
        cc__ = 0.5*(amul(C1_, C1_mul, I)+amul(C1_, C1_mul, I_));
        duy[I] += cc__*wy*wy*(uy[I_]-uy[I]);

        I_ = idx(hclampx(ix+1), iy, iz);
        cc__ = amul(C2_, C2_mul, I_);
        d__ = ux[I_];
        I_ = idx(hclampx(ix+1), iy+1, iz);
        cc__ += amul(C2_, C2_mul, I_);
        d__ += ux[I_];
        I_ = idx(lclampx(ix-1), iy, iz);
        cc__ += amul(C2_, C2_mul, I_);
        d__ -= ux[I_];
        I_ = idx(lclampx(ix-1), iy+1, iz);
        cc__ += amul(C2_, C2_mul, I_);
        d__ -= ux[I_];
        cc__ = cc__/4;
        duy[I] += cc__*wx*wy*0.25*d__;
        
        
    } else if (iy==Ny-1 && PBCy==0) {
        //Upper-inteface
        I_ = idx(hclampx(ix+1), iy, iz);
        cc__ = amul(C1_, C1_mul, I_);
        d__ = ux[I_];
        I_ = idx(lclampx(ix-1), iy, iz);
        cc__ += amul(C1_, C1_mul, I_);
        d__ += ux[I_];
        cc__ += 2*amul(C1_, C1_mul, I);
        d__ -= 2*ux[I];
        cc__ = cc__/4;
        dux[I] += cc__*wx*wx*d__;

        I_ = idx(hclampx(ix+1), iy, iz);
        cc__ = amul(C2_, C2_mul, I_);
        d__ = uy[I_];
        I_ = idx(hclampx(ix+1), iy-1, iz);
        cc__ += amul(C2_, C2_mul, I_);
        d__ -= uy[I_];
        I_ = idx(lclampx(ix-1), iy, iz);
        cc__ += amul(C2_, C2_mul, I_);
        d__ -= uy[I_];
        I_ = idx(lclampx(ix-1), iy-1, iz);
        cc__ += amul(C2_, C2_mul, I_);
        d__ += uy[I_];
        cc__ = cc__/4;
        dux[I] += cc__*wx*wy*0.5*d__;

        I_ = idx(hclampx(ix+1), iy, iz);
        cc__ = amul(C3_, C3_mul, I_);
        d__ = uy[I_];
        I_ = idx(hclampx(ix+1), iy-1, iz);
        cc__ += amul(C3_, C3_mul, I_);
        d__ += uy[I_];
        I_ = idx(lclampx(ix-1), iy, iz);
        cc__ += amul(C3_, C3_mul, I_);
        d__ -= uy[I_];
        I_ = idx(lclampx(ix-1), iy-1, iz);
        cc__ += amul(C3_, C3_mul, I_);
        d__ -= uy[I_];
        cc__ = cc__/4;
        dux[I] += -cc__*wx*wy*0.25*d__;

        I_ = idx(ix, iy-1, iz);
        cc__ = 0.5*(amul(C3_, C3_mul, I)+amul(C3_, C3_mul, I_));
        dux[I] += cc__*wy*wy*(ux[I_]-ux[I]);

        I_ = idx(ix, iy-1, iz);
        cc__ = 0.5*(amul(C1_, C1_mul, I)+amul(C1_, C1_mul, I_));
        duy[I] += cc__*wy*wy*(uy[I_]-uy[I]);

        I_ = idx(hclampx(ix+1), iy, iz);
        cc__ = amul(C2_, C2_mul, I_);
        d__ = ux[I_];
        I_ = idx(hclampx(ix+1), iy-1, iz);
        cc__ += amul(C2_, C2_mul, I_);
        d__ += ux[I_];
        I_ = idx(lclampx(ix-1), iy, iz);
        cc__ += amul(C2_, C2_mul, I_);
        d__ -= ux[I_];
        I_ = idx(lclampx(ix-1), iy-1, iz);
        cc__ += amul(C2_, C2_mul, I_);
        d__ -= ux[I_];
        cc__ = cc__/4;
        duy[I] += -cc__*wx*wy*0.25*d__;
        
    } else {
        //Bulk
        //Double derivative in x-direction
        d_ = make_float3(0.0,0.0,0.0);
        cc = make_float3(amul(C1_, C1_mul, I),amul(C3_, C3_mul, I),amul(C3_, C3_mul, I));
        //Right neighbor
        I_ = idx(hclampx(ix+1), iy, iz);
        u_ = make_float3(ux[I_], uy[I_], uz[I_]);
        cc_ = make_float3(amul(C1_, C1_mul, I_),amul(C3_, C3_mul, I_), amul(C3_, C3_mul, I_));
        cc_ = 0.5*(cc+cc_);
        d_ = wx*wx*had(cc_,(u_-u0));
        //Left neighbou
        I_ = idx(lclampx(ix-1), iy, iz);
        u_ = make_float3(ux[I_], uy[I_], uz[I_]);
        cc_ = make_float3(amul(C1_, C1_mul, I_),amul(C3_, C3_mul, I_), amul(C3_, C3_mul, I_));
        cc_ = 0.5*(cc+cc_);
        d_ += wx*wx*had(cc_,(u_-u0));

        dux[I] += d_.x ;
        duy[I] += d_.y ;
        duz[I] += 0 ;

        //Double derivative in y-direction
        d_ = make_float3(0.0,0.0,0.0);
        cc = make_float3(amul(C3_, C3_mul, I),amul(C1_, C1_mul, I),amul(C3_, C3_mul, I));
        //Right neighbor
        I_ = idx(ix, hclampy(iy+1), iz);
        u_ = make_float3(ux[I_], uy[I_], uz[I_]);
        cc_ = make_float3(amul(C3_, C3_mul, I_),amul(C1_, C1_mul, I_), amul(C3_, C3_mul, I_));
        cc_ = 0.5*(cc+cc_);
        d_ = wy*wy*had(cc_,(u_-u0));
        //Left neighbour
        I_ = idx(ix, lclampy(iy-1), iz);
        u_ = make_float3(ux[I_], uy[I_], uz[I_]);
        cc_ = make_float3(amul(C3_, C3_mul, I_), amul(C1_, C1_mul, I_),amul(C3_, C3_mul, I_));
        cc_ = 0.5*(cc+cc_);
        d_ += wy*wy*had(cc_,(u_-u0));

        dux[I] += d_.x ;
        duy[I] += d_.y ;
        duz[I] += 0 ;

        //dxy without boundaries
        d_ = make_float3(0.0,0.0,0.0);
        cc__=0;
        //(i+1,j+1)
        I_ = idx(hclampx(ix+1),hclampy(iy+1), iz);
        d_ += make_float3(ux[I_], uy[I_], uz[I_]);
        cc__ += amul(C2_, C2_mul, I_)+amul(C3_, C3_mul, I_);
        //(i-1,j-1)
        I_ = idx(lclampx(ix-1),lclampy(iy-1), iz);
        d_ += make_float3(ux[I_], uy[I_], uz[I_]);
        cc__ += amul(C2_, C2_mul, I_)+amul(C3_, C3_mul, I_);
        //(i+1,j-1)
        I_ = idx(hclampx(ix+1),lclampy(iy-1), iz);
        d_ -= make_float3(ux[I_], uy[I_], uz[I_]);
        cc__ += amul(C2_, C2_mul, I_)+amul(C3_, C3_mul, I_);
        //(i-1,j+1)
        I_ = idx(lclampx(ix-1),hclampy(iy+1), iz);
        d_ -= make_float3(ux[I_], uy[I_], uz[I_]);
        cc__ += amul(C2_, C2_mul, I_)+amul(C3_, C3_mul, I_);

        cc__ += 4*(amul(C2_, C2_mul, I)+amul(C3_, C3_mul, I));
        cc__ = cc__/8;
        d_ = cc__*d_*0.25*wx*wy;

        dux[I] += d_.y ;
        duy[I] += d_.x ;
        duz[I] += 0.0 ;
    }

    //Out of plane component has just Nueman boundary conditions
    //Double derivative in x-direction
    //Right neighbor
    I_ = idx(hclampx(ix+1), iy, iz);
    cc__ = 0.5*(amul(C3_, C3_mul, I_)+amul(C3_, C3_mul, I));
    d__ = wx*wx*cc__*(uz[I_] - uz[I]);
    duz[I] += d__;
    //Left neighbour
    I_ = idx(lclampx(ix-1), iy, iz);
    cc__ = 0.5*(amul(C3_, C3_mul, I_)+amul(C3_, C3_mul, I));
    d__ = wx*wx*cc__*(uz[I_] - uz[I]);
    duz[I] += d__;

    //Double derivative in y-direction
    //Right neighbor
    I_ = idx(ix, hclampy(iy+1), iz);
    cc__ = 0.5*(amul(C3_, C3_mul, I_)+amul(C3_, C3_mul, I));
    d__ = wy*wy*cc__*(uz[I_] - uz[I]);
    duz[I] += d__;
    //Left neighbour
    I_ = idx(ix, lclampy(iy-1), iz);
    cc__ = 0.5*(amul(C3_, C3_mul, I_)+amul(C3_, C3_mul, I));
    d__ = wx*wx*cc__*(uz[I_] - uz[I]);
    duz[I] += d__;
}