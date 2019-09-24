#include <stdint.h>
#include <stdio.h>
#include "amul.h"
#include "float3.h"

// Add magneto-elastic coupling field to B.
// H = - δUmel / δM, 
// where Umel is magneto-elastic energy denstiy given by the eq. (12.18) of Gurevich&Melkov "Magnetization Oscillations and Waves", CRC Press, 1996
extern "C" __global__ void
addmagnetoelasticfield(float* __restrict__  Bx, float* __restrict__  By, float* __restrict__  Bz,
                      float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
					  float* __restrict__ exx, float* __restrict__ eyy, float* __restrict__ ezz,
                 	  float* __restrict__ exy, float* __restrict__ eyz, float* __restrict__ exz,	
					  float* __restrict__ B1_, float B1_mul, 
					  float* __restrict__ B2_, float B2_mul,
					  float* __restrict__ Ms_, float Ms_mul,
                      int N) {

	int I =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

	if (I < N) {

	    float Exx = exx[I];
	    float Eyy = eyy[I];
	    float Ezz = ezz[I];
	    
	    float Exy = exy[I];
	    float Eyx = Exy;

	    float Exz = exz[I];
	    float Ezx = Exz;

	    float Eyz = eyz[I];
	    float Ezy = Eyz;

		float invMs = inv_Msat(Ms_, Ms_mul, I);

		float B1 = amul(B1_, B1_mul, I) * invMs;
	    float B2 = amul(B2_, B2_mul, I) * invMs;

	    float3 m = {mx[I], my[I], mz[I]};

	    Bx[I] += -(2.0f*B1*m.x*Exx + B2*(m.y*Exy + m.z*Exz));
	    By[I] += -(2.0f*B1*m.y*Eyy + B2*(m.x*Eyx + m.z*Eyz));
	    Bz[I] += -(2.0f*B1*m.z*Ezz + B2*(m.x*Ezx + m.y*Ezy));
	}
}


