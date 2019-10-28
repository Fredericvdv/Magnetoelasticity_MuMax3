// dst[i] = scale*src[i] + offset
extern "C" __global__ void
scale(float* __restrict__  x, float scale, float offset, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {
        x[i] = scale*x[i] + offset;
    }
}
