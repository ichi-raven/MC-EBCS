
//#include <pbrt/gpu/aggregate.h>
//#include <pbrt/gpu/optix.h>

//#include <cuda_runtime.h>
//
//#include <cstdio>
//
//__global__ void testKernel(int val) {
//    // printf("[%d, %d]:\t\tValue is:%d\n", blockIdx.y * gridDim.x + blockIdx.x,
//    //        threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
//    //            threadIdx.x,
//    //        val);
//
//    printf("gridDim = (%d, %d), blockIdx = (%d, %d), blockDim = (%d, %d), threadIdx = "
//           "(%d, %d, %d)\n",
//           gridDim.x, gridDim.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y,
//           threadIdx.x, threadIdx.y, threadIdx.z);
//}
//
//extern "C" void invokeTestKernel(int val) 
//{
//    dim3 dimGrid(10, 10);
//    dim3 dimBlock(2, 2, 2);
//    testKernel<<<dimGrid, dimBlock>>>(val);
//    cudaDeviceSynchronize();
//}