#ifndef __CUDACC__  
    #define __CUDACC__
#endif
#include "cuda_runtime.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include "kernel.h"

#define Nsample 1000
#define Nchan  128
#define Frame 100
#define NUM 128000
//#define NUM 89600
using namespace std;
__constant__ const float startDepth = 0.0;
__constant__ const float fs = 20.0;
__constant__ const float fc = 5.0;
__constant__ const float soundv = 1.54;
__constant__ const float lambda = 0.3;
__constant__ const float pitch =  0.3;
__constant__ const float dz =  0.077; //mm
__constant__ const float f_num = 3.0;

__global__ void delayAndSum(float *input, float *out, float* x_axis, float* z_axis, float* toff , float *apod){		
	float idx;
	int idx_floor;
	float delay;
	int index = blockIdx.y * NUM + blockIdx.x * blockDim.x + threadIdx.x; // 2d 
	float s = 0.0f;
	float point = 0.0f;
	float point_next = 0.0f;
	for(int i = 0; i < Nchan; i++){
		delay = (((sqrtf( z_axis[blockIdx.x * blockDim.x + threadIdx.x] * z_axis[blockIdx.x * blockDim.x + threadIdx.x] + (x_axis[i*blockDim.x+threadIdx.x] - x_axis[blockIdx.x * blockDim.x + threadIdx.x]) * (x_axis[i*blockDim.x+threadIdx.x] - x_axis[blockIdx.x * blockDim.x + threadIdx.x]))))+z_axis[blockIdx.x * blockDim.x + threadIdx.x]) / soundv;
		idx = ( (delay - (*toff)*1000000)*fs );
		if( (idx >= 0) && (idx < Nsample - 1) ){
			idx_floor = floorf((idx));  
			point = input[blockIdx.y * NUM + i * blockDim.x + idx_floor];
			point_next = input[blockIdx.y * NUM + i * blockDim.x + idx_floor +1];

			if ( fabs(x_axis[i*blockDim.x + threadIdx.x] - x_axis[blockIdx.x * blockDim.x + threadIdx.x]) <= z_axis[blockIdx.x * blockDim.x + threadIdx.x] / f_num ){		
				s += (  (float)(point_next - point) * (idx - (float)idx_floor) + point ) ;

			}
		}   
		out[index] = s;
	}

}

extern "C" int bf2(float* input, float* out, float *z_axis, float *x_axis, float *toff, float *apod){
	cudaError_t cudaStatus;
	float *input_d, *out_d, *z_axis_d, *x_axis_d, *toff_d, *apod_d;
	cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
		cudaDeviceReset();
		return 1;      
    }
    cudaStatus = cudaMalloc((void**)&input_d, NUM*Frame * sizeof(float));
    if (cudaStatus != cudaSuccess) {
		cudaDeviceReset();
        return 2;
    }
    cudaStatus = cudaMalloc((void**)&out_d, NUM*Frame * sizeof(float));
    if (cudaStatus != cudaSuccess) {
		cudaDeviceReset();
        return 3;
    }
	cudaStatus = cudaMalloc((void**)&z_axis_d, NUM * sizeof(float));
    if (cudaStatus != cudaSuccess) {
		cudaDeviceReset();
         return 4;
    }
	cudaStatus = cudaMalloc((void**)&x_axis_d, NUM* sizeof(float));
    if (cudaStatus != cudaSuccess) {
		cudaDeviceReset();
         return 5;
    }

	cudaStatus = cudaMalloc((void**)&toff_d, 1*sizeof(float));
    if (cudaStatus != cudaSuccess) {
		cudaDeviceReset();
         return 6;
    }
	cudaStatus = cudaMalloc((void**)&apod_d, Nchan*sizeof(float));
    if (cudaStatus != cudaSuccess) {
		cudaDeviceReset();
         return 7;
    }
	cudaMemset(out_d,0,NUM*Frame*sizeof(float));
	cudaMemset(z_axis_d,0,NUM*sizeof(float));
	cudaMemset(x_axis_d,0,NUM*sizeof(float));
	
	cudaStatus = cudaMemcpy(out_d, out, NUM * Frame * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
		cudaDeviceReset();
         return 11;
    }
	cudaStatus = cudaMemcpy(z_axis_d, z_axis, NUM  * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
		cudaDeviceReset();
         return 12;
    }
	cudaStatus = cudaMemcpy(x_axis_d, x_axis, NUM  * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
         return 13;
    }

	cudaStatus = cudaMemcpy(input_d, input, NUM * Frame * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
		cudaDeviceReset();
        return 14; 
    }
	cudaStatus = cudaMemcpy(toff_d, toff, sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
		cudaDeviceReset();
        return 15; 
    }
	cudaStatus = cudaMemcpy(apod_d, apod, Nchan*sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
		cudaDeviceReset();
        return 16; 
    }
	dim3 bb(Nchan,Frame);
	delayAndSum<<< bb, Nsample>>>(input_d,out_d,x_axis_d,z_axis_d,toff_d,apod_d);

	cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
		cudaDeviceReset();
        return 10; 
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
		cudaDeviceReset();
       return 11;  
    }

	cudaStatus = cudaMemcpy(out, out_d, NUM * Frame * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
		cudaDeviceReset();
        return 12; 
    }

	cudaFree(input_d);
	cudaFree(out_d);
	cudaFree(x_axis_d);
	cudaFree(z_axis_d);
	cudaFree(toff_d);
	cudaFree(apod_d);
	cudaDeviceReset();
	return 0;
}





