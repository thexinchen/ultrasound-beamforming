#include "mex.h"
#include "math.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define Nsample  1000
#define Nchan  128
#define Frame 5000
#define NUM 128000
#define IN  prhs[0]
#define IN2 prhs[1]
#define OUT plhs[0]


float*z_axis = (float*)malloc(sizeof(float)*Nsample);
float*x_axis = (float*)malloc(sizeof(float)*Nchan);

const float fs = 20;
const float fc = 5;
const float soundv = 1.54;
const float lambda = soundv/fc;
const float pitch = 0.3;
const float dz = 0.25 * lambda; //mm
const float f_num = 3;




void bf2(float*input, float *toff, float*out){

    int idx_floor = 0;
    float idx = 0;
    float delay = 0;
    float Rmin =  (*toff)*1000000 * soundv / 2; //distance  mm 
    

    for (int i = 0 ; i < Nsample; i++){
        z_axis[i] = ( ((float)i+1 )/fs*soundv/2 ) + Rmin;
    }
 
    for (int i = 0; i < Nchan; i++){
        x_axis[i] = ((-1)*(float)(Nchan-1)/2 + (float)i )*pitch;
    }    

    for(int frame = 0; frame < Frame; frame++){

        for(int I = 0 ; I < Nchan; I++){
            float point = 0;
            float point_next = 0;
            #pragma omp parallel for
            for(int i1 = 0 ; i1 < Nchan; i1++){
                #pragma omp parallel for
                for(int j1 = 0; j1 < Nsample; j1++){
                        // time us  
                    delay = ( sqrt(z_axis[j1]*z_axis[j1] + (x_axis[I] - x_axis[i1])*(x_axis[I] - x_axis[i1]) ) + z_axis[j1]) / soundv; // us 
                    idx = ( (delay - (*toff)*1000000)*fs );

                    if( (idx >= 0) && (idx < Nsample-1) ){
                        idx_floor = (idx);
                        point = input[frame*Nchan*Nsample + (i1*Nsample+idx_floor)];
                        point_next = input[frame*Nchan*Nsample + (i1*Nsample+idx_floor+1)];

						if( fabs(x_axis[I] - x_axis[i1]) <= (z_axis[j1])/f_num ){

							out[frame*Nchan*Nsample + I*Nsample + j1] += (point_next - point) * (idx - idx_floor) + point;
						}else{
							out[frame*Nchan*Nsample + I*Nsample + j1] += 0;
						}
                        
                    }
                }
                    
            }


        }
        
       
        
    }
    free(z_axis);
    free(x_axis);;
}



void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[] )
{ 
    size_t m1,n1;
    m1 = mxGetM(IN); 
    n1 = mxGetN(IN);
    OUT = mxCreateNumericMatrix( (mwSize)m1, (mwSize)n1, mxSINGLE_CLASS, mxREAL);
    float* out1 =  (float*)mxGetData(OUT);
    
    float* in1 = (float*)mxGetData(IN);
    float* toff = (float*)mxGetData(IN2);
    
    mexPrintf("toff = %f\n",*toff);

    bf2(in1,toff,out1);

}
    