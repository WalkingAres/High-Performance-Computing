#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/time.h> 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


__global__
void conv2(float *A, float *kernel,int inputSize, int depth, int kernelSize , int stride, int pad, float *B, int outputSize) {

    // 计算元素output(i,j)的值 一次卷积运算
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if( !(i < outputSize) || !(j < outputSize) ) return;
    
    int Ai = i*stride;
    int Aj = j*stride;

    // 除去填充的0
    int startk = (pad-Ai) < 0? 0 : pad-Ai;
    int endk = kernelSize < (inputSize + pad - Ai) ? kernelSize : (inputSize + pad - Ai);
    int startl = (pad-Aj) < 0? 0 : pad-Aj;
    int endl = kernelSize < (inputSize + pad - Aj) ? kernelSize : (inputSize + pad - Aj);
    float sum = 0;

    for(int d = 0; d < depth; d++) {
        for( int k = startk ; k < endk; k++) {
            for( int l = startl; l < endl; l++) {
                sum += A[d*inputSize*inputSize + (Ai+k-pad)*inputSize + Aj+l-pad]*kernel[d*kernelSize*kernelSize + k*kernelSize+l];
            }
        }
        B[d*outputSize*outputSize + i*outputSize + j] = sum;
    }
    B[i*outputSize + j] = sum;
}


int main(int argc, char * argv[] ) {

    // input: inputSize*inputSize*depth
    // kernel: kernelSize*kernelSize*depth
    // output: outputSize*outputSize

    int inputSize = 7;
    int depth = 3;
    int kernelSize = 3;
    int kernelNum = 3;
    int stride[3] = {1 , 2 , 3 };
    int pad[3] = {0,0,0};
    int outputSize[3];

    // 计算不同stride下需要的padding数量pad和output的规模outputSize
    
    for(int i = 0; i < kernelNum; i++) {
        if((inputSize - kernelSize)%stride[i] != 0) {
            pad[i] = (stride[i] - ((inputSize - kernelSize)%stride[i])) / 2;
        }
        outputSize[i] = (inputSize - kernelSize + 2*pad[i] ) / stride[i] + 1;
    }

    // ============================= 资源申请的初始化 =========================

    // ==== CPU资源申请和初始化
    // input:A kernel:kernel output:B
    
    float *A, *kernel[3], *B[3];
    A = (float *)malloc(sizeof(float)*inputSize*inputSize*depth);
    for(int i = 0; i < 3; i++) {
        kernel[i] = (float *)malloc(sizeof(float)*kernelSize*kernelSize*depth);
        B[i] = (float *)malloc(sizeof(float)*outputSize[i]*outputSize[i]*depth);
    }

   // 初始化input A
    for(int d = 0; d < depth; d++) {
        for(int i=0; i<inputSize*inputSize; i++) {
            A[d*inputSize*inputSize + i] = i;
        }
    }

    // 初始化kernel
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < kernelSize*kernelSize*depth; j++) {
            kernel[i][j] = 1;
        }
    }

    // ==== GPU资源申请和初始化

    float *d_A, *d_kernel[3], *d_B[3];

    cudaMalloc((void**)&d_A,sizeof(float)*inputSize*inputSize*depth);

    for(int i = 0; i < 3; i++) {
        cudaMalloc((void**)&d_kernel[i], sizeof(float)*kernelSize*kernelSize*depth);
        cudaMalloc((void**)&d_B[i],sizeof(float)*outputSize[i]*outputSize[i]);
    }

    cudaMemcpy(d_A,A,sizeof(float)*inputSize*inputSize*depth,cudaMemcpyHostToDevice);
    for(int i = 0; i < 3; i++) {
        cudaMemcpy(d_kernel[i],kernel[i],sizeof(float)*kernelSize*kernelSize*depth,cudaMemcpyHostToDevice);
    }

    // ============================= 调用核函数 =========================
    struct timeval start, end;
    gettimeofday( &start, NULL );

    for( int i = 0; i < 3; i++ ) {
        int blockx = (int) (log2(outputSize[i])+ 1);
        int blocky = blockx;

        dim3 Block(blockx,blocky);
        dim3 Grid((inputSize+Block.x-1) / Block.x,(inputSize+Block.y-1) / Block.y ); 

        conv2 <<< Grid, Block >>> (d_A,d_kernel[i],inputSize,depth,kernelSize,stride[i],pad[i],d_B[i],outputSize[i]);

    }
    
    // 结果回传
    for( int i = 0; i < 3; i++ ) {
        cudaMemcpy(B[i],d_B[i],sizeof(float)*outputSize[i]*outputSize[i],cudaMemcpyDeviceToHost);
    }
    
    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    //printf("Block(%d,%d)   Grid(%d,%d).\n", Block.x, Block.y, Grid.x, Grid.y);
    printf("total time is %f ms\n", timeuse/(float)1000);

    // 输出结果
    FILE *b[3];
    b[0] = fopen("matrixB11.m", "wb");
    b[1] = fopen("matrixB12.m", "wb");
    b[2] = fopen("matrixB13.m", "wb");


    for(int k = 0; k < 3; k++ ) {
        fprintf(b[k], "B = [ \n");
        for (int i = 0; i < outputSize[k]; i++)
        {
            for (int j = 0; j < outputSize[k]; j++)
                fprintf(b[k], "%f ", B[k][i * outputSize[k] + j]);
            fprintf(b[k], "\n");
        }
        fprintf(b[k], "];");
    }

    // ============================= 资源释放 =========================

    free(A);
    cudaFree(d_A);

    for(int i = 0; i < 3; i++) {

        free(kernel[i]);
        free(B[i]);

        cudaFree(d_B[i]);
        cudaFree(d_kernel[i]);
        
        fclose(b[i]);
    }

    return 0;
}