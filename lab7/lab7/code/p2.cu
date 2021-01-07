#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/time.h> 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__
void im2col(float *A, int inputSize, int depth, int kernelSize, int stride, int pad, float *col, int outputSize) {
    
    // 一个线程完成一次卷积操作中的转换  也就是说 一个线程转换生成col中的一个行向量
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if( !(i < outputSize) || !(j < outputSize) ) return;

    int Ai = i * stride;
    int Aj = j * stride;

    for( int d = 0; d < depth; d++ ) {
        for(int k = 0; k < kernelSize; k++ ) {
            for( int l = 0; l < kernelSize; l++) {
                if( Ai + k - pad < 0 || !(Ai + k - pad < inputSize) || Aj + l - pad < 0 || !( Aj + l - pad < inputSize)) {
                    //col[ d*outputSize*outputSize*kernelSize*kernelSize + (i*outputSize + j)*kernelSize*kernelSize + k*kernelSize + l] = 0;
                    col[ (i*outputSize + j)*(kernelSize*kernelSize*depth)+ d*kernelSize*kernelSize + k*kernelSize + l] = 0;
                }
                else  col[ (i*outputSize + j)*(kernelSize*kernelSize*depth)+ d*kernelSize*kernelSize + k*kernelSize + l] \
                        = A[d*inputSize*inputSize + (Ai + k - pad)*inputSize + Aj + l - pad ];
            }
        }
    }

}

// 计算 C = A*v A size m*n v size n*1
__global__
void gemm(float *A, float *B, float *C, int m, int n) {

    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if( !( i < m ) ) return;

    float sum = 0;
    for( int l = 0; l < n; l++ ) {
        sum += A[i*n + l] * B[l];
    }

    C[i] = sum;
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

    float *d_A, *d_kernel[3], *d_B[3], *d_col[3];

    cudaMalloc((void**)&d_A,sizeof(float)*inputSize*inputSize*depth);
    for(int i = 0; i < 3; i++) {
        cudaMalloc((void**)&d_kernel[i], sizeof(float)*kernelSize*kernelSize*depth);
        cudaMalloc((void**)&d_B[i],sizeof(float)*outputSize[i]*outputSize[i]*depth);
        cudaMalloc((void**)&d_col[i], sizeof(float)*outputSize[i]*outputSize[i]*kernelSize*kernelSize*depth);
    }

    cudaMemcpy(d_A,A,sizeof(float)*inputSize*inputSize*depth,cudaMemcpyHostToDevice);
    for(int i = 0; i < 3; i++) {
        cudaMemcpy(d_kernel[i],kernel[i],sizeof(float)*kernelSize*kernelSize*depth,cudaMemcpyHostToDevice);
    }

    // ============================= 调用核函数 =========================

    // ===== 调用im2col

    for( int i = 0; i < 3; i++ ) {
        int blockx = (int) (log2(outputSize[i])+ 1);
        int blocky = blockx;

        dim3 Block(blockx,blocky);
        dim3 Grid((inputSize+Block.x-1) / Block.x,(inputSize+Block.y-1) / Block.y ); 

        im2col <<< Grid, Block >>> (d_A,inputSize,depth,kernelSize,stride[i],pad[i],d_col[i],outputSize[i]);

    }

    cudaDeviceSynchronize();

    // ==== 调用gemm

    struct timeval start, end;
    gettimeofday( &start, NULL );

    for( int i = 0; i < 3; i++ ) {
        int blockx = (int) (log2(outputSize[i]*outputSize[i])+ 1);
        dim3 Block(blockx);
        dim3 Grid((outputSize[i]*outputSize[i]+Block.x-1) / Block.x); 
        gemm <<< Grid, Block >>> (d_col[i],d_kernel[i],d_B[i],outputSize[i]*outputSize[i],kernelSize*kernelSize*depth);
    }
    
    // 结果回传
    for( int i = 0; i < 3; i++ ) {
        cudaMemcpy(B[i],d_B[i],sizeof(float)*outputSize[i]*outputSize[i]*depth,cudaMemcpyDeviceToHost);
    }
    
    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    
    printf("total time is %f ms\n", timeuse/(float)1000);

    // 输出结果
    FILE *b[3];
    b[0] = fopen("matrixB21.m", "wb");
    b[1] = fopen("matrixB22.m", "wb");
    b[2] = fopen("matrixB23.m", "wb");


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
        cudaFree(d_col[i]);

        fclose(b[i]);
    }

    return 0;
}