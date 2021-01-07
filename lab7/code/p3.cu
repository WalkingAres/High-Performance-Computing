#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/time.h> 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cudnn.h>
#include <iostream>

#define checkCUDNN(expression)                              \
{                                                           \
    cudnnStatus_t status = (expression);                    \
    if (status != CUDNN_STATUS_SUCCESS) {                   \
        std::cerr << "Error on line " << __LINE__ << ": "   \
            << cudnnGetErrorString(status) << std::endl;    \
        std::exit(EXIT_FAILURE);                            \
    }                                                       \
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
        cudaMalloc((void**)&d_B[i],sizeof(float)*outputSize[i]*outputSize[i]*depth);
    }

    cudaMemcpy(d_A,A,sizeof(float)*inputSize*inputSize*depth,cudaMemcpyHostToDevice);
    for(int i = 0; i < 3; i++) {
        cudaMemcpy(d_kernel[i],kernel[i],sizeof(float)*kernelSize*kernelSize*depth,cudaMemcpyHostToDevice);
    }
    
    // ========================== cuDNN 调用 ===================
    cudnnHandle_t cudnn[3];
    for( int i = 0; i < 3; i++ )
        checkCUDNN(cudnnCreate(&cudnn[i]));

    cudnnTensorDescriptor_t input_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(
                input_desc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,
                1,depth,inputSize,inputSize));
    
    cudnnFilterDescriptor_t filter_desc;
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    checkCUDNN(cudnnSetFilter4dDescriptor(
                filter_desc,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,
                1,depth,kernelSize,kernelSize));
    
    cudnnConvolutionDescriptor_t conv_desc[3];
    for( int i = 0; i < 3; i++) {
        checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc[i]));
        checkCUDNN(cudnnSetConvolution2dDescriptor(
                    conv_desc[i],
                    pad[i],pad[i],stride[i],stride[i],1,1,
                    CUDNN_CONVOLUTION,CUDNN_DATA_FLOAT));
    }

    cudnnTensorDescriptor_t output_desc[3];
    for( int i = 0; i < 3; i++ ) {
        checkCUDNN(cudnnCreateTensorDescriptor(&output_desc[i]));
        checkCUDNN(cudnnSetTensor4dDescriptor(
                    output_desc[i],CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,
                    1,1,outputSize[i],outputSize[i]));
    }
    
    cudnnConvolutionFwdAlgo_t algo[3];
    size_t ws_size[3];
    float *ws_data[3];
    for( int i = 0; i < 3; i++ ) {
       cudnnGetConvolutionForwardAlgorithm(
                    cudnn[i],
                    input_desc,filter_desc,conv_desc[i],output_desc[i],
                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo[i]);
    }
    
    for(int i = 0; i < 3; i++ )
    {
            checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
                    cudnn[i],
                    input_desc,filter_desc,conv_desc[i],output_desc[i],
                    algo[i],&ws_size[i]));
        cudaMalloc((void**)&ws_data[i],ws_size[i]);
    }
    
    //printf("1: %d \n 2: %d \n 3: %d \n", ws_size[0],ws_size[1],ws_size[2]);


    struct timeval start, end;
    gettimeofday( &start, NULL );

    float alpha = 1.0;
    float beta = 0.0;
    
    for(int i = 0; i < 3; i++ ) {
        checkCUDNN(cudnnConvolutionForward(
                    cudnn[i],
                    &alpha,
                    input_desc,d_A,
                    filter_desc,d_kernel[i],
                    conv_desc[i],algo[i],ws_data[i],ws_size[i],
                    &beta,
                    output_desc[i],d_B[i]));
    }

    for( int i = 0; i < 3; i++ ) {
        cudaMemcpy(B[i],d_B[i],sizeof(float)*outputSize[i]*outputSize[i]*depth,cudaMemcpyDeviceToHost);
    }
    
    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    //printf("Block(%d,%d)   Grid(%d,%d).\n", Block.x, Block.y, Grid.x, Grid.y);
    printf("total time is %f ms\n", timeuse/(float)1000);

    FILE *b[3];
    b[0] = fopen("matrixB31.m", "wb");
    b[1] = fopen("matrixB32.m", "wb");
    b[2] = fopen("matrixB33.m", "wb");


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
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyFilterDescriptor(filter_desc);

    for(int i = 0; i < 3; i++) {

        free(kernel[i]);
        free(B[i]);

        cudaFree(d_B[i]);
        cudaFree(d_kernel[i]);
        cudaFree(ws_data[i]);

        cudnnDestroyTensorDescriptor(output_desc[i]);
        cudnnDestroyConvolutionDescriptor(conv_desc[i]);

    
        fclose(b[i]);
    }

    return 0;
}