#include <iostream>

#include <stdio.h>

#include <math.h>

#include <cuda.h>

#include <cuda_runtime.h>

#include "device_launch_parameters.h"







// Problem description: compute threads with catalan numbers

//

// Divide up the work as evenly as possible









/* Write device code / function here pls */

/* CUDA kernel function goes here */

__global__

void catalan(unsigned int catNums, double* data)

{

    unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);



    int j, k;

    double result = 1.0;



    k = i + 1;



    // thread 0: i = 0 -> calculate first cat #

    // thread 1: i = 1 -> calculate second cat #

    // thread 2: i = 2 -> calculate the next one...

    // thread 1000: i = 999



    // all kernel threads (GPU) will enter and execute this kernel function

    // i <- unique to each thread



    //printf("thread number: %d\n", i);



    if (i <= catNums)

    {

        for (j = 0; j < k; j++)

        {

            // compute the binomial coefficient

            result *= ((2 * k) - j);

            result /= (j + 1);

        }



        // resulting catalan number

        result = result / (k + 1);



        // store resulting catalan number into array

        data[i] = result;



    }

    // result = (2*(2*i-1)*c)/(i+1);

}





__global__

void printThreads(int N)

{

    int i = (blockIdx.x * blockDim.x + threadIdx.x);



    if (i < N)

    {

        printf("Thread number: %d\n", i);

    }

}





/* Function to find and display information on installed GPU devices */

void printDeviceInfo()

{

    struct cudaDeviceProp dp;

    int gpuCount;

    int i;



    cudaGetDeviceCount(&gpuCount);

    printf("%d GPU(s) found.\n", gpuCount);

    for (i = 0; i < gpuCount; i++)

    {

        cudaError_t err = cudaGetDeviceProperties(&dp, i);

        if (err == cudaSuccess)

        {

            printf("GPU #%d [Compute Capability %d.%d] (%lg GB of Global Memory): %s\n", i, dp.major, dp.minor, dp.totalGlobalMem / 1073741824.0, dp.name);

            //printf("GPU #%d connected to PCI Bus #%d as Device #%d\n", i, dp.pciBusID, dp.pciDeviceID);

        }

    }

}





/* Host entry point */

int main(int argc, char** argv)

{

    unsigned int i, threadNums, catalanNums, blocks = 0;



    double* catData, * dev_catData;

    FILE* fp;



    printDeviceInfo();



    /*if (argc != 2)

    {

        printf("Usage: catalan catalanNums\n");

        return -1;

    }*/



    catalanNums = 10000;

    threadNums = catalanNums;



    // allocate memory on the host

    catData = (double*)malloc(sizeof(double) * catalanNums);



    // allocate memory on the device

    cudaMalloc(&dev_catData, sizeof(double) * catalanNums);



    // copy host memory to device

    cudaMemcpy(dev_catData, catData, catalanNums * sizeof(double), cudaMemcpyHostToDevice);



    // total number of 1024 thread blocks

    if (threadNums % 1024 != 0)

    {

        blocks = (threadNums / 1024) + 1;

    }

    else

    {

        blocks = threadNums / 1024;

    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // launch the threads onto the GPU

    catalan<<<blocks,1024>>>(catalanNums, dev_catData);

    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%f ms\n", milliseconds);

    printf("Exited kernel\n");



    cudaMemcpy(catData, dev_catData, catalanNums * sizeof(double), cudaMemcpyDeviceToHost);



    if ((fp = fopen("catalan.dat", "a")) == NULL)

    {

        printf("Failed to open file: catalan.dat\n");

    }

    else

    {

        for (i = 0; i < catalanNums; i++)

        {

            fprintf(fp, "%.0lf\n", catData[i]);

        }

    }



    fclose(fp);



    cudaFree(dev_catData);



    free(catData);



    return 0;


}


