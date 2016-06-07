/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include <stdio.h>
#include <math.h>
#define THREADS_PER_BLOCK 1024

__global__ void slowHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals, int numBins)
{
  // Shared memory allocation for the local histogram
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  if(id >= numVals) return;
  
  atomicAdd(&histo[vals[id]], 1);
}

__global__ void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals, int numBins)
{
  // Shared memory allocation for the local histogram
  extern __shared__ unsigned int sh[];
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(id >= numVals) return;

  if(threadIdx.x < numBins) sh[threadIdx.x] = 0;
  __syncthreads();

  atomicAdd(&sh[vals[id]], 1);
  __syncthreads();

  // atomically add the local histogram to the global one
  if(threadIdx.x < numBins) {
    atomicAdd(&histo[threadIdx.x], sh[threadIdx.x]);
  }
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //TODO Launch the yourHisto kernel
  //if you want to use/launch more than one kernel,
  //feel free
  dim3 threads(THREADS_PER_BLOCK);
  dim3 blocks(ceil((float)numElems / THREADS_PER_BLOCK));
  size_t memSize = numBins * sizeof(unsigned int);
  if(numBins <= THREADS_PER_BLOCK) {
    yourHisto<<<blocks, threads, memSize>>>(d_vals, d_histo, numElems, numBins);
  }
  else printf("we got a problem... \n");

  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
