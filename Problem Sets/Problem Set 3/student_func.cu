/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


#include "reference_calc.cpp"
#include "utils.h"

#define THREADS_PER_BLOCK 512

/*
  In order to generalize my reduction i use function pointers.
  This will now work with any such function as long as it is a
  associative and binary, such as min,max or sum.
*/
typedef float (*ReductionOperator) (float, float);

__device__ float minOperator(float x,float y)
{
  return min(x,y);
}

__device__ float maxOperator(float x,float y)
{
  return max(x,y);
}

__device__ ReductionOperator pMinOperator = minOperator;
__device__ ReductionOperator pMaxOperator = maxOperator;

/*
  A general reduction kernel using shared memory to increase performance.
*/
__global__ void reduce(float* input,int inputSize,ReductionOperator op) 
{
  // initialize variables
  extern __shared__ float sData[];
  int tId = threadIdx.x;
  int id = blockIdx.x * blockDim.x + tId;
  if(id >= inputSize) {
    return;
  }

  // load data into shared memory
  sData[tId] = input[id];
  __syncthreads();
  
  for(unsigned int d=blockDim.x/2; d>=1; d/=2) {
    if(tId < d) {
      //sData[tId] = min(sData[tId], sData[tId + d]);
      sData[tId] = (*op)(sData[tId], sData[tId+d]);
    }
    __syncthreads();
  }

  // after the final iteration sData[0] holds the final result
  if(tId == 0) {
    input[blockIdx.x] = sData[0];
  }
}

/*
  A case specific caller for the reduction kernel finding either a max
  or min value depending on the boolean argument.
*/
float extremeVal(const float* const d_logLuminance,int inputSize,bool minimum)
{
  ReductionOperator op;
  if(minimum) {
    cudaMemcpyFromSymbol(&op, pMinOperator, sizeof(ReductionOperator) );
  }
  else {
    cudaMemcpyFromSymbol(&op, pMaxOperator, sizeof(ReductionOperator) );
  }

  // A temp table in global memory is needed since my input is const
  float* d_temp;
  checkCudaErrors(cudaMalloc((void**)&d_temp, sizeof(float) * inputSize)); 
  checkCudaErrors(cudaMemcpy(d_temp, d_logLuminance, sizeof(float) * inputSize, cudaMemcpyDeviceToDevice));
  
  while(inputSize > THREADS_PER_BLOCK) {
    int numBlocks = ceil((float)inputSize/THREADS_PER_BLOCK);
    int memSize = sizeof(float)*THREADS_PER_BLOCK;
    reduce<<<numBlocks,THREADS_PER_BLOCK,memSize>>>(d_temp,inputSize,op);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    inputSize = numBlocks;
  } 
  
  
  // launch 1 block to wrap up the calculation
  int numBlocks = 1;
  int memSize = sizeof(float)*inputSize;
  reduce<<<numBlocks,inputSize,memSize>>>(d_temp,inputSize,op);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // clean up
  float extremeValue;
  checkCudaErrors(cudaMemcpy(&extremeValue,d_temp,sizeof(float),cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_temp));
  return extremeValue;
}
/*
  A fast histogram calculation kernel using shared memory. Each blocks computes its own
  local copy of the histogram and then atomically contributes it to the global one.
*/
__global__ void histogram(const float* const input,unsigned int* const histogram,const unsigned int inputSize,
                     const unsigned int numBins,const float lumMin,const float lumMax)
{
  // initialize variables
  extern __shared__ int localBins[];
  int tId = threadIdx.x;
  int id = blockIdx.x * blockDim.x + tId;
  if(id >= inputSize) {
    return;
  }
  if(tId < numBins) {
    localBins[tId] = 0;
  }
  __syncthreads();

  float range = lumMax - lumMin;

  int binIdx = int(numBins * (input[id] - lumMin) / range);
  binIdx = min(numBins - 1, binIdx);

  atomicAdd(&localBins[binIdx],1);
  __syncthreads();
  if(tId < numBins) {
    atomicAdd(&histogram[tId],localBins[tId]);
  }
}

/*
  A slow implementation of the histogram kernel used to verify the faster one. 
  Every thread directly (and atomically) writes to the global memory, thus
  many collisions should be expected.
*/
__global__ void slowHistogram(const float* const input,unsigned int* const histogram,const unsigned int inputSize,
                     const unsigned int numBins,const float lumMin,const float lumMax)
{
  float range = lumMax - lumMin;
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id > inputSize) {
    return;
  }
  int binIdx = int(numBins * (input[id] - lumMin) / range);

  binIdx = min(numBins - 1, binIdx);
  atomicAdd(&(histogram[binIdx]), 1);
}

/*
  A caller of the histogram kernel, responsible for managing device memory pointers.
*/
void createHistogram(const float* const d_logLuminance,unsigned int* const d_cdf,
                     const int inputSize,const int numBins,float minVal,float maxVal)
{
  cudaMemset(d_cdf, 0, sizeof(unsigned int) * numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  int numBlocks = ceil((float)inputSize/THREADS_PER_BLOCK);
  //int memSize = sizeof(unsigned int) * numBins;
  //histogram<<<numBlocks,THREADS_PER_BLOCK,memSize>>>(d_logLuminance,d_cdf,inputSize,numBins,minVal,maxVal);
  slowHistogram<<<numBlocks,THREADS_PER_BLOCK>>>(d_logLuminance,d_cdf,inputSize,numBins,minVal,maxVal);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

/*
  Swap two (int) pointers, this could be a template. Used to 
  swap between what my program considers an input or an output
  without copying any elements back and forth.
*/
__device__ void swapPointers(unsigned int **a,unsigned int **b)
{
  unsigned int* temp = *a;
  *a = *b;
  *b = temp;
}

/*
  A fast prefix Sum kernel using the Hillis - Steele algorithm. Uses 
  two shared memory chunks which play the role of the input or the output
  in turns at every iteration. Achieving more than 2x speed improvement
  against an implementation that does not exploit shared memory.
  NOTE: This will not work if the number of bins exceeds 1024.
*/
__global__ void prefixSum(unsigned int* input,unsigned int* output,int numBins)
{
  extern __shared__ unsigned int sharedMemory[];
  unsigned int* sInput = sharedMemory;
  unsigned int* sOutput = sharedMemory + numBins;
  
  // assuming that only 1 block exists.
  int id = threadIdx.x;
  if(id >= numBins) {
    return;
  }
  sInput[id] = input[id];
  __syncthreads();

  for(int stride=1;stride < numBins;stride *= 2) {
    if(id + stride < numBins) {
      sOutput[id + stride] = sInput[id] + sInput[id + stride];
    }
    if(id < stride) {
      sOutput[id] = sInput[id];
    }  
    __syncthreads();
    swapPointers(&sInput,&sOutput);
  }
  
  output[id] = sInput[id];
  
}
/*
  A kernel caller for my prefix sum function, responsible for managing
  device pointers. Care has been taken to transform the inclusive scan
  into exclusive with minimal memory access.
*/
void prefixSum(unsigned int* const d_cdf,const size_t numBins)
{  
  // A mutable copy of the input is needed
  unsigned int* d_input;
  checkCudaErrors(cudaMalloc((void**)&d_input, sizeof(unsigned int) * numBins)); 
  checkCudaErrors(cudaMemcpy(d_input, d_cdf, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToDevice));

  unsigned int* d_output;
  checkCudaErrors(cudaMalloc((void**)&d_output, sizeof(unsigned int) * numBins)); 

  int memSize = 2 * numBins * sizeof(unsigned int);
  if(numBins > 1024) {
    return;
  }
  prefixSum<<<1,numBins,memSize>>>(d_input,d_output,numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Transform inclusive scan into exclusive.
  checkCudaErrors(cudaMemcpy(d_cdf + 1, d_output, sizeof(unsigned int) * (numBins - 1), cudaMemcpyDeviceToDevice));

  // clean up
  checkCudaErrors(cudaFree(d_input));
  checkCudaErrors(cudaFree(d_output));
  checkCudaErrors(cudaMemset(d_cdf,0,sizeof(unsigned int)));
}

/*
  Main function called to perform histogram equilization on an image.
*/
void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{ 
  const int inputSize = numRows * numCols;

  /*  1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum  */
  min_logLum = extremeVal(d_logLuminance,inputSize,1);
  max_logLum = extremeVal(d_logLuminance,inputSize,0); 

  /*  3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins */
  createHistogram(d_logLuminance,d_cdf,inputSize,numBins,min_logLum,max_logLum);

  /* 4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)    */
  prefixSum(d_cdf,numBins);
}
