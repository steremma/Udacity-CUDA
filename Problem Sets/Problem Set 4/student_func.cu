//Udacity HW 4
//Radix Sorting

#include "utils.h"

#define THREADS_PER_BLOCK 512

/* Red Eye Removal
   ===============

   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

*/

/*
  Negate the specified bit of the input number.
*/
__device__ int isThisBitZero(const unsigned int number, const unsigned int bitIndex)
{
  return !(number & (1<<bitIndex));
}

/*
  Fill the (essentially boolean) output array with 1's if the corresponding input element 
  has a 0 at the specified bit, or 0 otherwise.
*/
__global__ void filter(unsigned int* d_input, unsigned int* d_output, unsigned int size, unsigned int bitIndex)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id > size) {
    return;
  }
  d_output[id] = isThisBitZero(d_input[id],bitIndex);
}

/*
  Swap two pointers (not pointed values). This function is used to swap between
  what my program considers an input or an output at every iteration.
*/
__host__ __device__ void swapPointers(unsigned int **a,unsigned int **b)
{
  unsigned int* temp = *a;
  *a = *b;
  *b = temp;
}

/*
  A fast prefix Sum step kernel using the Hillis - Steele algorithm. Shared memory
  cannot be easily exploited if size is not guaranteed to be small enough.
*/
__global__ void scanStep(unsigned int* input,unsigned int* output,const size_t size, int stride)
{

  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id >= size) {
    return;
  }

  if(id + stride < size) {
    output[id + stride] = input[id] + input[id + stride];
  }
  if(id < stride) {
    output[id] = input[id];
  }  
}

/*
  A kernel caller for my prefix sum function, responsible for managing
  device pointers. Care has been taken to transform the inclusive scan
  into exclusive with minimal memory access.
*/
__host__ void prefixSum(unsigned int* d_input,const size_t size)
{  
  // A mutable copy of the input is needed
  unsigned int* d_tempInput;
  checkCudaErrors(cudaMalloc((void**)&d_tempInput, sizeof(unsigned int) * size)); 
  checkCudaErrors(cudaMemcpy(d_tempInput, d_input, sizeof(unsigned int) * size, cudaMemcpyDeviceToDevice));

  unsigned int* d_tempOutput;
  checkCudaErrors(cudaMalloc((void**)&d_tempOutput, sizeof(unsigned int) * size)); 
  
  dim3 numBlocks(size/THREADS_PER_BLOCK + 1);
  for(int stride=1; stride < size; stride <<= 1) {
    scanStep<<<numBlocks,THREADS_PER_BLOCK>>>(d_tempInput,d_tempOutput,size, stride);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    swapPointers(&d_tempInput,&d_tempOutput);
  }
  swapPointers(&d_tempInput,&d_tempOutput);

  // Transform inclusive scan into exclusive.
  checkCudaErrors(cudaMemcpy(d_input + 1, d_tempOutput, sizeof(unsigned int) * (size - 1), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemset(d_input,0,sizeof(unsigned int)));

  // clean up
  checkCudaErrors(cudaFree(d_tempInput));
  checkCudaErrors(cudaFree(d_tempOutput)); 
}

/*
  Find the position of the 1st element having 1 in the respective bit.
  This is either equal, or 1 below the position of the last 0 number,
  depending on whether the last element had a 0 or an 1.
*/
int findOffset(unsigned int* d_isZero, unsigned int* d_cdf, size_t size)
{
  // an essentially boolean variable equal to 1 if last inspected element is zero, 0 otherwise.
  unsigned int lastElement;
  // The position where the last element is sorted. If that element was in the 0 bucket this should be incremented
  unsigned int lastPosition;
  checkCudaErrors(cudaMemcpy(&lastElement, d_isZero + size - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&lastPosition, d_cdf + size - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost));  
  return lastPosition + lastElement;
}

/*
  Computes the address where each input element should go at the current iteration.
*/
__global__ void computeOutputAddress(const unsigned int* const d_cdf, const unsigned int* const d_isZero,
                                     const size_t size, unsigned int* d_outputPos, int offset)
{
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  if(id > size) return;
  
  if(d_isZero[id]) d_outputPos[id] = d_cdf[id];
  else d_outputPos[id] = id - d_cdf[id] + offset;
}

/*
  moves the elements to their corrent address specified by the permutation vector.
*/
__global__ void scatter(unsigned int* d_output, const unsigned int* d_input,
                        unsigned int* d_outputPos, unsigned int* d_inputPos,
                        unsigned int* d_permutation, const size_t size)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id >= size) return;

  d_output[d_outputPos[id]] = d_input[id];
  d_permutation[d_outputPos[id]] = d_inputPos[id];
}

/*
  call the scatter kernel and manage its device pointers.
*/
void scatterCaller(const unsigned int* d_input, unsigned int* d_output,
                   unsigned int* d_inputPos, unsigned int* d_outputPos,
                   unsigned int* d_permutation, const unsigned int* const d_isZero,
                   const unsigned int* const d_cdf, const unsigned offset, const size_t size)
{
  dim3 numBlocks(size/THREADS_PER_BLOCK + 1);
  computeOutputAddress<<<numBlocks, THREADS_PER_BLOCK>>>(d_cdf, d_isZero, size, d_outputPos, offset);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  scatter<<<numBlocks, THREADS_PER_BLOCK>>>(d_output, d_input, d_outputPos, d_inputPos, d_permutation, size);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)

{
  /*
    Proposed algorithm: For each bit in the input (32 in the common case):
      1) filter input by putting 1 where the corresponding bit is 0.
      2) perform exclusive prefix sum on the filtered vector.
      3) use the scanned vector to compute the corrent position for each element.
      4) move its element to the corresponding position
      5) swap between input and output pointers and repeat for the next MSB.
  */

  // allocate device memory for my helper vectors, released before function ends.
  unsigned int* d_isZero;
  checkCudaErrors(cudaMalloc((void**)&d_isZero, numElems*sizeof(unsigned)));

  unsigned int* d_cdf;
  checkCudaErrors(cudaMalloc((void**)&d_cdf, numElems*sizeof(unsigned)));

  unsigned int* d_permutation;
  checkCudaErrors(cudaMalloc((void**)&d_permutation, numElems*sizeof(unsigned int)));
  checkCudaErrors(cudaMemcpy(d_permutation, d_inputPos, numElems*sizeof(unsigned int),
                             cudaMemcpyDeviceToDevice));


  // Need non-const pointers in order to swap them at every iteration.
  unsigned int* d_input = d_inputVals;
  unsigned int* d_inPos = d_inputPos;
  unsigned int* d_output = d_outputVals;
  unsigned int* d_outPos = d_outputPos;

  for(unsigned int bit = 0; bit < 8 * sizeof(unsigned int); bit++) {

    dim3 numBlocks(numElems/THREADS_PER_BLOCK + 1);
    filter<<<numBlocks,THREADS_PER_BLOCK>>>(d_input, d_isZero, numElems, bit);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(d_cdf, d_isZero, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

    prefixSum(d_cdf,numElems);

    unsigned int offset = findOffset(d_isZero, d_cdf, numElems);
    scatterCaller(d_input, d_output, d_inPos, d_outPos, d_permutation, d_isZero, d_cdf, offset, numElems);

    // swap pointers.
    swapPointers(&d_input, &d_output);
    swapPointers(&d_inPos, &d_permutation);
  }

  checkCudaErrors(cudaMemcpy(d_outputPos, d_inPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

  // clean up.
  checkCudaErrors(cudaFree(d_isZero));
  checkCudaErrors(cudaFree(d_cdf));
  checkCudaErrors(cudaFree(d_permutation));
  
}

