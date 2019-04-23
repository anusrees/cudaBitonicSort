#include <cuda_runtime.h>
#include <bits/stdc++.h>

#define BLOCKS 32768*2
#define THREADS 256
#define SIZE BLOCKS*THREADS

using namespace std;

__host__ void printArr(int *arr, int size);
__host__ void randomArrGenerator(int *arr, int size);
__host__ void checkSorted(int *arr, int size);

__device__ void swapCu(int &a, int &b)
{
	int temp = a;
	a = b;
	b = temp;
}

//bitonic sort on GPU
__global__ void bitonicSortCu(int *arr, int i, int j, int size)
{
	int k = threadIdx.x + blockIdx.x*blockDim.x;

	if(k<size && k%(j<<1) < j)
	{
		bool descending = (k/i)%2;

		if(descending && arr[k] < arr[k+j])
			swapCu(arr[k], arr[k+j]);
		else if(!descending && arr[k] > arr[k+j])
			swapCu(arr[k], arr[k+j]);
	}
}

void bitonicSortParallel(int *arr, int size)
{
	for(int i=2; i<=size; i*=2)
		for(int j=i/2; j>=1; j/=2)
			bitonicSortCu<<<BLOCKS, THREADS>>>(arr, i, j, size);
}

int main(int argc, char const *argv[])
{
	int *d_arr;
	int *arr = new int[SIZE];

	randomArrGenerator(arr, SIZE);

	cudaMalloc(&d_arr, sizeof(int)*SIZE);
	cudaDeviceSynchronize();

//start timer here
	cudaMemcpyAsync(d_arr, arr, sizeof(int)*SIZE, cudaMemcpyHostToDevice);
	bitonicSortParallel(d_arr, SIZE);
	cudaMemcpyAsync(arr, d_arr, sizeof(int)*SIZE, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
//end timer here

	checkSorted(arr, SIZE);

	return 0;
}

//Auxilliary CPU functions
__host__ void checkSorted(int *arr, int size)
{
	for(int i=1; i<size; i++)
		if(arr[i] < arr[i-1])
		{
			cout << "sorting unsuccessful\n";
			return;
		}

	cout << "sorting successful\n";
}

__host__ void randomArrGenerator(int *arr, int size)
{
	for(int i=0; i<size; i++)
		arr[i] = rand()%1000;
}

__host__ void printArr(int *arr, int size)
{
	for(int i=0; i<size; i++)
		cout << arr[i] << " ";
	cout << endl;
}

__host__ void swap(int &a, int &b)
{
	int temp = a;
	a = b;
	b = temp;
}

//bitonic sort on CPU
__host__ void bitonicSort(int *arr, int size)
{
	if(size > 1)
	{
		for(int i=2; i<=size; i*=2)
		{
			for(int j=i/2; j>=1; j/=2)
			{
				for(int k=0; k<size; k++)
				{
					if(k%(j<<1) < j)
					{
						bool descending = (k/i)%2;

						if(descending && arr[k] < arr[k+j])
							swap(arr[k], arr[k+j]);
						else if(!descending && arr[k] > arr[k+j])
							swap(arr[k], arr[k+j]);
					}
				}
			}
		}
	}
}
