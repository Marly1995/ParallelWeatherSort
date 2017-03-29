﻿// https://www.math.kth.se/na/SF2568/parpro-17/F7.pdf
// https://github.com/Gram21/GPUSorting/blob/master/Code/Sort.cl

// USEFUL FUNCTIONS
void swap(__global int *A, __global int *B)
{ 
	int temp = *B;
	*B = *A;
	*A = temp;
}

void compare(__global int *A, __global int *B)
{ 
	if(*A > *B) // other function needed for direction style check
		swap(A, B);
}



////////////////////////
///////////////////////
// START ODD EVEN SORT
void cmpxchg(__global int *A, __global int *B)
{ 
	if(*A > *B)
	{ 
		int t = *A; 
		*A = *B; 
		*B = t;
	}
}

__kernel void oddeven_sort(__global int *A)
{ 
	int id = get_global_id(0);
	int N = get_global_size(0);

	for (int i = 0; i < N; i += 2)
	{ 
		if(id%2 == 1 && id+1 < N)
		{ 
			cmpxchg(&A[id], &A[id+1]); // odd
		}

		barrier(CLK_GLOBAL_MEM_FENCE);

		if(id%2 == 0 && id+1 < N)
		{
			cmpxchg(&A[id], &A[id+1]); //  even
		}
	}
}
//////////////////////
/////////////////////
// END ODD EVEN SORT

///////////////////////
//////////////////////
// START BITONIC SORT
void btic_cmpxchg(__global int *A, __global int *B, bool dir)
{ 
	if((!dir && *A > *B) || (dir && *A < *B))
	{
		int t = *A;
		*A =*B;
		*B = t;
	}
}

void bitonic_merge(int id, __global int *A, int N, bool dir)
{
	for	(int i = N/2; i > 0; i/= 2)
	{ 
		if((id % (i*2)) < i)
		{ 
			btic_cmpxchg(&A[id], &A[id+i], dir);
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

__kernel void bitonic_sort(__global int *A)
{ 
	int id = get_global_id(0);
	int N = get_global_size(0);

	for	(int i = 1; i < N/2; i*=2)
	{ 
		if(id%(i*4) < i*2)
			bitonic_merge(id, A, i*2, false);
		else if((id + i*2)%(i*4) < i*2)
			bitonic_merge(id, A, i*2, true);
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	bitonic_merge(id, A, N, false);
}
/////////////////////
////////////////////
// END BITONIC SORT

//	OLD CODE FROM WORKSHOPS
///////////////////////////
//////////////////////////

//fixed 4 step reduce
__kernel void reduce_add_1(__global const int* A, __global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id]; //copy input to output

	barrier(CLK_GLOBAL_MEM_FENCE); //wait for all threads to finish copying
	 
	//perform reduce on the output array
	//modulo operator is used to skip a set of values (e.g. 2 in the next line)
	//we also check if the added element is within bounds (i.e. < N)
	if (((id % 2) == 0) && ((id + 1) < N)) 
		B[id] += B[id + 1];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 4) == 0) && ((id + 2) < N)) 
		B[id] += B[id + 2];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 8) == 0) && ((id + 4) < N)) 
		B[id] += B[id + 4];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 16) == 0) && ((id + 8) < N)) 
		B[id] += B[id + 8];
}

//flexible step reduce 
__kernel void reduce_add_2(__global const int* A, __global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) { //i is a stride
		if (!(id % (i * 2)) && ((id + i) < N)) 
			B[id] += B[id + i];

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

//reduce using local memory (so called privatisation)
__kernel void reduce_max_float(__global const float* A, __global float* B, __local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			if(scratch[lid] < scratch[lid+i])
				scratch[lid] = scratch[lid+i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	if(!lid){
		if(scratch[lid] > B[0])
		B[0] = scratch[lid];
		}
}

__kernel void reduce_min_float(__global const float* A, __global float* B, __local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			if(scratch[lid] > scratch[lid+i])
				scratch[lid] = scratch[lid+i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	if(!lid){
		if(scratch[lid] < B[0])
		B[0] = scratch[lid];
		}
}

//reduce using local memory + accumulation of local sums into a single location
//works with any number of groups - not optimal!
__kernel void reduce_add_4(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid) {
		atomic_add(&B[0],scratch[lid]);
	}
}

__kernel void reduce_add_float(__global const float* A, __global float* B, __local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

		B[id] = scratch[lid];
}

__kernel void reduce_max(__global const int *A, __global int *B, __local int *scratch)
{ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];
	/*
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			if(scratch[lid] < scratch[lid+i])
				scratch[lid] = scratch[lid+i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}*/

	if (!lid) {
		atomic_max(&B[0],scratch[lid]);
	}
}

__kernel void reduce_min(__global const int *A, __global int *B, __local int *scratch)
{ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];
	
	/*
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			if(scratch[lid] > scratch[lid+i])
				scratch[lid] = scratch[lid+i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	*/
	if (!lid)
		atomic_min(&B[0],scratch[lid]);
}
__kernel void get_variance(__global const int *A, __global int *B, int M)
{ 
	int id = get_global_id(0);
	int lid = get_local_id(0);

	B[id] = A[id] - M;

	B[id] =  (B[id] * B[id])/1000;
}
// sd reduce
__kernel void reduce_standard_deviation(__global const int *A, __global int *B, __local int *scratch, int M)
{ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = ((A[id] - M) * (A[id] - M))/10000;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i*=2)
	{ 
		if(!(lid % (i*2)) && ((lid + i) < N))
		{ 
			scratch[lid] += (scratch[lid+i]);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(!lid)
	{ 
		atomic_add(&B[0], scratch[lid]);
	}

}

//a very simple histogram implementation
__kernel void hist_simple(__global const int* A, __global int* H) { 
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index

	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}

//a double-buffered version of the Hillis-Steele inclusive scan
//requires two additional input arguments which correspond to two local buffers
__kernel void scan_add(__global const int* A, __global int* B, __local int* scratch_1, __local int* scratch_2) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	__local int *scratch_3;//used for buffer swap

	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		//buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	//copy the cache to output array
	B[id] = scratch_1[lid];
}

//calculates the block sums
__kernel void block_sum(__global const int* A, __global int* B, int local_size) {
	int id = get_global_id(0);
	B[id] = A[(id+1)*local_size-1];
}

//simple exclusive serial scan based on atomic operations - sufficient for small number of elements
__kernel void scan_add_atomic(__global int* A, __global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id+1; i < N; i++)
		atomic_add(&B[i], A[id]);
}

//adjust the values stored in partial scans by adding block sums to corresponding blocks
__kernel void scan_add_adjust(__global int* A, __global const int* B) {
	int id = get_global_id(0);
	int gid = get_group_id(0);
	A[id] += B[gid];
}
