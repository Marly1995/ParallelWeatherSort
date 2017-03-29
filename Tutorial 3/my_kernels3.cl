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

__kernel void selection_sort(__global const int *A, __global int *B)
{ 
	int id = get_global_id(0);
	int N = get_global_size(0);

	int ikey = A[id];

	int pos = 0;
	for	(int j = 0; j < N; j++)
	{
		int jkey = A[j];
		bool smaller = (jkey < ikey) || (jkey == ikey && j < id);
		pos += (smaller)?1:0;
	}
	B[pos] = ikey;
}
/////////////////////
////////////////////
// END BITONIC SORT

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
	int gid = get_group_id(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid) {
		B[gid] = scratch[lid];
	}
}

__kernel void reduce_max(__global const int *A, __global int *B, __local int *scratch)
{ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];

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

	scratch[lid] = ((A[id] - M) * (A[id] - M))/1000;

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

__kernel void reduce_standard_deviation_float(__global const float *A, __global float *B, __local float *scratch, float M)
{ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	int gid = get_group_id(0);

	scratch[lid] = ((A[id] - M) * (A[id] - M));//pown(A[id] - M, 2);

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i*=2)
	{ 
		if(!(lid % (i*2)) && ((lid + i) < N))
		{ 
			scratch[lid] += (scratch[lid+i]);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid) 
	{
		B[gid] = scratch[lid];
	}
}