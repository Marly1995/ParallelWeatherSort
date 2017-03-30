////////////////////////
///////////////////////
// START ODD EVEN SORT

// compare and exchange function to swap values if ordered wrong
void cmpxchg(__global int *A, __global int *B)
{ 
	if(*A > *B)
	{ 
		int t = *A; 
		*A = *B; 
		*B = t;
	}
}

// Odd even sort to sort individual workgroups
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
// attempted combination of selection sort and oddeven sort to increase speed of selection sort
/*
__kernel void oddeven_selection_sort(__global const int *A, __global int *B)
{ 
	int id = get_global_id(0);
	int N = get_global_size(0);

	int ikey = A[id];

	int pos = 0;
	for (int i = 0; i < N; i += 2)
	{ 
		if(id%2 == 1 && id+1 < N)
		{ 
			int jkey = A[i];
			//cmpxchg(&A[id], &A[id+1]); // odd
			bool smaller = (jkey < ikey) || (jkey == ikey && i < id);
			pos -= (smaller)?1:0;
		}

		barrier(CLK_GLOBAL_MEM_FENCE);

		if(id%2 == 0 && id+1 < N)
		{
			int jkey = A[i];			
			bool smaller = (jkey < ikey) || (jkey == ikey && i < id);
			pos += (smaller)?1:0;
			//cmpxchg(&A[id], &A[id+1]); //  even
		}
	}

	B[pos] = ikey;
}*/

//////////////////////
/////////////////////
// END ODD EVEN SORT

/////////////////////////
////////////////////////
// START SELECTION SORT

// selection sort using global memory
__kernel void selection_sort(__global const int *A, __global int *B)
{ 
	int id = get_global_id(0);
	int N = get_global_size(0);

	// local values stored as it increases speed
	int ikey = A[id];

	int pos = 0;
	// iterate through all other data for comparison
	for	(int j = 0; j < N; j++)
	{
		int jkey = A[j];
		bool smaller = (jkey < ikey) || (jkey == ikey && j < id);
		// shuffle position of data
		pos += (smaller)?1:0;
	}
	// assign data new position
	B[pos] = ikey;
}

// selection sort using local memory to increase speed
__kernel void selection_sort_local(__global const int *A, __global int *B, __local int *scratch)
{ 
	int id = get_global_id(0);
	int N = get_global_size(0);
	int LN = get_local_size(0);
	int blocksize = LN;

	int ikey = A[id];

	int pos = 0;
	for	(int j = 0; j < N; j+=blocksize)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		// sort through workgroups and assign to local memory
		for (int index = get_local_id(0); index<blocksize; index+=LN)
		{ 
			scratch[index] = A[j+index];
		}
		// wait for all workgroup to be complete
		barrier(CLK_LOCAL_MEM_FENCE);

		// same comparison as global execpt going from work group to workgroup
		for	(int index = 0; index<blocksize; index++)
		{ 
			int jkey = scratch[index];
			bool smaller = (jkey < ikey) || (jkey == ikey && (j+index) < id);
			pos += (smaller)?1:0;
		}	
	}
	// assign data to new position
	B[pos] = ikey;
}

// same sort but designed for floats
__kernel void selection_sort_local_float(__global const float *A, __global float *B, __local float *scratch)
{ 
	int id = get_global_id(0);
	int N = get_global_size(0);
	int LN = get_local_size(0);
	int blocksize = LN;

	float ikey = A[id];

	int pos = 0;
	for	(int j = 0; j < N; j+=blocksize)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int index = get_local_id(0); index<blocksize; index+=LN)
		{ 
			scratch[index] = A[j+index];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for	(int index = 0; index<blocksize; index++)
		{ 
			float jkey = scratch[index];
			bool smaller = (jkey < ikey) || (jkey == ikey && (j+index) < id);
			pos += (smaller)?1:0;
		}	
	}
	B[pos] = ikey;
}
///////////////////////
//////////////////////
// END SELECTION SORT

// reduction kernel to find max data using floats
__kernel void reduce_max_float(__global const float* A, __global float* B, __local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	// compare all local values and keep highest one
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			if(scratch[lid] < scratch[lid+i])
				scratch[lid] = scratch[lid+i]; 

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// check if current local highest value is heigher than global highest value
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

	barrier(CLK_LOCAL_MEM_FENCE);

	// compare all local values and keep lowest one
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			if(scratch[lid] > scratch[lid+i])
				scratch[lid] = scratch[lid+i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// check if current local lowest value is lower than global lowest value
	if(!lid){
		if(scratch[lid] < B[0])
		B[0] = scratch[lid];
		}
}

// reduction kernel to sum all values using ints
__kernel void reduce_add(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	// sum all local values in work group
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// using atomic all sum all values
	if (!lid) {
		atomic_add(&B[0],scratch[lid]);
	}
}

// reduction kernel to sum all values using floats
__kernel void reduce_add_float(__global const float* A, __global float* B, __local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	int gid = get_group_id(0); //  get group id

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	// sum all values in workgroup
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// return sum of workgroup to array using group id
	if (!lid) {
		B[gid] = scratch[lid];
	}
}

// atomic reduce max
// loads all data to local memory and then uses atomic max to find maximum
// on average 4 times faster then reduction method with large dataset 
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

// atomic reduce min
// loads all data to local memory and then uses atomic min to find maximum
// on average 4 times faster then reduction method with large dataset 
__kernel void reduce_min(__global const int *A, __global int *B, __local int *scratch)
{ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];

	if (!lid)
		atomic_min(&B[0],scratch[lid]);
}

// gets all values in data set and returns them to a new vector as sum of squared difference
__kernel void get_variance(__global const int *A, __global int *B, int M)
{ 
	int id = get_global_id(0);
	int lid = get_local_id(0);

	B[id] = A[id] - M;

	B[id] =  (B[id] * B[id])/1000; // divide by 1000 to remove any integer accuracy loss here
}

// standard deviation using reduction for ints
// same as normal reduce add but with variance calculation
__kernel void reduce_standard_deviation(__global const int *A, __global int *B, __local int *scratch, int M)
{ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = ((A[id] - M) * (A[id] - M))/1000;// calculate variance and then divide by 1000 to prevent data loss

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

// standard deviation using floats much the same as float reduce add
__kernel void reduce_standard_deviation_float(__global const float *A, __global float *B, __local float *scratch, float M)
{ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	int gid = get_group_id(0);


	// variance calculation
	scratch[lid] = ((A[id] - M) * (A[id] - M));//pown(A[id] - M, 2) can be used as works on floats

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i*=2)
	{ 
		if(!(lid % (i*2)) && ((lid + i) < N))
		{ 
			scratch[lid] += (scratch[lid+i]);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// assign local sum to array indexed by workgroup
	if (!lid) 
	{
		B[gid] = scratch[lid];
	}
}