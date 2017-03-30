#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

// File parse using fscanf efficient during debug but less so in release.
// Data stored direct into float array or int array
// time debug: ~9s
// time release: ~2.5s
float* fscanFile(char* fileDirectory, int dataSize)
{
	float* data = new float[dataSize];
	FILE * myFile;

	myFile = fopen(fileDirectory, "r");
	fseek(myFile, 0L, SEEK_SET);
	for (int i = 0; i < dataSize; i++)
	{
		fscanf(myFile, "%*s %*lf %*lf %*lf %*lf %f", &data[i]); // values with a star are read but not stored
	}
	fclose(myFile);
	return data;
}

// File reading using ifstream slow in debug but fast in release
// Data stored into single string
// time debug: ~30s
// time release: ~1s
string loadFile(char* fileDirectory)
{
	ifstream file;
	string data;
	string line;
	if (fileDirectory != nullptr)
	{
		file.open(fileDirectory);
	}
	if (file.is_open())
	{
		std::cout << "File Opened!" << std::endl;
		while (getline(file, line))
		{
			data += line;
		}
	}
	file.close();
	return data;
}

// File parse for data in single string
vector<float> parseData(string data, int dataSize)
{
	vector<float> temps;
	int column = 0;
	string num;
	for (int i = 0; i < data.size(); i++)
	{		
		if (column == 6)
		{
			temps.push_back(stof(num));
			num.clear();
			column = 1;
		}
		if (column == 5)
		{
			num += data[i];
		}
		if (data[i] == ' ')
		{
			column++;
		}
	}
	return temps;
}

// Function to print results
void printData(string info, float result, cl::Event event)
{
	std::cout << "///////////////////////////////////////////////////////////////////////////////" << std::endl;
	std::cout << info << result << std::endl;
	//std::cout << "" << std::endl;
	//std::cout << GetFullProfilingInfo(event, ProfilingResolution::PROF_US) << endl;
	//std::cout << "" << std::endl;
	std::cout << "Kernel execution time [ns]: " << event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
	std::cout << "///////////////////////////////////////////////////////////////////////////////" << std::endl;

}

int main(int argc, char **argv) {
	int platform_id = 0;
	int device_id = 0;

	clock_t time = clock();

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//detect any potential exceptions
	try {
		// File parsing using fscanf or if stream
		int dataSize = 1873106;
		time = clock() - time;
		//float* data = fscanFile("../../temp_lincolnshire.txt", dataSize);
		string dataString = loadFile("../../temp_lincolnshire.txt");
		std::vector<float> data = parseData(dataString, dataSize);
		time = clock() - time;
		std::cout << "Time to read and parse file [ms]: " << time << std::endl;

		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;
		AddSources(sources, "../../Tutorial 3/my_kernels3.cl");
		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		// define types for easy alteration of multiple functions
		typedef int mytype;
		typedef float myfloat;

		// events used for profilig differnt kernal operations
		// events marked with an f for floating operations
		cl::Event max_event;
		cl::Event fmax_event;
		cl::Event min_event;
		cl::Event fmin_event;
		cl::Event mean_event;
		cl::Event fmean_event;
		cl::Event sd_event;
		cl::Event fsd_event;
		cl::Event variance_event;
		cl::Event sd2_event;
		cl::Event sort_event;
		cl::Event fsort_event;

		// Create vectors to store data for operations
		// Different vector for: floats, integers and multliplied integers for approximate float values
		std::vector<mytype> A(dataSize, 0);
		std::vector<mytype> I(dataSize, 0);
		std::vector<myfloat> FA(dataSize, 0);
		// load all data into vectors
		for (int i = 0; i < dataSize; i++)
		{
			I[i] = data[i];
			A[i] = data[i]*100; 
			FA[i] = data[i];
		}
		time = clock() - time;

		// define a workgroup size
		// 32 used as it is fastest with current operations for this data type
		// 32 is also optimal for use on the gtx platform
		// TODO: reasons for this
		size_t local_size = 32;

		// padding given by the remainer of number of elements when split into work groups
		size_t padding_size = A.size() % local_size;

		if (padding_size) {
			// create empty vector with size of the remaining space in the last workgroup
			// TODO: fill plain integer vector
			std::vector<int> A_ext(local_size-padding_size, 0);
			std::vector<float> FA_ext(local_size - padding_size, 0);
			// add this vector to the end of our data
			A.insert(A.end(), A_ext.begin(), A_ext.end());
			FA.insert(FA.end(), FA_ext.begin(), FA_ext.end());
		}

		size_t input_elements = A.size(); // number of elements in data set including padding
		size_t input_size = A.size()*sizeof(mytype); // size in bytes of the data set (integers)
		size_t finput_size = FA.size() * sizeof(myfloat); // size in bytes of the data set (floats)
		size_t nr_groups = input_elements / local_size; // number of work groups 

		//host - output
		// TODO: redefine single values as pointers not vectors
		// Values to store outputs 
		std::vector<mytype> B(1);	
		std::vector<mytype> C(input_elements);
		std::vector<mytype> D(input_elements);
		std::vector<myfloat> FB(1);
		std::vector<myfloat> FC(input_elements);		
		std::vector<myfloat> FD(input_elements);
		// Sizes of outputs in bytes	
		size_t output_size = B.size()*sizeof(mytype); 
		size_t foutput_size = B.size() * sizeof(myfloat);
		// TODO: think this can go
		size_t fmean_output_size = FC.size() * sizeof(myfloat);
		
		
		

		//device - buffers
		// create all buffers at start
		// buffers for integer operations
		// buffers to store dataset in integers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_I(context, CL_MEM_READ_ONLY, input_size);

		// buffers to accept integer outputs
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, input_size);
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, input_size);
		
		// buffer to store dataset in floats
		cl::Buffer buffer_FA(context, CL_MEM_READ_ONLY, finput_size);

		// buffers to accept float outputs
		cl::Buffer buffer_FB(context, CL_MEM_READ_WRITE, foutput_size);
		cl::Buffer buffer_FC(context, CL_MEM_READ_WRITE, fmean_output_size);
		cl::Buffer buffer_FD(context, CL_MEM_READ_WRITE, finput_size);

		// queue all buffers prior to running kernals
		// queue write buffers for integer values
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueWriteBuffer(buffer_I, CL_TRUE, 0, input_size, &I[0]);

		// queue fill and read buffers for integer values
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);
		queue.enqueueReadBuffer(buffer_C, 0, 0, input_size, &C[0]);
		queue.enqueueReadBuffer(buffer_D, 0, 0, input_size, &D[0]);

		// queue write buffers for float values
		queue.enqueueWriteBuffer(buffer_FA, CL_TRUE, 0, finput_size, &FA[0]);

		// queue fill and read buffers for float values
		queue.enqueueFillBuffer(buffer_FB, 0, 0, foutput_size);
		queue.enqueueFillBuffer(buffer_FC, 0, 0, fmean_output_size);
		queue.enqueueReadBuffer(buffer_FD, CL_TRUE, 0, finput_size, &FD[0]);

		// TODO: make all of these functions
		// integer maximum using atomic method
		// with current dataset faster than reduction method
		// time: ~120000 ns
		cl::Kernel kernel_0 = cl::Kernel(program, "reduce_max");
		kernel_0.setArg(0, buffer_A);
		kernel_0.setArg(1, buffer_B);
		kernel_0.setArg(2, cl::Local(local_size * sizeof(mytype)));
		queue.enqueueNDRangeKernel(kernel_0, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &max_event);
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		printData("Maximum Temperature:", B[0]/100, max_event); 

		// floating Maximum using reduction and global adressing
		// slower than atmoic method but this is not usable on floats
		// time: ~ 360000 ns
		cl::Kernel kernel_fmax = cl::Kernel(program, "reduce_max_float");
		kernel_fmax.setArg(0, buffer_FA);
		kernel_fmax.setArg(1, buffer_FB);
		kernel_fmax.setArg(2, cl::Local(local_size * sizeof(myfloat)));
		queue.enqueueNDRangeKernel(kernel_fmax, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &fmax_event);
		queue.enqueueReadBuffer(buffer_FB, CL_TRUE, 0, foutput_size, &FB[0]);
		printData("Maximum floating : ", FB[0], fmax_event);

		// integer minimum using atomic method
		// with current dataset faster than reduction method
		// time: ~120000 ns
		cl::Kernel kernel_min = cl::Kernel(program, "reduce_min");
		kernel_min.setArg(0, buffer_A);
		kernel_min.setArg(1, buffer_B);
		kernel_min.setArg(2, cl::Local(local_size * sizeof(mytype)));
		queue.enqueueNDRangeKernel(kernel_min, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &min_event);
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		printData("Minimum Temperature:", B[0] / 100, min_event);

		// floating minimum using reduction and global adressing
		// slower than atmoic method but this is not usable on floats
		// time: ~360000 ns
		cl::Kernel kernel_fmin = cl::Kernel(program, "reduce_min_float");
		kernel_fmin.setArg(0, buffer_FA);
		kernel_fmin.setArg(1, buffer_FB);
		kernel_fmin.setArg(2, cl::Local(local_size * sizeof(myfloat)));
		queue.enqueueNDRangeKernel(kernel_fmin, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &fmin_event);
		queue.enqueueReadBuffer(buffer_FB, CL_TRUE, 0, foutput_size, &FB[0]);
		printData("Minimum floating: ", FB[0], fmin_event);

		// Integer mean using atmoic add
		// faster than reduction method but only usable on ints
		// sum of data is recieved nd divided by dataset size to get mean
		// time: ~320000 ns
		cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &mean_event);
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		int mean = 0;
		mean = (B[0] / (dataSize)); 
		float fmean = (float)(B[0] / dataSize)/100.0f;
		printData("Mean Temperature:", fmean, mean_event);

		// float mean using reduction to return array of work group sums
		// fast on current data set but additional kernals could be used to further reduce the data
		// work group sums and added and then devied by the datasize to get the mean
		// time: ~320000 ns + ~30000 ns from sequential addition
		cl::Kernel kernel_fmean = cl::Kernel(program, "reduce_add_float");
		kernel_fmean.setArg(0, buffer_FA);
		kernel_fmean.setArg(1, buffer_FC);
		kernel_fmean.setArg(2, cl::Local(local_size * sizeof(myfloat)));//local memory size
		queue.enqueueNDRangeKernel(kernel_fmean, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &fmean_event);
		queue.enqueueReadBuffer(buffer_FC, CL_TRUE, 0, fmean_output_size, &FC[0]);
		float fsum = 0.0f;
		for (int i = 0; i <= nr_groups; i++) { fsum += FC[i]; }
		fmean = fsum / (dataSize);
		printData("Mean floating:", fmean, fmean_event);

		// need to requeue buffer to remove current data
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);
		// integer standard deviation using atomic add 
		cl::Kernel kernel_2 = cl::Kernel(program, "reduce_standard_deviation");
		kernel_2.setArg(0, buffer_A);
		kernel_2.setArg(1, buffer_B);
		kernel_2.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size
		kernel_2.setArg(3, mean);
		queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &sd_event);
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		float SD = (sqrt((B[0] /10)/ dataSize)); // need to divide by 10 here as division earlier displaces int values too much
		printData("SD: ", SD, sd_event);

		// float standard deviation using reduction and sequential summation of work groups
		queue.enqueueFillBuffer(buffer_FC, 0, 0, fmean_output_size);
		cl::Kernel kernel_fsd = cl::Kernel(program, "reduce_standard_deviation_float");
		kernel_fsd.setArg(0, buffer_FA);
		kernel_fsd.setArg(1, buffer_FC);
		kernel_fsd.setArg(2, cl::Local(local_size * sizeof(myfloat)));//local memory size
		kernel_fsd.setArg(3, fmean);
		queue.enqueueNDRangeKernel(kernel_fsd, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &fsd_event);
		queue.enqueueReadBuffer(buffer_FC, CL_TRUE, 0, fmean_output_size, &FC[0]);
		fsum = 0.0f;
		for (int i = 0; i <= nr_groups; i++) { fsum += FC[i]; }
		fmean = fsum / (dataSize);
		SD = (sqrt(fmean));
		printData("floating SD: ", SD, fsd_event);

		// alternative method for variance and standard deviation calculation with separate kernel for variance
		// slower and more cumbersome than other method
		cl::Kernel kernel_variance = cl::Kernel(program, "get_variance");
		kernel_variance.setArg(0, buffer_A);
		kernel_variance.setArg(1, buffer_C);
		kernel_variance.setArg(2, mean);
		queue.enqueueNDRangeKernel(kernel_variance, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &variance_event);
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, output_size, &C[0]);
		printData("Variance: ", 35.0f, variance_event);
		
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);
		cl::Kernel kernel_vsd = cl::Kernel(program, "reduce_add");
		kernel_vsd.setArg(0, buffer_C);
		kernel_vsd.setArg(1, buffer_B);
		kernel_vsd.setArg(2, cl::Local(local_size * sizeof(mytype)));
		queue.enqueueNDRangeKernel(kernel_vsd, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &sd2_event);
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		mean = (B[0] / (dataSize));
		float sd2 = sqrt(mean/10);
		printData("Separated SD:  ", sd2, sd2_event);


		// sort integer values using selection sort
		// sort inefficient and does not really benefit from paralization
		// time: ~22500000000 ns
		cl::Kernel kernel_sel_sort = cl::Kernel(program, "selection_sort_local");
		kernel_sel_sort.setArg(0, buffer_I);
		kernel_sel_sort.setArg(1, buffer_D);
		kernel_sel_sort.setArg(2, cl::Local(local_size * sizeof(mytype)));
		queue.enqueueNDRangeKernel(kernel_sel_sort, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &sort_event);
		queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, input_size, &D[0]);
		printData("Sort: ", 0.0f, sort_event);

		int range = D.size()/4;
		std::cout << "Min = " << D[0] << std::endl;
		std::cout << "Max = " << D[D.size()-1] << std::endl;
		std::cout << "Median = " << D[range*2] << std::endl;
		std::cout << "LQ = " << D[range] << std::endl;
		std::cout << "UQ = " << D[D.size() - (range +1)] << std::endl;


		// same selection sort but using floats
		// time: ~50500000000 ns
		cl::Kernel kernel_sel_sort_float = cl::Kernel(program, "selection_sort_local_float");
		kernel_sel_sort_float.setArg(0, buffer_FA);
		kernel_sel_sort_float.setArg(1, buffer_FD);
		kernel_sel_sort_float.setArg(2, cl::Local(local_size * sizeof(myfloat)));
		queue.enqueueNDRangeKernel(kernel_sel_sort_float, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &sort_event);
		queue.enqueueReadBuffer(buffer_FD, CL_TRUE, 0, finput_size, &FD[0]);
		printData("Float Sort: ", 0.0f, sort_event);

		std::cout << "Min = " << FD[0] << std::endl;
		std::cout << "Max = " << FD[FD.size() - 1] << std::endl;
		std::cout << "Median = " << FD[range * 2] << std::endl;
		std::cout << "LQ = " << FD[range] << std::endl;
		std::cout << "UQ = " << FD[FD.size() - (range + 1)] << std::endl;




		time = clock() - time;
		std::cout << "Total Calculation Time = " << time << std::endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}


/* 
///////////////////////////////////////////////////////////////////////////////
Maximum 126400
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
Minimum 140256
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
Mean 347584
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
Standard Deviation 365152
///////////////////////////////////////////////////////////////////////////////
Total Calculation Time = 352
*/
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////