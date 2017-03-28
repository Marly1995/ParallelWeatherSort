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
float* fscanFile(char* fileDirectory, int dataSize)
{
	float* data = new float[dataSize];
	FILE * myFile;

	myFile = fopen(fileDirectory, "r");
	fseek(myFile, 0L, SEEK_SET);
	for (int i = 0; i < dataSize; i++)
	{
		fscanf(myFile, "%*s %*lf %*lf %*lf %*lf %f", &data[i]);
	}
	fclose(myFile);
	return data;
}

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
	//Part 1 - handle command line options such as device selection, verbosity, etc.
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
		// File parsing
		int dataSize = 1873106;
		time = clock() - time;
		float* data = fscanFile("../../temp_lincolnshire.txt", dataSize);
		//string data = loadFile("../temp_lincolnshire_short.txt");
		/*float temps[100] = { 0.0f };
		int column = 0;
		int index = 0;
		string num;
		for (int i = 0; i < data.size(); i++)
		{
		if (column == 6)
		{
		temps[index] = stof(num);
		index++;
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
		if (!isalpha(data[i]))
		{
		column++;
		}
		}*/
		time = clock() - time;
		std::cout << "read and parse time = " << time << std::endl;

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

		typedef int mytype;

		//Part 4 - memory allocation
		//host - input	
		std::vector<mytype> A(dataSize, 0);
		for (int i = 0; i < dataSize; i++)
		{
			A[i] = data[i]*100; 
		}
		time = clock() - time;
		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t local_size = 32;

		size_t padding_size = A.size() % local_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(local_size-padding_size, 0);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

		size_t input_elements = A.size();//number of input elements
		size_t input_size = A.size()*sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;

		cl::Event max_event;
		cl::Event min_event;
		cl::Event mean_event;
		cl::Event sd_event;
		cl::Event variance_event;
		cl::Event sd2_event;

		//host - output
		//int *B;
		//size_t output_size = sizeof(int);
		std::vector<mytype> B(1);		
		size_t output_size = B.size()*sizeof(mytype);//size in bytes

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

		//Part 5 - device operations

		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

		cl::Kernel kernel_0 = cl::Kernel(program, "reduce_max");
		kernel_0.setArg(0, buffer_A);
		kernel_0.setArg(1, buffer_B);
		kernel_0.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

																   //call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_0, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &max_event);

		//5.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

		printData("Maximum Temperature:", B[0]/100, max_event);

		cl::Kernel kernel_min = cl::Kernel(program, "reduce_min");
		kernel_min.setArg(0, buffer_A);
		kernel_min.setArg(1, buffer_B);
		kernel_min.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

																   //call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_min, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &min_event);

		//5.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

		printData("Minimum Temperature:", B[0] / 100, min_event);

		//5.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_4");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size

		//call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &mean_event);

		//5.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

		int mean = 0;
		mean = (B[0] / (dataSize)); 
		float fmean = (float)(B[0] / dataSize)/100.0f;

		printData("Mean Temperature:", fmean, mean_event);

		std::vector<mytype> C(1);

		output_size = C.size() * sizeof(mytype);//size in bytes
													   //device - buffers
		//cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_D(context, CL_MEM_READ_ONLY, output_size);

		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_C, 0, 0, output_size);//zero B buffer on device memory

															 //5.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_2 = cl::Kernel(program, "reduce_standard_deviation");
		kernel_2.setArg(0, buffer_A);
		kernel_2.setArg(1, buffer_C);
		kernel_2.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size
		kernel_2.setArg(3, mean);
		queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &sd_event);
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, output_size, &C[0]);
		float SD = (sqrt(C[0] / dataSize));
		printData("Standard Deviation of Temperatures:", SD, sd_event);


		vector<mytype> E(input_elements);
		cl::Buffer buffer_E(context, CL_MEM_READ_WRITE, input_size);
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueReadBuffer(buffer_E, 0, 0, input_size, &E[0]);
		cl::Kernel kernel_variance = cl::Kernel(program, "get_variance");
		kernel_variance.setArg(0, buffer_A);
		kernel_variance.setArg(1, buffer_E);
		kernel_variance.setArg(2, mean);
		//call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_variance, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &variance_event);
		//5.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_E, CL_TRUE, 0, output_size, &E[0]);
		//float SD = (sqrt([0] / dataSize));
		printData("Variance", 0.0f, variance_event);



		vector<mytype> F(1);
		cl::Buffer buffer_F(context, CL_MEM_READ_WRITE, output_size);
		cl::Kernel kernel_vsd = cl::Kernel(program, "reduce_add_4");
		kernel_vsd.setArg(0, buffer_E);
		kernel_vsd.setArg(1, buffer_F);
		kernel_vsd.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size
																   //call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_vsd, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &sd2_event);

		//5.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_F, CL_TRUE, 0, output_size, &F[0]);

		mean = (F[0] / (dataSize));
		fmean = (float)(F[0] / dataSize) / 100.0f;
		float sd2 = sqrt(mean/10);
		printData("Different SD:  ", sd2, sd2_event);



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