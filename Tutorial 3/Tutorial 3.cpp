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
		typedef float myfloat;

		std::vector<mytype> A(dataSize, 0);
		std::vector<mytype> I(dataSize, 0);
		std::vector<myfloat> FA(dataSize, 0);
		for (int i = 0; i < dataSize; i++)
		{
			I[i] = data[i];
			A[i] = data[i]*100; 
			FA[i] = data[i];
		}
		time = clock() - time;

		size_t local_size = 32;

		size_t padding_size = A.size() % local_size;

		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(local_size-padding_size, 0);
			std::vector<float> FA_ext(local_size - padding_size, 0);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end());
			FA.insert(FA.end(), FA_ext.begin(), FA_ext.end());
		}

		size_t input_elements = A.size();//number of input elements
		size_t input_size = A.size()*sizeof(mytype);//size in bytes
		size_t finput_size = FA.size() * sizeof(myfloat);
		size_t nr_groups = input_elements / local_size;

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

		//host - output
		//int *B;
		//size_t output_size = sizeof(int);
		std::vector<mytype> B(1);		
		size_t output_size = B.size()*sizeof(mytype);//size in bytes
		std::vector<myfloat> FB(1);
		size_t foutput_size = B.size() * sizeof(myfloat);//size in bytes
		std::vector<myfloat> FC(input_elements);
		size_t fmean_output_size = FC.size() * sizeof(myfloat);
		std::vector<mytype> C(input_elements);
		std::vector<mytype> D(input_elements);
		std::vector<myfloat> FD(input_elements);
		
		

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, input_size);
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, input_size);
		cl::Buffer buffer_I(context, CL_MEM_READ_ONLY, input_size);

		cl::Buffer buffer_FA(context, CL_MEM_READ_ONLY, finput_size);
		cl::Buffer buffer_FB(context, CL_MEM_READ_WRITE, foutput_size);
		cl::Buffer buffer_FC(context, CL_MEM_READ_WRITE, fmean_output_size);
		cl::Buffer buffer_FD(context, CL_MEM_READ_WRITE, finput_size);

		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueWriteBuffer(buffer_I, CL_TRUE, 0, input_size, &I[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);
		queue.enqueueReadBuffer(buffer_C, 0, 0, input_size, &C[0]);
		queue.enqueueReadBuffer(buffer_D, 0, 0, input_size, &D[0]);

		queue.enqueueWriteBuffer(buffer_FA, CL_TRUE, 0, finput_size, &FA[0]);
		queue.enqueueFillBuffer(buffer_FB, 0, 0, foutput_size);
		queue.enqueueFillBuffer(buffer_FC, 0, 0, fmean_output_size);
		queue.enqueueReadBuffer(buffer_FD, CL_TRUE, 0, finput_size, &FD[0]);

		// integer maximum	
		cl::Kernel kernel_0 = cl::Kernel(program, "reduce_max");
		kernel_0.setArg(0, buffer_A);
		kernel_0.setArg(1, buffer_B);
		kernel_0.setArg(2, cl::Local(local_size * sizeof(mytype)));
		queue.enqueueNDRangeKernel(kernel_0, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &max_event);
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		printData("Maximum Temperature:", B[0]/100, max_event);

		// floating Maximum
		cl::Kernel kernel_fmax = cl::Kernel(program, "reduce_max_float");
		kernel_fmax.setArg(0, buffer_FA);
		kernel_fmax.setArg(1, buffer_FB);
		kernel_fmax.setArg(2, cl::Local(local_size * sizeof(myfloat)));
		queue.enqueueNDRangeKernel(kernel_fmax, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &fmax_event);
		queue.enqueueReadBuffer(buffer_FB, CL_TRUE, 0, foutput_size, &FB[0]);
		printData("Maximum floating : ", FB[0], fmax_event);

		// initeger minimum
		cl::Kernel kernel_min = cl::Kernel(program, "reduce_min");
		kernel_min.setArg(0, buffer_A);
		kernel_min.setArg(1, buffer_B);
		kernel_min.setArg(2, cl::Local(local_size * sizeof(mytype)));
		queue.enqueueNDRangeKernel(kernel_min, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &min_event);
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		printData("Minimum Temperature:", B[0] / 100, min_event);

		// floating Minimum
		cl::Kernel kernel_fmin = cl::Kernel(program, "reduce_min_float");
		kernel_fmin.setArg(0, buffer_FA);
		kernel_fmin.setArg(1, buffer_FB);
		kernel_fmin.setArg(2, cl::Local(local_size * sizeof(myfloat)));
		queue.enqueueNDRangeKernel(kernel_fmin, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &fmin_event);
		queue.enqueueReadBuffer(buffer_FB, CL_TRUE, 0, foutput_size, &FB[0]);
		printData("Minimum floating: ", FB[0], fmin_event);

		// integer mean
		cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_4");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &mean_event);
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		int mean = 0;
		mean = (B[0] / (dataSize)); 
		float fmean = (float)(B[0] / dataSize)/100.0f;
		printData("Mean Temperature:", fmean, mean_event);

		// floating mean
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

		// FIND OUT WHY THIS IS NEEDED
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);
		// integer sd
		cl::Kernel kernel_2 = cl::Kernel(program, "reduce_standard_deviation");
		kernel_2.setArg(0, buffer_A);
		kernel_2.setArg(1, buffer_B);
		kernel_2.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size
		kernel_2.setArg(3, mean);
		queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &sd_event);
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		float SD = (sqrt((B[0] /10)/ dataSize));
		printData("SD: ", SD, sd_event);

		// floating sd
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

		// integer variance
		cl::Kernel kernel_variance = cl::Kernel(program, "get_variance");
		kernel_variance.setArg(0, buffer_A);
		kernel_variance.setArg(1, buffer_C);
		kernel_variance.setArg(2, mean);
		queue.enqueueNDRangeKernel(kernel_variance, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &variance_event);
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, output_size, &C[0]);
		printData("Variance: ", 35.0f, variance_event);
		// correct integer sd
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);
		cl::Kernel kernel_vsd = cl::Kernel(program, "reduce_add_4");
		kernel_vsd.setArg(0, buffer_C);
		kernel_vsd.setArg(1, buffer_B);
		kernel_vsd.setArg(2, cl::Local(local_size * sizeof(mytype)));
		queue.enqueueNDRangeKernel(kernel_vsd, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &sd2_event);
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		mean = (B[0] / (dataSize));
		float sd2 = sqrt(mean/10);
		printData("Separated SD:  ", sd2, sd2_event);


		// sort integer
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

		// sort float
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