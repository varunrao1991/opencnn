#include "logger.h"

#include "cl/cl_ext.h"
#include "cl/cl.h"

#include <cstdarg>
#include <iostream>
#include <assert.h> 
#include <cstring>
#include <windows.h>

#define LOG_TAG "[Semantic Seg]"

void logNoNewline(char* format, ...)
{
	va_list argList;
	va_start(argList, format);
	std::printf(format, argList);
	va_end(argList);
}

void log(char* format, ...)
{
	va_list argList;
	va_start(argList, format);
	std::printf(LOG_TAG);
	std::printf(format, argList);
	std::printf("\n");
	va_end(argList);
}
void checkErr(cl_int err, int line, const char* n, bool verbosity)
{
	if (err != CL_SUCCESS) {
		std::cout << n << ": line:" << line << " " << getErrorString(err) << std::endl;
		assert(0);
	}
	else if (n != NULL) {
		if (verbosity) std::cerr << n << "\r\t" << "OK" << std::endl;
	}
}
void checkError(cl_int clErrNum, cl_int clSuccess)
{
	if (clErrNum != clSuccess)
	{
		ALOG_GPUML("Error %d : %s", clErrNum, getErrorString(clErrNum));
	}
}

const char* getErrorString(cl_int error)
{
	static const char* errorString[] = {
		"CL_SUCCESS",
		"CL_DEVICE_NOT_FOUND",
		"CL_DEVICE_NOT_AVAILABLE",
		"CL_COMPILER_NOT_AVAILABLE",
		"CL_MEM_OBJECT_ALLOCATION_FAILURE",
		"CL_OUT_OF_RESOURCES",
		"CL_OUT_OF_HOST_MEMORY",
		"CL_PROFILING_INFO_NOT_AVAILABLE",
		"CL_MEM_COPY_OVERLAP",
		"CL_IMAGE_FORMAT_MISMATCH",
		"CL_IMAGE_FORMAT_NOT_SUPPORTED",
		"CL_BUILD_PROGRAM_FAILURE",
		"CL_MAP_FAILURE",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"CL_INVALID_VALUE",
		"CL_INVALID_DEVICE_TYPE",
		"CL_INVALID_PLATFORM",
		"CL_INVALID_DEVICE",
		"CL_INVALID_CONTEXT",
		"CL_INVALID_QUEUE_PROPERTIES",
		"CL_INVALID_COMMAND_QUEUE",
		"CL_INVALID_HOST_PTR",
		"CL_INVALID_MEM_OBJECT",
		"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
		"CL_INVALID_IMAGE_SIZE",
		"CL_INVALID_SAMPLER",
		"CL_INVALID_BINARY",
		"CL_INVALID_BUILD_OPTIONS",
		"CL_INVALID_PROGRAM",
		"CL_INVALID_PROGRAM_EXECUTABLE",
		"CL_INVALID_KERNEL_NAME",
		"CL_INVALID_KERNEL_DEFINITION",
		"CL_INVALID_KERNEL",
		"CL_INVALID_ARG_INDEX",
		"CL_INVALID_ARG_VALUE",
		"CL_INVALID_ARG_SIZE",
		"CL_INVALID_KERNEL_ARGS",
		"CL_INVALID_WORK_DIMENSION",
		"CL_INVALID_WORK_GROUP_SIZE",
		"CL_INVALID_WORK_ITEM_SIZE",
		"CL_INVALID_GLOBAL_OFFSET",
		"CL_INVALID_EVENT_WAIT_LIST",
		"CL_INVALID_EVENT",
		"CL_INVALID_OPERATION",
		"CL_INVALID_GL_OBJECT",
		"CL_INVALID_BUFFER_SIZE",
		"CL_INVALID_MIP_LEVEL",
		"CL_INVALID_GLOBAL_WORK_SIZE",
	};

	const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

	const int index = -error;

	return (index >= 0 && index < errorCount) ? errorString[index] : "Unspecified Error";
}

void printDevInfo(cl_device_id device)
{
	char device_string[1024] = "Failed to fetch";
	bool nv_device_attibute_query = false;

	clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_string), device_string, NULL);
	ALOG_GPUML("  CL_DEVICE_NAME: \t\t\t%s", device_string);

	clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(device_string), device_string, NULL);
	ALOG_GPUML("  CL_DEVICE_VENDOR: \t\t\t%s", device_string);

	clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(device_string), device_string, NULL);
	ALOG_GPUML("  CL_DRIVER_VERSION: \t\t\t%s", device_string);

	clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(device_string), device_string, NULL);
	ALOG_GPUML("  CL_DEVICE_VERSION: \t\t\t%s", device_string);

#if !defined(__APPLE__) && !defined(__MACOSX)
	if (strncmp("OpenCL 1.0", device_string, 10) != 0)
	{
#ifndef CL_DEVICE_OPENCL_C_VERSION
#define CL_DEVICE_OPENCL_C_VERSION 0x103D
#endif

		clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(device_string), device_string, NULL);
		ALOG_GPUML("  CL_DEVICE_OPENCL_C_VERSION: \t\t%s", device_string);
	}
#endif

	cl_device_type type;
	clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
	if (type & CL_DEVICE_TYPE_CPU)
		ALOG_GPUML("  CL_DEVICE_TYPE:\t\t\t%s", "CL_DEVICE_TYPE_CPU");
	if (type & CL_DEVICE_TYPE_GPU)
		ALOG_GPUML("  CL_DEVICE_TYPE:\t\t\t%s", "CL_DEVICE_TYPE_GPU");
	if (type & CL_DEVICE_TYPE_ACCELERATOR)
		ALOG_GPUML("  CL_DEVICE_TYPE:\t\t\t%s", "CL_DEVICE_TYPE_ACCELERATOR");
	if (type & CL_DEVICE_TYPE_DEFAULT)
		ALOG_GPUML("  CL_DEVICE_TYPE:\t\t\t%s", "CL_DEVICE_TYPE_DEFAULT");

	cl_uint compute_units;
	clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
	ALOG_GPUML("  CL_DEVICE_MAX_COMPUTE_UNITS:\t\t%u", compute_units);

	size_t workitem_dims;
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(workitem_dims), &workitem_dims, NULL);
	ALOG_GPUML("  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:\t%u", workitem_dims);

	size_t workitem_size[3];
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
	ALOG_GPUML("  CL_DEVICE_MAX_WORK_ITEM_SIZES:\t%u / %u / %u ", workitem_size[0], workitem_size[1], workitem_size[2]);

	size_t workgroup_size;
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);
	ALOG_GPUML("  CL_DEVICE_MAX_WORK_GROUP_SIZE:\t%u", workgroup_size);

	cl_uint clock_frequency;
	clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
	ALOG_GPUML("  CL_DEVICE_MAX_CLOCK_FREQUENCY:\t%u MHz", clock_frequency);

	cl_uint addr_bits;
	clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(addr_bits), &addr_bits, NULL);
	ALOG_GPUML("  CL_DEVICE_ADDRESS_BITS:\t\t%u", addr_bits);

	cl_ulong max_mem_alloc_size;
	clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
	ALOG_GPUML("  CL_DEVICE_MAX_MEM_ALLOC_SIZE:\t\t%u MByte", (unsigned int)(max_mem_alloc_size / (1024 * 1024)));

	cl_ulong mem_size;
	clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
	ALOG_GPUML("  CL_DEVICE_GLOBAL_MEM_SIZE:\t\t%u MByte", (unsigned int)(mem_size / (1024 * 1024)));

	cl_bool error_correction_support;
	clGetDeviceInfo(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(error_correction_support), &error_correction_support, NULL);
	ALOG_GPUML("  CL_DEVICE_ERROR_CORRECTION_SUPPORT:\t%s", error_correction_support == CL_TRUE ? "yes" : "no");

	cl_device_local_mem_type local_mem_type;
	clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, NULL);
	ALOG_GPUML("  CL_DEVICE_LOCAL_MEM_TYPE:\t\t%s", local_mem_type == 1 ? "local" : "global");

	clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
	ALOG_GPUML("  CL_DEVICE_LOCAL_MEM_SIZE:\t\t%u KByte", (unsigned int)(mem_size / 1024));

	clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(mem_size), &mem_size, NULL);
	ALOG_GPUML("  CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:\t%u KByte", (unsigned int)(mem_size / 1024));

	cl_command_queue_properties queue_properties;
	clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(queue_properties), &queue_properties, NULL);
	if (queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
		ALOG_GPUML("  CL_DEVICE_QUEUE_PROPERTIES:\t\t%s", "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE");
	if (queue_properties & CL_QUEUE_PROFILING_ENABLE)
		ALOG_GPUML("  CL_DEVICE_QUEUE_PROPERTIES:\t\t%s", "CL_QUEUE_PROFILING_ENABLE");

	cl_bool image_support;
	clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
	ALOG_GPUML("  CL_DEVICE_IMAGE_SUPPORT:\t\t%u", image_support);

	cl_uint max_read_image_args;
	clGetDeviceInfo(device, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(max_read_image_args), &max_read_image_args, NULL);
	ALOG_GPUML("  CL_DEVICE_MAX_READ_IMAGE_ARGS:\t%u", max_read_image_args);

	cl_uint max_write_image_args;
	clGetDeviceInfo(device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(max_write_image_args), &max_write_image_args, NULL);
	ALOG_GPUML("  CL_DEVICE_MAX_WRITE_IMAGE_ARGS:\t%u\n", max_write_image_args);

	cl_device_fp_config fp_config;
	clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(cl_device_fp_config), &fp_config, NULL);
	ALOG_GPUML("  CL_DEVICE_SINGLE_FP_CONFIG:\t\t%s%s%s%s%s%s",
		fp_config & CL_FP_DENORM ? "denorms " : "",
		fp_config & CL_FP_INF_NAN ? "INF-quietNaNs " : "",
		fp_config & CL_FP_ROUND_TO_NEAREST ? "round-to-nearest " : "",
		fp_config & CL_FP_ROUND_TO_ZERO ? "round-to-zero " : "",
		fp_config & CL_FP_ROUND_TO_INF ? "round-to-inf " : "",
		fp_config & CL_FP_FMA ? "fma " : "");

	size_t szMaxDims[5];
	ALOG_GPUML("CL_DEVICE_IMAGE <dim>");
	clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &szMaxDims[0], NULL);
	ALOG_GPUML("\t\t\t2D_MAX_WIDTH\t %u", szMaxDims[0]);
	clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[1], NULL);
	ALOG_GPUML("\t\t\t\t\t2D_MAX_HEIGHT\t %u", szMaxDims[1]);
	clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), &szMaxDims[2], NULL);
	ALOG_GPUML("\t\t\t\t\t3D_MAX_WIDTH\t %u", szMaxDims[2]);
	clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[3], NULL);
	ALOG_GPUML("\t\t\t\t\t3D_MAX_HEIGHT\t %u", szMaxDims[3]);
	clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), &szMaxDims[4], NULL);
	ALOG_GPUML("\t\t\t\t\t3D_MAX_DEPTH\t %u", szMaxDims[4]);

	clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(device_string), &device_string, NULL);
	if (device_string != 0)
	{
		ALOG_GPUML("CL_DEVICE_EXTENSIONS:");
		std::string stdDevString;
		stdDevString = std::string(device_string);
		size_t szOldPos = 0;
		size_t szSpacePos = stdDevString.find(' ', szOldPos); 		while (szSpacePos != stdDevString.npos)
		{
			if (strcmp("cl_nv_device_attribute_query", stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str()) == 0)
				nv_device_attibute_query = true;

			if (szOldPos > 0)
			{
				ALOG_GPUML("\t\t");
			}
			ALOG_GPUML("\t\t\t%s", stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str());

			do {
				szOldPos = szSpacePos + 1;
				szSpacePos = stdDevString.find(' ', szOldPos);
			} while (szSpacePos == szOldPos);
		}
		ALOG_GPUML("");
	}
	else
	{
		ALOG_GPUML("  CL_DEVICE_EXTENSIONS: None");
	}

	if (nv_device_attibute_query)
	{
		cl_uint compute_capability_major, compute_capability_minor;
		clGetDeviceInfo(device, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(cl_uint), &compute_capability_major, NULL);
		clGetDeviceInfo(device, CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(cl_uint), &compute_capability_minor, NULL);
		ALOG_GPUML("CL_DEVICE_COMPUTE_CAPABILITY_NV:\t%u.%u", compute_capability_major, compute_capability_minor);

		ALOG_GPUML("  NUMBER OF MULTIPROCESSORS:\t\t%u", compute_units);
		cl_uint regs_per_block;
		clGetDeviceInfo(device, CL_DEVICE_REGISTERS_PER_BLOCK_NV, sizeof(cl_uint), &regs_per_block, NULL);
		ALOG_GPUML("  CL_DEVICE_REGISTERS_PER_BLOCK_NV:\t%u", regs_per_block);

		cl_uint warp_size;
		clGetDeviceInfo(device, CL_DEVICE_WARP_SIZE_NV, sizeof(cl_uint), &warp_size, NULL);
		ALOG_GPUML("  CL_DEVICE_WARP_SIZE_NV:\t\t%u", warp_size);

		cl_bool gpu_overlap;
		clGetDeviceInfo(device, CL_DEVICE_GPU_OVERLAP_NV, sizeof(cl_bool), &gpu_overlap, NULL);
		ALOG_GPUML("  CL_DEVICE_GPU_OVERLAP_NV:\t\t%s", gpu_overlap == CL_TRUE ? "CL_TRUE" : "CL_FALSE");

		cl_bool exec_timeout;
		clGetDeviceInfo(device, CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV, sizeof(cl_bool), &exec_timeout, NULL);
		ALOG_GPUML("  CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV:\t%s", exec_timeout == CL_TRUE ? "CL_TRUE" : "CL_FALSE");

		cl_bool integrated_memory;
		clGetDeviceInfo(device, CL_DEVICE_INTEGRATED_MEMORY_NV, sizeof(cl_bool), &integrated_memory, NULL);
		ALOG_GPUML("  CL_DEVICE_INTEGRATED_MEMORY_NV:\t%s", integrated_memory == CL_TRUE ? "CL_TRUE" : "CL_FALSE");
	}

	ALOG_GPUML("  CL_DEVICE_PREFERRED_VECTOR_WIDTH_<t>\t", false);
	cl_uint vec_width[6];
	clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(cl_uint), &vec_width[0], NULL);
	clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(cl_uint), &vec_width[1], NULL);
	clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(cl_uint), &vec_width[2], NULL);
	clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(cl_uint), &vec_width[3], NULL);
	clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(cl_uint), &vec_width[4], NULL);
	clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &vec_width[5], NULL);
	ALOG_GPUML("CHAR %u, SHORT %u, INT %u, LONG %u, FLOAT %u, DOUBLE %u\n\n\n",
		vec_width[0], vec_width[1], vec_width[2], vec_width[3], vec_width[4], vec_width[5]);
}
