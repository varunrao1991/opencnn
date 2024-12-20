#include "OpenclWrapper.h"

#include <iostream>

OpenclWrapper::OpenclWrapper() : m_context(nullptr), m_commandQueue(nullptr), m_deviceId(0)
{
	cl_int err;
	cl_platform_id platform_id;
	cl_uint num_platforms;
	cl_device_id device_id;
	cl_uint num_devices;

	err = clGetPlatformIDs(1, &platform_id, &num_platforms);
	if (err != CL_SUCCESS) {
		std::cerr << "Error getting platform IDs: " << err << std::endl;
	}

	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, &num_devices);
	if (err != CL_SUCCESS) {
		std::cerr << "Error getting device IDs: " << err << std::endl;
	}

	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		std::cerr << "Error creating OpenCL context: " << err << std::endl;
	}

	char device_name[1024];
	err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL);
	if (err != CL_SUCCESS) {
		std::cerr << "Error getting device info: " << err << std::endl;
	}
	std::cout << "Device name: " << device_name << std::endl;
	char versionInfo[1024];
	err = clGetDeviceInfo(device_id, CL_DRIVER_VERSION, sizeof(versionInfo), &versionInfo, NULL);
	if (err != CL_SUCCESS) {
		std::cerr << "Error getting device info: " << err << std::endl;
	}

	std::cout << "Version : " << versionInfo << std::endl;

	err = CL_SUCCESS;
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	if (err != CL_SUCCESS) {
		std::cerr << "Error creating command queue: " << err << std::endl;
	}
	m_deviceId = device_id;
	m_context = context;
	m_commandQueue = command_queue;
}

OpenclWrapper::~OpenclWrapper()
{
	if (m_commandQueue) {
		clReleaseCommandQueue(m_commandQueue);
	}
	if (m_context) {
		clReleaseContext(m_context);
	}
}