#pragma once

#include "CL/cl.h"

class OpenclWrapper
{
public:
	explicit OpenclWrapper();
	virtual ~OpenclWrapper();

	cl_context m_context;
	cl_command_queue m_commandQueue;
	cl_device_id m_deviceId;
};

