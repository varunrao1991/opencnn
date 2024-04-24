__kernel void batchnormalization1(  
    __global const float* bias,
    __global const float* gamma,
    __global const float* input,
    __global float* output,
	__const int input_width,
	__const int input_height,
	__const int channels)
{
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
	const int iz = get_global_id(2); // goes up to channels

	const int index = ix * input_height * channels + iy * channels + iz;
	float valueIn = input[index];
	valueIn *= gamma[iz];
	valueIn += bias[iz];
	output[index] = valueIn;
}