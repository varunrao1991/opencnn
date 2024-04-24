__kernel void batchnormalization(  
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
	const int iz = get_global_id(2); // goes up to channels/8

	const int input_idx = ix * input_height * channels + iy * channels;
	float8 valueIn = vload8(iz, input + input_idx);
	valueIn *= vload8(iz,gamma);
	valueIn += vload8(iz,bias);
	int output_index = ix*channels*input_height+channels*iy;
	vstore8(valueIn,iz,output+output_index);
}