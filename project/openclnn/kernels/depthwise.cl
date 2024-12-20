__kernel void depthwise(__global const float* filter,
					   __global const float* input,
					   __global float* output,
                               const int input_width,
                               const int input_height,
                               const int input_channels,
                               const int stride)
{
    const int output_height = input_height / stride;
    const int output_width = input_width / stride;
    
    const int tid = get_global_id(0);
    const int c = tid % input_channels;
    const int y = (tid / input_channels) % output_height;
    const int x = tid / input_channels / output_height;

	const int stridem2 = stride % 2;
	const int endIndex = 2 - stridem2;	
    
    float sum = 0.0;
    
	for (int k1 = -stridem2; k1 <= endIndex; k1++) {
		for (int l1 = -stridem2; l1 <= endIndex; l1++) {
			const int k = k1 + stridem2;
			const int l = l1 + stridem2;
			const int input_x = x * stride + k1;
			const int input_y = y * stride + l1;
			if(input_x >= 0 && input_x < input_width && input_y >= 0 && input_y < input_height)
			{
				const int input_idx = input_x * input_height * input_channels + input_y * input_channels + c;
				const int filter_idx = (l * 3 + k) * input_channels + c;
				sum += input[input_idx] * filter[filter_idx];
			}
        }
    }    
    output[tid] = sum;
}
