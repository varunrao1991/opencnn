__kernel void convolution(  
    __global const float* filter,
    __global const float* input,
    __global float* output,
	__const int input_width,
	__const int input_height,
	__const int input_channels,
	__const int output_channels,
	__const int stride,
	__const char divisibility,
	__const char reluEnabled)
{
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
	const int iz = get_global_id(2); // goes up to output_channels/16
	
	int output_height = (input_height + stride - 1) / stride;
	const int stridem2 = stride % 2;
	const int endIndex = 2 - stridem2;
		
	if(divisibility)
	{
		float16 sum = 0;
		for(int j=0;j<input_channels;j++)
		{
			for (int k1 = -stridem2; k1 <= endIndex; k1++) {
				for (int l1 = -stridem2; l1 <= endIndex; l1++) {
					const int k = k1 + stridem2;
					const int l = l1 + stridem2;
					const int input_x = ix * stride + k1;
					const int input_y = iy * stride + l1;
					float value;
					if(input_x >= 0 && input_x < input_width && input_y >= 0 && input_y < input_height)
					{
						const int input_idx = input_x * input_height * input_channels + input_y * input_channels + j;
						float16 weight = vload16(iz,filter+(l * 3 + k)*input_channels*output_channels+j*output_channels);
						sum+=input[input_idx] * weight;
					}
				}
			}
		}
		int output_index = ix*output_channels*output_height+output_channels*iy;
		vstore16(reluEnabled?max(0.0f, sum):sum,iz,output+output_index);
	}
	else
	{
		float2 sum = 0;
		for (int k1 = -stridem2; k1 <= endIndex; k1++) {
			for (int l1 = -stridem2; l1 <= endIndex; l1++) {	
			
				float16 values;
				float* value = (float*)&values;

				const int k = k1 + stridem2;
				const int l = l1 + stridem2;
				const int input_x = ix * stride + k1;
				const int input_y = iy * stride + l1;
				
				for(int j=0;j<input_channels;j++)
				{
					if(input_x >= 0 && input_x < input_width && input_y >= 0 && input_y < input_height)
					{
						if(j%16==0){
							const int input_idx = input_x * input_height * input_channels + input_y * input_channels;
							values = vload16(j/16,input+input_idx);
						}
						float2 weight = vload2(iz,filter+(l * 3 + k)*input_channels*output_channels+j*output_channels);
						sum+=value[j%16]*weight;
					}
				}
			}
		}
		int output_index = ix*output_channels*output_height+output_channels*iy;		
		vstore2(reluEnabled?max(0.0f, sum):sum,iz,output+output_index);
	}
}