inline float dot16(float16 a, float16 b)
{
	return a.s0*b.s0+ a.s1*b.s1+ a.s2*b.s2+ a.s3*b.s3+ a.s4*b.s4+ a.s5*b.s5+ a.s6*b.s6+ a.s7*b.s7+ a.s8*b.s8+ a.s9*b.s9+ a.sa*b.sa+ a.sb*b.sb+ a.sc*b.sc+ a.sd*b.sd+ a.se*b.se+ a.sf*b.sf;
}
#define VECTOR_SIZE 16

__kernel void deconvolution(  
    __global float* filter,
    __global float* input,
    __global float* output,
	__const int input_width,
	__const int input_height,
	__const int input_channels, 
	__const int output_channels)
{
    const int ix = get_global_id(0); // goes upto input_width
    const int iy = get_global_id(1); // goes upto input_height
    const int iz = get_global_id(2); // goes upto output_channels/16

	int index = ix*input_channels*input_height+iy*input_channels;
	
	if((ix+1)>=input_width || (iy+1)>=input_height)
		return;
	
	int output_height = 2 * input_height;
	int output_width = 2 * input_width;

	for(int k=0;k<4;k++) // Each element in filter
	{
			int temp1=k/2;
			int temp2=k%2;
			float16 sum = 0;
			float* temp = (float*)(&sum);
			for(int ind=0;ind<VECTOR_SIZE;ind++)
			{	
				float locSum = 0;
				for(int im = 0;im < input_channels/VECTOR_SIZE;im++)
				{
					float16 valueFil = vload16(im,filter+k*input_channels*output_channels+input_channels*(iz*VECTOR_SIZE+ind));
					float16 valueIn= vload16(im,input+index);
					locSum +=dot16(valueIn,valueFil);
				}
				*temp = locSum;
				temp++;
			}
			if((2*ix+temp2)>=output_width || (2*iy+temp1)>=output_height)
				continue;
			int indexOut = (2*ix + temp2)*output_channels*output_height+output_channels*(2*iy+temp1);
			vstore16(sum,iz,output+indexOut);
	}
}