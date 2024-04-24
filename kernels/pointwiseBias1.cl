__kernel void pointwiseBias1(__global const float* filter, // filter
			__global const float* bias, // bias
			__global const float* input, // input
			__global float* output,
			__const int p, // filter filter shape [1 ,1 , m, n]
			__const int q, // input shape input  [p, q, m]
			__const int m, // Works only for 1X1 convolution
			__const int n, // Works only for 1X1 convolution
			__const char reluEnabled)  // output share output [p, q, n]
{
    const int ixy = get_global_id(0); // goes up to pq
	const int iz = get_global_id(1); // goes up to n
	
	float sum = 0;
	for(int j=0;j<m;j++) // input channels
	{
		sum+=input[ixy*m+j]*filter[j*n+iz];
	}
	sum += bias[iz];
	output[ixy*n+iz] = reluEnabled?max(0.0f, sum):sum;
}