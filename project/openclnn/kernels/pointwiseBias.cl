__kernel void pointwiseBias(__global const float* filter, // filter
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
	const int iz = get_global_id(1); // goes up to n/2
	
	float2 sum = 0;
	for(int j=0;j<m;j++) // input channels
	{
		sum+=input[ixy*m+j]*vload2(iz,filter+j*n);
	}
	sum += vload2(iz,bias);
	vstore2(reluEnabled?max(0.0f, sum):sum,iz,output+ixy*n);
}

__kernel void pointwise2(__global const float* filter, // filter
			__global const float* bias, // bias
			__global const float* input, // input
			__global float* output,
			__const int p, // filter filter shape [1 ,1 , m, n]
			__const int q, // input shape input  [p, q, m]
			__const int m, // Works only for 3X3 convolution
			__const int n)  // output share output [p, q, n]
{
    const int ix = get_global_id(0); // goes up to p
    const int iy = get_global_id(1); // goes up to q
	const int iz = get_global_id(2); // goes up to n/2
	
	float2 sum = 0;
	for(int j=0;j<m;j++) // input channels
	{
		sum+=input[ix*m*q+iy*m+j]*vload2(iz,filter+j*n);
	}
	sum += vload2(iz,bias);
	vstore2(sum,iz,output+ix*n*q+n*iy);
}

__kernel void pointwise4(__global const float* filter, // filter
			__global const float* bias, // bias
			__global const float* input, // input
			__global float* output,
			__const int p, // filter filter shape [1 ,1 , m, n]
			__const int q, // input shape input  [p, q, m]
			__const int m, // Works only for 3X3 convolution
			__const int n)  // output share output [p, q, n]
{
    const int ix = get_global_id(0); // goes up to p
    const int iy = get_global_id(1); // goes up to q
	const int iz = get_global_id(2); // goes up to n/4
	
	float4 sum = 0;
	for(int j=0;j<m;j++) // input channels
	{
		sum+=input[ix*m*q+iy*m+j]*vload4(iz,filter+j*n);
	}
	sum += vload4(iz,bias);
	vstore4(sum,iz,output+ix*n*q+n*iy);
}

__kernel void pointwise16(__global const float* filter, // filter
			__global const float* bias, // bias
			__global const float* input, // input
			__global float* output,
			__const int p, // filter filter shape [1 ,1 , m, n]
			__const int q, // input shape input  [p, q, m]
			__const int m, // Works only for 3X3 convolution
			__const int n)  // output share output [p, q, n]
{
    const int ix = get_global_id(0); // goes up to p
    const int iy = get_global_id(1); // goes up to q
	const int iz = get_global_id(2); // goes up to n/16
	
	float16 sum = 0;
	for(int j=0;j<m;j++) // input channels
	{
		sum+=input[ix*m*q+iy*m+j]*vload16(iz,filter+j*n);
	}
	sum += vload16(iz,bias);
	vstore16(sum,iz,output+ix*n*q+n*iy);
}