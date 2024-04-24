	__kernel void concatenation(  
    __global float* src1,
    __global float* src2,
    __global float* dst,
	__const int p, // input1 src1 shape [m, q, p] [channels, height, width]
	__const int q, // input2 src2 shape [n, q, p] [channels, height, width]
	__const int m, 
	__const int n)  // output share dst [m+n, q, p] [channels, height, width]; m should be always divisible by 16 n is either 3 or divisible by 16
{
	const int m_n = m+n;	
	const int ixy = get_global_id(0); // goes up to p*q
	const int iz = get_global_id(1); // goes up to (m+n)
	
	if(iz < m)
	{
		dst[ixy * m_n + iz] = src1[ixy * m + iz];
	}
	else
	{
		dst[ixy * m_n + iz] = src2[ixy * n + iz-m];
	}
}