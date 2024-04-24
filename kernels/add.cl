	__kernel void add(  
    __global float* src1,
    __global float* src2,
    __global float* dst,
	__const int p, // input1 src1 shape [m, q, p] [channels, height, width]
	__const int q, // input2 src2 shape [m, q, p] [channels, height, width]
	__const int m)  // output share dst [m, q, p] [channels, height, width]
{
	const int ixy = get_global_id(0); // goes up to pq
	const int iz = get_global_id(1); // goes up to m/4
	
	vstore4(vload4(iz,src1+ixy*m) + vload4(iz,src2+ixy*m),iz, dst+ixy*m);
}