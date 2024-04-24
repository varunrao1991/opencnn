__kernel void preprocess( 
    __global float* src,
    __global float* dst,
	__const unsigned int r)
{
    const int ixy = get_global_id(0); //pq
    const int iz = get_global_id(1);
	const int index = ixy * r + iz;
	vstore2(vload2(iz, src + ixy * r), iz, dst + ixy * r);	
}

__kernel void preprocess1( 
    __global float* src,
    __global float* dst)  // input1 src shape [p, q, r]
				    // input2 dst shape  [p, q, r]
					// input2 dst shape  [p, q, r]
{
    const int ixyz = get_global_id(0); // goes up to p*q*r/16
	//vstore16((vload16(ixyz,src) / 127.5f)-1.0f ,ixyz,dst);	
	//vstore16(vload16(ixyz,src)* 0.0f + 1.0f,ixyz,dst);	
	vstore16(vload16(ixyz,src),ixyz,dst);	
}