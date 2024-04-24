__kernel void relu( 
    __global float* src,
    __global float* dst,
	__const int r,
	__const float maxVlaue)  // input1 src shape [p, q, r]
				    // input2 dst shape  [p, q, r]
					// input2 dst shape  [p, q, r]
{
    const int ix = get_global_id(0); // goes up to p*q/16
	if (maxVlaue > 0)
	{
		vstore16(min(maxVlaue, max(0.0f, vload16(ix,src))),ix,dst);	
	}else{
		vstore16(max(0.0f, vload16(ix,src)),ix,dst);		
	}
}
