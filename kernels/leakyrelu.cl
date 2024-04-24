__kernel void leakyrelu( 
    __global float* src,
    __global float* dst,
	__const int r,
	__const float leakyValue)  // input1 src shape [p, q, r]
				    // input2 dst shape  [p, q, r]
					// input2 dst shape  [p, q, r]
{
    const int ix = get_global_id(0); // goes up to p*q/16
	vstore16(max(leakyValue * vload16(ix,src), vload16(ix,src)),ix,dst);		
}
