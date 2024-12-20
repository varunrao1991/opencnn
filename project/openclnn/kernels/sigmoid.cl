__kernel void sigmoid( 
    __global float* src,
    __global float* dst,
	__const int r)  // input1 src shape [p, q, r]
				    // input2 dst shape  [p, q, r]
{
    const int ix = get_global_id(0); // goes up to p*q*r/16
	float16 a; 
	a = vload16(ix,src);
	float16 c;
	#if 1
		c.s0 = 1.0f/(1.0f+exp(-a.s0));
		c.s1 = 1.0f/(1.0f+exp(-a.s1));
		c.s2 = 1.0f/(1.0f+exp(-a.s2));
		c.s3 = 1.0f/(1.0f+exp(-a.s3));
		c.s4 = 1.0f/(1.0f+exp(-a.s4));
		c.s5 = 1.0f/(1.0f+exp(-a.s5));
		c.s6 = 1.0f/(1.0f+exp(-a.s6));
		c.s7 = 1.0f/(1.0f+exp(-a.s7));
		c.s8 = 1.0f/(1.0f+exp(-a.s8));
		c.s9 = 1.0f/(1.0f+exp(-a.s9));
		c.sa = 1.0f/(1.0f+exp(-a.sa));
		c.sb = 1.0f/(1.0f+exp(-a.sb));
		c.sc = 1.0f/(1.0f+exp(-a.sc));
		c.sd = 1.0f/(1.0f+exp(-a.sd));
		c.se = 1.0f/(1.0f+exp(-a.se));
		c.sf = 1.0f/(1.0f+exp(-a.sf));
	#else
		c = 1.0f/(1.0f+exp(-a));
	#endif
	vstore16(c,ix,dst);
}