inline float dot16(float16 a, float16 b)
{
	return a.s0*b.s0+ a.s1*b.s1+ a.s2*b.s2+ a.s3*b.s3+ a.s4*b.s4+ a.s5*b.s5+ a.s6*b.s6+ a.s7*b.s7+ a.s8*b.s8+ a.s9*b.s9+ a.sa*b.sa+ a.sb*b.sb+ a.sc*b.sc+ a.sd*b.sd+ a.se*b.se+ a.sf*b.sf;
}
__kernel void deconvolution3x3(  
    __global float* src1, //filter
    __global float* src2, // input
    __global float* dst, // output share dst [2*p, 2*q, m]
	__const int p, // filter src2 shape [l ,l ,m, r] ([3,3,m,r]
	__const int q, // input shape src1  [p, q, m]
	__const int m, 
	__const int r,
	__const int run) // Works only for 3X3 convolution  
{
    const int ix = get_global_id(0); // goes upto p/2
    const int iy = get_global_id(1); // goes upto q/2
    const int iz = get_global_id(2); // goes upto r/16

	// Assuming filter passed is rearranged to 4*4 i.e. src2 with zero padded

	float value;
	int index;
	int indexOut;

	if(run==0)
		index = (ix*2)*m*q+(iy*2)*m;
	else if(run==1)
		index = (ix*2)*m*q+(iy*2+1)*m;
	else if(run==2)
		index = (ix*2+1)*m*q+(iy*2)*m;
	else
		index = (ix*2+1)*m*q+(iy*2+1)*m;

	if((ix*2+1)>=p || (iy*2+1)>=q)
		return;

	for(int k=0;k<9;k++) // Each element in filter
	{
			int temp1=k%3;
			int temp2=k/3;
			float16 sum = 0;
			float* temp = (float*)(&sum);
			for(int ind=0;ind<16;ind++) // Each element in filter
			{	
				float locSum = 0;
				for(int im = 0;im < m/16;im++) // each input channel
				{
					float16 valueIn= vload16(im,src2+index);
					float16 valueFil = vload16(im,src1+k*m*r+m*(iz*16+ind)); // filter
					locSum +=dot16(valueIn,valueFil);
				}
				*temp = locSum;
				temp++;
			}

			if(run==0)
			{
				if((4*ix+temp2)>=(2*p) || (4*iy+temp1)>=(2*q))
					continue;
				indexOut = (4*ix+temp2)*r*2*q+r*(4*iy+temp1);
				vstore16(sum,iz,dst+indexOut);
			}			
			else if(run==1)
			{
				if((4*ix+temp2)>=(2*p) || (4*iy+2+temp1)>=(2*q))
					continue;
				indexOut = (4*ix+temp2)*r*2*q+r*(4*iy+2+temp1);
				vstore16(sum+vload16(iz,dst+indexOut),iz,dst+indexOut);
			}	
			else if(run==2)
			{
				if((4*ix+2+temp2)>=(2*p) || (4*iy+temp1)>=(2*q))
					continue;
				indexOut = (4*ix+2+temp2)*r*2*q+r*(4*iy+temp1);
				vstore16(sum+vload16(iz,dst+indexOut),iz,dst+indexOut);
			}	
			else
			{
				if((4*ix+2+temp2)>=(2*p) || (4*iy+2+temp1)>=(2*q))
					continue;
				indexOut = (4*ix+2+temp2)*r*2*q+r*(4*iy+2+temp1);
				vstore16(sum+vload16(iz,dst+indexOut),iz,dst+indexOut);
			}
	}
}