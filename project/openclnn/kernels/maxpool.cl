__kernel void maxpool(  // Boundary case wont come in maxpool as p and q always divisible by 2
    __global float* src,
    __global float* dst,
	__const int p, // input1 src shape [p, q, r]
	__const int q, // input2 dst shape  [p/2, q/2, r]
	__const int r, // input2 dst shape  [p/2, q/2, r]
	__const int stride)
{
    const int ix = get_global_id(0); // goes upto p/2
    const int iy = get_global_id(1); // goes upto q/2
    const int iz = get_global_id(2); // goes upto r/16
	
	#if 0
		float a, b, c, d;
		int w, x, y, z;
		int qr = q*r;
		int qrByStride = q*r/stride;
		w = (ix*stride)*qr+(iy*stride)*r + iz;
		x = (ix*stride)*qr+(iy*stride+1)*r + iz;
		y = (ix*stride+1)*qr+(iy*stride)*r + iz;
		z = (ix*stride+1)*qr+(iy*stride+1)*r + iz;
		
		if((ix*stride+1)>=p && (iy*stride+1)>=q)
		{
			a = src[w];
			dst[ix*qrByStride+iy*r+iz] = a;
		}
		else if((iy*stride+1)>=q)
		{
			a = src[w];
			b = src[y];
			dst[ix*qrByStride+iy*r+iz] = max(a,b);
		}
		else if((ix*stride+1)>=p)
		{
			a = src[w];
			b = src[x];
			dst[ix*qrByStride+iy*r+iz] = max(a,b);
		}
		else
		{
			a = src[w];
			b = src[x];
			c = src[y];
			d = src[z];
			dst[ix*qrByStride+iy*r+iz] = max(max(a,b),max(c,d));
		}
	#else
		int w, x, y, z;
		int qr = q*r;
		int qrByStride = q*r/stride;
		w = (ix*stride)*qr+(iy*stride)*r;
		x = (ix*stride)*qr+(iy*stride+1)*r;
		y = (ix*stride+1)*qr+(iy*stride)*r;
		z = (ix*stride+1)*qr+(iy*stride+1)*r;
		
		if((ix*stride+1)>=p && (iy*stride+1)>=q)
		{
			float16 a;	
			a = vload16(iz,src+w);
			vstore16(a,iz,dst+ix*qrByStride+iy*r);
		}
		else if((iy*stride+1)>=q)
		{
			float16 a,b,c;				
			a = vload16(iz,src+w);
			b = vload16(iz,src+y);
			c = max(a,b);
			vstore16(c,iz,dst+ix*qrByStride+iy*r);
		}
		else if((ix*stride+1)>=p)
		{
			float16 a,b,c;							
			a = vload16(iz,src+w);
			b = vload16(iz,src+x);
			c = max(a,b);
			vstore16(c,iz,dst+ix*qrByStride+iy*r);
		}
		else
		{
			float16 a,b,c,d,e;					
			a = vload16(iz,src+w);
			b = vload16(iz,src+x);
			c = vload16(iz,src+y);
			d = vload16(iz,src+z);
			e = max(max(a,b),max(c,d));
			vstore16(e,iz,dst+ix*qrByStride+iy*r);
		}		
	#endif
}