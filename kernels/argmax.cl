__kernel void argmax1(  
    __global float* src,
    __global unsigned char* dst,
	__const int p,
	__const int q,
    __const int num_channels)
{
	const int ix = get_global_id(0); // goes up to p
	const int iy = get_global_id(1); // goes up to q
	dst[ix+iy*p] = convert_uchar_rtz(( 1 + src[ix*2*q+iy*2+1]-src[ix*2*q+iy*2+0]) * 128.0f);
}

__kernel void argmax(
    __global float* src,
    __global unsigned char* dst,
    __const int p,
    __const int q,
    __const int num_channels)
{
    const int ix = get_global_id(0); // goes up to p*q
    const int iy = get_global_id(1); // goes up to q
    const int channel_offset = ix * q + iy; // offset for each channel
    float maxValue = 0;
	float maxIndex = 0;
    for (int channel = 0; channel < num_channels; channel++) {
        const int src_offset = (channel_offset * num_channels) + channel;
		float currentValue = src[src_offset];
		if(currentValue > maxValue)
		{
			maxValue = currentValue;
			maxIndex = channel;
		}
	}
	const int dst_offset = ix + iy * p;
	dst[dst_offset] = convert_uchar_rtz(maxIndex * 255.0f / num_channels);
}
